"""
recommender_service.py

Self-contained helper layer around an **implicit ALS** model trained on
(log-scaled, confidence-weighted) play-count data.

Core features
-------------
* add_user(...)            - create a brand-new user (with or without history)
* update_user_profile(...) - append / overwrite plays for an existing user
* recommend(...)           - top-N song IDs (empty list + flag if no data)
* All artefacts (encoders, CSR matrix, ALS model) auto-persist after every
  change so a container restart never loses state.

Assumptions
-----------
* item catalog is frozen; new songs are ignored with a log entry.
* α (confidence scaling) is fixed at 40, but can be overridden per instance.
* We partial_fit ALS **every time** the matrix changes.

Author : <you>
Date   : 2025-07-10
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import os
from dotenv import load_dotenv


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack, lil_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import LabelEncoder


# ── Config  ──────────────────
BASE_DIR    = Path(__file__).resolve().parent
DOTENV_PATH = BASE_DIR / ".envs" / ".data_paths"
load_dotenv(dotenv_path=str(DOTENV_PATH))


USER_ENC_PATH = BASE_DIR / os.getenv("COLLABORATIVE-FILTERING-USER-ENCODER", "")
ITEM_ENC_PATH = BASE_DIR /os.getenv("COLLABORATIVE-FILTERING-ITEM-ENCODER", "") # immutable
MATRIX_PATH = BASE_DIR / os.getenv("COLLABORATIVE-FILTERING-USER-ITEM-MATRIX", "") # CSR    
MODEL_PATH = BASE_DIR / os.getenv("COLLABORATIVE-FILTERING-ALS-MODEL", "")
MUSIC_INFO_PATH = BASE_DIR / os.getenv("COLLABORATIVE-FILTERING-MUSIC-INFO-TRIMMED", "")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("RecommenderService")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _ensure_label_present(encoder: LabelEncoder, label: str) -> int:
    """
    Add `label` to a fitted sklearn LabelEncoder **in-place** if missing,
    return its integer index.
    """
    if label in encoder.classes_:
        return int(np.where(encoder.classes_ == label)[0][0])

    # Extend classes_
    encoder.classes_ = np.append(encoder.classes_, label)
    return len(encoder.classes_) - 1


def _save_pickle(obj, path: Path):
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


# ── Main class ───────────────────────────────────────────────────────────────
class RecommenderService:
    """
    Wrapper around an implicit ALS model with add/update/recommend APIs.
    """

    # --------------------------------------------------------------------- #
    def __init__(
        self,
        *,
        alpha: float = 40.0,
        factors: int = 64,
        regularization: float = 0.1,
        iterations: int = 20,
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state

        # Load artifacts or bail if something crucial is missing
        self.user_enc: LabelEncoder = self._load_encoder(USER_ENC_PATH)
        self.item_enc: LabelEncoder = self._load_encoder(ITEM_ENC_PATH, must_exist=True)
        self.matrix: csr_matrix = self._load_matrix(MATRIX_PATH, must_exist=True)

        self.model = self._maybe_load_model()
        self.music_info: Optional[pd.DataFrame] = None
        if MUSIC_INFO_PATH.exists():
            self.music_info = pd.read_parquet(MUSIC_INFO_PATH)

        log.info(
            "RecommenderService initialised "
            f"(users={self.matrix.shape[0]}, items={self.matrix.shape[1]})"
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_user(
        self,
        user_id: str,
        history: Optional[List[Tuple[str, int]]] = None,
    ) -> None:
        """
        Register a brand-new user.  `history` is an optional list of
        (spotify_id, playcount).  If history is empty, we’ll create an all-zero
        row and persist, but **won’t** retrain until real data arrives.
        """
        if user_id in self.user_enc.classes_:
            raise ValueError(
                f"user_id {user_id!r} already exists - " "use update_user_profile"
            )

        user_idx = _ensure_label_present(self.user_enc, user_id)

        # ---------------------------------------------------------------- #
        # 1. Expand the CSR matrix with an empty row
        # ---------------------------------------------------------------- #
        empty_row = csr_matrix((1, self.matrix.shape[1]), dtype=np.float32)
        self.matrix = vstack([self.matrix, empty_row], format="csr")

        # ---------------------------------------------------------------- #
        # 2. If history provided, write it immediately
        # ---------------------------------------------------------------- #
        if history:
            self._apply_history(user_idx, history)

        # ---------------------------------------------------------------- #
        # 3. Persist & optionally update model
        # ---------------------------------------------------------------- #
        if history:
            self._incremental_fit_users([user_idx])
            self._persist_all(save_model=True)        # save encoders, matrix, model
        else:
            self._persist_all(save_model=False)       # no model change yet

        log.info(
            "Added user %s (index=%d, interactions=%d)",
            user_id,
            user_idx,
            len(history or []),
        )

    # ------------------------------------------------------------------ #
    def update_user_profile(
        self,
        user_id: str,
        history: List[Tuple[str, int]],
        overwrite: bool = False,
    ) -> None:
        """
        Append or overwrite `(spotify_id, playcount)` records for an existing
        user.  If `overwrite=True` we zero-out previous interactions first.
        """
        if user_id not in self.user_enc.classes_:
            raise ValueError(
                f"user_id {user_id!r} does not exist – " "call add_user first"
            )

        if not history:
            log.warning(
                "update_user_profile called with empty history; " "nothing done."
            )
            return

        user_idx = int(self.user_enc.transform([user_id])[0])

        if overwrite:
            self.matrix[user_idx] = 0.0

        self._apply_history(user_idx, history)
        self._incremental_fit_users([user_idx])      
        self._persist_all(save_model=True)            # encoders + matrix + model
        
        log.info(
            "Updated user %s (index=%d) with %d interactions " "(overwrite=%s)",
            user_id,
            user_idx,
            len(history),
            overwrite,
        )

    # ------------------------------------------------------------------ #
    def recommend(
        self,
        user_id: str,
        n: int = 10,
    ) -> Tuple[List[str], bool]:
        """
        Return up to `n` Spotify IDs recommended for `user_id`.

        Returns
        -------
        (recommended_ids, insufficient_data_flag)
        """
        if user_id not in self.user_enc.classes_:
            raise ValueError(f"user_id {user_id!r} not found")

        user_idx = int(self.user_enc.transform([user_id])[0])

        if self.matrix[user_idx].nnz == 0:
            log.info("User %s has no interactions – returning empty list", user_id)
            return [], True  # insufficient data

        rec_idx, _ = self.model.recommend(
            userid=user_idx,
            user_items=self.matrix[user_idx],
            N=n,
            filter_already_liked_items=True,
        )
        rec_ids = self.item_enc.inverse_transform(rec_idx)
        return rec_ids.tolist(), False

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _load_encoder(self, path: Path, *, must_exist: bool = False) -> LabelEncoder:
        if path.exists():
            return _load_pickle(path)
        if must_exist:
            raise FileNotFoundError(path)
        # Fresh, unfitted encoder
        enc = LabelEncoder()
        enc.classes_ = np.array([], dtype="<U32")  # empty but initialised
        return enc

    # ------------------------------------------------------------------ #
    def _load_matrix(self, path: Path, *, must_exist: bool = False) -> csr_matrix:
        if path.exists():
            return _load_pickle(path)
        if must_exist:
            raise FileNotFoundError(path)
        # Make an empty matrix with correct number of items
        return csr_matrix((0, len(self.item_enc.classes_)), dtype=np.float32)

    # ------------------------------------------------------------------ #
    def _maybe_load_model(self) -> AlternatingLeastSquares:
        if MODEL_PATH.exists():
            return _load_pickle(MODEL_PATH)
        return self._train_model()  # first run

    # ------------------------------------------------------------------ #
    def _compute_conf(self, playcount: int) -> float:
        return 1.0 + self.alpha * np.log1p(playcount)

    # ------------------------------------------------------------------ #
    def _apply_history(self, user_idx: int, history: List[Tuple[str, int]]) -> None:
        """
        Mutates `self.matrix` in-place: adds (or sets) confidence values
        derived from `history`.
        """
        # Prepare mutable view
        lil: lil_matrix = self.matrix.tolil()

        for spotify_id, plays in history:
            if spotify_id not in self.item_enc.classes_:
                log.warning(
                    "Song %s not in catalog – ignored " "(user_idx=%d)",
                    spotify_id,
                    user_idx,
                )
                continue

            item_idx = int(self.item_enc.transform([spotify_id])[0])
            conf_val = self._compute_conf(plays)
            lil[user_idx, item_idx] = conf_val

        self.matrix = lil.tocsr()

    # ------------------------------------------------------------------ #
    def _incremental_fit_users(self, user_indices: List[int]) -> None:
        """
        Update the ALS model **just for the given users** instead of a full
        retrain.  Requires implicit ≥0.6 where `partial_fit_users` exists.
        """
        sub_matrix = self.matrix[user_indices]            # shape: (k, n_items)
        self.model.partial_fit_users(user_indices, sub_matrix)

        _save_pickle(self.model, MODEL_PATH)


    # ------------------------------------------------------------------ #
    def _train_model(self) -> AlternatingLeastSquares:
        log.info(
            "Training ALS (factors=%d, iters=%d, reg=%.3f)",
            self.factors,
            self.iterations,
            self.regularization,
        )
        model = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=self.random_state,
        )
        model.fit(self.matrix) #TODO:
        return model

    # ------------------------------------------------------------------ #
    def _persist_all(self, *, save_model: bool) -> None:
        _save_pickle(self.user_enc, USER_ENC_PATH)
        _save_pickle(self.matrix, MATRIX_PATH)
        if save_model:
            _save_pickle(self.model, MODEL_PATH)
        log.debug("Artefacts persisted")

    # ------------------------------------------------------------------ #
    # pretty-print recommendations with metadata
    # ------------------------------------------------------------------ #
    def pretty_recommend(
        self,
        user_id: str,
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Same as `recommend` but returns a DataFrame joined with `music_info`
        (if available) for human inspection.
        """
        rec_ids, flag = self.recommend(user_id, n)
        if flag or not self.music_info is not None:
            return pd.DataFrame({"spotify_id": rec_ids})
        return self.music_info.set_index("spotify_id").loc[rec_ids].reset_index()

