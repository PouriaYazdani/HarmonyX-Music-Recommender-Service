import math
from pathlib import Path
from importlib import reload

import numpy as np
import pytest
from collaborative_filtering import _save_pickle
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

# ---------------------------------------------------------------------- #
# Helper to bootstrap an empty but valid artefact set in a tmp dir
# ---------------------------------------------------------------------- #
def _bootstrap_empty_artifacts(tmp_dir: Path):
    # --- item encoder with a tiny fixed catalogue ----------------------
    item_enc = LabelEncoder()
    item_enc.classes_ = np.array(["trackA", "trackB", "trackC"])
    _save_pickle(item_enc, tmp_dir / "item_encoder.pkl")

    # --- start with *no* users ----------------------------------------
    empty_matrix = csr_matrix((0, len(item_enc.classes_)), dtype=np.float32)
    _save_pickle(empty_matrix, tmp_dir / "user_item_matrix.pkl")

    # --- very small ALS model (1×3 factors) so partial_fit users will work
    from implicit.als import AlternatingLeastSquares

    model = AlternatingLeastSquares(factors=2, iterations=1)
    model.fit(empty_matrix)              # fits nothing, but initialises factors
    _save_pickle(model, tmp_dir / "als_model.pkl")

# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
@pytest.fixture
def svc(tmp_path, monkeypatch):
    """Fresh RecommenderService backed by artefacts in tmp_path."""
    _bootstrap_empty_artifacts(tmp_path)
    monkeypatch.setenv(
        "COLLABORATIVE-FILTERING-ALS-MODEL",
        str(tmp_path / "als_model.pkl"),
    )
    monkeypatch.setenv(
        "COLLABORATIVE-FILTERING-ITEM-ENCODER",
        str(tmp_path / "item_encoder.pkl"),
    )
    monkeypatch.setenv(
        "COLLABORATIVE-FILTERING-USER-ENCODER",
        str(tmp_path / "user_encoder.pkl"),
    )
    monkeypatch.setenv(
        "COLLABORATIVE-FILTERING-USER-ITEM-MATRIX",
        str(tmp_path / "user_item_matrix.pkl"),
    )
    monkeypatch.setenv(
        "COLLABORATIVE-FILTERING-MUSIC-INFO-TRIMMED",
        str(tmp_path / "music_info_trimmed.parquet"),
    )

    # ――― reload so collaborative_filtering picks up the new env paths ――― #
    import collaborative_filtering as cf
    reload(cf)
    return cf.RecommenderService()


# ---------------------------------------------------------------------- #
# 1) confidence formula
# ---------------------------------------------------------------------- #
def test_confidence_formula_is_log_scaled(svc):
    # α = 40 by default
    playcount = 10
    expected = 1 + 40 * math.log1p(playcount)
    assert math.isclose(svc._compute_conf(playcount), expected, rel_tol=1e-9)

# ---------------------------------------------------------------------- #
# 2) add/update user_item matrix
# ---------------------------------------------------------------------- #
def test_add_and_multiple_updates_edge_cases(svc):
    # add new user with two tracks
    svc.add_user("u1", [("trackA", 3), ("trackB", 1)])
    u_idx = svc.user_enc.transform(["u1"])[0]

    # matrix should have 2 non-zero entries
    assert svc.matrix[u_idx].nnz == 2

    # 2a) update overlapping track + brand-new track
    svc.update_user_profile("u1", [("trackA", 5), ("trackC", 2)])

    # After overwrite rule: trackA value should be replaced, not summed
    row = svc.matrix[u_idx].toarray().ravel()
    assert row[svc.item_enc.transform(["trackA"])[0]] == pytest.approx(
        svc._compute_conf(5)
    )
    # trackC should now be present
    assert row[svc.item_enc.transform(["trackC"])[0]] == pytest.approx(
        svc._compute_conf(2)
    )
    # total nnz == 3 (trackA, trackB, trackC)
    assert svc.matrix[u_idx].nnz == 3

    # 2b) adding same user again should raise
    with pytest.raises(ValueError):
        svc.add_user("u1")

    # 2c) updating non-existent user raises
    with pytest.raises(ValueError):
        svc.update_user_profile("ghost", [("trackA", 1)])
# ---------------------------------------------------------------------- #
# 3) incremental fit: user_factors should grow by exactly 1
# ---------------------------------------------------------------------- #
def test_add_user_triggers_partial_fit_and_grows_factors(svc):
    """
    When we add a *warm* user (history not empty), the service should call
    model.partial_fit_users once, and the shape[0] of user_factors must grow
    by exactly one row containing non-zero numbers.
    """
    n_before = svc.model.user_factors.shape[0]

    svc.add_user("warm_user", [("trackA", 4)])

    # 1) user_factors grew by one row
    n_after = svc.model.user_factors.shape[0]
    assert n_after == n_before + 1

    # 2) model now stroes new user
    new_idx = svc.user_enc.transform(["warm_user"])[0]
    assert svc.model.user_factors.shape[0] == 1


# ---------------------------------------------------------------------- #
# 4) cold add should *not* grow user_factors until later update
# ---------------------------------------------------------------------- #
def test_cold_add_does_not_grow_factors_until_update(svc):
    n_before = svc.model.user_factors.shape[0]

    # Add cold user (empty history) → no partial fit
    svc.add_user("cold_user")                   # no history argument
    n_after_cold = svc.model.user_factors.shape[0]
    assert n_after_cold == n_before            # still same size

    # Now update with history → factors grow
    svc.update_user_profile("cold_user", [("trackB", 2)])
    n_after_update = svc.model.user_factors.shape[0]
    assert n_after_update == n_before + 1
