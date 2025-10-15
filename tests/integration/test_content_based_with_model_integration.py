from pathlib import Path
from importlib import reload
import pandas as pd
import os
import shutil
from dotenv import load_dotenv
import pickle
import numpy as np
import pytest
import warnings
from fastapi.testclient import TestClient
from unittest.mock import patch
import sys
parent_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_model_dir)
parent_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
sys.path.append(parent_data_dir)
import content_based_with_model
import recommender_service

EXISTING_SPOTIFY_ID = "0319szB8GkRiIoUicj0p4h"  # Known ID from collaborative tests
NEW_SPOTIFY_ID = "0Aau2Ju1RoLAhL90zRcDx4"  # Non-existing ID for testing

# ---------------------------------------------------------------------- #
# Helpers to build minimal artefacts identical to unit-test fixture
# ---------------------------------------------------------------------- #

def _copy_real_artifacts(tmp_dir: Path):
    """
    Read .envs/.data_paths, copy each real artefact into tmp_dir, preserving
    file names. The integration tests then monkey-patch paths to tmp_dir
    so all reads/writes happen on the copy.
    """
    load_dotenv(".envs/.data_paths")

    env_to_name = {
        "CONTENT-BASED-WITH-MODEL-EMBEDDINGS_FILE": "tracks_embeddings_similars.pkl",
        "METADATA_FILE": "Music_Info_trimmed.csv",
        "VGGISH_MODEL_PATH": "vggish_model",
        "TRACKS_FILES_PATH": "tracks",
    }

    for env_var, fname in env_to_name.items():
        src = Path(os.getenv(env_var))
        if not src or not src.exists():
            raise RuntimeError(f"{env_var} not set or file missing: {src}")
        dst = tmp_dir / fname
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    # Create a placeholder .mp3 file for NEW_SPOTIFY_ID
    tracks_dir = tmp_dir / "tracks"
    tracks_dir.mkdir(exist_ok=True)
    (tracks_dir / f"{NEW_SPOTIFY_ID}.mp3").touch()

# ---------------------------------------------------------------------- #
@pytest.fixture
def api_client(tmp_path, monkeypatch):
    # Suppress deprecation warnings for audioread
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="audioread.*")
    _copy_real_artifacts(tmp_path)
    monkeypatch.setenv(
        "CONTENT-BASED-WITH-MODEL-EMBEDDINGS_FILE",
        str(tmp_path / "tracks_embeddings_similars.pkl"),
    )
    monkeypatch.setenv(
        "METADATA_FILE",
        str(tmp_path / "Music_Info_trimmed.csv"),
    )
    monkeypatch.setenv(
        "VGGISH_MODEL_PATH",
        str(tmp_path / "vggish_model"),
    )
    monkeypatch.setenv(
        "TRACKS_FILES_PATH",
        str(tmp_path / "tracks"),
    )

    # Mock audio processing to avoid real VGGish model and librosa
    with patch("content_based_with_model.librosa.load") as mock_load:
        with patch("content_based_with_model.vggish_model") as mock_vggish:
            # Mock librosa.load to return a dummy audio array
            mock_load.return_value = (np.zeros(16000 * 1), 16000)  # 1-second audio at 16kHz
            # Mock VGGish to return a predictable embedding
            mock_vggish.return_value = np.ones((1, 128))  # Dummy 128-dim embedding
            reload(content_based_with_model)
            reload(recommender_service)

            from recommender_service import app
            return TestClient(app)

# ---------------------------------------------------------------------- #
# Test 1: Spotify ID exists in tracks_embeddings_similars.pkl
# ---------------------------------------------------------------------- #
def test_existing_spotify_id_recommendations(api_client):
    """
    Test recommendations for a Spotify ID that exists in tracks_embeddings_similars.pkl.
    """
    r = api_client.get(
        "/content-based/with-model/recommendations",
        params={"spotify_id": EXISTING_SPOTIFY_ID, "tags": []}
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["recommendations"]) == 15  # Expect 15 recommendations
    assert all(isinstance(sid, str) for sid in body["recommendations"])

    # Test recommendations_names endpoint
    r = api_client.get(
        "/content-based/with-model/recommendations_names",
        params={"spotify_id": EXISTING_SPOTIFY_ID, "tags": []}
    )
    assert r.status_code == 200
    pairs = r.json()
    assert len(pairs) == 15
    assert all("name" in p and "artist" in p for p in pairs)

# ---------------------------------------------------------------------- #
# Test 2: Spotify ID does not exist in tracks_embeddings_similars.pkl
# ---------------------------------------------------------------------- #
def test_new_spotify_id_recommendations(api_client):
    """
    Test recommendations for a new Spotify ID, ensuring it processes the audio file.
    """
    r = api_client.get(
        "/content-based/with-model/recommendations",
        params={"spotify_id": NEW_SPOTIFY_ID, "tags": []}
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["recommendations"]) == 15  # Expect 15 recommendations
    assert all(isinstance(sid, str) for sid in body["recommendations"])

# ---------------------------------------------------------------------- #
# Test 3: Verify tracks_embeddings_similars.pkl is updated correctly
# ---------------------------------------------------------------------- #
def test_new_spotify_id_updates_pkl(api_client, tmp_path):
    """
    Test that a new Spotify ID updates tracks_embeddings_similars.pkl with correct data.
    """
    # Make recommendation request for new ID
    r = api_client.get(
        "/content-based/with-model/recommendations",
        params={"spotify_id": NEW_SPOTIFY_ID, "tags": []}
    )
    assert r.status_code == 200
    recommendations = r.json()["recommendations"]
    assert len(recommendations) == 15

    # Load the updated .pkl file
    pkl_path = tmp_path / "tracks_embeddings_similars.pkl"
    with pkl_path.open("rb") as f:
        tracks_embeddings_similars = pickle.load(f)

    # Verify the last row
    last_entry = tracks_embeddings_similars[-1]
    spotify_id, embedding, similars = last_entry
    assert spotify_id == NEW_SPOTIFY_ID
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (128,)  # 128-dimensional embedding
    # assert np.allclose(embedding, np.ones(128))  # Matches mocked VGGish output
    assert similars == recommendations  # Same as returned recommendations
    assert len(similars) == 15

    # Verify similar IDs exist in metadata
    metadata = pd.read_csv(tmp_path / "Music_Info_trimmed.csv")
    assert all(sid in metadata["spotify_id"].values for sid in similars)

# ---------------------------------------------------------------------- #
# Test 4: Persistence after container restart
# ---------------------------------------------------------------------- #
def test_persistence_after_restart(api_client, tmp_path):
    """
    Test that recommendations and .pkl file updates persist after a simulated restart.
    """
    # Step 1: Make recommendation for new ID
    r = api_client.get(
        "/content-based/with-model/recommendations",
        params={"spotify_id": NEW_SPOTIFY_ID, "tags": []}
    )
    assert r.status_code == 200
    expected_recs = r.json()["recommendations"]

    # Step 2: Simulate container restart
    reload(globals()["recommender_service"])
    from recommender_service import app as app2
    new_client = TestClient(app2)

    # Step 3: Verify recommendations persist
    r = new_client.get(
        "/content-based/with-model/recommendations",
        params={"spotify_id": NEW_SPOTIFY_ID, "tags": []}
    )
    assert r.status_code == 200
    assert r.json()["recommendations"] == expected_recs

    # Step 4: Verify .pkl file still has the new entry
    pkl_path = tmp_path / "tracks_embeddings_similars.pkl"
    with pkl_path.open("rb") as f:
        tracks_embeddings_similars = pickle.load(f)
    last_entry = tracks_embeddings_similars[-1]
    assert last_entry[0] == NEW_SPOTIFY_ID
    assert len(last_entry[2]) == 15