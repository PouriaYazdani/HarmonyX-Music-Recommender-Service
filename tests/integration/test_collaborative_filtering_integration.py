from pathlib import Path
from importlib import reload
import os, shutil
from dotenv import load_dotenv

import pytest
from fastapi.testclient import TestClient
import collaborative_filtering as cf
import recommender_service

EXISITNG_USER_ID = "user_0000"
EXISITNG_USER_ID_2 = "b80344d063b5ccb3212f76538f3d9e43d87dca9e"
NON_EXISTING_USER_ID = "user_0001"

# ---------------------------------------------------------------------- #
# Helpers to build minimal artefacts identical to unit-test fixture
# ---------------------------------------------------------------------- #

def _copy_real_artifacts(tmp_dir: Path):
    """
    Read .envs/.data_paths, copy each real artefact into tmp_dir, preserving
    file-names.  The integration tests then monkey-patch DATA_DIR to tmp_dir
    so all reads/writes happen on the copy.
    """
    load_dotenv(".envs/.data_paths")

    env_to_name = {
        "COLLABORATIVE-FILTERING-ALS-MODEL"          : "als_model.pkl",
        "COLLABORATIVE-FILTERING-ITEM-ENCODER"       : "item_encoder.pkl",
        "COLLABORATIVE-FILTERING-USER-ENCODER"       : "user_encoder.pkl",
        "COLLABORATIVE-FILTERING-USER-ITEM-MATRIX"   : "user_item_matrix.pkl",
        "COLLABORATIVE-FILTERING-MUSIC-INFO-TRIMMED" : "music_info_trimmed.parquet",
    }

    for env_var, fname in env_to_name.items():
        src = Path(os.getenv(env_var))
        if not src or not src.exists():
            raise RuntimeError(f"{env_var} not set or file missing: {src}")
        shutil.copy2(src, tmp_dir / fname)

# ---------------------------------------------------------------------- #
@pytest.fixture
def api_client(tmp_path, monkeypatch):
    _copy_real_artifacts(tmp_path)
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

    
    reload(cf)
    reload(recommender_service)

    from recommender_service import app
    return TestClient(app)

""" 'user_0000' is inserted into model and matrix and is available. """

# ---------------------------------------------------------------------- #
# Happy-path: add user with history → recommendations → persistence
# ---------------------------------------------------------------------- #
def test_add_with_history_recommendations_and_persistence(api_client):
    # 1) add user with history
    payload = {
        "user_id": NON_EXISTING_USER_ID,
        "listening_histories": [["002dcNTbYNed2wEmFMm3kI", 1], ["003N1g1EqX9dbMw7wCVjse", 1],
                                ["003hZS58wfgow6mFNfBSeF", 1], ["00CNIPMT5VXKAApLVLpzd0", 1],
                                ["00CS57fbmT3RmBFdj9wjp9", 1]],
        "likes": [],
        "dislikes": []
    }
    r = api_client.post("/collaborative/add_user", json=payload)
    assert r.status_code == 200 and r.json()["success"]

    # 2) get recommendations (should succeed)
    r = api_client.get("/collaborative/recommendations",
                       params={"user_id": NON_EXISTING_USER_ID, "n": 3})
    assert r.status_code == 200
    expected = r.json()['recommendations']
    assert len(expected) == 3
    
    # 3) restart the app (simulate container restart)
    reload(globals()["recommender_service"])
    from recommender_service import app as app2
    new_client = TestClient(app2)

    # 4) data should persist
    r = new_client.get("/collaborative/recommendations",
                       params={"user_id": NON_EXISTING_USER_ID, "n": 3})
    assert r.status_code == 200
    assert set(r.json()["recommendations"]) == set(expected)

    # 5) check with local run results
    assert set(expected) == set(['02kxfjlaQ6yYjTZAFfn2Kh',
                                 '0oneCGouDqYCFwUgqV6EVZ',
                                 '0aSdH48aHnE8KdSf39vWFF']
                            )

# ---------------------------------------------------------------------- #
# Happy-path2: add user without history → update profile → recommendations → persistence
# ---------------------------------------------------------------------- #
def test_add_without_history_update_recommendations_and_persistance(api_client):
    # 1) add user without history
    payload = {
        "user_id": NON_EXISTING_USER_ID,
        "listening_histories": [],
        "likes": [],
        "dislikes": []
    }
    r = api_client.post("/collaborative/add_user", json=payload)
    assert r.status_code == 200 and r.json()["success"]
    
    # 2) update user profile
    payload = {"user_id": NON_EXISTING_USER_ID, # now should exists
               "listening_histories": [["002dcNTbYNed2wEmFMm3kI", 1], ["003N1g1EqX9dbMw7wCVjse", 1],
                                       ["003hZS58wfgow6mFNfBSeF", 1], ["00CNIPMT5VXKAApLVLpzd0", 1],
                                       ["00CS57fbmT3RmBFdj9wjp9", 1]],
    }
    r = api_client.post("/collaborative/update_user_profile", json=payload)
    assert r.status_code == 200
    r = api_client.get("/collaborative/recommendations",
                       params={"user_id": NON_EXISTING_USER_ID, "n": 3})
    expected = r.json()["recommendations"]

    # 3) restart the app (simulate container restart)
    reload(globals()["recommender_service"])
    from recommender_service import app as app2
    new_client = TestClient(app2)

    # 4) data should persist
    r = new_client.get("/collaborative/recommendations",
                       params={"user_id": NON_EXISTING_USER_ID, "n": 3})
    assert r.status_code == 200
    assert set(r.json()["recommendations"]) == set(expected)

    # 5) check with local run results
    assert set(expected) == set(['02kxfjlaQ6yYjTZAFfn2Kh',
                                 '0oneCGouDqYCFwUgqV6EVZ',
                                 '0aSdH48aHnE8KdSf39vWFF']
                            )

# ---------------------------------------------------------------------- #
# Error: duplicate add
# ---------------------------------------------------------------------- #
def test_add_duplicate_user(api_client):
    payload = {"user_id": "u1", "listening_histories": [["trackA", 1]]}
    assert api_client.post("/collaborative/add_user", json=payload).status_code == 200
    r = api_client.post("/collaborative/add_user", json=payload)
    assert r.status_code == 400 and "already exists" in r.json()["detail"]

# ---------------------------------------------------------------------- #
# Error: Add already existing user
# ---------------------------------------------------------------------- #
def test_already_added_user(api_client):
    payload = {"user_id": EXISITNG_USER_ID, "listening_histories": [["trackA", 1]]}
    r = api_client.post("/collaborative/add_user", json=payload)
    assert r.status_code == 400 and "already exists" in r.json()["detail"]

# ---------------------------------------------------------------------- #
# Error: cold user recommendations
# ---------------------------------------------------------------------- #
def test_cold_user_returns_404(api_client):
    # add user with no history
    r = api_client.post("/collaborative/add_user",
                        json={"user_id": "cold", "listening_histories": []})
    assert r.status_code == 200
    # Ask for recs -> 404 insufficient data
    r = api_client.get("/collaborative/recommendations",
                       params={"user_id": "cold", "n": 3})
    assert r.status_code == 404 and "insufficient data" in r.json()["detail"]

# ---------------------------------------------------------------------- #
# Error: update with empty histories
# ---------------------------------------------------------------------- #
def test_update_empty_histories(api_client):
    payload = {"user_id": EXISITNG_USER_ID, "listening_histories": []}
    r = api_client.post("/collaborative/update_user_profile", json=payload)
    assert r.status_code == 400 and "empty" in r.json()["detail"]

def test_update_non_exsiting_user(api_client):
    payload = {"user_id": NON_EXISTING_USER_ID, "listening_histories": [["trackA", 1]]}
    r = api_client.post("/collaborative/update_user_profile", json=payload)
    assert r.status_code == 404 and "does not exist" in r.json()["detail"]

# ---------------------------------------------------------------------- #
# Check Recommedations with artist and song names
# ---------------------------------------------------------------------- #
def test_existing_user_recommend_names(api_client):
    r = api_client.get("/collaborative/recommendations_names",
                       params={"user_id": EXISITNG_USER_ID_2, "n": 2})
    assert r.status_code == 200
    body = r.json()

    expected = [
        {"name": "Revelry", "artist": "Kings of Leon"},
        {"name": "Bubble Toes", "artist": "Jack Johnson"},
    ] 
    assert sorted(body, key=lambda x: (x["name"], x["artist"])) == \
        sorted(expected, key=lambda x: (x["name"], x["artist"]))
