from typing import List, Tuple
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import logging
from collaborative_filtering import RecommenderService 


logger = logging.getLogger("api.collaborative")
recommender = RecommenderService()        # one singleton for the whole app


# 1) Pydantic models for response
class RecResponse(BaseModel):
    recommendations: List[str]

class NameArtist(BaseModel):
    name: str
    artist: str

class UserProfilePayload(BaseModel):
    """
    POST body for /collaborative/add_user and /collaborative/update_user_profile
    """
    user_id: str = Field(..., description="Unique ID for the user")

    # list-of-two-element arrays:  [["spotify_id", 23], ["id2", 5], ...]
    listening_histories: List[Tuple[str, int]] = Field(
        default_factory=list,
        description="(spotify_id, playcount) pairs"
    )

    likes:   List[str] = Field(default_factory=list,  description="(ignored for now)")
    dislikes: List[str] = Field(default_factory=list, description="(ignored for now)")


app = FastAPI(
    title="HarmonyX Recommender API",
    description="Expose Recommender Services via FastAPI",
    version="1.0",
)

from content_based_no_model import (
    get_recommendations as no_model_content_based_recommendations,
    get_recommendations_names as no_model_content_based_recommendations_names,
    get_playlist_recommendations as no_model_content_based_playlist_recommendations,
    get_playlist_recommendations_names as no_model_content_based_playlist_recommendations_names,
)
from content_based_with_model import (
    get_recommendations as with_model_content_based_recommendations,
    get_recommendations_names as with_model_content_based_recommendations_names,
    get_playlist_recommendations as with_model_content_based_playlist_recommendations,
    get_playlist_recommendations_names as with_model_content_based_playlist_recommendations_names,
)

# --------------------------------------------------------------------------- #
# ── content-based-no-model endpoints                                         #
# --------------------------------------------------------------------------- #

@app.get(
    "/content-based/no-model/recommendations",
    response_model=RecResponse,
    summary="Get a list of spotify_ids of the recommended songs using non-model version of content-based recommender"
)
def content_based_no_model_recommend_endpoint(
    spotify_id: str = Query(..., description="The Spotify track ID to get recommendations for"),
    tags:       List[str] = Query([], description="Optional list of tags to bias the recommendations")
):
    recs = no_model_content_based_recommendations({"spotify_id": spotify_id, "tags": tags})
    if not recs:
        raise HTTPException(
            status_code=404,
            detail="Track not found and no fallback available"
        )
    return RecResponse(recommendations=recs)


@app.get(
    "/content-based/no-model/recommendations_names",
    response_model=List[NameArtist],
    summary="Get (name, artist) recommendation pairs instead of just IDs using non-model version of content-based recommender"
)
def content_based_no_model_recommend_names_endpoint(
    spotify_id: str = Query(..., description="The Spotify track ID to get recommendations for"),
    tags:       List[str] = Query([], description="Optional list of tags to bias the recommendations")
):
    pairs = no_model_content_based_recommendations_names({"spotify_id": spotify_id, "tags": tags})
    if not pairs:
        raise HTTPException(
            status_code=404,
            detail="Track not found and no fallback available"
        )
    return [NameArtist(name=n, artist=a) for n, a in pairs]

@app.get(
    "/content-based/no-model/playlist_recommendations",
    response_model=RecResponse,
    summary=(
        "Get a playlist of **m** recommended Spotify IDs that covers each "
        "seed track uniformly (content-based, no-model)."
    ),
    responses={
        422: {
            "description": "Validation Error - 'm' must be greater than number of unique seed tracks.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "m(10) should be greter equal than n(12)"
                    }
                }
            },
        }
    }
)
def content_based_no_model_playlist_endpoint(
    spotify_ids: List[str] = Query(
        ...,
        min_length=5,
        description="Seed Spotify track IDs (min 5 unique IDs)",
    ),
    m: int = Query(
        ...,
        gt=5,
        le=25,
        description="Total number of tracks to return (must be > len(spotify_ids) and ≤ 25)",
    ),
):
    # dynamic validation because m must exceed the actual seed count
    seed_ids = list(dict.fromkeys(spotify_ids))  # de-dupe again for clarity
    if m <= len(seed_ids):
        raise HTTPException(
            status_code=422,
            detail=f"m ({m}) must be greater than the number of unique seed tracks ({len(seed_ids)})",
        )

    try:
        recs = no_model_content_based_playlist_recommendations(seed_ids, m)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return RecResponse(recommendations=recs)


@app.get(
    "/content-based/no-model/playlist_recommendations_names",
    response_model=List[NameArtist],
    summary=(
        "Same as the playlist endpoint above, but returns (name, artist) "
        "tuples instead of raw Spotify IDs."
    ),
    responses={
        422: {
            "description": "Validation Error - 'm' must be greater than number of unique seed tracks.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "m(10) should be greter equal than n(12)"
                    }
                }
            },
        }
    }

)
def content_based_no_model_playlist_names_endpoint(
    spotify_ids: List[str] = Query(
        ...,
        min_length=5,
        description="Seed Spotify track IDs (min 5 unique IDs)",
    ),
    m: int = Query(
        ...,
        gt=5,
        le=25,
        description="Total number of tracks to return (must be > len(spotify_ids) and ≤ 25)",
    ),
):
    seed_ids = list(dict.fromkeys(spotify_ids))
    if m <= len(seed_ids):
        raise HTTPException(
            status_code=422,
            detail=f"m ({m}) must be greater than the number of unique seed tracks ({len(seed_ids)})",
        )

    try:
        pairs = no_model_content_based_playlist_recommendations_names(seed_ids, m)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return [NameArtist(name=n, artist=a) for n, a in pairs]


# --------------------------------------------------------------------------- #
# ── Content-based-with-model endpoints                                      #
# --------------------------------------------------------------------------- #

@app.get(
    "/content-based/with-model/recommendations",
    response_model=RecResponse,
    summary="Get a list of spotify_ids of the recommended songs using VGGish model-based content recommender",
    responses={
        404: {
            "description": "Track Not Found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Track not found and no fallback available"
                    }
                }
            }
        }
    }
)
def content_based_with_model_recommend_endpoint(
    spotify_id: str = Query(..., description="The Spotify track ID to get recommendations for"),
    tags: List[str] = Query([], description="Optional list of tags to bias the recommendations")
):
    recs = with_model_content_based_recommendations({"spotify_id": spotify_id, "tags": tags})
    if not recs:
        raise HTTPException(
            status_code=404,
            detail="Track not found and no fallback available"
        )
    return RecResponse(recommendations=recs)

@app.get(
    "/content-based/with-model/recommendations_names",
    response_model=List[NameArtist],
    summary="Get (name, artist) recommendation pairs instead of just IDs using VGGish model-based content recommender",
    responses={
        404: {
            "description": "Track Not Found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Track not found and no fallback available"
                    }
                }
            }
        }
    }
)
def content_based_with_model_recommend_names_endpoint(
    spotify_id: str = Query(..., description="The Spotify track ID to get recommendations for"),
    tags: List[str] = Query([], description="Optional list of tags to bias the recommendations")
):
    pairs = with_model_content_based_recommendations_names({"spotify_id": spotify_id, "tags": tags})
    if not pairs:
        raise HTTPException(
            status_code=404,
            detail="Track not found and no fallback available"
        )
    return [NameArtist(name=n, artist=a) for n, a in pairs]

@app.get(
    "/content-based/with-model/playlist_recommendations",
    response_model=RecResponse,
    summary=(
        "Get a playlist of **m** recommended Spotify IDs that covers each "
        "seed track uniformly using VGGish model-based content recommender."
    ),
    responses={
        404: {
            "description": "Track Not Found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Failed to generate recommendations for spotify_id <id>"
                    }
                }
            }
        },
        422: {
            "description": "Validation Error - Invalid input parameters",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "m(10) must be greater than the number of unique seed tracks (12)"
                    }
                }
            },
        }
    }
)
def content_based_with_model_playlist_endpoint(
    spotify_ids: List[str] = Query(
        ...,
        min_length=5,
        description="Seed Spotify track IDs (min 5 unique IDs)",
    ),
    m: int = Query(
        ...,
        gt=5,
        le=25,
        description="Total number of tracks to return (must be > len(spotify_ids) and ≤ 25)",
    ),
):
    seed_ids = list(dict.fromkeys(spotify_ids))  # de-dupe again for clarity
    if m <= len(seed_ids):
        raise HTTPException(
            status_code=422,
            detail=f"m ({m}) must be greater than the number of unique seed tracks ({len(seed_ids)})",
        )

    try:
        recs = with_model_content_based_playlist_recommendations(seed_ids, m)
    except ValueError as e:
        if "Failed to generate" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=422, detail=str(e))

    return RecResponse(recommendations=recs)


@app.get(
    "/content-based/with-model/playlist_recommendations_names",
    response_model=List[NameArtist],
    summary=(
        "Same as the playlist endpoint above, but returns (name, artist) "
        "tuples instead of raw Spotify IDs."
    ),
    responses={
        404: {
            "description": "Track Not Found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Failed to generate recommendations for spotify_id <id>"
                    }
                }
            }
        },
        422: {
            "description": "Validation Error - Invalid input parameters",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "m(10) must be greater than the number of unique seed tracks (12)"
                    }
                }
            },
        }
    }
)
def content_based_with_model_playlist_names_endpoint(
    spotify_ids: List[str] = Query(
        ...,
        min_length=5,
        description="Seed Spotify track IDs (min 5 unique IDs)",
    ),
    m: int = Query(
        ...,
        gt=5,
        le=25,
        description="Total number of tracks to return (must be > len(spotify_ids) and ≤ 25)",
    ),
):
    seed_ids = list(dict.fromkeys(spotify_ids))
    if m <= len(seed_ids):
        raise HTTPException(
            status_code=422,
            detail=f"m ({m}) must be greater than the number of unique seed tracks ({len(seed_ids)})",
        )

    try:
        pairs = with_model_content_based_playlist_recommendations_names(seed_ids, m)
    except ValueError as e:
        if "Failed to generate" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=422, detail=str(e))

    return [NameArtist(name=n, artist=a) for n, a in pairs]

# --------------------------------------------------------------------------- #
# ── Collaborative-filtering endpoints                                        #
# --------------------------------------------------------------------------- #

@app.post(
    "/collaborative/add_user",
    summary="Add a brand-new user (optionally with initial listening history)",
    responses={
        400: {
            "description": "Bad Request - User already exists",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "User with ID already exists"
                    }
                }
            }
        }
    }
)
def collaborative_add_user(payload: UserProfilePayload):
    # Log but currently ignore likes / dislikes
    if payload.likes or payload.dislikes:
        logger.info("Received likes/dislikes for user %s - ignored by ALS engine",
                    payload.user_id)

    try:
        recommender.add_user(
            user_id=payload.user_id,
            history=payload.listening_histories or None  # None ↔ cold start
        )
        return {"success": True, "detail": "user added"}
    except ValueError as exc:          # user already exists
        raise HTTPException(status_code=400, detail=str(exc))


@app.post(
    "/collaborative/update_user_profile",
    summary="Append / Update listening history for an existing user",
    responses={
        400: {
            "description": "Bad Request - Empty listening history",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "listening_histories cannot be empty for update"
                    }
                }
            }
        },
        404: {
            "description": "User Not Found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "User ID not found"
                    }
                }
            }
        }
    }
)
def collaborative_update_user(payload: UserProfilePayload):
    if not payload.listening_histories:
        raise HTTPException(
            status_code=400,
            detail="listening_histories cannot be empty for update"
        )

    if payload.likes or payload.dislikes:
        logger.info("Received likes/dislikes for user %s - ignored by ALS engine",
                    payload.user_id)

    try:
        recommender.update_user_profile(
            user_id=payload.user_id,
            history=payload.listening_histories,
            overwrite=False,
        )
        return {"success": True, "detail": "profile updated"}
    except ValueError as exc:          # user doesn’t exist
        raise HTTPException(status_code=404, detail=str(exc))


@app.get(
    "/collaborative/recommendations",
    response_model=RecResponse,
    summary="Top-N song IDs from collaborative recommender"
)
def collaborative_recommend_endpoint(
    user_id: str = Query(..., description="User to recommend for"),
    n:       int = Query(10,  ge=1, le=50, description="Number of recommendations"),
):
    rec_ids, cold_flag = recommender.recommend(user_id, n)
    if cold_flag:
        raise HTTPException(
            status_code=404,
            detail="User has no interactions - insufficient data for recommendation"
        )
    return RecResponse(recommendations=rec_ids)


@app.get(
    "/collaborative/recommendations_names",
    response_model=List[NameArtist],
    summary="Top-N (name, artist) pairs from collaborative recommender"
)
def collaborative_recommend_names_endpoint(
    user_id: str = Query(..., description="User to recommend for"),
    n:       int = Query(10,  ge=1, le=50, description="Number of recommendations"),
):
    df = recommender.pretty_recommend(user_id, n)
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No metadata available or insufficient data for this user"
        )

    return [NameArtist(name=row["name"], artist=row["artist"])
            for _, row in df[["name", "artist"]].iterrows()]
