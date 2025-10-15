import pandas as pd
from typing import List, TypedDict
from dotenv import load_dotenv
from pathlib import Path
import os
from collections import defaultdict

class RecInput(TypedDict):
    spotify_id: str
    tags: List[str]


BASE_DIR    = Path(__file__).resolve().parent
DOTENV_PATH = BASE_DIR / ".envs" / ".data_paths"
load_dotenv(dotenv_path=str(DOTENV_PATH))

# build full paths (with project root as /app inside container)
RECOMMENDATIONS_PATH = BASE_DIR / os.getenv("CONTENT-BASED-NO-MODEL-RECOMMENDATIONS_FILE", "data/content_based_no_model/recommendations.pkl")
SONGS_METADATA_PATH    = BASE_DIR / os.getenv("METADATA_FILE",       "data/Music_Info_trimmed.csv")

recommendations   = pd.read_pickle(RECOMMENDATIONS_PATH)
songs_metadata    = pd.read_csv(SONGS_METADATA_PATH)
songs_metadata['tags'] = songs_metadata['tags'].str.split(', ')

def get_recommendations(input_data: RecInput) -> list[str]:
    """
    Get recommendations for a given song, either from precomputed recs
    or via a simple tag-overlap rule.
    
    :param input_data: {
        "spotify_id": str,
        "tags": list[str], could be empty list or nothing at all
    }
    :return: list of 12 recommended spotify_ids
    """
    spotify_id = input_data["spotify_id"]
    input_tags = set(input_data.get("tags", []))
    
    # 1) Try precomputed
    recs_series = recommendations.loc[
        recommendations["spotify_id"] == spotify_id,
        "recommendations"
    ]
    if not recs_series.empty:
        return recs_series.iloc[0]
    
    if len(input_data) > 0:
        # 2) Fallback: tag‐overlap
        candidates = songs_metadata.copy()
        candidates["tag_overlap"] = candidates["tags"].apply(
            lambda tags: len(set(tags) & input_tags)
        )
        
        matches = candidates[candidates["tag_overlap"] > 0]
        
        if len(matches) >= 12:
            # weighted sampling: more overlap → higher chance
            sampled = matches.sample(n=12, weights="tag_overlap", replace=False)
        elif 0 < len(matches) < 12:
            # take all matches, then fill the rest at random
            sampled = matches.copy()
            remaining = 12 - len(matches)
            others = candidates.drop(matches.index)
            fill = others.sample(n=remaining, replace=False)
            sampled = pd.concat([sampled, fill])
        else:
            # no matches at all: pure random fallback
            sampled = candidates.sample(n=12, replace=False)
        
        return sampled["spotify_id"].tolist()
    else:
        pass

def get_recommendations_names(input_data: RecInput) -> list[tuple[str, str]]:
    return _get_name_artist_pairs(get_recommendations(input_data=input_data))

def get_playlist_recommendations(
    seed_ids: list[str],
    m: int,
    *,
    max_extra_batches: int = 3,
) -> list[str]:
    """
    Build a composite playlist that covers every seed song as uniformly as
    possible.

    Parameters
    ----------
    seed_ids       : list[str]
        Distinct Spotify IDs provided by the caller (n ≥ 5).
    m              : int
        Desired final length of the recommendation list (n < m ≤ 25).
    max_extra_batches : int, optional
        Safety valve: extra batches we are willing to fetch per seed to
        satisfy the uniform-coverage requirement before falling back to a
        global random sample.

    Returns
    -------
    list[str]
        Exactly *m* **unique** Spotify IDs, order arbitrary.
    """
    seed_ids = list(dict.fromkeys(seed_ids))          # de-dupe while
    n = len(seed_ids)                                 # preserving order
    if n < 5:
        raise ValueError("Need at least 5 seed tracks")
    if not (n < m <= 25):
        raise ValueError("m must satisfy n < m ≤ 25")

    base = m // n
    remainder = m % n
    allocation = [base + (1 if i < remainder else 0) for i in range(n)]

    candidate_lists: dict[str, list[str]] = {}
    for sid in seed_ids:
        recs = get_recommendations({"spotify_id": sid, "tags": []})
        candidate_lists[sid] = recs

    # Pick tracks round-robin, enforcing global uniqueness
    chosen: set[str] = set(seed_ids)   # never recommend the seeds themselves
    result: list[str] = []

    # index cursor per seed
    cursors = defaultdict(int)

    def _exhausted(seed: str) -> bool:
        """Have we looked at all currently downloaded recs for this seed?"""
        return cursors[seed] >= len(candidate_lists[seed])

    for _ in range(max_extra_batches + 1):   # bounded safety loop
        made_progress = False

        for idx, seed in enumerate(seed_ids):
            need = allocation[idx]
            while need and not _exhausted(seed):
                candidate = candidate_lists[seed][cursors[seed]]
                cursors[seed] += 1
                if candidate not in chosen:
                    chosen.add(candidate)
                    result.append(candidate)
                    need -= 1
                    made_progress = True
            allocation[idx] = need  # update remaining quota

            # fetched everything we had? Request another batch if allowed
            if need and _exhausted(seed) and _ < max_extra_batches:
                candidate_lists[seed].extend(
                    get_recommendations({"spotify_id": seed, "tags": []})
                )

        if not any(allocation):       # all quotas satisfied → done
            break
        if not made_progress:
            break

    # Still short? Fill at random from the full catalogue.
    shortfall = m - len(result)
    if shortfall:
        # fast random sample from songs_metadata, excluding already chosen
        pool = songs_metadata[~songs_metadata["spotify_id"].isin(chosen)]
        extra = pool["spotify_id"].sample(n=shortfall, replace=False).tolist()
        result.extend(extra)

    return result

def get_playlist_recommendations_names(seed_ids: list[str], m: int) -> list[tuple[str, str]]:
    """Thin wrapper that returns (name, artist) pairs."""
    return _get_name_artist_pairs(get_playlist_recommendations(seed_ids, m))

def _get_name_artist_pairs(spotify_ids: list[str]) -> list[tuple[str, str]]:
    """
    Returns (name, artist) for each song id in spotify_ids, in the same order.
    
    :param spotify_ids: The list of song ids to look up.
    :return: A list of (name, artist) tuples.
    """
    # re-index for fast lookup
    meta = songs_metadata.set_index('spotify_id')
    
    result: list[tuple[str, str]] = []
    for tid in spotify_ids:
        if tid in meta.index:
            row = meta.loc[tid]
            result.append((row['name'], row['artist']))
    
    return result