import pandas as pd
from typing import List, TypedDict

class RecInput(TypedDict):
    spotify_id: str
    tags: List[str]

recommendations = pd.read_pickle("recommendations.pkl")
songs_metadata = pd.read_csv("Music_Info_trimmed.csv")
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

def _get_name_artist_pairs(spotify_ids: list[str]) -> list[tuple[str, str]]:
    """
    Returns (name, artist) for each song id in spotify_ids, in the same order.
    
    :param spotify_ids: The list of song ids to look up.
    :return: A list of (name, artist) tuples.
    """
    # re-index for fast lookup
    # import ipdb; ipdb.set_trace()    
    meta = songs_metadata.set_index('spotify_id')
    
    result: list[tuple[str, str]] = []
    for tid in spotify_ids:
        if tid in meta.index:
            row = meta.loc[tid]
            result.append((row['name'], row['artist']))
        # else: you could append (None, None) or raise if missing
    
    return result