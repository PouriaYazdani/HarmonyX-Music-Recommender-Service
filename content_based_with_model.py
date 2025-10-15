import pandas as pd
from typing import List, TypedDict
from dotenv import load_dotenv
from pathlib import Path
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
import librosa
from collections import defaultdict


class RecInput(TypedDict):
    spotify_id: str
    tags: List[str]


BASE_DIR = Path(__file__).resolve().parent
DOTENV_PATH = BASE_DIR / ".envs" / ".data_paths"
load_dotenv(dotenv_path=str(DOTENV_PATH))

# Build full paths
TRACKS_EMBEDDINGS_SIMILARS_PATH = BASE_DIR / os.getenv("CONTENT-BASED-WITH-MODEL-EMBEDDINGS_FILE",
                                                       "data/content_based_with_model/tracks_embeddings_similars.pkl")
SONGS_METADATA_PATH = BASE_DIR / os.getenv("METADATA_FILE", "data/Music_Info_trimmed.csv")
VGGISH_MODEL_PATH = BASE_DIR / os.getenv("VGGISH_MODEL_PATH", "data/content_based_with_model/vggish_model")
TRACKS_FILES_PATH = BASE_DIR / os.getenv("TRACKS_FILES_PATH", "data/content_based_with_model/tracks")

# Load songs metadata
songs_metadata = pd.read_csv(SONGS_METADATA_PATH)
songs_metadata['tags'] = songs_metadata['tags'].str.split(', ')

# Load VGGish model
vggish_model = hub.KerasLayer(str(VGGISH_MODEL_PATH))

def preprocess_audio(file_path: str) -> np.ndarray:
    """
    Preprocess audio file for VGGish model.

    :param file_path: Path to the audio file.
    :return: Preprocessed audio array or None if processing fails.
    """
    try:
        # Load audio file with librosa
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        # VGGish expects 0.96-second chunks (VGGish frame length)
        if len(audio) < 0.96 * sr:
            audio = np.pad(audio, (0, int(0.96 * sr) - len(audio)), mode='constant')
        return audio
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def extract_vggish_embedding(audio: np.ndarray) -> np.ndarray:
    """
    Extract VGGish embedding from audio.

    :param audio: Preprocessed audio array.
    :return: 128-dimensional embedding or None if extraction fails.
    """
    try:
        # VGGish expects input in shape (num_samples,)
        embeddings = vggish_model(audio)
        # Mean pooling over time to get a single 128-dimensional embedding
        embedding = np.mean(embeddings, axis=0)
        return embedding
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None


def get_recommendations(input_data: RecInput) -> List[str]:
    """
    Get recommendations for a given song using VGGish embeddings.

    :param input_data: {
        "spotify_id": str,
        "tags": list[str], could be empty list or nothing at all
    }
    :return: List of 15 recommended spotify_ids
    """
    spotify_id = input_data["spotify_id"]
    input_tags = set(input_data.get("tags", []))

    # Load precomputed embeddings
    try:
        with open(TRACKS_EMBEDDINGS_SIMILARS_PATH, 'rb') as file:
            tracks_embeddings_similars = pickle.load(file)
    except FileNotFoundError:
        tracks_embeddings_similars = []
        print(f"Embeddings file not found: {TRACKS_EMBEDDINGS_SIMILARS_PATH}")

    # Check if spotify_id exists in precomputed embeddings
    for track_spotify_id, _, similars in tracks_embeddings_similars:
        if track_spotify_id == spotify_id:
            return similars

    # If not found, process new track
    audio_file = f"{spotify_id}.mp3"
    audio_file_path = TRACKS_FILES_PATH / audio_file

    # Process audio and extract embedding
    audio = preprocess_audio(audio_file_path)
    if audio is None:
        return []

    new_embedding = extract_vggish_embedding(audio)
    if new_embedding is None:
        return []

    # Compute cosine similarities with existing embeddings
    spotify_ids = [sid for sid, _, _ in tracks_embeddings_similars]
    existing_embeddings = np.array([emb for _, emb, _ in tracks_embeddings_similars])

    new_embedding = new_embedding.reshape(1, -1)
    similarities = cosine_similarity(new_embedding, existing_embeddings)[0]

    # Get top 15 similar songs
    top_n_indices = np.argsort(similarities)[-15:][::-1]
    similar_ids = [spotify_ids[idx] for idx in top_n_indices]

    # Append new entry to tracks_embeddings_similars
    tracks_embeddings_similars.append((spotify_id, new_embedding[0], similar_ids))

    # Save updated embeddings
    try:
        with open(TRACKS_EMBEDDINGS_SIMILARS_PATH, 'wb') as f:
            pickle.dump(tracks_embeddings_similars, f)
    except Exception as e:
        print(f"Error saving updated .pkl file: {e}")

    return similar_ids


def get_recommendations_names(input_data: RecInput) -> List[tuple[str, str]]:
    """
    Get (name, artist) pairs for recommended songs.

    :param input_data: Input data with spotify_id and optional tags.
    :return: List of (name, artist) tuples.
    """
    return _get_name_artist_pairs(get_recommendations(input_data=input_data))

def get_playlist_recommendations(
    seed_ids: list[str],
    m: int,
    *,
    max_extra_batches: int = 3,
) -> list[str]:
    """
    Build a composite playlist that covers every seed song as uniformly as
    possible using VGGish model-based content recommender. All seed IDs must
    have valid recommendations, or an error is raised.

    Parameters
    ----------
    seed_ids       : list[str]
        Distinct Spotify IDs provided by the caller (n ≥ 5).
    m              : int
        Desired final length of the recommendation list (n < m ≤ 25).
    max_extra_batches : int, optional
        Safety valve: extra batches we are willing to fetch per seed to
        satisfy the uniform-coverage requirement.

    Returns
    -------
    list[str]
        Exactly *m* **unique** Spotify IDs, order arbitrary.

    Raises
    ------
    ValueError
        If any seed ID cannot be processed (e.g., missing audio file) or
        if input validation fails (n < 5 or m not in range n < m ≤ 25).
    """
    seed_ids = list(dict.fromkeys(seed_ids))  # de-dupe while preserving order
    n = len(seed_ids)
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
        if not recs:
            raise ValueError(f"Failed to generate recommendations for spotify_id {sid}")
        candidate_lists[sid] = recs

    # Pick tracks round-robin, enforcing global uniqueness
    chosen: set[str] = set(seed_ids)  # never recommend the seeds themselves
    result: list[str] = []

    # index cursor per seed
    cursors = defaultdict(int)

    def _exhausted(seed: str) -> bool:
        """Have we looked at all currently downloaded recs for this seed?"""
        return cursors[seed] >= len(candidate_lists[seed])

    for _ in range(max_extra_batches + 1):  # bounded safety loop
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
                recs = get_recommendations({"spotify_id": seed, "tags": []})
                if not recs:
                    raise ValueError(f"Failed to generate additional recommendations for spotify_id {seed}")
                candidate_lists[seed].extend(recs)

        if not any(allocation):  # all quotas satisfied → done
            break
        if not made_progress:
            raise ValueError("Unable to satisfy allocation with available recommendations")

    if len(result) < m:
        raise ValueError(f"Could not generate {m} unique recommendations")

    return result

def get_playlist_recommendations_names(seed_ids: list[str], m: int) -> list[tuple[str, str]]:
    """Thin wrapper that returns (name, artist) pairs."""
    return _get_name_artist_pairs(get_playlist_recommendations(seed_ids, m))

def _get_name_artist_pairs(spotify_ids: List[str]) -> List[tuple[str, str]]:
    """
    Returns (name, artist) for each song id in spotify_ids, in the same order.

    :param spotify_ids: The list of song ids to look up.
    :return: A list of (name, artist) tuples.
    """
    meta = songs_metadata.set_index('spotify_id')
    result: List[tuple[str, str]] = []
    for tid in spotify_ids:
        if tid in meta.index:
            row = meta.loc[tid]
            result.append((row['name'], row['artist']))

    return result