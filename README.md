## Music Recommender Service
A self-contained recommendation microservice that powers a Spotify-like streaming platform built for a university Software Engineering course. It exposes three complementary recommenders behind a FastAPI application and uses Spotify track IDs as the canonical item identifier.

> This service is one part of a broader SOA project; it focuses exclusively on recommendation logic and a thin HTTP interface.

---

### Features at a Glance
- **Two content-based recommenders**
  - **No-model variant:** serves top-k recommendations from a precomputed matrix; falls back to a tag-overlap sampler when a seed track is missing from the matrix. 
  - **Model-based variant (audio):** extracts 128-dimensional VGGish embeddings from short audio segments, computes cosine similarity to an existing embedding bank, and persists newly seen tracks (embedding + similar IDs) to an on-disk store. 
- **Collaborative filtering**
  - Implicit-feedback ALS recommender with confidence-weighted playcounts, incremental user updates, and artifact persistence (encoders, CSR matrix, model). Provides “top-N IDs” and a convenience endpoint that joins metadata for human-readable (name, artist) results.
 
- **Uniform Playlist Builders**
  - Exposes interfaces including Round Robin style selection among recommended songs based on mentioned content-based recommenders. 

- **FastAPI service**
  - Endpoints for all three recommenders, plus playlist builders and (name, artist) variants for testing.  
    Validation and error semantics are included (e.g., cold start, missing items, and input contracts).

- **Unit & integration tests**
   - Tests cover audio preprocessing/embedding, persistence on update, playlist uniformity/uniqueness, and collaborative filtering flows (add/update user, cold-start handling, restart     invariants).

---

### Recommenders
#### 1) Content-Based (No Model)

Looks up recommendations from a precomputed mapping: `spotify_id → [top 12 similar IDs]`.  
When a seed ID is not present, it falls back to a **tag-overlap** strategy (round robin style): sample by overlap weight; if too few matches exist, fill uniformly at random to reach 12.  
Also includes a playlist builder returning composed uniform, duplicate-free set across multiple seeds, reserving quota per seed and fetching extra batches as needed. (wraps single recommendations methods)

For this variant, the dataset was taken from:  
- [million-song-dataset-spotify-lastfm](https://www.kaggle.com/datasets/undefinednull/million-song-dataset-spotify-lastfm)

See data preprocessing details in /preprocess/content_based_no_model

#### 2) Content-Based (With Model)
**Data & ID reconciliation.**  
Audio previews are sourced from MP3 files (e.g., via Spotify API previews or provided dataset samples). **Track IDs** from the **Million Song Dataset (MSD)** are mapped to **Spotify track IDs** using metadata in a CSV file, creating a dictionary for quick lookup and ensuring Spotify IDs serve as the canonical item identifier.

**Model & objective.**  
We use the **VGGish model** (loaded from `TensorFlow Hub`) to extract audio embeddings. Embeddings are 128-dimensional vectors derived from short audio segments, with cosine similarity computed to measure track similarity and identify content-based recommendations.

**Training workflow (high level).**  
1. Load audio MP3 files from the dataset folder and map track IDs to Spotify IDs using the metadata CSV.
2. Preprocess each audio file (resample to 16kHz mono, pad/truncate to meet VGGish frame requirements).
3. Extract embeddings by passing preprocessed audio through VGGish and applying mean pooling over frame-level outputs to obtain a single 128-dimensional vector per track.
4. Compute pairwise cosine similarities across all embeddings using scikit-learn.
5. For each track, select the top 15 most similar Spotify IDs (excluding the track itself).
6. Persist the results (Spotify ID, embedding, and top similar IDs) to a .pkl file for fast lookup and state continuity.

**Online updates & cold-start.** 
- **New tracks:** If a seed Spotify ID is not in the .pkl file, load and preprocess the corresponding MP3 audio, extract its VGGish embedding, compute cosine similarities against the existing embedding bank, and return the top 15 similar IDs. The new embedding and similar IDs are appended to the .pkl file for persistence.  
- **Recommendation:** `recommend(spotify_id, n)` returns top-n similar Spotify IDs from the precomputed list or computes them on-the-fly for unseen tracks; if no audio is available, it handles errors gracefully without recommendations.

Specifics of **data preprocessing, audio loading/mapping, embedding extraction, similarity computation, hyperparameters (e.g., top-N=15)**, and the **training pipeline** are documented in the accompanying notebook:
`/preprocess and train/content_based_with_model/content-based-filtering.ipynb`

For this variant, the dataset was taken from:
- [million-song-dataset-spotify-lastfm](https://www.kaggle.com/datasets/undefinednull/million-song-dataset-spotify-lastfm)

#### 3) Collaborative Filtering 
**Data & ID reconciliation.**  
User listening histories come from the **Million Song Dataset (MSD)**. We use MSD metadata to match tracks to their artists, and then reconcile those records to **Spotify track IDs** (the service’s canonical item identifier).

**Model & objective.**  
We use **implicit-feedback Alternating Least Squares (ALS)** (via the `implicit` library). Interaction strengths are interpreted as playcounts and converted into confidences with a log-scaled function, which empirically stabilizes training while preserving rank signal.

**Training workflow (high level).**  
1. Construct a user–item matrix from the reconciled MSD histories (users × Spotify track IDs, values = confidence-weighted playcounts).  
2. Fit the implicit ALS model on this sparse matrix.  
3. Persist the learned factors and auxiliary artefacts (e.g., encoders and the matrix) to enable fast startup and state continuity across runs.  

**Online updates & cold-start.**  
- **New users:** `add_user` registers a user immediately. If no history is provided, an empty row is added and **model factors are not expanded** until actual listening data appears.  
- **Partial updates:** `update_user_profile` appends or overwrites a user’s recent plays and triggers a **targeted incremental update** (i.e., a **partial fit**) so the user receives immediate, personalized recommendations without a full retrain.  
- **Recommendation:** `recommend(user_id, n)` returns top-n Spotify track IDs; if the user has no usable history, the service returns an empty list with a flag indicating cold-start.

Specifics of **data preprocessing, MSD→Spotify ID matching, matrix construction, hyperparameters**, and the **training pipeline** are documented in the accompanying notebook:  
`notebooks/collaborative_filtering.ipynb`

For this variant, the dataset was taken from:  
- [Million Song Dataset (MSD)](http://millionsongdataset.com/tasteprofile/)


---

### Testing
Since this service was developed as part of a Software Engineering course, it includes both **unit tests** and **integration tests**. You can run the suite with **either `pytest` or Python’s built-in `unittest`**.
- **Content-based (with model)**
  - Audio is padded/truncated to the VGGish frame requirements before embedding.
  - Embeddings are produced and cached; unseen IDs trigger on-the-fly extraction and persistence.
  - Recommendation paths for both “already embedded” and “newly embedded” tracks return the expected shape and ordering.
  - Playlist builders enforce uniform per-seed allocation, prevent duplicates, and honor `n`/`m` bounds.
  - State (embeddings and similar-lists) survives a simulated restart.

- **Collaborative filtering**
  - Playcount → confidence transformation is applied consistently.
  - Encoders (user/item) and matrices persist and reload correctly.
  - `partial_fit` updates newly-observed users without a full retrain and immediately affects recommendations.
  - Cold-start users return an empty result (and/or a specific flag), while warm users return top-N Spotify IDs.
  - API round-trips (add → update → recommend) behave correctly with proper 4xx errors for invalid payloads.
  - Model and auxiliary artifacts remain consistent across simulated service restarts.
- Tests are designed to be hermetic (no external network calls) and rely on local test doubles/stubs for data and persistence.
- You can run a single test module or test case by passing its path/name to either `pytest` or `unittest` as needed.

---

### How to Run

```
docker build -t harmonyx:local .
docker run --rm -it --env-file ".envs\.env.docker" -p 8000:8000 --name harmonyx harmonyx:local
```

