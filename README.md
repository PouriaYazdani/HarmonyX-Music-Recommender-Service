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




