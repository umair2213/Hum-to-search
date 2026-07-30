# Hum-to-Search: AI Melody-Based Song Recognition

A production-grade, offline query-by-humming (QBH) system that identifies songs from a hummed melody or an uploaded audio clip. The system matches on **melodic contour alone** — independent of singer identity, lyrics, instrumentation, or musical key — using a six-stage AI/ML pipeline: blind source separation → neural pitch tracking → domain-specific signal cleaning → transposition-invariant interval encoding → deterministic 768-dimensional feature embedding → approximate nearest neighbor (ANN) vector search.

---

## How It Works

The system implements an **asymmetric dual-pipeline architecture**: a heavier offline indexing pipeline that performs full audio source separation, and a lightweight online query pipeline that skips separation because a hum is already a monophonic vocal signal. Both pipelines converge on the same embedding and retrieval stages, ensuring representational symmetry.

### 1 · Vocal Separation — Demucs (`htdemucs`)
Applies blind source separation (BSS) to isolate the vocal stem from a full polyphonic mix. The `htdemucs` architecture uses a hybrid time-frequency Transformer trained on MUSDB18. Separated vocals are cached to disk (`separated/htdemucs/<song_stem>/vocals.wav`); re-indexing skips already-processed songs, eliminating the 15–30 s per-song Demucs cost.

> **Why this matters:** Direct pitch tracking on a full mix detects notes from every instrument simultaneously. Isolating the vocal stem ensures the indexed representation reflects the sung melody — the same signal modality that a human hum produces.

### 2 · Melody Extraction — BasicPitch (Spotify, ICASSP 2022)
Runs a pretrained lightweight CNN (`ICASSP_2022_MODEL_PATH`) on the vocal audio to produce three frame-level probability maps (onset, frame, offset), post-processed into discrete MIDI note events: `{pitch: int, start: float, end: float}`. Used as a frozen feature extractor — no fine-tuning required.

### 3 · Melody Filtering — Five-Stage Domain-Specific Cleaning
Raw output from neural pitch trackers contains predictable artifact classes. A deterministic five-stage pipeline (`src/audio/melody_filter.py`) removes them in strict sequence:

| Stage | Function | Filter Logic | Artifact Targeted |
|---|---|---|---|
| 1 | `filter_invalid_notes` | MIDI range [40–80], duration ≥ 0.12 s | Out-of-range pitches; sub-120 ms spurious detections |
| 2 | `remove_overlapping_duplicates` | 0.25 s window, keep highest-pitch candidate | Multiple overlapping candidates for a single sung note |
| 3 | `remove_pitch_jumps` | Sequential jump > 12 semitones (1 octave) | Non-vocal transient detection errors |
| 4 | `remove_harmonic_notes` | Exact jumps of 12, 24, or 36 semitones | F0 tracker octave errors (harmonic overtone misidentification) |
| 5 | `smooth_vibrato` | Merge adjacent notes within ≤ 2 semitones | Vibrato-induced note splitting (5–7 Hz pitch oscillation) |

### 4 · Interval Normalization — Transposition Invariance
Converts the absolute MIDI pitch sequence into a semitone-interval sequence via first-order differencing: `intervals[i] = pitch[i] − pitch[i−1]`. This transformation is **invariant under uniform transposition**: shifting all pitches by a constant `k` leaves every interval unchanged, enabling cross-key matching between a hum and the indexed vocal melody.

### 5 · Embedding — Deterministic 768-Dimensional Feature Vector
`src/embedding/melody_embedder.py` converts a variable-length interval sequence into a fixed-length, L2-normalized `float32` vector via three concatenated feature blocks:

| Block | Dimensions | Representation | Captures |
|---|---|---|---|
| **Interval histogram** | 49 | Normalized frequency of semitone jumps in [−24, +24] | Melodic texture (stepwise vs. leaping motion) |
| **Pitch-contour shape** | 712 | Cumulative sum of intervals, resampled to fixed length via linear interpolation, max-normalized | Melodic shape (rises, falls, arches, directionality) |
| **Summary statistics** | 7 | Mean, std, min, max, count, positive ratio, negative ratio of interval sequence | Compact global melodic descriptors |

The 768-dim vector is L2-normalized, projecting it onto the unit hypersphere so cosine similarity is equivalent to a dot product: `cos(A, B) = A · B` when `‖A‖ = ‖B‖ = 1`.

### 6 · Vector Search — Qdrant HNSW ANN Index
Each song is stored in a local Qdrant collection (`Distance.COSINE`) as a 768-dim vector with a JSON payload (`title`, `path`). Qdrant builds an **HNSW (Hierarchical Navigable Small World)** graph index enabling approximate nearest neighbor retrieval in O(log N) time. A cosine similarity threshold of 0.20 is applied post-retrieval to suppress low-confidence matches.

---

## System Architecture

```
INDEXING PIPELINE (Offline · Parallelized)

  Song File
    │
    ▼
  Demucs (htdemucs)              ← Blind source separation; hybrid time-freq Transformer
    │  vocals.wav cached on disk
    ▼
  BasicPitch (ICASSP 2022 CNN)   ← Neural pitch tracking; outputs (onset, offset, pitch) events
    │  raw note events
    ▼
  Melody Filter (5 stages)       ← Domain-specific artifact removal
    │  clean note sequence
    ▼
  Interval Normalization          ← First-order differencing; transposition invariance
    │  semitone interval sequence
    ▼
  Embedding (768-dim)            ← Histogram + contour + statistics; L2-normalized
    │  float32 vector
    ▼
  Qdrant (local embedded)        ← HNSW cosine index; persistent on disk


QUERY PIPELINE (Online · Low-Latency)

  Hum / Uploaded Audio
    │
    ▼
  BasicPitch                     ← No Demucs — hum is already monophonic
    │
    ▼
  Melody Filter → Interval Normalization → Embedding (768-dim)
    │
    ▼
  Qdrant cosine ANN search       ← Top-K results with similarity ≥ 0.20
```

---

## Project Structure

```
song_finder/
├── app.py                         # Streamlit web UI (record, upload, search, re-index)
├── requirements.txt               # Pinned dependencies
├── songs/                         # Song library (.mp3 / .wav / .flac / .ogg / .m4a)
├── separated/                     # Demucs vocal stem cache
│   └── htdemucs/<song_stem>/vocals.wav
├── database/
│   └── qdrant/                    # Local Qdrant vector database (embedded mode)
├── src/
│   ├── config.py                  # Centralized configuration constants
│   ├── index_songs.py             # Indexing pipeline orchestration (ThreadPoolExecutor)
│   ├── search_song.py             # Query pipeline orchestration
│   ├── audio/
│   │   ├── separator.py           # Demucs vocal separation + disk-cache logic
│   │   ├── melody.py              # BasicPitch note extraction wrapper
│   │   ├── melody_filter.py       # Five-stage deterministic signal cleaning
│   │   ├── melody_normalizer.py   # Pitch → semitone interval conversion
│   │   └── recorder.py            # Microphone capture (sounddevice, 24 kHz mono)
│   ├── embedding/
│   │   └── melody_embedder.py     # 768-dim deterministic feature vector construction
│   ├── search/
│   │   └── search_engine.py       # Qdrant search wrapper (top-K with threshold)
│   └── database/
│       └── qdrant_client.py       # Qdrant CRUD: collection management, upsert, query
├── tests/                         # pytest unit tests for all deterministic components
└── docs/                          # Technical project report (PDF + Markdown)
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Required for type hint syntax used in the codebase |
| FFmpeg | Any recent | Required by Demucs for audio decoding |
| Microphone | — | Required for hum-to-search only; file upload works without one |

**Install FFmpeg:**

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg
```

---

## Installation

```bash
# Clone and navigate to the project root
cd song_finder

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install all pinned dependencies
pip install -r requirements.txt
```

> **GPU acceleration:** `torch`, `torchaudio`, and `tensorflow` are listed in `requirements.txt` for CPU inference. For GPU-accelerated Demucs, install a CUDA-enabled PyTorch build separately before installing other requirements.

---

## Usage

### Web Application

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The interface provides:

- **Search by Humming** — Records 20 seconds of microphone audio at 24 kHz, runs the full query pipeline, and displays ranked matches with cosine similarity scores.
- **Search Using Audio File** — Accepts MP3, WAV, or FLAC uploads; writes to a temporary file, runs the same query pipeline, then cleans up.
- **Sidebar — Rebuild Song Index** — Drops and recreates the Qdrant collection, then re-indexes all files in `songs/` with a live progress bar.
- **Metrics Bar** — Displays songs on disk, vectors indexed, embedding dimension, and last search latency.

### Indexing Songs

Place audio files in the `songs/` directory, then run:

```bash
python -m src.index_songs
```

The indexer scans `songs/` recursively for all supported formats, then processes each file through the full pipeline in parallel:

- **Parallelism:** `ThreadPoolExecutor` with `cpu_count − 1` workers. Demucs (PyTorch) and BasicPitch (TensorFlow) both release the Python GIL during inference, enabling true thread-level parallelism on multi-core CPUs without the overhead of multiprocessing.
- **Caching:** If `separated/htdemucs/<song_stem>/vocals.wav` already exists, the Demucs stage is skipped for that song.
- **Minimum note threshold:** Songs yielding fewer than 4 clean notes after filtering are skipped and not indexed.

### CLI Search

```bash
python -m src.search_song
```

Captures 20 seconds from the microphone and prints the top 10 matches (cosine score + title) to stdout.

---

## Configuration

All tunable constants are centralized in `src/config.py`:

| Constant | Default | Description |
|---|---|---|
| `SONG_DIR` | `./songs` | Root directory scanned recursively for audio files |
| `SEPARATED_DIR` | `./separated` | Demucs output directory (vocal stem cache) |
| `QDRANT_PATH` | `./database/qdrant` | Local Qdrant embedded database path |
| `COLLECTION_NAME` | `"songs"` | Qdrant collection name |
| `VECTOR_SIZE` | `768` | Embedding dimension (must match embedder constants) |
| `SUPPORTED_FORMATS` | `.mp3 .wav .flac .ogg .m4a` | Audio formats discovered during indexing |
| `DEMUCS_MODEL` | `"htdemucs"` | Demucs model variant |

**Recording** (`src/audio/recorder.py`):

| Constant | Default | Description |
|---|---|---|
| `SAMPLE_RATE` | `24000` | Microphone capture sample rate (Hz) |
| `DURATION` | `20` | Recording duration (seconds) |

**Retrieval** (`src/database/qdrant_client.py`):

| Constant | Default | Description |
|---|---|---|
| Cosine threshold | `0.20` | Minimum similarity score; results below this are discarded |
| `limit` (default) | `5` (UI) / `10` (CLI) | Maximum number of results returned |

**Parallelism** (`src/index_songs.py`):

| Constant | Default | Description |
|---|---|---|
| `MAX_WORKERS` | `cpu_count − 1` | Thread pool size for parallel indexing |
| `MIN_NOTES` | `4` | Minimum clean notes required to index a song |

---

## Testing

Unit tests cover all deterministic pipeline components (filtering, normalization, embedding, and vector database operations). Stochastic components (Demucs, BasicPitch) are excluded from unit testing.

```bash
# Run the full test suite
pytest

# Run individual test modules
pytest tests/test_melody_filter.py       # 5-stage filter logic and edge cases
pytest tests/test_melody_normalizer.py  # Pitch-to-interval conversion correctness
pytest tests/test_melody_embedder.py    # Vector dimensions, L2 norm, determinism
pytest tests/test_qdrant_client.py      # Collection CRUD, search results, threshold filtering
```

---

## Technology Stack

| Component | Technology | Version |
|---|---|---|
| Vocal separation | [Demucs](https://github.com/facebookresearch/demucs) (htdemucs) | 4.1.0 |
| Neural pitch tracking | [BasicPitch](https://github.com/spotify/basic-pitch) (Spotify ICASSP 2022) | 0.4.0 |
| Vector database | [Qdrant](https://qdrant.tech/) (local embedded mode) | 1.18.0 |
| Deep learning — Demucs | [PyTorch](https://pytorch.org/) + torchaudio | 2.13.0 |
| Deep learning — BasicPitch | [TensorFlow](https://www.tensorflow.org/) | 2.15.0 |
| Numerical computing | [NumPy](https://numpy.org/) | 1.26.4 |
| Audio I/O | [soundfile](https://python-soundfile.readthedocs.io/), [sounddevice](https://python-sounddevice.readthedocs.io/) | 0.14.0, 0.5.5 |
| Audio analysis | [librosa](https://librosa.org/), [SciPy](https://scipy.org/) | 0.11.0, 1.15.3 |
| Web UI | [Streamlit](https://streamlit.io/) | 1.60.0 |
| Testing | [pytest](https://docs.pytest.org/) | 9.1.1 |
