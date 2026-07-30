"""
Interactive step-by-step pipeline test.

Edit SONG_PATHS and RECORDING_PATH below, then run:

    python test_pipeline.py

Press Enter after each step to continue.
"""

import atexit
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio.separator import separate_vocals
from src.audio.melody import extract_melody
from src.audio.melody_filter import clean_notes
from src.audio.melody_normalizer import notes_to_intervals
from src.embedding.melody_embedder import embed_intervals
from src.database.qdrant_client import (
    recreate_collection,
    upload_song,
    search_song,
    client as qdrant_client,
)
from src.config import VECTOR_SIZE

# ── EDIT THESE ──────────────────────────────────────────────
SONG_PATHS = [
    "/home/hasnain/Desktop/song_finder/songs/Rahat Fateh Ali Khan - Zaroori Tha - Most Broken Heart Song.mp3",
    "/home/hasnain/Desktop/song_finder/songs/Tumhe Dillagi Full Song with Lyrics  Rahat Fateh Ali Khan  Huma Qureshi, Vidyut Jammwal.mp3",
    "/home/hasnain/Desktop/song_finder/songs/Woh Lamhe Woh Baatein - Lyrical Video  Emraan Hashmi  Atif Aslam  Shamita Shetty  Zeher.mp3",
    "/home/hasnain/Desktop/song_finder/songs/Coke Studio Season 9  Paar Chanaa De  Shilpa Rao & Noori.mp3"
]
RECORDING_PATH = "/home/hasnain/Desktop/song_finder/humming.mp3"
# ────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.20
PAUSE = False


def _cleanup_qdrant():
    try:
        qdrant_client.close()
    except Exception:
        pass


atexit.register(_cleanup_qdrant)


def pause(msg="Press Enter to continue..."):
    if PAUSE:
        input(f"\n  >>> {msg}")
    print()


def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_notes(notes, max_n=10):
    print(f"  Total notes: {len(notes)}")
    shown = notes[:max_n]
    for n in shown:
        dur = n["end"] - n["start"]
        print(f"    pitch={n['pitch']:3d}  start={n['start']:.2f}s  "
              f"end={n['end']:.2f}s  dur={dur:.2f}s")
    if len(notes) > max_n:
        print(f"    ... ({len(notes) - max_n} more)")


def print_embedding(vec, label=""):
    print(f"  Shape: {vec.shape}")
    print(f"  L2 norm: {np.linalg.norm(vec):.6f}")
    print(f"  First 10 values: {vec[:10]}")
    print(f"  Min: {vec.min():.6f}  Max: {vec.max():.6f}  Mean: {vec.mean():.6f}")


def main():

    # ─── SONG PIPELINE (loop over all songs) ─────────────────

    banner("RECREATE QDRANT COLLECTION")
    print(f"  Wiping and recreating collection (vector_size={VECTOR_SIZE})...")
    recreate_collection(VECTOR_SIZE)
    print("  Done.")
    pause("Collection ready. Press Enter to start processing songs...")

    song_embeddings = {}

    for song_idx, song_path in enumerate(SONG_PATHS):

        banner(f"SONG {song_idx + 1}/{len(SONG_PATHS)}: {song_path}")

        if not os.path.exists(song_path):
            print(f"  ERROR: {song_path} not found! Skipping.")
            continue

        banner("  STEP 1: Vocal Separation (Demucs)")
        print(f"  Input song: {song_path}")
        print("  Running Demucs (htdemucs)...")
        vocals_path = separate_vocals(song_path)
        print(f"  Vocals saved to: {vocals_path}")

        info = sf.info(vocals_path)
        print(f"  Duration: {info.duration:.1f}s  Sample rate: {info.samplerate}")
        pause("Step 1 complete. Press Enter for BasicPitch extraction...")

        banner("  STEP 2: Melody Extraction (BasicPitch)")
        print(f"  Input: {vocals_path}")
        print("  Running BasicPitch...")
        raw_notes = extract_melody(vocals_path)
        print_notes(raw_notes, max_n=10)
        pause("Step 2 complete. Press Enter for melody cleaning...")

        banner("  STEP 3: Melody Cleaning (melody_filter)")
        print(f"  Input: {len(raw_notes)} raw notes")
        clean = clean_notes(raw_notes)
        removed = len(raw_notes) - len(clean)
        print(f"  Removed {removed} notes ({removed / max(len(raw_notes), 1) * 100:.1f}%)")
        print_notes(clean, max_n=10)
        pause("Step 3 complete. Press Enter for interval normalization...")

        banner("  STEP 4: Interval Normalization")
        intervals = notes_to_intervals(clean)
        print(f"  {len(clean)} notes -> {len(intervals)} intervals")
        print(f"  First 20 intervals: {intervals[:20]}")
        pause("Step 4 complete. Press Enter for embedding...")

        banner("  STEP 5: Embedding (melody_embedder)")
        song_embedding = embed_intervals(intervals)
        print(f"  Song embedding:")
        print_embedding(song_embedding, "song")
        pause("Step 5 complete. Press Enter to upload to Qdrant...")

        banner("  STEP 6: Upload to Qdrant")
        payload = {
            "title": Path(song_path).stem,
            "path": song_path,
        }
        upload_song(song_idx, song_embedding, payload)
        print(f"  Uploaded (id={song_idx}): {payload['title']}")
        song_embeddings[song_path] = song_embedding
        pause(f"Step 6 complete. Song {song_idx + 1} done!")

    # ─── RECORDING / QUERY PIPELINE ──────────────────────────

    banner("STEP 7: Load Recording")
    print(f"  Recording file: {RECORDING_PATH}")
    if not os.path.exists(RECORDING_PATH):
        print(f"  ERROR: {RECORDING_PATH} not found!")
        print("  Please record a query first or update RECORDING_PATH.")
        sys.exit(1)
    info = sf.info(RECORDING_PATH)
    print(f"  Duration: {info.duration:.1f}s  Sample rate: {info.samplerate}")
    pause("Step 7 complete. Press Enter for BasicPitch on recording...")

    banner("STEP 8: Melody Extraction from Recording (BasicPitch)")
    print(f"  Input: {RECORDING_PATH}")
    print("  Running BasicPitch...")
    query_raw_notes = extract_melody(RECORDING_PATH)
    print_notes(query_raw_notes, max_n=10)
    pause("Step 8 complete. Press Enter for melody cleaning...")

    banner("STEP 9: Melody Cleaning (recording)")
    print(f"  Input: {len(query_raw_notes)} raw notes")
    query_clean = clean_notes(query_raw_notes)
    removed = len(query_raw_notes) - len(query_clean)
    print(f"  Removed {removed} notes ({removed / max(len(query_raw_notes), 1) * 100:.1f}%)")
    print_notes(query_clean, max_n=10)
    pause("Step 9 complete. Press Enter for intervals + embedding...")

    banner("STEP 10: Intervals + Embedding (recording)")
    query_intervals = notes_to_intervals(query_clean)
    print(f"  {len(query_clean)} notes -> {len(query_intervals)} intervals")
    print(f"  First 20 intervals: {query_intervals[:20]}")
    print()
    query_embedding = embed_intervals(query_intervals)
    print(f"  Query embedding:")
    print_embedding(query_embedding, "query")
    pause("Step 10 complete. Press Enter to search Qdrant...")

    banner("STEP 11: Search Qdrant")
    print(f"  Searching for matches (cosine similarity > {SIMILARITY_THRESHOLD})...")
    results = search_song(query_embedding, limit=10)

    if not results:
        print(f"  No matches found (all scores below {SIMILARITY_THRESHOLD} threshold).")
    else:
        print(f"  Found {len(results)} matches:")
        print()
        print(f"  {'Score':>8}   {'Title'}")
        print(f"  {'-----':>8}   {'-----'}")
        for r in results:
            print(f"  {r.score:8.4f}   {r.payload.get('title', 'Unknown')}")

    # ─── FINAL SUMMARY ───────────────────────────────────────

    banner("SUMMARY: Query vs All Songs")
    print(f"  Query: {RECORDING_PATH}")
    print()
    print(f"  {'Cosine':>8}   {'Threshold':>10}   {'Song'}")
    print(f"  {'------':>8}   {'---------':>10}   {'----'}")
    for song_path, emb in song_embeddings.items():
        cosine = float(np.dot(emb, query_embedding))
        status = "MATCH" if cosine > SIMILARITY_THRESHOLD else "below"
        print(f"  {cosine:8.4f}   {status:>10}   {Path(song_path).stem}")
    print()
    print("  Pipeline test complete.")


if __name__ == "__main__":
    main()
