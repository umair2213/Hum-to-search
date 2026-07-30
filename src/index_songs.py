"""
Database indexing pipeline.

    Song --> Demucs --> BasicPitch --> Melody Filter
         --> Interval Encoding --> Melody Embedding --> Qdrant
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.config import SONG_DIR
from src.config import SUPPORTED_FORMATS
from src.config import VECTOR_SIZE

from src.audio.separator import separate_vocals
from src.audio.melody import extract_melody
from src.audio.melody_filter import clean_notes
from src.audio.melody_normalizer import notes_to_intervals

from src.embedding.melody_embedder import embed_intervals

from src.database.qdrant_client import (
    recreate_collection,
    upload_song
)


MIN_NOTES = 4
MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def get_audio_files():

    files = []

    for ext in SUPPORTED_FORMATS:
        files.extend(
            SONG_DIR.rglob(f"*{ext}")
        )

    return files


def build_song_embedding(song_path):
    """
    Run the full melody pipeline for a single song and return its
    embedding, or None if too few notes were detected.
    """

    vocals_path = separate_vocals(song_path)

    raw_notes = extract_melody(vocals_path)

    clean = clean_notes(raw_notes)

    if len(clean) < MIN_NOTES:
        return None

    intervals = notes_to_intervals(clean)

    return embed_intervals(intervals)


def _process_song(song_path: str):
    """
    Process a single song: separate, extract, filter, embed.
    Returns (song_path, embedding, payload) or (song_path, None, None).
    """
    embedding = build_song_embedding(song_path)
    if embedding is None:
        return song_path, None, None

    p = Path(song_path)
    payload = {
        "title": p.stem,
        "path": str(p),
    }
    return song_path, embedding, payload


def main(progress_callback=None, max_workers: int = MAX_WORKERS):
    """
    Index songs into Qdrant using parallel thread-pool processing.

    Demucs (PyTorch) and BasicPitch (TensorFlow) release the GIL
    during computation, so threads achieve real parallelism on
    multi-core CPUs.

    If progress_callback is provided, it is called with
    (completed, total, current_song_name) after each song finishes.
    """
    recreate_collection(VECTOR_SIZE)

    songs = get_audio_files()

    if not songs:
        print("No songs found.")
        if progress_callback:
            progress_callback(0, 0, "")
        return

    point_id = 0
    indexed_count = 0
    completed_count = 0
    total = len(songs)

    print(f"Indexing {total} song(s) with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_song, str(song)): song
            for song in songs
        }

        for future in tqdm(
            as_completed(futures),
            total=total,
            desc="Indexing songs",
        ):
            song_path, embedding, payload = future.result()
            completed_count += 1

            song_name = Path(song_path).stem if song_path else ""

            if embedding is None:
                if progress_callback:
                    progress_callback(completed_count, total, song_name)
                continue

            upload_song(
                point_id,
                embedding,
                payload
            )

            point_id += 1
            indexed_count += 1

            if progress_callback:
                progress_callback(completed_count, total, song_name)

    print(
        f"Indexed {indexed_count} song(s). Total vectors: {point_id}"
    )


if __name__ == "__main__":
    main()