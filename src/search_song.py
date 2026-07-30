"""
User search pipeline.

    Hum --> BasicPitch --> Melody Filter
        --> Interval Encoding --> Melody Embedding --> Qdrant Search
"""

from src.audio.recorder import record_audio
from src.audio.melody import extract_melody
from src.audio.melody_filter import clean_notes
from src.audio.melody_normalizer import notes_to_intervals

from src.embedding.melody_embedder import embed_intervals

from src.search.search_engine import search_by_vector


def get_query_embedding(audio_path):

    raw_notes = extract_melody(audio_path)

    notes = clean_notes(raw_notes)

    intervals = notes_to_intervals(notes)

    return embed_intervals(intervals)


def main():

    audio = record_audio()

    embedding = get_query_embedding(audio)

    results = search_by_vector(
        embedding,
        top_k=10
    )

    print()

    print("Best Matches")

    print("=" * 50)

    for result in results:

        print(
            f"{result.score:.3f}",
            result.payload["title"]
        )


if __name__ == "__main__":
    main()