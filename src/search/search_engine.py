"""
Search engine.

Responsibility:
    Vector  -->  Qdrant  -->  Top K

No audio processing lives here.
"""

from src.database.qdrant_client import search_song


def search_by_vector(embedding, top_k: int = 10):

    return search_song(
        embedding,
        limit=top_k
    )
