import numpy as np
import pytest
from qdrant_client import QdrantClient


@pytest.fixture
def qdrant_module(monkeypatch):
    """
    Swap the module-level Qdrant client for an isolated, in-memory
    instance so tests don't touch the real on-disk database.
    """
    import src.database.qdrant_client as qdrant_client

    test_client = QdrantClient(location=":memory:")

    monkeypatch.setattr(qdrant_client, "client", test_client)

    yield qdrant_client

    test_client.close()


def test_create_and_search_round_trip(qdrant_module):
    vector_size = 8

    qdrant_module.recreate_collection(vector_size)

    embedding = np.ones(vector_size, dtype=np.float32) / np.sqrt(vector_size)

    qdrant_module.upload_song(
        0,
        embedding,
        {"title": "Test Song", "path": "songs/test.mp3"},
    )

    results = qdrant_module.search_song(embedding, limit=5)

    assert len(results) == 1
    assert results[0].payload["title"] == "Test Song"
    assert results[0].score > 0.7


def test_recreate_collection_wipes_previous_data(qdrant_module):
    vector_size = 8
    embedding = np.ones(vector_size, dtype=np.float32) / np.sqrt(vector_size)

    qdrant_module.recreate_collection(vector_size)
    qdrant_module.upload_song(0, embedding, {"title": "Old Song"})

    qdrant_module.recreate_collection(vector_size)

    results = qdrant_module.search_song(embedding, limit=5)
    assert results == []
