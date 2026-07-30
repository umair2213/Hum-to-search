import numpy as np

from src.embedding.melody_embedder import embed_intervals, EMBEDDING_DIM


def test_embedding_shape_and_normalization():
    vector = embed_intervals([2, 2, -2, 1, -1])
    assert vector.shape == (EMBEDDING_DIM,)
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)


def test_embedding_deterministic():
    intervals = [2, 2, -2, 1, -1, 3, -3]
    v1 = embed_intervals(intervals)
    v2 = embed_intervals(intervals)
    assert np.array_equal(v1, v2)


def test_embedding_empty_intervals_returns_zero_vector():
    vector = embed_intervals([])
    assert vector.shape == (EMBEDDING_DIM,)
    assert np.allclose(vector, 0.0)


def test_similar_melodies_have_high_cosine_similarity():
    melody_a = [2, 2, -2, 1, -1, 3, -3, 2]
    melody_b = [2, 2, -2, 1, -1, 3, -3, 2]  # identical
    melody_c = [-5, 7, -3, 6, -2, 4, -6, 1]  # very different

    va = embed_intervals(melody_a)
    vb = embed_intervals(melody_b)
    vc = embed_intervals(melody_c)

    sim_ab = np.dot(va, vb)
    sim_ac = np.dot(va, vc)

    assert sim_ab > sim_ac
    assert np.isclose(sim_ab, 1.0, atol=1e-5)
