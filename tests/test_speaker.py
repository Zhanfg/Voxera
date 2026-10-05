from __future__ import annotations

import numpy as np
import pytest

from voxera.speaker import SpeakerEmbedding, cosine_similarity


def test_embedding_is_normalized() -> None:
    embedding = SpeakerEmbedding(np.array([3.0, 4.0], dtype=np.float32))

    assert embedding.dimension == 2
    assert np.isclose(np.linalg.norm(embedding.values), 1.0)
    assert np.allclose(embedding.values, np.array([0.6, 0.8], dtype=np.float32))


def test_cosine_similarity_uses_normalized_values() -> None:
    left = SpeakerEmbedding(np.array([1.0, 0.0], dtype=np.float32))
    right = SpeakerEmbedding(np.array([1.0, 1.0], dtype=np.float32))

    assert np.isclose(cosine_similarity(left, right), 1.0 / np.sqrt(2.0))


def test_zero_embedding_is_rejected() -> None:
    with pytest.raises(ValueError):
        SpeakerEmbedding(np.zeros(256, dtype=np.float32))


def test_dimension_mismatch_is_rejected() -> None:
    left = SpeakerEmbedding(np.ones(3, dtype=np.float32))
    right = SpeakerEmbedding(np.ones(4, dtype=np.float32))

    with pytest.raises(ValueError):
        cosine_similarity(left, right)
