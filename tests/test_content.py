from __future__ import annotations

import numpy as np
import pytest

from voxera.content import ContentCadence, StreamingCadenceSelector


def test_default_cadence_selects_every_fourth_frame() -> None:
    features = np.arange(12 * 3, dtype=np.float32).reshape(12, 3)
    selected = ContentCadence().select(features)

    assert np.array_equal(selected, features[[3, 7, 11]])


def test_streaming_selector_matches_offline_for_irregular_chunks() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(size=(103, 256)).astype(np.float32)
    cadence = ContentCadence()
    expected = cadence.select(features)
    selector = StreamingCadenceSelector(cadence)

    chunks = (1, 2, 17, 3, 31, 4, 7, 38)
    outputs = []
    cursor = 0
    for size in chunks:
        end = min(features.shape[0], cursor + size)
        outputs.append(selector.push(features[cursor:end]))
        cursor = end
        if cursor == features.shape[0]:
            break

    if cursor < features.shape[0]:
        outputs.append(selector.push(features[cursor:]))

    actual = np.concatenate(outputs, axis=0)
    assert np.array_equal(actual, expected)
    assert selector.phase == features.shape[0] % cadence.stride


def test_reset_restores_cadence_phase() -> None:
    selector = StreamingCadenceSelector()
    features = np.ones((5, 8), dtype=np.float32)

    first = selector.push(features)
    selector.reset()
    second = selector.push(features)

    assert np.array_equal(first, second)
    assert selector.phase == 1


@pytest.mark.parametrize(
    ("stride", "offset"),
    [
        (0, 0),
        (-1, 0),
        (4, -1),
        (4, 4),
    ],
)
def test_invalid_cadence_is_rejected(stride: int, offset: int) -> None:
    with pytest.raises(ValueError):
        ContentCadence(stride=stride, offset=offset)
