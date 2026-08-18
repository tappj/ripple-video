import numpy as np
import pytest

from cutdetect.signals import procrustes_residual, scale_delta, word_gap_anomaly


def test_procrustes_residual_removes_similarity_transform() -> None:
    first = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)
    second = first @ rotation * 2.0 + np.asarray([5.0, 8.0])
    landmarks = np.zeros((2, 3, 3), dtype=np.float32)
    landmarks[0, :, :2] = first
    landmarks[1, :, :2] = second

    result = procrustes_residual(landmarks, np.asarray([True, True]))

    assert result[0] == pytest.approx(0.0, abs=1e-7)


def test_scale_delta_uses_boundary_coordinates() -> None:
    result = scale_delta(np.asarray([1.0, 1.0, 2.0]), np.asarray([True, True, True]))

    assert result.shape == (2,)
    assert result[0] == 0.0
    assert result[1] == pytest.approx(np.log(2.0))


def test_word_gap_anomaly_marks_the_gap_midpoint_boundary() -> None:
    words = [
        {"start": 0.0, "end": 0.2},
        {"start": 0.6, "end": 0.8},
        {"start": 0.81, "end": 1.0},
        {"start": 1.4, "end": 1.6},
    ]
    boundary_times = np.arange(0.0, 1.8, 0.1)

    result = word_gap_anomaly(words, boundary_times)

    assert int(np.nanargmax(result)) == 8
