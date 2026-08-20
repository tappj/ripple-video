import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from cutdetect.config import SignalConfig
from cutdetect.signals import (
    background_delta,
    compute_signal_bundle,
    procrustes_residual,
    scale_delta,
    word_gap_anomaly,
)


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


def test_background_delta_streams_temporal_variance_without_changing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grayscale = np.arange(5 * 8 * 6, dtype=np.uint8).reshape(5, 8, 6)
    bboxes = np.tile(np.asarray([2.0, 2.0, 3.0, 3.0], dtype=np.float32), (5, 1))
    captured: dict[str, npt.NDArray[np.float64]] = {}

    def capture_percentile(values: npt.NDArray[np.float64], _percentile: float) -> float:
        captured["temporal"] = values.copy()
        return float("inf")

    monkeypatch.setattr("cutdetect.signals.np.percentile", capture_percentile)

    result, captions_masked = background_delta(
        grayscale,
        bboxes,
        (6, 8),
        4,
        SignalConfig(),
    )

    expected = np.std(grayscale.astype(np.float32), axis=0)
    assert np.allclose(captured["temporal"], expected, atol=1e-6)
    assert result.shape == (4,)
    assert not captions_masked


def test_signal_bundle_reduces_dense_flow_without_promoting_full_cube(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_count = 3
    feature_path = tmp_path / "features.npz"
    np.savez_compressed(
        feature_path,
        metadata=np.asarray(json.dumps({"source_width": 4, "source_height": 4})),
        frame_times_sec=np.arange(frame_count, dtype=np.float64) / 30,
        face_available=np.zeros(frame_count, dtype=np.bool_),
        face_landmarks_px=np.empty((frame_count, 0, 3), dtype=np.float32),
        facial_transformation_matrices=np.full((frame_count, 4, 4), np.nan, dtype=np.float32),
        face_blendshapes=np.empty((frame_count, 0), dtype=np.float32),
        face_blendshape_names=np.empty(0, dtype="U64"),
        face_bbox_px=np.tile(np.asarray([1, 1, 2, 2], dtype=np.float32), (frame_count, 1)),
        face_scale_px=np.full(frame_count, np.nan, dtype=np.float32),
        face_centroid_px=np.full((frame_count, 2), np.nan, dtype=np.float32),
        grayscale=np.arange(frame_count * 16, dtype=np.uint8).reshape(frame_count, 4, 4),
        hsv_histograms=np.zeros((frame_count, 6), dtype=np.float32),
        dense_flow=np.zeros((frame_count - 1, 4, 4, 2), dtype=np.float16),
        audio_times_sec=np.empty(0, dtype=np.float64),
        audio_spectral_flux=np.empty(0, dtype=np.float32),
        audio_high_band_rms=np.empty(0, dtype=np.float32),
        audio_waveform=np.empty(0, dtype=np.float32),
        audio_sample_rate=np.asarray(0, dtype=np.int32),
        audio_f0_hz=np.empty(0, dtype=np.float32),
        audio_mfcc=np.empty((0, 13), dtype=np.float32),
    )
    observed: dict[str, np.dtype[np.generic]] = {}

    def reduce_flow(
        flow: npt.NDArray[np.float16], _config: SignalConfig
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]:
        observed["dtype"] = flow.dtype
        return np.zeros(frame_count - 1), {}

    monkeypatch.setattr("cutdetect.signals.flow_incoherence", reduce_flow)

    bundle = compute_signal_bundle(feature_path, tmp_path / "missing.mp4")

    assert observed["dtype"] == np.dtype(np.float16)
    assert bundle.raw["flow_incoherence"].shape == (frame_count - 1,)

    with np.load(feature_path, allow_pickle=False) as legacy:
        compact = {
            name: legacy[name]
            for name in legacy.files
            if name not in {"dense_flow", "grayscale"}
        }
    compact.update(
        flow_incoherence=np.zeros(frame_count - 1, dtype=np.float64),
        flow_magnitude=np.zeros(frame_count - 1, dtype=np.float64),
        flow_inlier_ratio=np.ones(frame_count - 1, dtype=np.float64),
        stabilized_residual=np.full(frame_count - 1, np.nan, dtype=np.float64),
        background_delta=np.zeros(frame_count - 1, dtype=np.float64),
        captions_region_masked=np.asarray(False, dtype=np.bool_),
    )
    compact_path = tmp_path / "features-compact.npz"
    np.savez_compressed(compact_path, **compact)

    def reject_legacy(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("compact archives must not load full-frame legacy tensors")

    monkeypatch.setattr("cutdetect.signals.flow_incoherence", reject_legacy)
    monkeypatch.setattr("cutdetect.signals.stabilized_residual", reject_legacy)
    monkeypatch.setattr("cutdetect.signals.background_delta", reject_legacy)

    compact_bundle = compute_signal_bundle(compact_path, tmp_path / "missing.mp4")

    assert compact_bundle.raw["background_delta"].shape == (frame_count - 1,)
