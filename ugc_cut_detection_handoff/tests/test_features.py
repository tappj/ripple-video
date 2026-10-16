import json
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from cutdetect.config import FeatureConfig
from cutdetect.features import (
    _audio_features,
    _disk_array,
    _frame_grid,
    _mel_filterbank,
    load_feature_metadata,
)
from cutdetect.ingest import VideoContext


def _context(tmp_path: Path, audio_path: Path | None = None) -> VideoContext:
    timestamps = (0.0, 1 / 30)
    return VideoContext(
        source_path=tmp_path / "source.mp4",
        working_video_path=tmp_path / "source.mp4",
        artifact_dir=tmp_path,
        audio_path=audio_path,
        cache_key="abc",
        duration_sec=2 / 30,
        fps=Fraction(30, 1),
        frame_count=2,
        width=720,
        height=1280,
        was_vfr=False,
        has_audio=audio_path is not None,
        video_codec="h264",
        audio_codec="pcm_s16le" if audio_path else None,
        source_rotation_deg=0,
        working_timestamps_sec=timestamps,
        original_timestamps_sec=timestamps,
    )


def test_frame_grid_preserves_vertical_aspect_ratio(tmp_path: Path) -> None:
    assert _frame_grid(_context(tmp_path), 96) == (96, 171)


def test_large_feature_arrays_are_disk_backed(tmp_path: Path) -> None:
    array = _disk_array(tmp_path, "flow", (120, 96, 54, 2), np.float16, fill=np.nan)

    assert isinstance(array, np.memmap)
    assert array.shape == (120, 96, 54, 2)
    assert array.filename is not None
    assert Path(array.filename) == tmp_path / "flow.npy"


def test_mel_filterbank_is_nonnegative_and_has_expected_shape() -> None:
    filters = _mel_filterbank(48_000, 2048, 40, 50.0, 12_000.0)

    assert filters.shape == (40, 1025)
    assert np.all(filters >= 0.0)
    assert np.all(filters.sum(axis=1) > 0.0)


def test_audio_features_use_five_millisecond_grid(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    sample_rate = 48_000
    samples = (np.sin(2 * np.pi * 200 * np.arange(sample_rate // 10) / sample_rate) * 8000).astype(
        "<i2"
    )
    audio_path = tmp_path / "audio.wav"
    with wave.open(str(audio_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()

    features = _audio_features(_context(tmp_path, audio_path), FeatureConfig(), scratch_dir)

    times = features["audio_times_sec"]
    assert times.shape[0] > 10
    assert float(times[1] - times[0]) == pytest.approx(0.005)
    assert features["audio_log_mel"].shape[1] == 40
    assert features["audio_mfcc"].shape[1] == 13
    assert isinstance(features["audio_waveform"], np.memmap)
    assert (scratch_dir / "audio_power.npy").is_file()


def test_load_feature_metadata(tmp_path: Path) -> None:
    path = tmp_path / "features.npz"
    expected = {"extractor_version": "1.0", "face_detection_rate": 1.0}
    np.savez(path, metadata=np.asarray(json.dumps(expected)))

    assert load_feature_metadata(path) == expected
