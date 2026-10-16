from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

import pytest

from cutdetect.config import IngestConfig
from cutdetect.ingest import decode_frame_timestamps, ingest_video, iter_rgb_frames, probe_video


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


@pytest.fixture
def cfr_video(tmp_path: Path) -> Path:
    video = tmp_path / "cfr.mp4"
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=30:duration=0.5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.5",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(video),
        ]
    )
    return video


@pytest.fixture
def vfr_video(tmp_path: Path) -> Path:
    video = tmp_path / "vfr.mp4"
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=30:duration=1",
            "-vf",
            "select=if(lt(t\\,0.5)\\,not(mod(n\\,2))\\,not(mod(n\\,3)))",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ]
    )
    return video


def test_probe_reports_cfr_media_properties(cfr_video: Path) -> None:
    probe = probe_video(cfr_video)

    assert probe.frame_count == 15
    assert math.isclose(float(probe.fps), 30.0)
    assert (probe.width, probe.height) == (160, 90)
    assert probe.has_audio is True
    assert probe.audio_codec == "aac"
    assert probe.was_vfr is False


def test_ingest_extracts_canonical_audio_and_decodes_sequentially(
    cfr_video: Path, tmp_path: Path
) -> None:
    context = ingest_video(cfr_video, IngestConfig(cache_dir=tmp_path / "cache"))

    assert context.working_video_path == cfr_video
    assert context.audio_path is not None and context.audio_path.is_file()
    with wave.open(str(context.audio_path), "rb") as audio:
        assert audio.getframerate() == 48_000
        assert audio.getnchannels() == 1
    timestamps = decode_frame_timestamps(context.working_video_path)
    frames = list(iter_rgb_frames(context))
    assert len(timestamps) == context.frame_count == len(frames) == 15
    assert frames[0][2].shape == (90, 160, 3)


def test_vfr_input_is_normalized_and_mapped_to_source_pts(vfr_video: Path, tmp_path: Path) -> None:
    source_probe = probe_video(vfr_video)
    context = ingest_video(vfr_video, IngestConfig(cache_dir=tmp_path / "cache"))

    assert source_probe.was_vfr is True
    assert context.was_vfr is True
    assert context.working_video_path.name == "normalized.mp4"
    assert context.working_video_path.is_file()
    assert len(context.original_timestamps_sec) == context.frame_count
    assert set(context.original_timestamps_sec).issubset(set(source_probe.original_timestamps_sec))
    deltas = [
        right - left
        for left, right in zip(
            context.working_timestamps_sec, context.working_timestamps_sec[1:], strict=False
        )
    ]
    assert max(deltas) - min(deltas) < 0.000_1


def test_rotation_metadata_controls_display_dimensions_and_decoding(
    cfr_video: Path, tmp_path: Path
) -> None:
    rotated = tmp_path / "rotated.mp4"
    _run(
        [
            "ffmpeg",
            "-y",
            "-display_rotation:v:0",
            "90",
            "-i",
            str(cfr_video),
            "-c",
            "copy",
            str(rotated),
        ]
    )

    probe = probe_video(rotated)
    context = ingest_video(rotated, IngestConfig(cache_dir=tmp_path / "cache"))
    first_frame = next(iter_rgb_frames(context))[2]

    assert probe.rotation_deg == 90
    assert (probe.width, probe.height) == (90, 160)
    assert first_frame.shape == (160, 90, 3)
