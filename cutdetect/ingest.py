"""Frame-accurate video ingest and CFR normalization.

The implementation intentionally uses ffprobe for container inspection and
PyAV for sequential decoding. No frame-index seeking is used.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import numpy.typing as npt

from cutdetect.config import IngestConfig


class IngestError(RuntimeError):
    """Raised when media cannot be probed or normalized."""


@dataclass(frozen=True, slots=True)
class VideoProbe:
    """Properties reported for the source video before normalization."""

    path: Path
    duration_sec: float
    fps: Fraction
    frame_count: int
    width: int
    height: int
    was_vfr: bool
    has_audio: bool
    video_codec: str
    audio_codec: str | None
    rotation_deg: int
    original_timestamps_sec: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "path": str(self.path),
            "duration_sec": self.duration_sec,
            "fps": float(self.fps),
            "fps_fraction": str(self.fps),
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "was_vfr": self.was_vfr,
            "has_audio": self.has_audio,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "rotation_deg": self.rotation_deg,
        }


@dataclass(frozen=True, slots=True)
class VideoContext:
    """Canonical media paths and frame-to-source-timestamp mapping."""

    source_path: Path
    working_video_path: Path
    artifact_dir: Path
    audio_path: Path | None
    cache_key: str
    duration_sec: float
    fps: Fraction
    frame_count: int
    width: int
    height: int
    was_vfr: bool
    has_audio: bool
    video_codec: str
    audio_codec: str | None
    source_rotation_deg: int
    working_timestamps_sec: tuple[float, ...]
    original_timestamps_sec: tuple[float, ...]

    def source_time_for_frame(self, frame: int) -> float:
        """Map a working-domain frame index to its nearest source PTS."""
        if frame < 0 or frame >= self.frame_count:
            raise IndexError(f"frame {frame} outside [0, {self.frame_count})")
        return self.original_timestamps_sec[frame]

    def to_dict(self, *, include_timestamps: bool = True) -> dict[str, object]:
        """Return the stable, JSON-compatible context representation."""
        result: dict[str, object] = {
            "source_path": str(self.source_path),
            "working_video_path": str(self.working_video_path),
            "artifact_dir": str(self.artifact_dir),
            "audio_path": str(self.audio_path) if self.audio_path else None,
            "cache_key": self.cache_key,
            "duration_sec": self.duration_sec,
            "fps": float(self.fps),
            "fps_fraction": str(self.fps),
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "was_vfr": self.was_vfr,
            "has_audio": self.has_audio,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "source_rotation_deg": self.source_rotation_deg,
        }
        if include_timestamps:
            result["working_timestamps_sec"] = list(self.working_timestamps_sec)
            result["original_timestamps_sec"] = list(self.original_timestamps_sec)
        return result


def _require_binary(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise IngestError(f"required executable not found on PATH: {name}")
    return executable


def _run_json(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise IngestError(f"command failed ({command[0]}): {detail}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise IngestError(f"invalid JSON from {command[0]}: {error}") from error
    if not isinstance(value, dict):
        raise IngestError(f"unexpected JSON root from {command[0]}")
    return cast(dict[str, Any], value)


def _parse_fraction(value: object) -> Fraction:
    if not isinstance(value, str) or value in {"", "0/0", "N/A"}:
        return Fraction(0, 1)
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise IngestError(f"invalid frame rate from ffprobe: {value!r}") from error


def _parse_float(value: object, fallback: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def _rotation(stream: dict[str, Any]) -> int:
    raw: object = stream.get("tags", {}).get("rotate")
    for side_data in stream.get("side_data_list", []):
        if isinstance(side_data, dict) and "rotation" in side_data:
            raw = side_data["rotation"]
            break
    angle = round(_parse_float(raw)) % 360
    return min(
        (0, 90, 180, 270),
        key=lambda candidate: min(abs(candidate - angle), 360 - abs(candidate - angle)),
    )


def _timestamps_are_variable(timestamps: Sequence[float], config: IngestConfig) -> bool:
    if len(timestamps) < 3:
        return False
    deltas = [right - left for left, right in pairwise(timestamps)]
    if any(delta <= 0.0 for delta in deltas):
        return True
    ordered = sorted(deltas)
    median = ordered[len(ordered) // 2]
    tolerance = max(config.pts_absolute_tolerance_sec, median * config.pts_relative_tolerance)
    return any(abs(delta - median) > tolerance for delta in deltas)


def probe_video(path: str | Path, config: IngestConfig | None = None) -> VideoProbe:
    """Inspect media metadata and frame PTS values with ffprobe."""
    settings = config or IngestConfig()
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    ffprobe = _require_binary("ffprobe")
    metadata = _run_json(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-show_entries",
            (
                "stream=index,codec_type,codec_name,width,height,r_frame_rate,"
                "avg_frame_rate,nb_frames,duration:stream_tags=rotate:"
                "stream_side_data=rotation:format=duration"
            ),
            "-of",
            "json",
            str(source),
        ]
    )
    frames = _run_json(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(source),
        ]
    )
    streams = metadata.get("streams", [])
    video_streams = [item for item in streams if item.get("codec_type") == "video"]
    audio_streams = [item for item in streams if item.get("codec_type") == "audio"]
    if not video_streams:
        raise IngestError(f"no video stream found: {source}")
    video = cast(dict[str, Any], video_streams[0])
    frame_items = frames.get("frames", [])
    timestamps = tuple(
        _parse_float(item.get("best_effort_timestamp_time"))
        for item in frame_items
        if isinstance(item, dict) and "best_effort_timestamp_time" in item
    )
    average_rate = _parse_fraction(video.get("avg_frame_rate"))
    nominal_rate = _parse_fraction(video.get("r_frame_rate"))
    fps = average_rate or nominal_rate
    if fps <= 0:
        raise IngestError(f"could not determine a positive frame rate: {source}")
    rate_mismatch = False
    if average_rate > 0 and nominal_rate > 0:
        rate_mismatch = not math.isclose(
            float(average_rate),
            float(nominal_rate),
            rel_tol=settings.rate_relative_tolerance,
        )
    rotation = _rotation(video)
    coded_width = int(video.get("width", 0))
    coded_height = int(video.get("height", 0))
    width, height = (
        (coded_height, coded_width) if rotation in {90, 270} else (coded_width, coded_height)
    )
    duration = _parse_float(video.get("duration"))
    if duration <= 0.0:
        duration = _parse_float(metadata.get("format", {}).get("duration"))
    if duration <= 0.0 and timestamps:
        duration = timestamps[-1] - timestamps[0] + (1.0 / float(fps))
    declared_frames = int(video.get("nb_frames", 0) or 0)
    frame_count = len(timestamps) or declared_frames or round(duration * float(fps))
    if not timestamps:
        timestamps = tuple(index / float(fps) for index in range(frame_count))
    audio_codec = str(audio_streams[0].get("codec_name", "unknown")) if audio_streams else None
    return VideoProbe(
        path=source,
        duration_sec=duration,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        was_vfr=rate_mismatch or _timestamps_are_variable(timestamps, settings),
        has_audio=bool(audio_streams),
        video_codec=str(video.get("codec_name", "unknown")),
        audio_codec=audio_codec,
        rotation_deg=rotation,
        original_timestamps_sec=timestamps,
    )


def _content_hash(path: Path, chunk_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "cutdetect"


def _run_ffmpeg(command: Sequence[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise IngestError(f"ffmpeg failed: {detail}")


def _normalize_to_cfr(source: Path, destination: Path, fps: Fraction, config: IngestConfig) -> None:
    ffmpeg = _require_binary("ffmpeg")
    _run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-autorotate",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-r",
            str(fps),
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-preset",
            config.normalization_preset,
            "-crf",
            str(config.normalization_crf),
            "-metadata:s:v:0",
            "rotate=0",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )


def _extract_audio(source: Path, destination: Path, config: IngestConfig) -> None:
    ffmpeg = _require_binary("ffmpeg")
    _run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            str(config.audio_channels),
            "-ar",
            str(config.audio_sample_rate),
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )


def decode_frame_timestamps(path: str | Path) -> tuple[float, ...]:
    """Sequentially decode a video with PyAV and return presentation times."""
    result: list[float] = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise IngestError(f"no video stream found: {path}")
        stream = container.streams.video[0]
        stream.thread_count = 1
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                if stream.average_rate is None:
                    raise IngestError(f"decoded frame has no timestamp or average rate: {path}")
                result.append(len(result) / float(stream.average_rate))
            else:
                result.append(float(frame.pts * frame.time_base))
    return tuple(result)


def iter_rgb_frames(context: VideoContext) -> Iterator[tuple[int, float, npt.NDArray[np.uint8]]]:
    """Yield display-oriented RGB frames sequentially from the working video."""
    working_rotation = 0 if context.was_vfr else context.source_rotation_deg
    with av.open(str(context.working_video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            image = cast(npt.NDArray[np.uint8], frame.to_ndarray(format="rgb24"))
            if working_rotation:
                image = cast(npt.NDArray[np.uint8], np.rot90(image, k=working_rotation // 90))
            timestamp = (
                float(frame.pts * frame.time_base)
                if frame.pts is not None and frame.time_base is not None
                else index / float(context.fps)
            )
            yield index, timestamp, image


def _nearest_source_times(
    working_timestamps: Sequence[float], source_timestamps: Sequence[float]
) -> tuple[float, ...]:
    if not source_timestamps:
        return tuple(working_timestamps)
    result: list[float] = []
    for timestamp in working_timestamps:
        position = bisect.bisect_left(source_timestamps, timestamp)
        candidates = source_timestamps[max(0, position - 1) : position + 1]
        result.append(min(candidates, key=lambda source_time: abs(source_time - timestamp)))
    return tuple(result)


def ingest_video(path: str | Path, config: IngestConfig | None = None) -> VideoContext:
    """Probe, normalize when needed, decode timestamps, and extract audio."""
    settings = config or IngestConfig()
    probe = probe_video(path, settings)
    cache_key = _content_hash(probe.path, settings.hash_chunk_bytes)
    cache_root = (settings.cache_dir or _default_cache_dir()).expanduser().resolve()
    artifact_dir = cache_root / cache_key
    artifact_dir.mkdir(parents=True, exist_ok=True)

    working_video = probe.path
    if probe.was_vfr:
        working_video = artifact_dir / "normalized.mp4"
        if not working_video.is_file():
            _normalize_to_cfr(probe.path, working_video, probe.fps, settings)

    audio_path: Path | None = None
    if probe.has_audio:
        audio_path = artifact_dir / "audio.wav"
        if not audio_path.is_file():
            _extract_audio(probe.path, audio_path, settings)

    working_timestamps = decode_frame_timestamps(working_video)
    original_mapping = (
        _nearest_source_times(working_timestamps, probe.original_timestamps_sec)
        if probe.was_vfr
        else probe.original_timestamps_sec
    )
    if len(original_mapping) != len(working_timestamps):
        original_mapping = _nearest_source_times(working_timestamps, probe.original_timestamps_sec)

    working_probe = probe_video(working_video, settings) if probe.was_vfr else probe
    context = VideoContext(
        source_path=probe.path,
        working_video_path=working_video,
        artifact_dir=artifact_dir,
        audio_path=audio_path,
        cache_key=cache_key,
        duration_sec=working_probe.duration_sec,
        fps=working_probe.fps,
        frame_count=len(working_timestamps),
        width=working_probe.width,
        height=working_probe.height,
        was_vfr=probe.was_vfr,
        has_audio=probe.has_audio,
        video_codec=probe.video_codec,
        audio_codec=probe.audio_codec,
        source_rotation_deg=probe.rotation_deg,
        working_timestamps_sec=working_timestamps,
        original_timestamps_sec=original_mapping,
    )
    context_path = artifact_dir / "context.json"
    context_path.write_text(json.dumps(context.to_dict(), indent=2) + "\n", encoding="utf-8")
    return context
