"""Frame-accurate video splitting and downloadable clip bundles."""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import cast

from cutdetect.config import ExportConfig, IngestConfig
from cutdetect.ingest import IngestError, ingest_video


class ExportError(RuntimeError):
    """Raised when cut boundaries or FFmpeg exports are invalid."""


@dataclass(frozen=True, slots=True)
class ExportedClip:
    """One encoded segment between two detected boundaries."""

    index: int
    path: Path
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    duration_sec: float
    frame_count: int

    def to_dict(self) -> dict[str, object]:
        """Return manifest-ready clip metadata."""
        value = asdict(self)
        value["path"] = str(self.path)
        value["filename"] = self.path.name
        return cast(dict[str, object], value)


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Complete clip export and bundle metadata."""

    output_dir: Path
    manifest_path: Path
    zip_path: Path
    cut_count: int
    clips: tuple[ExportedClip, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible export summary."""
        return {
            "output_dir": str(self.output_dir),
            "manifest": str(self.manifest_path),
            "zip": str(self.zip_path),
            "cut_count": self.cut_count,
            "clip_count": len(self.clips),
            "clips": [clip.to_dict() for clip in self.clips],
        }


def load_cut_frames(path: str | Path) -> tuple[int, ...]:
    """Load cut frames from the stable prediction contract."""
    root: object = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(root, dict) or not isinstance(root.get("cuts"), list):
        raise ValueError("prediction JSON must contain a cuts array")
    frames: list[int] = []
    for item in root["cuts"]:
        if not isinstance(item, dict) or "frame" not in item:
            raise ValueError("each cut must contain a frame")
        frames.append(int(item["frame"]))
    return tuple(frames)


def _validate_boundaries(cut_frames: Sequence[int], frame_count: int) -> tuple[int, ...]:
    cuts = tuple(int(frame) for frame in cut_frames)
    if tuple(sorted(set(cuts))) != cuts:
        raise ValueError("cut frames must be strictly increasing and unique")
    if any(frame <= 0 or frame >= frame_count for frame in cuts):
        raise ValueError(f"cut frames must lie strictly inside [0, {frame_count})")
    return (0, *cuts, frame_count)


def _run_ffmpeg(command: Sequence[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ExportError(f"FFmpeg clip export failed: {detail}")


def split_video(
    video_path: str | Path,
    cut_frames: Sequence[int],
    output_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    config: ExportConfig | None = None,
) -> ExportResult:
    """Re-encode exact frame ranges into one independently playable file each."""
    settings = config or ExportConfig()
    context = ingest_video(
        video_path,
        IngestConfig(cache_dir=Path(cache_dir) if cache_dir is not None else None),
    )
    boundaries = _validate_boundaries(cut_frames, context.frame_count)
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise IngestError("required executable not found on PATH: ffmpeg")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    clips: list[ExportedClip] = []
    fps = float(context.fps)
    for index, (start_frame, end_frame) in enumerate(pairwise(boundaries), start=1):
        start_sec = start_frame / fps
        end_sec = end_frame / fps
        frame_count = end_frame - start_frame
        filename = (
            f"clip_{index:0{settings.filename_digits}d}_f{start_frame:06d}-f{end_frame - 1:06d}.mp4"
        )
        path = destination / filename
        command = [
            executable,
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start_sec:.12f}",
            "-i",
            str(context.working_video_path),
        ]
        if context.audio_path is not None:
            command.extend(["-ss", f"{start_sec:.12f}", "-i", str(context.audio_path)])
        command.extend(["-map", "0:v:0"])
        if context.audio_path is not None:
            command.extend(["-map", "1:a:0"])
        command.extend(
            [
                "-frames:v",
                str(frame_count),
                "-t",
                f"{end_sec - start_sec:.12f}",
                "-c:v",
                settings.video_codec,
                "-preset",
                settings.video_preset,
                "-crf",
                str(settings.video_crf),
                "-pix_fmt",
                "yuv420p",
            ]
        )
        if context.audio_path is not None:
            command.extend(["-c:a", settings.audio_codec, "-b:a", settings.audio_bitrate])
        command.extend(["-movflags", "+faststart", str(path)])
        _run_ffmpeg(command)
        clips.append(
            ExportedClip(
                index=index,
                path=path,
                start_frame=start_frame,
                end_frame=end_frame,
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=end_sec - start_sec,
                frame_count=frame_count,
            )
        )
    manifest_path = destination / "manifest.json"
    zip_path = destination / "clips.zip"
    result = ExportResult(destination, manifest_path, zip_path, len(cut_frames), tuple(clips))
    manifest_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(manifest_path, manifest_path.name)
        for clip in clips:
            archive.write(clip.path, clip.path.name)
    return result


def split_from_predictions(
    video_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    config: ExportConfig | None = None,
) -> ExportResult:
    """Split a video using cut frames from a prediction contract."""
    return split_video(
        video_path,
        load_cut_frames(predictions_path),
        output_dir,
        cache_dir=cache_dir,
        config=config,
    )
