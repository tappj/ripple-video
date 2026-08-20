"""Small media transformations shared by generation and review."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from cutdetect.ingest import probe_video
from cutdetect.pipeline.runway_client import PipelineError


def trim_generated_duration(generated: Path, duration_sec: float, destination: Path) -> Path:
    """Create a frame-accurate review copy ending at the source-group boundary."""
    if duration_sec <= 0:
        raise PipelineError("trim duration must be positive")
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise PipelineError("required executable not found on PATH: ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".mp4.part")
    completed = subprocess.run(
        [
            executable,
            "-y",
            "-v",
            "error",
            "-filter_threads",
            "1",
            "-i",
            str(generated),
            "-t",
            f"{duration_sec:.12f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(temporary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PipelineError(f"could not trim generated clip: {detail}")
    temporary.replace(destination)
    return destination


def preserve_source_audio(generated: Path, source: Path, destination: Path) -> Path:
    """Pair generated visuals with the exact audio from their source segment."""
    source_info = probe_video(source)
    if not source_info.has_audio:
        raise PipelineError("the source video has no audio; add an optional voice reference")
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise PipelineError("required executable not found on PATH: ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".mp4.part")
    completed = subprocess.run(
        [
            executable,
            "-y",
            "-v",
            "error",
            "-filter_threads",
            "1",
            "-i",
            str(generated),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-t",
            f"{source_info.duration_sec:.12f}",
            "-c:v",
            "copy",
            "-threads",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(temporary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PipelineError(f"could not preserve source audio: {detail}")
    temporary.replace(destination)
    return destination
