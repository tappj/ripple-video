"""Reversible Phase D review operations over durable Phase C jobs."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import av
import numpy as np
import numpy.typing as npt

from cutdetect.pipeline.orchestration import (
    PhaseCStore,
    SegmentRecord,
    SegmentState,
)
from cutdetect.pipeline.runway_client import PipelineError
from cutdetect.pipeline.storage import LocalDiskStorage

_REVIEW_PROXY_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class MediaInfo:
    duration_sec: float
    fps: float
    frame_count: int
    has_audio: bool


@dataclass(frozen=True, slots=True)
class TrimSuggestion:
    start_frame: int
    end_frame: int
    original_end_frame: int
    trailing_silence_start_sec: float | None
    trailing_low_motion_start_sec: float | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


def inspect_media(path: Path) -> MediaInfo:
    """Read the timing profile used by review without modifying the media."""
    try:
        with av.open(str(path)) as container:
            videos = [stream for stream in container.streams if stream.type == "video"]
            if not videos:
                raise PipelineError(f"review media has no video stream: {path}")
            video = videos[0]
            fps = float(video.average_rate or video.base_rate or 0)
            if fps <= 0:
                raise PipelineError(f"review media has no usable frame rate: {path}")
            decoded_frames = sum(1 for _frame in container.decode(video=video.index))
            duration = (
                float(video.duration * video.time_base)
                if video.duration is not None and video.time_base is not None
                else decoded_frames / fps
            )
            return MediaInfo(
                duration_sec=duration,
                fps=fps,
                frame_count=decoded_frames,
                has_audio=any(stream.type == "audio" for stream in container.streams),
            )
    except (av.error.FFmpegError, OSError) as error:
        raise PipelineError(f"could not inspect review media {path}: {error}") from error


def _trailing_silence_start(path: Path, duration_sec: float) -> float | None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise PipelineError("required executable not found on PATH: ffmpeg")
    completed = subprocess.run(
        [
            executable,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "silencedetect=noise=-38dB:d=0.25",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    starts = [float(value) for value in re.findall(r"silence_start: ([0-9.]+)", completed.stderr)]
    ends = [float(value) for value in re.findall(r"silence_end: ([0-9.]+)", completed.stderr)]
    if not starts or not ends or ends[-1] < duration_sec - 0.12:
        return None
    return starts[-1]


def _trailing_low_motion_start(path: Path, fps: float) -> float | None:
    """Find the final low-motion run using small grayscale frame differences."""
    times: list[float] = []
    scores: list[float] = []
    previous: npt.NDArray[np.int16] | None = None
    with av.open(str(path)) as container:
        video = next((stream for stream in container.streams if stream.type == "video"), None)
        if video is None:
            return None
        video.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(video=video.index)):
            gray = frame.to_ndarray(format="gray")[::8, ::8].astype(np.int16)
            current_time = float(frame.time) if frame.time is not None else index / fps
            if previous is not None:
                scores.append(float(np.mean(np.abs(gray - previous))) / 255.0)
                times.append(current_time)
            previous = gray
    if not scores:
        return None
    cursor = len(scores) - 1
    while cursor >= 0 and scores[cursor] <= 0.008:
        cursor -= 1
    start_index = cursor + 1
    if start_index >= len(times):
        return None
    start = times[start_index]
    end = times[-1] + 1.0 / fps
    return start if end - start >= 0.25 else None


def suggest_trim(path: Path) -> TrimSuggestion:
    """Suggest a conservative tail trim only where silence and low motion overlap."""
    info = inspect_media(path)
    silence = _trailing_silence_start(path, info.duration_sec) if info.has_audio else None
    low_motion = _trailing_low_motion_start(path, info.fps)
    if silence is None or low_motion is None:
        return TrimSuggestion(
            start_frame=0,
            end_frame=info.frame_count,
            original_end_frame=info.frame_count,
            trailing_silence_start_sec=silence,
            trailing_low_motion_start_sec=low_motion,
            reason="no confident silence-plus-low-motion tail",
        )
    trim_sec = max(silence, low_motion)
    end_frame = max(1, min(info.frame_count, math.floor(trim_sec * info.fps)))
    if info.frame_count - end_frame < max(2, round(0.2 * info.fps)):
        end_frame = info.frame_count
    return TrimSuggestion(
        start_frame=0,
        end_frame=end_frame,
        original_end_frame=info.frame_count,
        trailing_silence_start_sec=silence,
        trailing_low_motion_start_sec=low_motion,
        reason=(
            "trailing silence and low motion overlap"
            if end_frame < info.frame_count
            else "overlap was too short to trim safely"
        ),
    )


def prepare_review_proxy(source: Path) -> Path:
    """Create a browser-safe review MP4 without modifying the provider output."""
    destination = source.with_name(f"{source.stem}_review_h264.mp4")
    if destination.is_file() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return destination
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise PipelineError("required executable not found on PATH: ffmpeg")
    with _REVIEW_PROXY_LOCK:
        if destination.is_file() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            return destination
        try:
            with av.open(str(source)) as container:
                video = next(
                    (stream for stream in container.streams if stream.type == "video"), None
                )
                audio = next(
                    (stream for stream in container.streams if stream.type == "audio"), None
                )
                pixel_format = cast(
                    str | None,
                    getattr(video.codec_context, "pix_fmt", None) if video is not None else None,
                )
                browser_safe = (
                    video is not None
                    and video.codec_context.name == "h264"
                    and pixel_format in {"yuv420p", "yuvj420p"}
                    and (audio is None or audio.codec_context.name == "aac")
                )
        except (av.error.FFmpegError, OSError) as error:
            message = f"could not inspect provider output for playback: {error}"
            raise PipelineError(message) from error
        temporary = destination.with_suffix(".mp4.part")
        command = [
            executable,
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-sn",
            "-dn",
        ]
        if browser_safe:
            command.extend(["-c", "copy"])
        else:
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-threads",
                    "1",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "160k",
                ]
            )
        command.extend(["-movflags", "+faststart", "-f", "mp4", str(temporary)])
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise PipelineError(f"could not prepare browser review media: {detail}")
        temporary.replace(destination)
    return destination


class ReviewService:
    """State-checked review, trim, approval, and regeneration operations."""

    custom_voice_fix_available = False
    custom_voice_fix_reason = (
        "Runway speech-to-speech currently accepts Runway preset voices, not this job's "
        "custom reference-audio sample. Use segment regeneration to retain the selected voice."
    )

    def __init__(self, *, store: PhaseCStore, storage: LocalDiskStorage) -> None:
        self.store = store
        self.storage = storage

    def _segment(self, job_id: str, index: int) -> SegmentRecord:
        segments = self.store.segments(job_id)
        if index < 0 or index >= len(segments):
            raise PipelineError(f"unknown segment {index} for job {job_id}")
        return segments[index]

    def _output_path(self, segment: SegmentRecord) -> Path:
        path = self.storage.path(segment.output_key)
        if not path.is_file():
            raise PipelineError(f"generated output is not available locally: {path}")
        return path

    def suggest(self, job_id: str, index: int) -> TrimSuggestion:
        segment = self._segment(job_id, index)
        if segment.state not in {SegmentState.READY_FOR_REVIEW, SegmentState.APPROVED}:
            raise PipelineError(f"segment {index} is not ready for review")
        return suggest_trim(self._output_path(segment))

    def trim(
        self,
        job_id: str,
        index: int,
        *,
        start_frame: int,
        end_frame: int,
    ) -> SegmentRecord:
        """Write a derived output_final while keeping output_raw byte-identical."""
        segment = self._segment(job_id, index)
        if segment.state not in {SegmentState.READY_FOR_REVIEW, SegmentState.APPROVED}:
            raise PipelineError(f"segment {index} cannot trim from {segment.state.value}")
        source = self._output_path(segment)
        info = inspect_media(source)
        if start_frame < 0 or end_frame <= start_frame or end_frame > info.frame_count:
            raise PipelineError(f"trim frames must satisfy 0 <= start < end <= {info.frame_count}")
        final_key = str(Path(segment.output_key).with_name("output_final.mp4"))
        destination = self.storage.path(final_key)
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise PipelineError("required executable not found on PATH: ffmpeg")
        start_sec = start_frame / info.fps
        end_sec = end_frame / info.fps
        command = [
            executable,
            "-y",
            "-v",
            "error",
            "-filter_threads",
            "1",
            "-filter_complex_threads",
            "1",
            "-i",
            str(source),
        ]
        if info.has_audio:
            command.extend(
                [
                    "-filter_complex",
                    (
                        f"[0:v]trim=start_frame={start_frame}:end_frame={end_frame},"
                        "setpts=PTS-STARTPTS[v];"
                        f"[0:a]atrim=start={start_sec:.12f}:end={end_sec:.12f},"
                        "asetpts=PTS-STARTPTS[a]"
                    ),
                    "-map",
                    "[v]",
                    "-map",
                    "[a]",
                ]
            )
        else:
            command.extend(
                [
                    "-vf",
                    f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS",
                ]
            )
        command.extend(
            [
                "-c:v",
                "libx264",
                "-threads",
                "1",
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
                str(destination),
            ]
        )
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise PipelineError(f"frame-accurate trim failed: {detail}")
        final_info = inspect_media(destination)
        updated = self.store.set_segment_state(
            job_id,
            index,
            SegmentState.READY_FOR_REVIEW,
            final_output_key=final_key,
            trim_start_frame=start_frame,
            trim_end_frame=end_frame,
            approved_at=None,
            actual_duration_sec=final_info.duration_sec,
        )
        self.store.record_event(
            job_id,
            "segment.trimmed",
            segment_index=index,
            payload={
                "start_frame": start_frame,
                "end_frame": end_frame,
                "output_key": final_key,
            },
        )
        return updated

    def approve(self, job_id: str, index: int) -> SegmentRecord:
        segment = self._segment(job_id, index)
        if segment.state == SegmentState.APPROVED:
            return segment
        if segment.state != SegmentState.READY_FOR_REVIEW:
            raise PipelineError(f"segment {index} cannot approve from {segment.state.value}")
        source = self._output_path(segment)
        final_key = segment.final_output_key or str(
            Path(segment.output_key).with_name("output_final.mp4")
        )
        final_path = self.storage.path(final_key)
        if segment.final_output_key is None:
            shutil.copy2(source, final_path)
        info = inspect_media(final_path)
        approved_at = datetime.now(UTC)
        updated = self.store.set_segment_state(
            job_id,
            index,
            SegmentState.APPROVED,
            final_output_key=final_key,
            approved_at=approved_at,
            actual_duration_sec=info.duration_sec,
        )
        self.store.record_event(
            job_id,
            "segment.approved",
            segment_index=index,
            payload={"output_key": final_key},
        )
        return updated

    def approve_all(self, job_id: str) -> tuple[SegmentRecord, ...]:
        for segment in self.store.segments(job_id):
            if segment.state not in {SegmentState.READY_FOR_REVIEW, SegmentState.APPROVED}:
                raise PipelineError(
                    f"cannot approve all while segment {segment.index} is {segment.state.value}"
                )
        for segment in self.store.segments(job_id):
            if segment.state == SegmentState.READY_FOR_REVIEW:
                self.approve(job_id, segment.index)
        return self.store.segments(job_id)

    def regenerate(
        self,
        job_id: str,
        index: int,
        *,
        prompt: str,
        max_credits: int,
    ) -> SegmentRecord:
        return self.store.request_regeneration(
            job_id,
            index,
            prompt=prompt,
            max_credits=max_credits,
        )

    def snapshot(self, job_id: str) -> dict[str, object]:
        job = self.store.job(job_id)
        segments = self.store.segments(job_id)
        approved = sum(segment.state == SegmentState.APPROVED for segment in segments)
        return {
            "job_id": job.id,
            "state": job.state.value,
            "approved_count": approved,
            "segment_count": len(segments),
            "can_stitch": bool(segments) and approved == len(segments),
            "submitted_credits": job.submitted_credits,
            "max_credits": job.max_credits,
            "custom_voice_fix": {
                "available": self.custom_voice_fix_available,
                "reason": self.custom_voice_fix_reason,
            },
            "segments": [
                {
                    "index": segment.index,
                    "state": segment.state.value,
                    "source_path": str(segment.input_path),
                    "output_path": (
                        str(self.storage.path(segment.output_key))
                        if self.storage.path(segment.output_key).is_file()
                        else None
                    ),
                    "final_output_path": (
                        str(self.storage.path(segment.final_output_key))
                        if segment.final_output_key is not None
                        else None
                    ),
                    "planned_duration_sec": segment.duration_sec,
                    "actual_duration_sec": segment.actual_duration_sec,
                    "hard_cut_offsets_sec": segment.hard_cut_offsets_sec,
                    "trim_start_frame": segment.trim_start_frame,
                    "trim_end_frame": segment.trim_end_frame,
                    "incremental_regeneration_credits": segment.estimated_credits,
                    "prompt": segment.prompt_override or job.prompt,
                }
                for segment in segments
            ],
        }
