"""Canonical Phase E stitching, validation, and visual QC artifacts."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from cutdetect.ingest import probe_video
from cutdetect.pipeline.orchestration import JobState, PhaseCStore, SegmentState
from cutdetect.pipeline.review import inspect_media
from cutdetect.pipeline.runway_client import PipelineError
from cutdetect.pipeline.storage import LocalDiskStorage


@dataclass(frozen=True, slots=True)
class StitchValidation:
    expected_duration_sec: float
    video_duration_sec: float
    audio_duration_sec: float
    audio_video_delta_sec: float
    expected_frame_count: int
    actual_frame_count: int
    width: int
    height: int
    fps: float
    pixel_format: str
    sample_aspect_ratio: str
    valid: bool


@dataclass(frozen=True, slots=True)
class StitchResult:
    job_id: str
    final_path: Path
    qc_path: Path
    validation: StitchValidation

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "final_path": str(self.final_path),
            "qc_path": str(self.qc_path),
            "validation": asdict(self.validation),
        }


def _binary(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise PipelineError(f"required executable not found on PATH: {name}")
    return executable


def _run(command: list[str], operation: str) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PipelineError(f"{operation} failed: {detail}")


def _stream_metadata(path: Path) -> list[dict[str, object]]:
    completed = subprocess.run(
        [
            _binary("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            (
                "stream=codec_type,duration,nb_frames,r_frame_rate,width,height,pix_fmt,"
                "sample_aspect_ratio"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise PipelineError(f"could not validate stitched output: {completed.stderr.strip()}")
    root = json.loads(completed.stdout)
    streams = root.get("streams", []) if isinstance(root, dict) else []
    return cast(list[dict[str, object]], streams)


def _fraction(value: object) -> float:
    numerator, denominator = str(value).split("/", maxsplit=1)
    return float(numerator) / float(denominator)


def _validate(
    path: Path,
    *,
    expected_duration_sec: float,
    expected_width: int,
    expected_height: int,
    expected_fps: float,
) -> StitchValidation:
    streams = _stream_metadata(path)
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        raise PipelineError("stitched output must contain one video and one audio stream")
    video_duration = float(str(video.get("duration", 0)))
    audio_duration = float(str(audio.get("duration", 0)))
    fps = _fraction(video.get("r_frame_rate", "0/1"))
    actual_frames = int(str(video.get("nb_frames", 0)))
    expected_frames = round(expected_duration_sec * expected_fps)
    frame_tolerance = max(2, 2 * math.ceil(expected_duration_sec / 10))
    duration_tolerance = max(0.12, frame_tolerance / expected_fps)
    pixel_format = str(video.get("pix_fmt", ""))
    sample_aspect_ratio = str(video.get("sample_aspect_ratio", ""))
    valid = (
        int(str(video.get("width", 0))) == expected_width
        and int(str(video.get("height", 0))) == expected_height
        and abs(fps - expected_fps) <= 0.01
        and pixel_format == "yuv420p"
        and sample_aspect_ratio in {"1:1", "N/A"}
        and abs(video_duration - expected_duration_sec) <= duration_tolerance
        and abs(audio_duration - video_duration) <= 0.12
        and abs(actual_frames - expected_frames) <= frame_tolerance
    )
    return StitchValidation(
        expected_duration_sec=expected_duration_sec,
        video_duration_sec=video_duration,
        audio_duration_sec=audio_duration,
        audio_video_delta_sec=abs(audio_duration - video_duration),
        expected_frame_count=expected_frames,
        actual_frame_count=actual_frames,
        width=int(str(video.get("width", 0))),
        height=int(str(video.get("height", 0))),
        fps=fps,
        pixel_format=pixel_format,
        sample_aspect_ratio=sample_aspect_ratio,
        valid=valid,
    )


def _cut_flash_filter(times: list[float]) -> str:
    filters = []
    for time_sec in times:
        start = max(0.0, time_sec - 0.06)
        end = time_sec + 0.06
        filters.append(
            "drawbox=x=0:y=0:w=iw:h=ih:color=0xff5138@0.9:t=8:"
            f"enable='between(t,{start:.6f},{end:.6f})'"
        )
    return ",".join(filters)


def _qc_artifact(
    source: Path,
    final: Path,
    destination: Path,
    *,
    source_cut_times: list[float],
    output_cut_times: list[float],
) -> None:
    source_info = inspect_media(source)
    output_info = inspect_media(final)
    duration = max(source_info.duration_sec, output_info.duration_sec)
    cell_width, cell_height = 480, 854
    source_pad = max(0.0, duration - source_info.duration_sec)
    output_pad = max(0.0, duration - output_info.duration_sec)
    common = (
        f"scale={cell_width}:{cell_height}:force_original_aspect_ratio=decrease,"
        f"pad={cell_width}:{cell_height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
    )
    source_filters = [common, f"tpad=stop_mode=clone:stop_duration={source_pad:.6f}"]
    output_filters = [common, f"tpad=stop_mode=clone:stop_duration={output_pad:.6f}"]
    source_flash = _cut_flash_filter(source_cut_times)
    output_flash = _cut_flash_filter(output_cut_times)
    if source_flash:
        source_filters.append(source_flash)
    if output_flash:
        output_filters.append(output_flash)
    filter_complex = (
        f"[0:v]{','.join(source_filters)}[source];"
        f"[1:v]{','.join(output_filters)}[output];"
        "[source][output]hstack=inputs=2,format=yuv420p[qc]"
    )
    _run(
        [
            _binary("ffmpeg"),
            "-y",
            "-v",
            "error",
            "-filter_threads",
            "1",
            "-filter_complex_threads",
            "1",
            "-i",
            str(source),
            "-i",
            str(final),
            "-filter_complex",
            filter_complex,
            "-map",
            "[qc]",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        "QC comparison render",
    )


def stitch_job(
    store: PhaseCStore,
    storage: LocalDiskStorage,
    job_id: str,
) -> StitchResult:
    """Normalize, concatenate, validate, and publish only fully approved clips."""
    job = store.job(job_id)
    if job.state == JobState.COMPLETE and job.final_output_key and job.qc_output_key:
        final = storage.path(job.final_output_key)
        qc = storage.path(job.qc_output_key)
        source_probe = probe_video(job.source_path)
        expected = sum(
            inspect_media(storage.path(segment.final_output_key or "")).duration_sec
            for segment in store.segments(job_id)
        )
        return StitchResult(
            job_id,
            final,
            qc,
            _validate(
                final,
                expected_duration_sec=expected,
                expected_width=source_probe.width,
                expected_height=source_probe.height,
                expected_fps=float(source_probe.fps),
            ),
        )
    segments = store.segments(job_id)
    if not segments or any(segment.state != SegmentState.APPROVED for segment in segments):
        raise PipelineError("stitching is locked until every generated clip is approved")
    clip_paths: list[Path] = []
    clip_durations: list[float] = []
    for segment in segments:
        if segment.final_output_key is None:
            raise PipelineError(f"approved segment {segment.index} has no final output")
        path = storage.path(segment.final_output_key)
        if not path.is_file():
            raise PipelineError(f"approved segment {segment.index} output is missing: {path}")
        media = inspect_media(path)
        if not media.has_audio:
            raise PipelineError(f"approved segment {segment.index} has no audio stream")
        clip_paths.append(path)
        clip_durations.append(media.duration_sec)

    source_profile = probe_video(job.source_path)
    width, height, fps = source_profile.width, source_profile.height, float(source_profile.fps)
    store.set_job_state(job_id, JobState.STITCHING)
    final_key = f"jobs/{job_id}/final.mp4"
    qc_key = f"jobs/{job_id}/qc_comparison.mp4"
    final_path = storage.path(final_key)
    qc_path = storage.path(qc_key)
    filters: list[str] = []
    concat_inputs = []
    for index in range(len(clip_paths)):
        filters.append(
            f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
            f"fps={fps:.12f},setsar=1,format=yuv420p,setpts=PTS-STARTPTS[v{index}]"
        )
        filters.append(
            f"[{index}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"asetpts=PTS-STARTPTS[a{index}]"
        )
        concat_inputs.append(f"[v{index}][a{index}]")
    filters.append(f"{''.join(concat_inputs)}concat=n={len(clip_paths)}:v=1:a=1[video][audio]")
    command = [
        _binary("ffmpeg"),
        "-y",
        "-v",
        "error",
        "-filter_threads",
        "1",
        "-filter_complex_threads",
        "1",
    ]
    for path in clip_paths:
        command.extend(["-i", str(path)])
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[video]",
            "-map",
            "[audio]",
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
            str(final_path),
        ]
    )
    try:
        _run(command, "canonical stitch")
        expected_duration = sum(clip_durations)
        validation = _validate(
            final_path,
            expected_duration_sec=expected_duration,
            expected_width=width,
            expected_height=height,
            expected_fps=fps,
        )
        if not validation.valid:
            raise PipelineError(f"stitched output validation failed: {asdict(validation)}")
        source_cuts = sorted(
            {
                *(segment.start_sec for segment in segments[1:]),
                *(
                    segment.start_sec + offset
                    for segment in segments
                    for offset in segment.hard_cut_offsets_sec
                ),
            }
        )
        output_cuts: list[float] = []
        cursor = 0.0
        for segment, duration in zip(segments, clip_durations, strict=True):
            output_cuts.extend(cursor + offset for offset in segment.hard_cut_offsets_sec)
            cursor += duration
            if segment.index < len(segments) - 1:
                output_cuts.append(cursor)
        _qc_artifact(
            job.source_path,
            final_path,
            qc_path,
            source_cut_times=source_cuts,
            output_cut_times=output_cuts,
        )
    except Exception:
        store.set_job_state(job_id, JobState.REVIEW)
        raise
    completed = store.mark_complete(
        job_id,
        final_output_key=final_key,
        qc_output_key=qc_key,
    )
    manifest_path = storage.path(f"jobs/{job_id}/job.json")
    manifest: dict[str, object] = {}
    if manifest_path.is_file():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            manifest = cast(dict[str, object], raw)
    storage.write_json(
        f"jobs/{job_id}/job.json",
        {
            **manifest,
            "state": completed.state.value,
            "final_output": str(final_path),
            "qc_comparison": str(qc_path),
            "validation": asdict(validation),
        },
    )
    return StitchResult(job_id, final_path, qc_path, validation)
