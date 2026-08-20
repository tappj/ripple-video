"""Single-segment proof path for the Runway regeneration pipeline."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from cutdetect.config import IngestConfig
from cutdetect.ingest import ingest_video, probe_video
from cutdetect.pipeline.capabilities import MODEL_CAPABILITIES, closest_seedance_ratio
from cutdetect.pipeline.runway_client import (
    GenerationRequest,
    JsonlCallLogger,
    PipelineError,
    RunwayClient,
    RunwayReferenceModel,
    RunwayTaskError,
    public_failure_code,
    request_manifest,
    seedance_ratio,
)
from cutdetect.pipeline.storage import LocalDiskStorage
from cutdetect.pipeline.templates import UGC_CLONE_V1


def cast_runway_reference_model(value: str) -> RunwayReferenceModel:
    """Narrow a capability-registry model ID after validation."""
    if value not in {"seedance2", "hailuo3"}:
        raise PipelineError(f"unsupported direct reference model: {value}")
    return cast(RunwayReferenceModel, value)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Phase A metrics and local artifacts for one generation."""

    job_id: str
    task_id: str
    input_path: Path
    output_path: Path
    manifest_path: Path
    log_path: Path
    requested_duration_sec: int
    source_slice_duration_sec: float
    output_duration_sec: float
    estimated_credits: int
    wall_clock_sec: float
    known_internal_cuts_sec: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for field in ("input_path", "output_path", "manifest_path", "log_path"):
            value[field] = str(value[field])
        return value


@dataclass(frozen=True, slots=True)
class GenerationPlan:
    """No-charge preview used to enforce the pre-generation cost gate."""

    source: Path
    reference_image: Path
    reference_audio: Path
    start_sec: float
    end_sec: float
    padded_end_sec: float
    requested_duration_sec: int
    model: str
    ratio: str
    resolution: str | None
    estimated_credits: int
    known_internal_cuts_sec: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for field in ("source", "reference_image", "reference_audio"):
            value[field] = str(value[field])
        return value


def _validate_asset(path: Path, role: str, suffixes: set[str]) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} not found: {resolved}")
    if resolved.suffix.lower() not in suffixes:
        allowed = ", ".join(sorted(suffixes))
        raise PipelineError(f"unsupported {role} type {resolved.suffix!r}; expected {allowed}")
    size = resolved.stat().st_size
    if not 512 <= size <= 200 * 1024 * 1024:
        raise PipelineError(f"{role} must be between 512 bytes and 200MB")
    return resolved


def _slice_source(
    source: Path,
    destination: Path,
    *,
    start_sec: float,
    end_sec: float,
    cache_dir: Path | None,
) -> float:
    context = ingest_video(source, IngestConfig(cache_dir=cache_dir))
    if start_sec < 0 or end_sec <= start_sec or end_sec > context.duration_sec + 1e-6:
        raise PipelineError(
            f"invalid range {start_sec:.3f}-{end_sec:.3f}s for {context.duration_sec:.3f}s source"
        )
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise PipelineError("required executable not found on PATH: ffmpeg")
    duration = end_sec - start_sec
    frame_count = round(duration * float(context.fps))
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-filter_threads",
        "1",
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
            f"{duration:.12f}",
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
        ]
    )
    if context.audio_path is not None:
        command.extend(["-c:a", "aac", "-b:a", "192k"])
    command.extend(["-movflags", "+faststart", str(destination)])
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PipelineError(f"failed to prepare Runway source slice: {detail}")
    return probe_video(destination).duration_sec


def _known_cuts(predictions: Path | None, start_sec: float, end_sec: float) -> tuple[float, ...]:
    if predictions is None or not predictions.is_file():
        return ()
    import json

    root = json.loads(predictions.read_text(encoding="utf-8"))
    cuts = root.get("cuts", []) if isinstance(root, dict) else []
    return tuple(
        float(item["time_sec"])
        for item in cuts
        if isinstance(item, dict) and start_sec < float(item.get("time_sec", -1)) < end_sec
    )


def plan_one(
    video: str | Path,
    image: str | Path,
    audio: str | Path,
    *,
    start_sec: float,
    end_sec: float,
    model: str = "seedance2",
    ratio: str | None = None,
    prompt: str = UGC_CLONE_V1.body,
    predictions: str | Path | None = "eval/phase3/predictions.json",
) -> GenerationPlan:
    """Validate inputs and estimate one request without calling Runway."""
    source = _validate_asset(Path(video), "reference video", {".mp4", ".mov", ".mkv", ".webm"})
    face = _validate_asset(Path(image), "reference image", {".jpg", ".jpeg", ".png", ".webp"})
    voice = _validate_asset(
        Path(audio), "reference audio", {".mp3", ".wav", ".flac", ".m4a", ".aac"}
    )
    if not prompt.strip():
        raise PipelineError("prompt must not be empty")
    if len(prompt) > 3500:
        raise PipelineError("Seedance prompt must be at most 3500 characters")
    source_probe = probe_video(source)
    if start_sec < 0 or end_sec <= start_sec or end_sec > source_probe.duration_sec + 1e-6:
        raise PipelineError(
            f"invalid range {start_sec:.3f}-{end_sec:.3f}s for "
            f"{source_probe.duration_sec:.3f}s source"
        )
    if model not in MODEL_CAPABILITIES:
        raise PipelineError(f"unsupported direct reference model: {model}")
    if model == "hailuo3":
        validated_ratio = ratio or "9:16"
        if validated_ratio not in MODEL_CAPABILITIES[model].supported_ratios:
            raise PipelineError(f"unsupported Hailuo ratio: {validated_ratio}")
        resolution = "768P"
    else:
        selected_ratio = ratio or closest_seedance_ratio(source_probe.width, source_probe.height)
        validated_ratio = seedance_ratio(selected_ratio)
        resolution = None
    requested_duration = math.ceil(end_sec - start_sec)
    caps = MODEL_CAPABILITIES[model]
    if not caps.min_duration_s <= requested_duration <= caps.max_duration_s:
        raise PipelineError(
            f"requested output duration {requested_duration}s is outside "
            f"{caps.min_duration_s:.0f}-{caps.max_duration_s:.0f}s"
        )
    known_cuts = _known_cuts(
        Path(predictions).expanduser().resolve() if predictions is not None else None,
        start_sec,
        end_sec,
    )
    from cutdetect.pipeline.capabilities import credit_cost

    return GenerationPlan(
        source=source,
        reference_image=face,
        reference_audio=voice,
        start_sec=start_sec,
        end_sec=end_sec,
        padded_end_sec=min(start_sec + requested_duration, source_probe.duration_sec),
        requested_duration_sec=requested_duration,
        model=model,
        ratio=validated_ratio,
        resolution=resolution,
        estimated_credits=credit_cost(
            model,
            requested_duration,
            resolution or validated_ratio,
            reference_video_duration_s=end_sec - start_sec,
        ),
        known_internal_cuts_sec=known_cuts,
    )


def generate_one(
    video: str | Path,
    image: str | Path,
    audio: str | Path,
    *,
    start_sec: float,
    end_sec: float,
    model: str = "seedance2",
    max_credits: int,
    output_root: str | Path = ".cutdetect/pipeline",
    cache_dir: str | Path | None = ".cutdetect/cache",
    ratio: str | None = None,
    prompt: str = UGC_CLONE_V1.body,
    predictions: str | Path | None = "eval/phase3/predictions.json",
) -> GenerationResult:
    """Prepare, submit, poll, download, and measure one Seedance proof clip."""
    plan = plan_one(
        video,
        image,
        audio,
        start_sec=start_sec,
        end_sec=end_sec,
        model=model,
        ratio=ratio,
        prompt=prompt,
        predictions=predictions,
    )
    if max_credits < 0:
        raise PipelineError("--max-credits must be non-negative")
    if plan.estimated_credits > max_credits:
        raise PipelineError(
            f"estimated cost {plan.estimated_credits} credits exceeds --max-credits {max_credits}"
        )
    job_id = uuid.uuid4().hex
    storage = LocalDiskStorage(output_root)
    job_key = f"jobs/{job_id}"
    input_path = storage.path(f"{job_key}/segments/0/input.mp4")
    slice_duration = _slice_source(
        plan.source,
        input_path,
        start_sec=start_sec,
        end_sec=plan.padded_end_sec,
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
    )
    log_path = storage.path(f"{job_key}/runway_calls.jsonl")
    manifest_key = f"{job_key}/job.json"
    base_manifest = {
        "job_id": job_id,
        "source": str(plan.source),
        "reference_image": str(plan.reference_image),
        "reference_audio": str(plan.reference_audio),
        "range": {"start_sec": start_sec, "end_sec": end_sec},
        "source_padding": {
            "enabled": plan.padded_end_sec > end_sec,
            "padded_end_sec": plan.padded_end_sec,
            "trim_output_to_duration_sec": end_sec - start_sec,
        },
        "known_internal_cuts_sec": plan.known_internal_cuts_sec,
        "estimated_credits": plan.estimated_credits,
    }
    manifest_path = storage.write_json(
        manifest_key,
        {**base_manifest, "state": "UPLOADING"},
    )
    client = RunwayClient(
        api_key=os.environ.get("RUNWAYML_API_SECRET", ""),
        logger=JsonlCallLogger(log_path),
    )
    request: GenerationRequest | None = None
    try:
        video_uri = client.upload(input_path, role="segment_video")
        image_uri = client.upload(plan.reference_image, role="reference_image")
        audio_uri = client.upload(plan.reference_audio, role="reference_audio")
        request = GenerationRequest(
            reference_video=video_uri,
            reference_image=image_uri,
            reference_audio=audio_uri,
            prompt_text=prompt,
            duration=plan.requested_duration_sec,
            ratio=plan.ratio,
            reference_video_duration_sec=slice_duration,
            model=cast_runway_reference_model(plan.model),
            resolution=plan.resolution,
        )
        storage.write_json(
            manifest_key,
            {**base_manifest, "state": "RUNNING", "request": request_manifest(request)},
        )
        output = client.generate(
            request,
            storage=storage,
            output_key=f"{job_key}/segments/0/output_raw.mp4",
        )
    except PipelineError as error:
        failure: dict[str, object] = {"message": str(error)}
        if isinstance(error, RunwayTaskError):
            failure.update(
                task_id=error.task_id,
                failure_code=public_failure_code(error.failure_code),
            )
        failed_manifest: dict[str, object] = {
            **base_manifest,
            "state": "FAILED",
            "failure": failure,
        }
        if request is not None:
            failed_manifest["request"] = request_manifest(request)
        storage.write_json(manifest_key, failed_manifest)
        raise
    output_duration = probe_video(output.path).duration_sec
    result = GenerationResult(
        job_id=job_id,
        task_id=output.task_id,
        input_path=input_path,
        output_path=output.path,
        manifest_path=manifest_path,
        log_path=log_path,
        requested_duration_sec=plan.requested_duration_sec,
        source_slice_duration_sec=slice_duration,
        output_duration_sec=output_duration,
        estimated_credits=output.estimated_credits,
        wall_clock_sec=output.wall_clock_sec,
        known_internal_cuts_sec=plan.known_internal_cuts_sec,
    )
    storage.write_json(
        manifest_key,
        {
            **base_manifest,
            "state": "READY_FOR_REVIEW",
            "request": request_manifest(request),
            "result": result.to_dict(),
        },
    )
    return result
