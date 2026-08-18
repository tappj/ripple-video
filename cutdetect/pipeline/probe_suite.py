"""Small paid diagnostic suite for isolating Runway moderation boundaries."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from runwayml import RunwayML
from runwayml.lib.polling import NewTaskCreatedResponse
from runwayml.types.task_retrieve_response import Failed, Succeeded
from runwayml.types.video_to_video_create_params import (
    Seedance2Reference,
    Seedance2ReferenceAudio,
)

from cutdetect.pipeline.gen_one import _slice_source, _validate_asset
from cutdetect.pipeline.runway_client import JsonlCallLogger, PipelineError, public_failure_code
from cutdetect.pipeline.storage import LocalDiskStorage

IMAGE_PROMPT = "A small blue ceramic cup centered on a plain warm-gray studio background."
GENERIC_VIDEO_PROMPT = "Locked camera. Soft daylight shifts gently across the cup."
SOURCE_ONLY_PROMPT = (
    "Preserve the same person, framing, background, timing, and movement. "
    "Apply only a subtle clean studio color grade."
)
NEUTRAL_REFERENCE_PROMPT = (
    "Create a natural talking-head video using the supplied media as authorized creative "
    "references. Preserve the source camera position, background, timing, gestures, and "
    "spoken pacing."
)

PROBE_CREDIT_CEILING = 367


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """One diagnostic call and its persisted terminal outcome."""

    name: str
    endpoint: str
    model: str
    prompt: str
    estimated_credits: int
    status: str
    task_id: str | None
    output_path: str | None
    failure_code: str | None
    wall_clock_sec: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProbeSuiteResult:
    """Complete capability probe report."""

    suite_id: str
    report_path: Path
    log_path: Path
    estimated_credits_attempted: int
    results: tuple[ProbeResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "suite_id": self.suite_id,
            "report_path": str(self.report_path),
            "log_path": str(self.log_path),
            "estimated_credits_attempted": self.estimated_credits_attempted,
            "results": [result.to_dict() for result in self.results],
        }


def _run_task(
    client: RunwayML,
    logger: JsonlCallLogger,
    storage: LocalDiskStorage,
    *,
    name: str,
    endpoint: str,
    model: str,
    prompt: str,
    estimated_credits: int,
    output_key: str,
    create: Callable[[], NewTaskCreatedResponse],
) -> tuple[ProbeResult, str | None]:
    started = time.monotonic()
    metadata = {
        "probe": name,
        "endpoint": endpoint,
        "model": model,
        "prompt": prompt,
        "estimated_credits": estimated_credits,
    }
    try:
        created = create()
    except Exception as error:
        elapsed = time.monotonic() - started
        logger.write(
            "runway.probe.create_failed",
            **metadata,
            error_type=type(error).__name__,
            error=str(error),
            wall_clock_sec=elapsed,
        )
        return (
            ProbeResult(
                name,
                endpoint,
                model,
                prompt,
                estimated_credits,
                "CREATE_FAILED",
                None,
                None,
                None,
                elapsed,
            ),
            None,
        )
    task_id = created.id
    logger.write("runway.probe.submitted", **metadata, task_id=task_id)
    while True:
        state = client.tasks.retrieve(task_id)
        logger.write("runway.probe.polled", **metadata, task_id=task_id, status=state.status)
        if isinstance(state, Failed):
            elapsed = time.monotonic() - started
            logger.write(
                "runway.probe.failed",
                **metadata,
                task_id=task_id,
                failure_code=state.failure_code,
                failure=state.failure,
                wall_clock_sec=elapsed,
            )
            return (
                ProbeResult(
                    name,
                    endpoint,
                    model,
                    prompt,
                    estimated_credits,
                    "FAILED",
                    task_id,
                    None,
                    public_failure_code(state.failure_code),
                    elapsed,
                ),
                None,
            )
        if isinstance(state, Succeeded):
            elapsed = time.monotonic() - started
            if not state.output:
                raise PipelineError(f"Runway probe {task_id} succeeded without output")
            path = storage.download_https(state.output[0], output_key)
            logger.write(
                "runway.probe.succeeded",
                **metadata,
                task_id=task_id,
                output_path=str(path),
                wall_clock_sec=elapsed,
            )
            return (
                ProbeResult(
                    name,
                    endpoint,
                    model,
                    prompt,
                    estimated_credits,
                    "SUCCEEDED",
                    task_id,
                    str(path),
                    None,
                    elapsed,
                ),
                state.output[0],
            )
        time.sleep(5.0)


def _upload(client: RunwayML, logger: JsonlCallLogger, path: Path, *, role: str) -> str:
    started = time.monotonic()
    response = client.uploads.create_ephemeral(file=path)
    logger.write(
        "runway.probe.uploaded",
        role=role,
        path=str(path),
        size_bytes=path.stat().st_size,
        runway_uri=response.uri,
        latency_sec=time.monotonic() - started,
    )
    return response.uri


def run_capability_probe_suite(
    video: str | Path,
    image: str | Path,
    audio: str | Path,
    *,
    max_credits: int,
    output_root: str | Path = ".cutdetect/pipeline/probes",
    cache_dir: str | Path | None = ".cutdetect/cache",
) -> ProbeSuiteResult:
    """Run five minimal paid probes, continuing after terminal task failures."""
    if max_credits < PROBE_CREDIT_CEILING:
        raise PipelineError(
            f"probe suite requires a {PROBE_CREDIT_CEILING}-credit ceiling; received {max_credits}"
        )
    source = _validate_asset(Path(video), "reference video", {".mp4", ".mov", ".mkv", ".webm"})
    face = _validate_asset(Path(image), "reference image", {".jpg", ".jpeg", ".png", ".webp"})
    voice = _validate_asset(
        Path(audio), "reference audio", {".mp3", ".wav", ".flac", ".m4a", ".aac"}
    )
    api_key = os.environ.get("RUNWAYML_API_SECRET", "")
    if not api_key:
        raise PipelineError("RUNWAYML_API_SECRET is not set")
    suite_id = uuid.uuid4().hex
    storage = LocalDiskStorage(Path(output_root) / suite_id)
    log_path = storage.path("runway_calls.jsonl")
    logger = JsonlCallLogger(log_path)
    client = RunwayML(api_key=api_key, max_retries=0)
    results: list[ProbeResult] = []

    image_result, generated_image_url = _run_task(
        client,
        logger,
        storage,
        name="benign_text_to_image",
        endpoint="/v1/text_to_image",
        model="gen4_image",
        prompt=IMAGE_PROMPT,
        estimated_credits=5,
        output_key="outputs/01_benign_image.jpg",
        create=lambda: client.text_to_image.create(
            model="gen4_image", prompt_text=IMAGE_PROMPT, ratio="1024:1024"
        ),
    )
    results.append(image_result)

    if generated_image_url is not None:
        generic_video_result, _ = _run_task(
            client,
            logger,
            storage,
            name="generic_image_to_video",
            endpoint="/v1/image_to_video",
            model="gen4_turbo",
            prompt=GENERIC_VIDEO_PROMPT,
            estimated_credits=10,
            output_key="outputs/02_generic_video.mp4",
            create=lambda: client.image_to_video.create(
                model="gen4_turbo",
                prompt_image=generated_image_url,
                prompt_text=GENERIC_VIDEO_PROMPT,
                ratio="960:960",
                duration=2,
            ),
        )
        results.append(generic_video_result)
    else:
        results.append(
            ProbeResult(
                "generic_image_to_video",
                "/v1/image_to_video",
                "gen4_turbo",
                GENERIC_VIDEO_PROMPT,
                0,
                "SKIPPED",
                None,
                None,
                None,
                0.0,
            )
        )

    source_slice = storage.path("inputs/source_4s.mp4")
    _slice_source(
        source,
        source_slice,
        start_sec=0.0,
        end_sec=4.0,
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
    )
    source_uri = _upload(client, logger, source_slice, role="source_4s")
    face_uri = _upload(client, logger, face, role="consented_face")
    voice_uri = _upload(client, logger, voice, role="consented_voice")

    mini_result, _ = _run_task(
        client,
        logger,
        storage,
        name="seedance_mini_source_only",
        endpoint="/v1/video_to_video",
        model="seedance2_mini",
        prompt=SOURCE_ONLY_PROMPT,
        estimated_credits=64,
        output_key="outputs/03_seedance_mini_source_only.mp4",
        create=lambda: client.video_to_video.create(
            model="seedance2_mini",
            prompt_video=source_uri,
            prompt_text=SOURCE_ONLY_PROMPT,
            ratio="720:1280",
            duration=4,
            audio=False,
        ),
    )
    results.append(mini_result)

    seedance_result, _ = _run_task(
        client,
        logger,
        storage,
        name="seedance_source_only",
        endpoint="/v1/video_to_video",
        model="seedance2",
        prompt=SOURCE_ONLY_PROMPT,
        estimated_credits=144,
        output_key="outputs/04_seedance_source_only.mp4",
        create=lambda: client.video_to_video.create(
            model="seedance2",
            prompt_video=source_uri,
            prompt_text=SOURCE_ONLY_PROMPT,
            ratio="720:1280",
            duration=4,
            audio=False,
        ),
    )
    results.append(seedance_result)

    images: list[Seedance2Reference] = [{"uri": face_uri}]
    audios: list[Seedance2ReferenceAudio] = [{"type": "audio", "uri": voice_uri}]
    reference_result, _ = _run_task(
        client,
        logger,
        storage,
        name="seedance_neutral_references",
        endpoint="/v1/video_to_video",
        model="seedance2",
        prompt=NEUTRAL_REFERENCE_PROMPT,
        estimated_credits=144,
        output_key="outputs/05_seedance_neutral_references.mp4",
        create=lambda: client.video_to_video.create(
            model="seedance2",
            prompt_video=source_uri,
            prompt_text=NEUTRAL_REFERENCE_PROMPT,
            ratio="720:1280",
            duration=4,
            audio=True,
            references=images,
            reference_audio=audios,
        ),
    )
    results.append(reference_result)

    attempted = sum(result.estimated_credits for result in results)
    report_path = storage.path("report.json")
    provisional = ProbeSuiteResult(suite_id, report_path, log_path, attempted, tuple(results))
    report_path.write_text(json.dumps(provisional.to_dict(), indent=2) + "\n", encoding="utf-8")
    return provisional
