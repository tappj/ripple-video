"""Typed Runway SDK adapter for Phase A single-clip generation."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from runwayml import RunwayML, omit
from runwayml.types.generate.video_create_params import Input as RouterVideoInput
from runwayml.types.generate.video_create_response import VideoCreateResponse
from runwayml.types.task_retrieve_response import Failed, Succeeded
from runwayml.types.text_to_video_create_params import (
    Seedance2Reference,
    Seedance2ReferenceAudio,
    Seedance2ReferenceVideo,
)
from runwayml.types.text_to_video_create_response import TextToVideoCreateResponse

from cutdetect.pipeline.capabilities import ROUTER_ASPECT_RATIOS, SEEDANCE_RATIOS, credit_cost
from cutdetect.pipeline.storage import Storage

SeedanceRatio = Literal[
    "992:432",
    "864:496",
    "752:560",
    "640:640",
    "560:752",
    "496:864",
    "1470:630",
    "1280:720",
    "1112:834",
    "960:960",
    "834:1112",
    "720:1280",
    "2206:946",
    "1920:1080",
    "1664:1248",
    "1440:1440",
    "1248:1664",
    "1080:1920",
    "3840:1646",
    "3840:2160",
    "3840:2880",
    "3840:3840",
    "2880:3840",
    "2160:3840",
]
RunwayReferenceModel = Literal["seedance2", "seedance2_5", "hailuo3"]
GenerationStatus = Literal["PENDING", "THROTTLED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"]
MODEL_ROUTER_ROUTE_PREFIX = "router:"
DEFAULT_MODEL_ROUTER_CONFIG_IDS: dict[RunwayReferenceModel, str] = {
    "seedance2": "ripple-seedance-2",
}
MODEL_ROUTER_ENV_KEYS: dict[RunwayReferenceModel, str] = {
    "seedance2": "RUNWAY_SEEDANCE_ROUTER_CONFIG_ID",
}


class PipelineError(RuntimeError):
    """Raised when a generation cannot safely proceed or complete."""


class RouterConfigurationError(PipelineError):
    """Raised before generation when a Model Router is missing or unsafe."""


class RunwayTaskError(PipelineError):
    """A terminal task failure with private diagnostics kept off the CLI."""

    def __init__(self, task_id: str, failure_code: str | None) -> None:
        self.task_id = task_id
        self.failure_code = failure_code
        public_code = public_failure_code(failure_code)
        if public_code.startswith("SAFETY."):
            detail = "content moderation rejected an input; do not retry automatically"
        else:
            detail = "see the structured call log for diagnostics"
        super().__init__(f"Runway task {task_id} failed ({public_code}): {detail}")


def public_failure_code(failure_code: str | None) -> str:
    """Hide provider diagnostics from safety codes shown to end users."""
    if failure_code is None:
        return "unknown"
    if failure_code.startswith(("SAFETY.INPUT.", "SAFETY.OUTPUT.")):
        return ".".join(failure_code.split(".")[:2])
    if ".SAFETY." in failure_code:
        return failure_code.split(".SAFETY.", maxsplit=1)[0] + ".SAFETY"
    return failure_code


def public_failure_message(failure_code: str | None) -> str:
    """Explain a public Runway failure code without exposing provider diagnostics."""
    code = public_failure_code(failure_code)
    if code.startswith("SAFETY.") or ".SAFETY" in code:
        return (
            f"Runway's model provider blocked this clip during content moderation ({code}). "
            "Do not retry it unchanged."
        )
    if code.startswith("ASSET.INVALID"):
        return f"Runway could not accept one of this clip's media inputs ({code})."
    if code.startswith("INTERNAL.BAD_OUTPUT"):
        return f"Runway rejected the generated result during quality checks ({code})."
    if code.startswith("THIRD_PARTY.UNAVAILABLE"):
        return f"The selected model provider is temporarily unavailable ({code})."
    if code == "unknown" or code.startswith("INTERNAL"):
        return f"Runway encountered an internal generation problem ({code})."
    return f"Runway generation failed ({code})."


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Validated Seedance multi-reference request independent of SDK naming conventions."""

    reference_video: str
    reference_image: str
    reference_audio: str | None
    prompt_text: str
    duration: int
    ratio: str
    reference_video_duration_sec: float
    model: RunwayReferenceModel = "seedance2"
    resolution: str | None = None
    reference_product: str | None = None

    @property
    def estimated_credits(self) -> int:
        pricing_dimension = self.resolution or self.ratio
        return credit_cost(
            self.model,
            self.duration,
            pricing_dimension,
            reference_video_duration_s=self.reference_video_duration_sec,
        )

    def api_payload(self) -> dict[str, object]:
        """Return the actual JSON field names sent by the SDK."""
        payload: dict[str, object] = {
            "model": self.model,
            "duration": self.duration,
            "promptText": self.prompt_text,
            "ratio": self.ratio,
            "references": [
                {"uri": uri}
                for uri in (self.reference_image, self.reference_product)
                if uri is not None
            ],
            "referenceVideos": [{"type": "video", "uri": self.reference_video}],
        }
        if self.model == "seedance2":
            payload["audio"] = self.reference_audio is not None
        if self.reference_audio is not None:
            payload["referenceAudio"] = [{"type": "audio", "uri": self.reference_audio}]
        if self.resolution is not None:
            payload["resolution"] = self.resolution
        return payload

    def router_payload(self) -> RouterVideoInput:
        """Return the SDK-shaped model-agnostic router payload."""
        if self.ratio not in ROUTER_ASPECT_RATIOS:
            raise RouterConfigurationError(f"unsupported routed aspect ratio: {self.ratio}")
        if self.resolution is None:
            raise RouterConfigurationError("routed generations require an output resolution")
        resolution = self.resolution.lower()
        if self.model == "hailuo3":
            # Router tiers are model-agnostic: 720p resolves to Hailuo 768P and
            # 1080p resolves to its 2K output tier.
            resolution = {"768p": "720p", "2k": "1080p"}.get(resolution, resolution)
        payload: dict[str, object] = {
            "prompt_text": self.prompt_text,
            "aspect_ratio": self.ratio,
            "resolution": resolution,
            "duration": self.duration,
            "audio": self.reference_audio is not None,
            "reference_images": [
                {"uri": uri, "role": "reference"}
                for uri in (self.reference_image, self.reference_product)
                if uri is not None
            ],
            "reference_videos": [{"uri": self.reference_video, "role": "source"}],
        }
        if self.reference_audio is not None:
            payload["reference_audio"] = [{"uri": self.reference_audio}]
        return cast(RouterVideoInput, payload)

    def router_http_payload(self) -> dict[str, object]:
        """Return the camel-case JSON aliases required by a raw HTTP dry-run."""
        sdk_payload = self.router_payload()
        aliases = {
            "prompt_text": "promptText",
            "aspect_ratio": "aspectRatio",
            "reference_images": "referenceImages",
            "reference_videos": "referenceVideos",
            "reference_audio": "referenceAudio",
        }
        return {aliases.get(key, key): value for key, value in sdk_payload.items()}


@dataclass(frozen=True, slots=True)
class RunwayOutput:
    """Persisted result from a successful Runway task."""

    task_id: str
    path: Path
    estimated_credits: int
    wall_clock_sec: float


@dataclass(frozen=True, slots=True)
class GenerationPoll:
    """Provider-neutral status for one direct Runway generation task."""

    status: GenerationStatus
    output_urls: tuple[str, ...] = ()
    progress: float | None = None
    failure_code: str | None = None
    failure_message: str | None = None


class JsonlCallLogger:
    """Append-only structured log for every external call."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: object) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def seedance_ratio(value: str) -> SeedanceRatio:
    """Validate a runtime string and narrow it to the SDK's ratio type."""
    if value not in SEEDANCE_RATIOS:
        raise ValueError(f"unsupported Seedance ratio: {value}")
    return cast(SeedanceRatio, value)


def model_router_config_id(model: RunwayReferenceModel) -> str:
    """Return the overridable stable config ID used for one selected model."""
    if model != "seedance2":
        raise RouterConfigurationError(f"{model} is not available in the Model Router catalog")
    return os.environ.get(MODEL_ROUTER_ENV_KEYS[model]) or DEFAULT_MODEL_ROUTER_CONFIG_IDS[model]


def model_router_route(model: RunwayReferenceModel) -> str:
    """Persist a router config ID in the existing route column."""
    return MODEL_ROUTER_ROUTE_PREFIX + model_router_config_id(model)


def router_config_id_from_route(route_id: str) -> str:
    """Extract a config ID from a persisted Model Router route."""
    if not route_id.startswith(MODEL_ROUTER_ROUTE_PREFIX):
        raise RouterConfigurationError(f"not a Model Router route: {route_id}")
    config_id = route_id.removeprefix(MODEL_ROUTER_ROUTE_PREFIX)
    if not config_id:
        raise RouterConfigurationError("Model Router route has no config ID")
    return config_id


class RunwayClient:
    """Small typed facade over the official SDK with durable output handling."""

    def __init__(self, *, api_key: str, logger: JsonlCallLogger) -> None:
        if not api_key:
            raise PipelineError("RUNWAYML_API_SECRET is not set")
        # Paid creation calls are not automatically retried here. Phase C owns retries.
        self._client = RunwayML(api_key=api_key, max_retries=0)
        self._logger = logger

    def upload(self, path: Path, *, role: str) -> str:
        started = time.monotonic()
        try:
            response = self._client.uploads.create_ephemeral(file=path)
        except Exception as error:
            self._logger.write(
                "runway.upload.failed",
                request={"path": str(path), "role": role, "size_bytes": path.stat().st_size},
                latency_sec=time.monotonic() - started,
                error_type=type(error).__name__,
            )
            raise PipelineError(f"Runway upload failed for {role}: {error}") from error
        self._logger.write(
            "runway.upload.succeeded",
            request={"path": str(path), "role": role, "size_bytes": path.stat().st_size},
            runway_uri=response.uri,
            latency_sec=time.monotonic() - started,
        )
        return response.uri

    def submit(self, request: GenerationRequest) -> str:
        """Create one paid task and return immediately with its durable task ID."""
        started = time.monotonic()
        images: list[Seedance2Reference] = [{"uri": request.reference_image}]
        videos: list[Seedance2ReferenceVideo] = [{"type": "video", "uri": request.reference_video}]
        audio: list[Seedance2ReferenceAudio] = (
            [{"type": "audio", "uri": request.reference_audio}]
            if request.reference_audio is not None
            else []
        )
        try:
            if request.model == "hailuo3":
                # Hailuo 3 reached the REST API before the generated Python SDK
                # added its model-specific overload. Use the official client's
                # transport with the live REST field names in the interim.
                raw_created = self._client.post(
                    "/v1/text_to_video",
                    cast_to=TextToVideoCreateResponse,
                    body=request.api_payload(),
                )
                task_id = raw_created.id
            else:
                created = self._client.text_to_video.create(
                    model="seedance2",
                    audio=request.reference_audio is not None,
                    duration=request.duration,
                    prompt_text=request.prompt_text,
                    ratio=seedance_ratio(request.ratio),
                    references=images,
                    reference_videos=videos,
                    reference_audio=audio if audio else omit,
                )
                task_id = created.id
        except Exception as error:
            self._logger.write(
                "runway.generation.create_failed",
                request=request.api_payload(),
                resolved_model=request.model,
                credit_cost=request.estimated_credits,
                latency_sec=time.monotonic() - started,
                error_type=type(error).__name__,
            )
            raise PipelineError(f"Runway generation submission failed: {error}") from error
        self._logger.write(
            "runway.generation.submitted",
            request=request.api_payload(),
            task_id=task_id,
            resolved_model=request.model,
            credit_cost=request.estimated_credits,
            latency_sec=time.monotonic() - started,
        )
        return task_id

    def validate_router(self, config_id: str, expected_model: RunwayReferenceModel) -> None:
        """Verify that a router exists and is pinned to the UI-selected model."""
        started = time.monotonic()
        try:
            router = next(
                (item for item in self._client.routers.list(limit=100) if item.slug == config_id),
                None,
            )
        except Exception as error:
            self._logger.write(
                "runway.router.validation_failed",
                config_id=config_id,
                expected_model=expected_model,
                latency_sec=time.monotonic() - started,
                error_type=type(error).__name__,
            )
            raise RouterConfigurationError(
                f"could not validate Model Router {config_id}: {error}"
            ) from error
        if router is None:
            raise RouterConfigurationError(
                f"Model Router {config_id} does not exist; create it in the Runway developer portal"
            )
        models = router.settings.models
        if (
            models is None
            or models.mode != "allowlist_only"
            or set(models.ids) != {expected_model}
        ):
            raise RouterConfigurationError(
                f"Model Router {config_id} must use an allow list containing only {expected_model}"
            )
        self._logger.write(
            "runway.router.validated",
            config_id=config_id,
            router_version=router.version,
            expected_model=expected_model,
            latency_sec=time.monotonic() - started,
        )

    def submit_router(self, config_id: str, request: GenerationRequest) -> str:
        """Dry-run, verify, and create one independent routed video task."""
        payload = request.router_payload()
        http_payload = request.router_http_payload()
        started = time.monotonic()
        try:
            preview = self._client.post(
                "/v1/generate/video",
                cast_to=VideoCreateResponse,
                body={"configId": config_id, "dryRun": True, "input": http_payload},
            )
        except Exception as error:
            self._logger.write(
                "runway.router.dry_run_failed",
                config_id=config_id,
                expected_model=request.model,
                latency_sec=time.monotonic() - started,
                error_type=type(error).__name__,
            )
            raise RouterConfigurationError(
                f"Model Router {config_id} rejected the requested inputs: {error}"
            ) from error
        routed_model = preview.routing.model
        routed_credits = math.ceil(preview.routing.estimated_cost.credits)
        self._logger.write(
            "runway.router.dry_run_succeeded",
            config_id=config_id,
            resolved_model=routed_model,
            estimated_credits=routed_credits,
            latency_sec=time.monotonic() - started,
        )
        if routed_model != request.model:
            raise RouterConfigurationError(
                f"Model Router {config_id} selected {routed_model}, not {request.model}"
            )
        if routed_credits != request.estimated_credits:
            raise RouterConfigurationError(
                "Runway's router estimate changed from "
                f"{request.estimated_credits} to {routed_credits} credits; prepare a new job"
            )
        submitted_at = time.monotonic()
        try:
            created = self._client.generate.video.create(config_id=config_id, input=payload)
        except Exception as error:
            self._logger.write(
                "runway.router.create_failed",
                config_id=config_id,
                expected_model=request.model,
                credit_cost=request.estimated_credits,
                latency_sec=time.monotonic() - submitted_at,
                error_type=type(error).__name__,
            )
            raise PipelineError(f"Runway routed generation submission failed: {error}") from error
        self._logger.write(
            "runway.router.submitted",
            config_id=config_id,
            task_id=created.id,
            resolved_model=routed_model,
            credit_cost=request.estimated_credits,
            latency_sec=time.monotonic() - submitted_at,
        )
        return created.id

    def poll(self, task_id: str) -> GenerationPoll:
        """Retrieve a direct task once without waiting or retrying."""
        started = time.monotonic()
        try:
            state = self._client.tasks.retrieve(task_id)
        except Exception as error:
            self._logger.write(
                "runway.task.poll_failed",
                request={"task_id": task_id},
                task_id=task_id,
                latency_sec=time.monotonic() - started,
                error_type=type(error).__name__,
            )
            raise PipelineError(f"Runway task poll failed: {error}") from error
        progress = getattr(state, "progress", None)
        self._logger.write(
            "runway.task.polled",
            request={"task_id": task_id},
            task_id=task_id,
            status=state.status,
            progress=progress,
            latency_sec=time.monotonic() - started,
        )
        if isinstance(state, Failed):
            self._logger.write(
                "runway.task.failed",
                request={"task_id": task_id},
                task_id=task_id,
                failure_code=state.failure_code,
                failure=state.failure,
                latency_sec=time.monotonic() - started,
            )
            return GenerationPoll(
                "FAILED",
                failure_code=public_failure_code(state.failure_code),
                failure_message=public_failure_message(state.failure_code),
            )
        if isinstance(state, Succeeded):
            return GenerationPoll("SUCCEEDED", output_urls=tuple(state.output))
        if state.status == "CANCELLED":
            return GenerationPoll("CANCELLED", failure_code="CANCELLED")
        if state.status == "THROTTLED":
            return GenerationPoll("THROTTLED")
        if state.status == "RUNNING":
            return GenerationPoll("RUNNING", progress=progress)
        return GenerationPoll("PENDING")

    def cancel(self, task_id: str) -> None:
        """Cancel a pending/running task, or delete it if it already finished."""
        started = time.monotonic()
        try:
            self._client.tasks.delete(task_id)
        except Exception as error:
            status = getattr(error, "status_code", None)
            if status == 404:
                return
            self._logger.write(
                "runway.task.cancel_failed",
                task_id=task_id,
                status_code=status,
                error_type=type(error).__name__,
            )
            raise PipelineError(f"Runway task cancellation failed: {error}") from error
        self._logger.write(
            "runway.task.cancelled",
            task_id=task_id,
            latency_sec=time.monotonic() - started,
        )

    def generate(
        self,
        request: GenerationRequest,
        *,
        storage: Storage,
        output_key: str,
        timeout_sec: float = 3600.0,
        poll_interval_sec: float = 5.0,
    ) -> RunwayOutput:
        started = time.monotonic()
        task_id = self.submit(request)
        while True:
            elapsed = time.monotonic() - started
            if elapsed > timeout_sec:
                raise PipelineError(f"Runway task {task_id} timed out after {timeout_sec:.0f}s")
            state = self.poll(task_id)
            if state.status in {"FAILED", "CANCELLED"}:
                self._logger.write(
                    "runway.generation.failed",
                    request={"task_id": task_id},
                    task_id=task_id,
                    status=state.status,
                    failure_code=state.failure_code,
                    failure=state.failure_message,
                    resolved_model=request.model,
                    credit_cost=request.estimated_credits,
                    wall_clock_sec=elapsed,
                )
                raise RunwayTaskError(task_id, state.failure_code)
            if state.status == "SUCCEEDED":
                if not state.output_urls:
                    raise PipelineError(f"Runway task {task_id} succeeded without an output URL")
                download_started = time.monotonic()
                path = storage.download_https(state.output_urls[0], output_key)
                total = time.monotonic() - started
                self._logger.write(
                    "runway.output.downloaded",
                    request={"task_id": task_id, "output_url": state.output_urls[0]},
                    task_id=task_id,
                    resolved_model=request.model,
                    credit_cost=request.estimated_credits,
                    latency_sec=time.monotonic() - download_started,
                    wall_clock_sec=total,
                    output_path=str(path),
                )
                return RunwayOutput(task_id, path, request.estimated_credits, total)
            time.sleep(poll_interval_sec)


class RunwayDirectGateway:
    """Async direct-task adapter used by the durable Phase C worker."""

    def __init__(
        self,
        *,
        api_key: str,
        logger: JsonlCallLogger,
        storage: Storage,
    ) -> None:
        self._client = RunwayClient(api_key=api_key, logger=logger)
        self._logger = logger
        self._storage = storage

    def upload(self, path: Path, *, role: str) -> str:
        return self._client.upload(path, role=role)

    def submit(self, request: GenerationRequest) -> str:
        return self._client.submit(request)

    def poll(self, task_id: str) -> GenerationPoll:
        return self._client.poll(task_id)

    def cancel(self, task_id: str) -> None:
        self._client.cancel(task_id)

    def download(self, url: str, destination_key: str) -> Path:
        started = time.monotonic()
        path = self._storage.download_https(url, destination_key)
        self._logger.write(
            "runway.output.downloaded",
            output_path=str(path),
            latency_sec=time.monotonic() - started,
        )
        return path


class RunwayRouterGateway:
    """Model Router adapter pinned to one UI-selected model."""

    def __init__(
        self,
        *,
        api_key: str,
        logger: JsonlCallLogger,
        storage: Storage,
        config_id: str,
        expected_model: RunwayReferenceModel,
    ) -> None:
        self._client = RunwayClient(api_key=api_key, logger=logger)
        self._logger = logger
        self._storage = storage
        self.config_id = config_id
        self.expected_model = expected_model
        self._client.validate_router(config_id, expected_model)

    def upload(self, path: Path, *, role: str) -> str:
        return self._client.upload(path, role=role)

    def submit(self, request: GenerationRequest) -> str:
        if request.model != self.expected_model:
            raise RouterConfigurationError(
                f"router {self.config_id} cannot run a {request.model} job"
            )
        return self._client.submit_router(self.config_id, request)

    def poll(self, task_id: str) -> GenerationPoll:
        return self._client.poll(task_id)

    def cancel(self, task_id: str) -> None:
        self._client.cancel(task_id)

    def download(self, url: str, destination_key: str) -> Path:
        started = time.monotonic()
        path = self._storage.download_https(url, destination_key)
        self._logger.write(
            "runway.router.output_downloaded",
            config_id=self.config_id,
            output_path=str(path),
            latency_sec=time.monotonic() - started,
        )
        return path


def request_manifest(request: GenerationRequest) -> dict[str, object]:
    """Return serializable request metadata for a job manifest."""
    return {**asdict(request), "estimated_credits": request.estimated_credits}
