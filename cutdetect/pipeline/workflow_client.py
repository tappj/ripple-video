"""Typed adapters for Ripple's published per-model Runway workflows."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from runwayml import RunwayML
from runwayml.types.workflow_invocation_retrieve_response import Failed, Succeeded
from runwayml.types.workflow_run_params import NodeOutputsNodeOutputsItem

from cutdetect.pipeline.runway_client import (
    GenerationPoll,
    GenerationRequest,
    JsonlCallLogger,
    PipelineError,
    RunwayReferenceModel,
)
from cutdetect.pipeline.storage import LocalDiskStorage

REFERENCE_VIDEO_NODE_ID = "2af124b7-f9c6-43b5-9451-66edc5dfd75d"
TARGET_FACE_NODE_ID = "f6de2210-8e15-435a-9348-b7d9cab5ef5a"
TARGET_VOICE_NODE_ID = "3082700d-5a09-41eb-8e5b-8d3a6eea1e9e"
PROMPT_NODE_ID = "46cc4af4-a180-4c37-bcfb-7cf08faca3b5"
WORKFLOW_ROUTE_PREFIX = "workflow:"


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    """Published graph identifiers and fixed settings for one selectable model."""

    model: RunwayReferenceModel
    workflow_id: str
    output_node_id: str
    duration_sec: int
    resolution: str
    reference_video_node_id: str = REFERENCE_VIDEO_NODE_ID
    target_face_node_id: str = TARGET_FACE_NODE_ID
    prompt_node_id: str = PROMPT_NODE_ID
    target_voice_node_id: str | None = TARGET_VOICE_NODE_ID
    target_product_node_id: str | None = None

    @property
    def route_id(self) -> str:
        return WORKFLOW_ROUTE_PREFIX + self.workflow_id


SEEDANCE2_WORKFLOW = WorkflowSpec(
    model="seedance2",
    workflow_id=os.environ.get("RUNWAY_SEEDANCE2_WORKFLOW_ID")
    or "f28115cf-16bd-453f-9f3c-e766982951a4",
    output_node_id="3df2ef68-f469-4792-a027-c87cefc6f9b8",
    duration_sec=15,
    resolution="720p",
)
SEEDANCE25_WORKFLOW = WorkflowSpec(
    model="seedance2_5",
    workflow_id=os.environ.get("RUNWAY_SEEDANCE25_WORKFLOW_ID")
    or "4af4fdf6-a371-4a73-b02d-fdbf116186d5",
    output_node_id="275b7c4a-7e35-495a-8f75-cba16c7abb13",
    duration_sec=15,
    resolution="720p",
)
HAILUO3_WORKFLOW = WorkflowSpec(
    model="hailuo3",
    workflow_id=os.environ.get("RUNWAY_HAILUO3_WORKFLOW_ID")
    or "9172f9ee-e4e9-4a25-92e1-29779d698556",
    output_node_id="ccfb49cf-4dd0-41ba-98a5-817dc6ee363d",
    duration_sec=15,
    resolution="768P",
)
PRODUCT_CLONE_WORKFLOW = WorkflowSpec(
    model="hailuo3",
    workflow_id=os.environ.get("RUNWAY_PRODUCT_CLONE_WORKFLOW_ID")
    or "0b9a4bd0-27a2-4ef7-a2d3-ba1d89a8a0d0",
    output_node_id="f6e6b3ae-b78d-4549-8d7d-c7af327b0e6e",
    duration_sec=15,
    resolution="768P",
    prompt_node_id="c4ca08e3-876e-4032-9f59-b9c765bb884e",
    target_voice_node_id=None,
    target_product_node_id="6137b730-55d0-45b1-89dc-2b442ef355e6",
)

# Preserve support for jobs created against Ripple's first 10-second workflow.
LEGACY_TALKING_WORKFLOW = WorkflowSpec(
    model="seedance2",
    workflow_id=os.environ.get("RUNWAY_TALKING_WORKFLOW_ID")
    or "1c916e0a-fb28-40a9-8eff-ce2552504072",
    output_node_id="3df2ef68-f469-4792-a027-c87cefc6f9b8",
    duration_sec=10,
    resolution="720p",
)
WORKFLOW_SPECS = (SEEDANCE2_WORKFLOW, SEEDANCE25_WORKFLOW, HAILUO3_WORKFLOW)
WORKFLOW_SPECS_BY_MODEL = {spec.model: spec for spec in WORKFLOW_SPECS}
WORKFLOW_SPECS_BY_ROUTE = {
    spec.route_id: spec
    for spec in (LEGACY_TALKING_WORKFLOW, PRODUCT_CLONE_WORKFLOW, *WORKFLOW_SPECS)
}

# Backward-compatible names used by persisted legacy jobs and external imports.
TALKING_WORKFLOW_ID = LEGACY_TALKING_WORKFLOW.workflow_id
TALKING_WORKFLOW_ROUTE = LEGACY_TALKING_WORKFLOW.route_id
GENERATED_CLIP_NODE_ID = LEGACY_TALKING_WORKFLOW.output_node_id
WORKFLOW_DURATION_SEC = LEGACY_TALKING_WORKFLOW.duration_sec
WORKFLOW_CREDITS_PER_RUN = 360


def workflow_spec_for_model(model: RunwayReferenceModel) -> WorkflowSpec:
    """Return the current published workflow selected for a model."""
    try:
        return WORKFLOW_SPECS_BY_MODEL[model]
    except KeyError as error:
        raise PipelineError(f"no published Workflow is configured for {model}") from error


def workflow_spec_for_route(route_id: str) -> WorkflowSpec:
    """Resolve a persisted workflow route, including the legacy workflow."""
    try:
        return WORKFLOW_SPECS_BY_ROUTE[route_id]
    except KeyError as error:
        raise PipelineError(f"unsupported Workflow route: {route_id}") from error


def is_workflow_route(route_id: str) -> bool:
    """Return whether the route identifies a known published workflow."""
    return route_id in WORKFLOW_SPECS_BY_ROUTE

WorkflowStatus = Literal["PENDING", "THROTTLED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"]


@dataclass(frozen=True, slots=True)
class WorkflowPoll:
    """Provider-neutral invocation status used by the durable worker."""

    status: WorkflowStatus
    output_urls: tuple[str, ...] = ()
    progress: float | None = None
    failure_code: str | None = None
    failure_message: str | None = None


class RunwayWorkflowClient:
    """Small, non-retrying facade over Runway's published workflow API."""

    def __init__(
        self,
        *,
        api_key: str,
        logger: JsonlCallLogger,
        storage: LocalDiskStorage,
        spec: WorkflowSpec,
    ) -> None:
        if not api_key:
            raise PipelineError("RUNWAYML_API_SECRET is not set")
        self._client = RunwayML(api_key=api_key, max_retries=0)
        self._logger = logger
        self._storage = storage
        self._spec = spec

    def upload(self, path: Path, *, role: str) -> str:
        """Upload an input without logging its signed ephemeral URI."""
        started = time.monotonic()
        try:
            response = self._client.uploads.create_ephemeral(file=path)
        except Exception as error:
            self._logger.write(
                "runway.workflow.upload_failed",
                role=role,
                path=str(path),
                size_bytes=path.stat().st_size,
                latency_sec=time.monotonic() - started,
                error_type=type(error).__name__,
            )
            raise PipelineError(f"Runway upload failed for {role}: {error}") from error
        self._logger.write(
            "runway.workflow.uploaded",
            role=role,
            path=str(path),
            size_bytes=path.stat().st_size,
            latency_sec=time.monotonic() - started,
        )
        return response.uri

    def configured_duration(self) -> int:
        """Return the published Seedance node duration without making a paid call."""
        started = time.monotonic()
        try:
            workflow = self._client.workflows.retrieve(self._spec.workflow_id)
        except Exception as error:
            self._logger.write(
                "runway.workflow.retrieve_failed",
                workflow_id=self._spec.workflow_id,
                latency_sec=time.monotonic() - started,
                error_type=type(error).__name__,
            )
            raise PipelineError(f"could not verify published workflow duration: {error}") from error
        payload = workflow.model_dump(by_alias=True)
        graph = payload.get("graph", {})
        nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
        for node in nodes:
            if not isinstance(node, dict) or node.get("id") != self._spec.output_node_id:
                continue
            inputs = node.get("nodeInputs", {})
            duration = inputs.get("duration", {}) if isinstance(inputs, dict) else {}
            value = duration.get("value") if isinstance(duration, dict) else None
            if isinstance(value, int | float):
                configured = int(value)
                self._logger.write(
                    "runway.workflow.verified",
                    workflow_id=self._spec.workflow_id,
                    workflow_version=payload.get("version"),
                    configured_duration_sec=configured,
                    latency_sec=time.monotonic() - started,
                )
                return configured
        raise PipelineError("published workflow has no readable Seedance duration")

    def submit(
        self,
        *,
        reference_video_uri: str,
        target_face_uri: str,
        target_voice_uri: str | None,
        prompt: str,
        target_product_uri: str | None = None,
    ) -> str:
        """Start one atomic-segment workflow invocation."""
        node_outputs: dict[str, dict[str, NodeOutputsNodeOutputsItem]] = {
            self._spec.reference_video_node_id: {
                "video": {"type": "video", "uri": reference_video_uri}
            },
            self._spec.target_face_node_id: {
                "image": {"type": "image", "uri": target_face_uri}
            },
            self._spec.prompt_node_id: {"prompt": {"type": "primitive", "value": prompt}},
        }
        if self._spec.target_product_node_id is not None:
            if target_product_uri is None:
                raise PipelineError("the product clone Workflow requires a product image")
            node_outputs[self._spec.target_product_node_id] = {
                "image": {"type": "image", "uri": target_product_uri}
            }
        if target_voice_uri is not None and self._spec.target_voice_node_id is not None:
            node_outputs[self._spec.target_voice_node_id] = {
                "audio": {"type": "audio", "uri": target_voice_uri}
            }
        started = time.monotonic()
        try:
            invocation = self._client.workflows.run(
                self._spec.workflow_id,
                node_outputs=node_outputs,
            )
        except Exception as error:
            self._logger.write(
                "runway.workflow.submit_failed",
                workflow_id=self._spec.workflow_id,
                latency_sec=time.monotonic() - started,
                error_type=type(error).__name__,
            )
            raise PipelineError(f"Runway workflow submission failed: {error}") from error
        self._logger.write(
            "runway.workflow.submitted",
            workflow_id=self._spec.workflow_id,
            invocation_id=invocation.id,
            latency_sec=time.monotonic() - started,
        )
        return invocation.id

    def poll(self, invocation_id: str) -> WorkflowPoll:
        """Retrieve one invocation without waiting or retrying."""
        started = time.monotonic()
        try:
            state = self._client.workflow_invocations.retrieve(invocation_id)
        except Exception as error:
            self._logger.write(
                "runway.workflow.poll_failed",
                invocation_id=invocation_id,
                latency_sec=time.monotonic() - started,
                error_type=type(error).__name__,
            )
            raise PipelineError(f"Runway workflow poll failed: {error}") from error
        status = state.status
        progress = getattr(state, "progress", None)
        self._logger.write(
            "runway.workflow.polled",
            invocation_id=invocation_id,
            status=status,
            progress=progress,
            latency_sec=time.monotonic() - started,
        )
        if isinstance(state, Failed):
            return WorkflowPoll(
                "FAILED",
                failure_code=state.failure_code,
                failure_message=state.failure,
            )
        if isinstance(state, Succeeded):
            return WorkflowPoll(
                "SUCCEEDED",
                output_urls=tuple(state.output.get(self._spec.output_node_id, [])),
            )
        if status == "CANCELLED":
            return WorkflowPoll("CANCELLED", failure_code="CANCELLED")
        if status == "THROTTLED":
            return WorkflowPoll("THROTTLED")
        if status == "RUNNING":
            return WorkflowPoll("RUNNING", progress=progress)
        return WorkflowPoll("PENDING")

    def download(self, url: str, destination_key: str) -> Path:
        """Persist an expiring workflow output URL immediately."""
        started = time.monotonic()
        path = self._storage.download_https(url, destination_key)
        self._logger.write(
            "runway.workflow.downloaded",
            output_path=str(path),
            latency_sec=time.monotonic() - started,
        )
        return path


class RunwayWorkflowGateway:
    """Adapt one published model workflow to the durable worker boundary."""

    def __init__(
        self,
        *,
        api_key: str,
        logger: JsonlCallLogger,
        storage: LocalDiskStorage,
        spec: WorkflowSpec,
    ) -> None:
        self._client = RunwayWorkflowClient(
            api_key=api_key,
            logger=logger,
            storage=storage,
            spec=spec,
        )
        self._spec = spec

    def upload(self, path: Path, *, role: str) -> str:
        return self._client.upload(path, role=role)

    def submit(self, request: GenerationRequest) -> str:
        if request.model != self._spec.model:
            raise PipelineError(
                f"Workflow {self._spec.workflow_id} requires {self._spec.model}"
            )
        if request.duration != self._spec.duration_sec:
            raise PipelineError(
                f"the published Workflow requires a {self._spec.duration_sec}s request"
            )
        return self._client.submit(
            reference_video_uri=request.reference_video,
            target_face_uri=request.reference_image,
            target_voice_uri=request.reference_audio,
            prompt=request.prompt_text,
            target_product_uri=request.reference_product,
        )

    def poll(self, task_id: str) -> GenerationPoll:
        result = self._client.poll(task_id)
        return GenerationPoll(
            result.status,
            output_urls=result.output_urls,
            progress=result.progress,
            failure_code=result.failure_code,
            failure_message=result.failure_message,
        )

    def download(self, url: str, destination_key: str) -> Path:
        return self._client.download(url, destination_key)
