"""Typed adapter for the published Clone UGC Talking Videos workflow."""

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
)
from cutdetect.pipeline.storage import LocalDiskStorage

TALKING_WORKFLOW_ID = (
    os.environ.get("RUNWAY_TALKING_WORKFLOW_ID") or "1c916e0a-fb28-40a9-8eff-ce2552504072"
)
TALKING_WORKFLOW_ROUTE = f"workflow:{TALKING_WORKFLOW_ID}"
REFERENCE_VIDEO_NODE_ID = "2af124b7-f9c6-43b5-9451-66edc5dfd75d"
TARGET_FACE_NODE_ID = "f6de2210-8e15-435a-9348-b7d9cab5ef5a"
TARGET_VOICE_NODE_ID = "3082700d-5a09-41eb-8e5b-8d3a6eea1e9e"
PROMPT_NODE_ID = "46cc4af4-a180-4c37-bcfb-7cf08faca3b5"
GENERATED_CLIP_NODE_ID = "3df2ef68-f469-4792-a027-c87cefc6f9b8"
WORKFLOW_DURATION_SEC = 10
WORKFLOW_CREDITS_PER_RUN = 360

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
    ) -> None:
        if not api_key:
            raise PipelineError("RUNWAYML_API_SECRET is not set")
        self._client = RunwayML(api_key=api_key, max_retries=0)
        self._logger = logger
        self._storage = storage

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
            workflow = self._client.workflows.retrieve(TALKING_WORKFLOW_ID)
        except Exception as error:
            self._logger.write(
                "runway.workflow.retrieve_failed",
                workflow_id=TALKING_WORKFLOW_ID,
                latency_sec=time.monotonic() - started,
                error_type=type(error).__name__,
            )
            raise PipelineError(f"could not verify published workflow duration: {error}") from error
        payload = workflow.model_dump(by_alias=True)
        graph = payload.get("graph", {})
        nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
        for node in nodes:
            if not isinstance(node, dict) or node.get("id") != GENERATED_CLIP_NODE_ID:
                continue
            inputs = node.get("nodeInputs", {})
            duration = inputs.get("duration", {}) if isinstance(inputs, dict) else {}
            value = duration.get("value") if isinstance(duration, dict) else None
            if isinstance(value, int | float):
                configured = int(value)
                self._logger.write(
                    "runway.workflow.verified",
                    workflow_id=TALKING_WORKFLOW_ID,
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
    ) -> str:
        """Start one atomic-segment workflow invocation."""
        node_outputs: dict[str, dict[str, NodeOutputsNodeOutputsItem]] = {
            REFERENCE_VIDEO_NODE_ID: {"video": {"type": "video", "uri": reference_video_uri}},
            TARGET_FACE_NODE_ID: {"image": {"type": "image", "uri": target_face_uri}},
            PROMPT_NODE_ID: {"prompt": {"type": "primitive", "value": prompt}},
        }
        if target_voice_uri is not None:
            node_outputs[TARGET_VOICE_NODE_ID] = {
                "audio": {"type": "audio", "uri": target_voice_uri}
            }
        started = time.monotonic()
        try:
            invocation = self._client.workflows.run(
                TALKING_WORKFLOW_ID,
                node_outputs=node_outputs,
            )
        except Exception as error:
            self._logger.write(
                "runway.workflow.submit_failed",
                workflow_id=TALKING_WORKFLOW_ID,
                estimated_credits=WORKFLOW_CREDITS_PER_RUN,
                latency_sec=time.monotonic() - started,
                error_type=type(error).__name__,
            )
            raise PipelineError(f"Runway workflow submission failed: {error}") from error
        self._logger.write(
            "runway.workflow.submitted",
            workflow_id=TALKING_WORKFLOW_ID,
            invocation_id=invocation.id,
            estimated_credits=WORKFLOW_CREDITS_PER_RUN,
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
                output_urls=tuple(state.output.get(GENERATED_CLIP_NODE_ID, [])),
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
    """Adapt the published talking-video Workflow to the durable worker boundary."""

    def __init__(
        self,
        *,
        api_key: str,
        logger: JsonlCallLogger,
        storage: LocalDiskStorage,
    ) -> None:
        self._client = RunwayWorkflowClient(
            api_key=api_key,
            logger=logger,
            storage=storage,
        )

    def upload(self, path: Path, *, role: str) -> str:
        return self._client.upload(path, role=role)

    def submit(self, request: GenerationRequest) -> str:
        if request.model != "seedance2":
            raise PipelineError("the talking-video Workflow requires Seedance 2")
        if request.duration != WORKFLOW_DURATION_SEC:
            raise PipelineError(
                f"the published Workflow requires a {WORKFLOW_DURATION_SEC}s request"
            )
        return self._client.submit(
            reference_video_uri=request.reference_video,
            target_face_uri=request.reference_image,
            target_voice_uri=request.reference_audio,
            prompt=request.prompt_text,
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
