from pathlib import Path
from types import SimpleNamespace

import pytest

from cutdetect.pipeline.runway_client import GenerationRequest, JsonlCallLogger, PipelineError
from cutdetect.pipeline.storage import LocalDiskStorage
from cutdetect.pipeline.templates import UGC_CLONE_V1
from cutdetect.pipeline.workflow_client import (
    PROMPT_NODE_ID,
    REFERENCE_VIDEO_NODE_ID,
    SEEDANCE25_WORKFLOW,
    TARGET_FACE_NODE_ID,
    TARGET_VOICE_NODE_ID,
    RunwayWorkflowClient,
    RunwayWorkflowGateway,
)


def test_workflow_submission_overrides_every_reference_and_backend_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, object]] = []

    class FakeWorkflows:
        def run(self, workflow_id: str, *, node_outputs: object) -> SimpleNamespace:
            calls.append((workflow_id, node_outputs))
            return SimpleNamespace(id="workflow-invocation")

    class FakeRunway:
        def __init__(self, **_kwargs: object) -> None:
            self.workflows = FakeWorkflows()

    monkeypatch.setattr("cutdetect.pipeline.workflow_client.RunwayML", FakeRunway)
    client = RunwayWorkflowClient(
        api_key="key_test",
        logger=JsonlCallLogger(tmp_path / "calls.jsonl"),
        storage=LocalDiskStorage(tmp_path),
        spec=SEEDANCE25_WORKFLOW,
    )

    invocation_id = client.submit(
        reference_video_uri="runway://video",
        target_face_uri="runway://face",
        target_voice_uri="runway://voice",
        prompt=UGC_CLONE_V1.body,
    )

    assert invocation_id == "workflow-invocation"
    workflow_id, raw_outputs = calls[0]
    outputs = raw_outputs
    assert workflow_id == SEEDANCE25_WORKFLOW.workflow_id
    assert isinstance(outputs, dict)
    assert outputs[REFERENCE_VIDEO_NODE_ID]["video"]["uri"] == "runway://video"
    assert outputs[TARGET_FACE_NODE_ID]["image"]["uri"] == "runway://face"
    assert outputs[TARGET_VOICE_NODE_ID]["audio"]["uri"] == "runway://voice"
    assert outputs[PROMPT_NODE_ID]["prompt"]["value"] == UGC_CLONE_V1.body


def test_workflow_gateway_rejects_a_request_with_the_wrong_fixed_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeRunway:
        def __init__(self, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr("cutdetect.pipeline.workflow_client.RunwayML", FakeRunway)
    gateway = RunwayWorkflowGateway(
        api_key="key_test",
        logger=JsonlCallLogger(tmp_path / "calls.jsonl"),
        storage=LocalDiskStorage(tmp_path),
        spec=SEEDANCE25_WORKFLOW,
    )
    request = GenerationRequest(
        reference_video="runway://video",
        reference_image="runway://face",
        reference_audio="runway://voice",
        prompt_text=UGC_CLONE_V1.body,
        duration=5,
        ratio="9:16",
        reference_video_duration_sec=5,
        model="seedance2_5",
        resolution="720p",
    )

    with pytest.raises(PipelineError, match="requires a 15s request"):
        gateway.submit(request)
