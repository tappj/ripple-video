from pathlib import Path
from types import SimpleNamespace

import pytest

from cutdetect.pipeline.runway_client import GenerationRequest, JsonlCallLogger, PipelineError
from cutdetect.pipeline.storage import LocalDiskStorage
from cutdetect.pipeline.templates import UGC_CLONE_V1
from cutdetect.pipeline.workflow_client import (
    DEFAULT_HAILUO3_WORKFLOW_ID,
    DEFAULT_SEEDANCE2_WORKFLOW_ID,
    DELETED_SEEDANCE2_WORKFLOW_ID,
    HAILUO3_WORKFLOW,
    PREVIOUS_HAILUO3_WORKFLOW_ID,
    PREVIOUS_SEEDANCE2_WORKFLOW_ID,
    PRODUCT_CLONE_WORKFLOW,
    PROMPT_NODE_ID,
    REFERENCE_VIDEO_NODE_ID,
    SEEDANCE2_WORKFLOW,
    SEEDANCE25_WORKFLOW,
    TARGET_FACE_NODE_ID,
    TARGET_VOICE_NODE_ID,
    VIDEO_ONLY_SEEDANCE2_WORKFLOW_ID,
    RunwayWorkflowClient,
    RunwayWorkflowGateway,
    workflow_spec_for_route,
)


def test_minimax_h3_workflow_maps_all_published_inputs_and_old_route() -> None:
    spec = HAILUO3_WORKFLOW

    assert spec.workflow_id == DEFAULT_HAILUO3_WORKFLOW_ID
    assert spec.reference_video_node_id == "6e4db3d7-8aa5-4def-abdb-6b0ec607f25e"
    assert spec.target_face_node_id == "97e7f919-1eb5-4fc1-ae62-388e404cd6b7"
    assert spec.prompt_node_id == "0c6c3f68-da4d-40fb-a0a0-f7ef86644435"
    assert spec.target_voice_node_id == "f8b888b1-1746-4b91-a05a-548c7a1350b5"
    assert spec.output_node_id == "de44988c-97a7-4133-b942-fca95c7e4e99"
    assert workflow_spec_for_route(
        "workflow:" + PREVIOUS_HAILUO3_WORKFLOW_ID
    ) is spec


def test_republished_seedance2_workflow_maps_its_target_audio_input() -> None:
    spec = SEEDANCE2_WORKFLOW

    assert spec.workflow_id == DEFAULT_SEEDANCE2_WORKFLOW_ID
    assert spec.reference_video_node_id == "6e4db3d7-8aa5-4def-abdb-6b0ec607f25e"
    assert spec.target_face_node_id == "97e7f919-1eb5-4fc1-ae62-388e404cd6b7"
    assert spec.prompt_node_id == "0c6c3f68-da4d-40fb-a0a0-f7ef86644435"
    assert spec.output_node_id == "b8caabf3-3000-43d0-a740-3fcef85c8153"
    assert spec.target_voice_node_id == "f8b888b1-1746-4b91-a05a-548c7a1350b5"
    assert workflow_spec_for_route(
        "workflow:" + DELETED_SEEDANCE2_WORKFLOW_ID
    ) is spec
    assert workflow_spec_for_route(
        "workflow:" + PREVIOUS_SEEDANCE2_WORKFLOW_ID
    ) is spec
    assert workflow_spec_for_route(
        "workflow:" + VIDEO_ONLY_SEEDANCE2_WORKFLOW_ID
    ) is spec


def test_republished_seedance2_submission_supplies_generated_target_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    class FakeWorkflows:
        def run(self, workflow_id: str, *, node_outputs: object) -> SimpleNamespace:
            calls.append({"workflow_id": workflow_id, "node_outputs": node_outputs})
            return SimpleNamespace(id="replacement-invocation")

    class FakeRunway:
        def __init__(self, **_kwargs: object) -> None:
            self.workflows = FakeWorkflows()

    monkeypatch.setattr("cutdetect.pipeline.workflow_client.RunwayML", FakeRunway)
    client = RunwayWorkflowClient(
        api_key="key_test",
        logger=JsonlCallLogger(tmp_path / "calls.jsonl"),
        storage=LocalDiskStorage(tmp_path),
        spec=SEEDANCE2_WORKFLOW,
    )

    client.submit(
        reference_video_uri="runway://video",
        target_face_uri="runway://face",
        target_voice_uri="runway://legacy-voice",
        prompt="visual prompt",
    )

    assert calls[0]["workflow_id"] == DEFAULT_SEEDANCE2_WORKFLOW_ID
    outputs = calls[0]["node_outputs"]
    assert isinstance(outputs, dict)
    assert set(outputs) == {
        SEEDANCE2_WORKFLOW.reference_video_node_id,
        SEEDANCE2_WORKFLOW.target_face_node_id,
        SEEDANCE2_WORKFLOW.prompt_node_id,
        SEEDANCE2_WORKFLOW.target_voice_node_id,
    }
    voice_node_id = SEEDANCE2_WORKFLOW.target_voice_node_id
    assert voice_node_id is not None
    assert outputs[voice_node_id]["audio"]["uri"] == "runway://legacy-voice"


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


def test_product_workflow_maps_its_distinct_avatar_product_and_prompt_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, object]] = []

    class FakeWorkflows:
        def run(self, workflow_id: str, *, node_outputs: object) -> SimpleNamespace:
            calls.append((workflow_id, node_outputs))
            return SimpleNamespace(id="product-invocation")

    class FakeRunway:
        def __init__(self, **_kwargs: object) -> None:
            self.workflows = FakeWorkflows()

    monkeypatch.setattr("cutdetect.pipeline.workflow_client.RunwayML", FakeRunway)
    client = RunwayWorkflowClient(
        api_key="key_test",
        logger=JsonlCallLogger(tmp_path / "calls.jsonl"),
        storage=LocalDiskStorage(tmp_path),
        spec=PRODUCT_CLONE_WORKFLOW,
    )

    invocation_id = client.submit(
        reference_video_uri="runway://video",
        target_face_uri="runway://avatar",
        target_voice_uri=None,
        target_product_uri="runway://product",
        prompt="product prompt",
    )

    assert invocation_id == "product-invocation"
    workflow_id, raw_outputs = calls[0]
    assert workflow_id == PRODUCT_CLONE_WORKFLOW.workflow_id
    assert isinstance(raw_outputs, dict)
    assert raw_outputs[PRODUCT_CLONE_WORKFLOW.reference_video_node_id]["video"]["uri"] == (
        "runway://video"
    )
    assert raw_outputs[PRODUCT_CLONE_WORKFLOW.target_face_node_id]["image"]["uri"] == (
        "runway://avatar"
    )
    product_node_id = PRODUCT_CLONE_WORKFLOW.target_product_node_id
    assert product_node_id is not None
    assert raw_outputs[product_node_id]["image"]["uri"] == "runway://product"
    assert raw_outputs[PRODUCT_CLONE_WORKFLOW.prompt_node_id]["prompt"]["value"] == (
        "product prompt"
    )
