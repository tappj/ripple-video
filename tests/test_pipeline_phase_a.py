from pathlib import Path
from types import SimpleNamespace

import pytest

from cutdetect.pipeline.capabilities import (
    MODEL_CAPABILITIES,
    closest_seedance_ratio,
    credit_cost,
)
from cutdetect.pipeline.orchestration import generation_route
from cutdetect.pipeline.probe_suite import PROBE_CREDIT_CEILING
from cutdetect.pipeline.runway_client import (
    DEFAULT_MODEL_ROUTER_CONFIG_IDS,
    GenerationRequest,
    JsonlCallLogger,
    RunwayRouterGateway,
    RunwayTaskError,
    model_router_route,
    public_failure_code,
    public_failure_message,
    seedance_ratio,
)
from cutdetect.pipeline.storage import LocalDiskStorage
from cutdetect.pipeline.templates import (
    UGC_CLONE_NO_VOICE_V1,
    UGC_CLONE_V1,
    strict_generation_prompt,
)
from cutdetect.pipeline.workflow_client import (
    HAILUO3_WORKFLOW,
    SEEDANCE2_WORKFLOW,
    SEEDANCE25_WORKFLOW,
)


def test_seedance_capabilities_and_vertical_default() -> None:
    caps = MODEL_CAPABILITIES["seedance2"]

    assert caps.min_duration_s == 4.0
    assert caps.max_duration_s == 15.0
    assert caps.supports_reference_audio
    assert not caps.supports_internal_cuts
    assert closest_seedance_ratio(720, 1280) == "720:1280"


def test_strict_clone_constraints_cannot_be_replaced_by_retry_direction() -> None:
    prompt = strict_generation_prompt("Make the delivery more energetic.")

    assert prompt.startswith(UGC_CLONE_V1.body)
    assert "stays in sync with Video 1's original dialogue" in prompt
    assert prompt.endswith("ADDITIONAL DIRECTION. Make the delivery more energetic.")


def test_clone_prompt_uses_audio_reference_only_when_it_is_supplied() -> None:
    with_voice = strict_generation_prompt("", has_voice=True)
    without_voice = strict_generation_prompt("", has_voice=False)

    assert "Use Audio 1 as the complete final audio for this clip" in with_voice
    assert "completely replaces and masks Video 1's original audio" in with_voice
    assert "Audio 1" not in without_voice
    assert without_voice == UGC_CLONE_NO_VOICE_V1.body
    assert with_voice != without_voice
    assert "Preserve every subtitle" in with_voice
    assert "No on-screen text" not in with_voice


def test_either_clone_base_is_recognised_as_an_unedited_default() -> None:
    assert strict_generation_prompt(UGC_CLONE_NO_VOICE_V1.body) == UGC_CLONE_V1.body
    assert (
        strict_generation_prompt(UGC_CLONE_V1.body, has_voice=False)
        == UGC_CLONE_NO_VOICE_V1.body
    )


def test_superseded_builtin_prompt_is_not_appended_to_current_prompt() -> None:
    previous_default = (
        "Recreate Video 1 exactly at its original duration. Video 1 is the only source for "
        "dialogue, wording, timing, motion, cuts, framing, background, and background audio. "
        "Do not slow, extend, loop, or add footage. Use Image 1 only for facial identity. "
        "Do not invent or alternate faces, glasses, hair, clothing, or accessories. If Audio "
        "1 is provided, use only its voice identity and tone. Ignore its words completely; "
        "speak only the exact words from Video 1 at the same timestamps. Change nothing else."
    )

    assert strict_generation_prompt(previous_default) == UGC_CLONE_V1.body


def test_seedance_payload_matches_live_api_contract() -> None:
    request = GenerationRequest(
        reference_video="runway://video",
        reference_image="runway://image",
        reference_audio="runway://audio",
        prompt_text=UGC_CLONE_V1.body,
        duration=8,
        ratio=seedance_ratio("720:1280"),
        reference_video_duration_sec=8.0,
    )

    assert request.api_payload() == {
        "model": "seedance2",
        "audio": True,
        "duration": 8,
        "promptText": UGC_CLONE_V1.body,
        "ratio": "720:1280",
        "references": [{"uri": "runway://image"}],
        "referenceVideos": [{"type": "video", "uri": "runway://video"}],
        "referenceAudio": [{"type": "audio", "uri": "runway://audio"}],
    }
    assert request.estimated_credits == 288
    assert credit_cost("seedance2", 8, "1080:1920") == 320
    assert credit_cost("seedance2", 8, "720:1280") == 288


def test_hailuo_payload_keeps_all_four_references_without_seedance_audio_flag() -> None:
    request = GenerationRequest(
        reference_video="runway://video",
        reference_image="runway://image",
        reference_audio="runway://audio",
        prompt_text=UGC_CLONE_V1.body,
        duration=6,
        ratio="9:16",
        reference_video_duration_sec=5.466667,
        model="hailuo3",
        resolution="768P",
    )

    payload = request.api_payload()
    assert "audio" not in payload
    assert payload["referenceVideos"] == [{"type": "video", "uri": "runway://video"}]
    assert payload["references"] == [{"uri": "runway://image"}]
    assert payload["referenceAudio"] == [{"type": "audio", "uri": "runway://audio"}]
    assert payload["promptText"] == UGC_CLONE_V1.body
    assert request.estimated_credits == 117


def test_model_router_payload_is_model_agnostic_and_keeps_all_references() -> None:
    request = GenerationRequest(
        reference_video="runway://video",
        reference_image="runway://image",
        reference_audio="runway://audio",
        prompt_text=UGC_CLONE_V1.body,
        duration=6,
        ratio="9:16",
        reference_video_duration_sec=5.466667,
        model="seedance2",
        resolution="720p",
    )

    assert request.router_payload() == {
        "prompt_text": UGC_CLONE_V1.body,
        "aspect_ratio": "9:16",
        "resolution": "720p",
        "duration": 6,
        "audio": True,
        "reference_images": [{"uri": "runway://image", "role": "reference"}],
        "reference_videos": [{"uri": "runway://video", "role": "source"}],
        "reference_audio": [{"uri": "runway://audio"}],
    }
    assert request.router_http_payload() == {
        "promptText": UGC_CLONE_V1.body,
        "aspectRatio": "9:16",
        "resolution": "720p",
        "duration": 6,
        "audio": True,
        "referenceImages": [{"uri": "runway://image", "role": "reference"}],
        "referenceVideos": [{"uri": "runway://video", "role": "source"}],
        "referenceAudio": [{"uri": "runway://audio"}],
    }
    assert model_router_route("seedance2") == (
        "router:" + DEFAULT_MODEL_ROUTER_CONFIG_IDS["seedance2"]
    )
    assert generation_route("seedance2") == SEEDANCE2_WORKFLOW.route_id
    assert generation_route("seedance2_5") == SEEDANCE25_WORKFLOW.route_id
    assert generation_route("hailuo3") == HAILUO3_WORKFLOW.route_id
    assert credit_cost("seedance2", 8, "720p") == 288
    assert credit_cost("seedance2_5", 15, "720p", reference_video_duration_s=5) == 525
    assert credit_cost("seedance2", 8, "4K") == 1200


def test_product_comparison_router_sends_avatar_and_product_in_order() -> None:
    request = GenerationRequest(
        reference_video="runway://video",
        reference_image="runway://avatar",
        reference_audio=None,
        reference_product="runway://product",
        prompt_text="Keep the source and replace the avatar and product.",
        duration=10,
        ratio="9:16",
        reference_video_duration_sec=10,
        model="seedance2",
        resolution="720p",
    )

    assert request.router_payload()["reference_images"] == [
        {"uri": "runway://avatar", "role": "reference"},
        {"uri": "runway://product", "role": "reference"},
    ]
    assert request.router_payload()["audio"] is False


def test_router_gateway_validates_dry_runs_then_starts_a_fresh_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, object]] = []
    config_id = DEFAULT_MODEL_ROUTER_CONFIG_IDS["seedance2"]

    class FakeRouters:
        def list(self, *, limit: int) -> list[SimpleNamespace]:
            calls.append(("list", limit))
            models = SimpleNamespace(mode="allowlist_only", ids=["seedance2"])
            return [
                SimpleNamespace(
                    slug=config_id,
                    version=1,
                    settings=SimpleNamespace(models=models),
                )
            ]

    class FakeVideo:
        def create(self, *, config_id: str, input: object) -> SimpleNamespace:
            calls.append(("create", {"config_id": config_id, "input": input}))
            return SimpleNamespace(id="fresh-task-id")

    class FakeRunway:
        def __init__(self, **_kwargs: object) -> None:
            self.routers = FakeRouters()
            self.generate = SimpleNamespace(video=FakeVideo())

        def post(self, path: str, *, cast_to: object, body: object) -> SimpleNamespace:
            calls.append(("dry-run", {"path": path, "cast_to": cast_to, "body": body}))
            routing = SimpleNamespace(
                model="seedance2",
                estimated_cost=SimpleNamespace(credits=216.0),
            )
            return SimpleNamespace(routing=routing)

    monkeypatch.setattr("cutdetect.pipeline.runway_client.RunwayML", FakeRunway)
    request = GenerationRequest(
        reference_video="runway://video",
        reference_image="runway://image",
        reference_audio="runway://audio",
        prompt_text=UGC_CLONE_V1.body,
        duration=6,
        ratio="9:16",
        reference_video_duration_sec=5.466667,
        model="seedance2",
        resolution="720p",
    )
    gateway = RunwayRouterGateway(
        api_key="key_test",
        logger=JsonlCallLogger(tmp_path / "calls.jsonl"),
        storage=LocalDiskStorage(tmp_path),
        config_id=config_id,
        expected_model="seedance2",
    )

    assert gateway.submit(request) == "fresh-task-id"
    assert [name for name, _value in calls] == ["list", "dry-run", "create"]
    dry_run = calls[1][1]
    assert isinstance(dry_run, dict)
    assert dry_run["body"]["dryRun"] is True
    assert dry_run["body"]["configId"] == config_id
    assert dry_run["body"]["input"] == request.router_http_payload()
    assert calls[2][1]["input"] == request.router_payload()


def test_local_storage_rejects_escape(tmp_path: Path) -> None:
    storage = LocalDiskStorage(tmp_path)

    with pytest.raises(ValueError, match="escapes root"):
        storage.path("../outside.mp4")


def test_safety_provider_diagnostic_is_not_user_visible() -> None:
    raw_code = "SAFETY.INPUT.PROVIDER_DETAIL"
    error = RunwayTaskError("task-id", raw_code)

    assert public_failure_code(raw_code) == "SAFETY.INPUT"
    assert "PROVIDER_DETAIL" not in str(error)
    assert "do not retry automatically" in str(error)
    assert (
        public_failure_code("INPUT_PREPROCESSING.SAFETY.THIRD_PARTY")
        == "INPUT_PREPROCESSING.SAFETY"
    )
    message = public_failure_message("INPUT_PREPROCESSING.SAFETY.THIRD_PARTY")
    assert "INPUT_PREPROCESSING.SAFETY" in message
    assert "Do not retry it unchanged" in message
    assert "THIRD_PARTY" not in message


def test_capability_probe_ceiling_is_sum_of_five_minimal_tests() -> None:
    assert PROBE_CREDIT_CEILING == 5 + 10 + 64 + 144 + 144
