from pathlib import Path
from types import SimpleNamespace

import pytest

from cutdetect.pipeline.capabilities import (
    MODEL_CAPABILITIES,
    closest_seedance_ratio,
    credit_cost,
)
from cutdetect.pipeline.probe_suite import PROBE_CREDIT_CEILING
from cutdetect.pipeline.runway_client import (
    DEFAULT_MODEL_ROUTER_CONFIG_IDS,
    GenerationRequest,
    JsonlCallLogger,
    RunwayRouterGateway,
    RunwayTaskError,
    model_router_route,
    public_failure_code,
    seedance_ratio,
)
from cutdetect.pipeline.storage import LocalDiskStorage
from cutdetect.pipeline.templates import UGC_CLONE_V1


def test_seedance_capabilities_and_vertical_default() -> None:
    caps = MODEL_CAPABILITIES["seedance2"]

    assert caps.min_duration_s == 4.0
    assert caps.max_duration_s == 15.0
    assert caps.supports_reference_audio
    assert not caps.supports_internal_cuts
    assert closest_seedance_ratio(720, 1280) == "720:1280"


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
        model="hailuo3",
        resolution="768P",
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
    assert model_router_route("hailuo3") == (
        "router:" + DEFAULT_MODEL_ROUTER_CONFIG_IDS["hailuo3"]
    )
    assert credit_cost("seedance2", 8, "720p") == 288
    assert credit_cost("seedance2", 8, "4K") == 1200


def test_router_gateway_validates_dry_runs_then_starts_a_fresh_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, object]] = []
    config_id = DEFAULT_MODEL_ROUTER_CONFIG_IDS["hailuo3"]

    class FakeRouters:
        def list(self, *, limit: int) -> list[SimpleNamespace]:
            calls.append(("list", limit))
            models = SimpleNamespace(mode="allowlist_only", ids=["hailuo3"])
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
                model="hailuo3",
                estimated_cost=SimpleNamespace(credits=117.0),
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
        model="hailuo3",
        resolution="768P",
    )
    gateway = RunwayRouterGateway(
        api_key="key_test",
        logger=JsonlCallLogger(tmp_path / "calls.jsonl"),
        storage=LocalDiskStorage(tmp_path),
        config_id=config_id,
        expected_model="hailuo3",
    )

    assert gateway.submit(request) == "fresh-task-id"
    assert [name for name, _value in calls] == ["list", "dry-run", "create"]
    dry_run = calls[1][1]
    assert isinstance(dry_run, dict)
    assert dry_run["body"]["dryRun"] is True
    assert dry_run["body"]["configId"] == config_id


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


def test_capability_probe_ceiling_is_sum_of_five_minimal_tests() -> None:
    assert PROBE_CREDIT_CEILING == 5 + 10 + 64 + 144 + 144
