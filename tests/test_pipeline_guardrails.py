from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cutdetect.pipeline import app
from cutdetect.pipeline.guardrails import AdmissionError, RequestGate
from cutdetect.pipeline.runway_client import PipelineError


def test_device_limit_recovers_after_one_minute() -> None:
    now = [0.0]
    gate = RequestGate(device_limit=1, clock=lambda: now[0])
    gate.check("first")
    with pytest.raises(AdmissionError) as rejected:
        gate.check("first")
    assert rejected.value.status == HTTPStatus.TOO_MANY_REQUESTS
    gate.check("second")
    now[0] = 60
    gate.check("first")


def test_rotating_identity_cannot_bypass_global_limit_under_concurrency() -> None:
    gate = RequestGate(global_limit=10)

    def admit(index: int) -> bool:
        try:
            gate.check(str(index))
        except AdmissionError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(admit, range(100))) == 10


@pytest.fixture
def handler(tmp_path: Path):
    server = app.create_pipeline_server(
        app.PipelineStudioConfig(port=0),
        output_root=tmp_path / "jobs",
        cache_dir=tmp_path / "cache",
    )
    instance = object.__new__(app._PipelineHandler)
    instance.server = server
    try:
        yield instance
    finally:
        server.server_close()


@pytest.mark.parametrize(
    ("credits", "duration", "message"),
    [(601, 10, "600 estimated Runway credits"), (500, 61, "60 seconds or shorter")],
)
def test_job_limits_reject_before_any_paid_work(
    handler, monkeypatch: pytest.MonkeyPatch, credits: int, duration: float, message: str
) -> None:
    job = SimpleNamespace(estimated_credits=credits, source_path=Path("source.mp4"))
    store = SimpleNamespace(job=lambda job_id: job)
    monkeypatch.setattr(app, "PhaseCStore", lambda path: nullcontext(store))
    monkeypatch.setattr(app, "probe_video", lambda path: SimpleNamespace(duration_sec=duration))
    gateway = Mock()
    monkeypatch.setattr(app._PipelineHandler, "_generation_gateway", gateway)

    with pytest.raises(PipelineError, match=message):
        handler._start_job("a" * 32, 999999)

    gateway.assert_not_called()
    assert not handler.server.active_jobs


def test_source_duration_rejected_before_cut_detection(handler, monkeypatch) -> None:
    video = handler.server.root / "source.mp4"
    image = handler.server.root / "face.jpg"
    video.touch()
    image.touch()
    session = "a" * 32
    handler.server.sessions[session] = {
        "video": str(video), "image": str(image), "request": {"auto_run": True}
    }
    monkeypatch.setattr(app, "probe_video", lambda path: SimpleNamespace(duration_sec=61))
    detection = Mock()
    monkeypatch.setattr(app, "_isolated_detection", detection)

    handler._prepare_work(session)

    detection.assert_not_called()
    assert handler.server.sessions[session]["status"] == "FAILED"
    assert "60 seconds or shorter" in handler.server.sessions[session]["error"]
    assert handler.server.analysis_lock.acquire(blocking=False)
    handler.server.analysis_lock.release()


def test_work_capacity_rejects_new_jobs_but_allows_duplicate_polling(handler) -> None:
    handler.server.active_jobs.update({"a" * 32, "b" * 32})
    handler._start_job("a" * 32, 600)
    with pytest.raises(AdmissionError) as rejected:
        handler._start_job("c" * 32, 600)
    assert rejected.value.status == HTTPStatus.SERVICE_UNAVAILABLE
    assert len(handler.server.active_jobs) == 2
    handler.server.active_preparations.update({"a" * 32, "b" * 32})
    with pytest.raises(AdmissionError):
        handler._launch_prepare("c" * 32)
    assert len(handler.server.active_preparations) == 2


def test_http_rate_limit_blocks_paid_preview_but_keeps_cancel_available(handler) -> None:
    server = handler.server
    device = "d" * 32
    owner = hashlib.sha256(device.encode()).hexdigest()
    server.request_gate = RequestGate(global_limit=1)
    server.request_gate.check(owner)
    session = "a" * 32
    server.sessions[session] = {"status": "UPLOADING", "owner_hash": owner}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def post(path: str):
        return urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}{path}",
                data=b"{}", headers={"X-Ripple-Device": device}, method="POST",
            ),
            timeout=5,
        )

    try:
        with pytest.raises(urllib.error.HTTPError) as rejected:
            post("/api/voice-previews/Maya")
        assert rejected.value.code == 429
        assert "Too many requests" in json.loads(rejected.value.read())["error"]
        assert not (server.root / "voice_previews").exists()
        with post(f"/api/sessions/{session}/cancel") as response:
            assert json.load(response)["status"] == "CANCELLED"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_hosting_transcription_requires_explicit_opt_in(monkeypatch) -> None:
    from cutdetect.pipeline import hosting

    monkeypatch.setattr(hosting, "load_env_file", lambda: None)
    run = Mock()
    monkeypatch.setattr(hosting, "run_pipeline_studio", run)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "available-but-not-enabled")
    monkeypatch.delenv("RIPPLE_ENABLE_TRANSCRIPTION", raising=False)
    hosting.main()
    assert run.call_args.kwargs["config"].enable_transcription is False
    monkeypatch.setenv("RIPPLE_ENABLE_TRANSCRIPTION", "true")
    hosting.main()
    assert run.call_args.kwargs["config"].enable_transcription is True
