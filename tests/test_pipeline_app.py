from __future__ import annotations

import base64
import json
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from cutdetect.pipeline.app import (
    PipelineStudioConfig,
    _health_payload,
    _isolated_detection,
    _load_sessions,
    _persist_session,
    create_pipeline_server,
    render_pipeline_html,
)


def test_ripple_interface_has_streamlined_generation_flow() -> None:
    html = render_pipeline_html()

    assert "Ripple — video recreation" in html
    assert ">ripple</div>" in html
    assert "Source video" in html
    assert "Target face" in html
    assert "Output voice" in html
    assert "Original source audio" in html
    assert "voice_preset" in html
    assert "/api/voice-previews/" in html
    assert "first preview is generated once and cached (1 api credit)" in html.lower()
    assert "UGC product test" in html
    assert "New product" in html
    assert "Comparison route" in html
    assert '"routeLabel":"Model Router API"' in html
    assert '"label":"Hailuo 3"' in html
    assert "ugc_product_clone_v1" in html
    assert "0b9a4bd0-27a2-4ef7-a2d3-ba1d89a8a0d0" not in html
    assert "Optional · preserve original audio" in html
    assert "consent" in html
    assert "Generate clips" in html
    assert "auto_run:true" in html
    assert "/api/jobs/${currentJob}/stitch" in html
    assert "model:model.value" in html
    assert "Seedance 2.0 · Workflow API" in html
    assert '<select id="model">' in html
    assert '<select id="ratio">' in html
    assert '<select id="resolution">' in html
    assert 'id="previewVideo"' in html
    assert 'id="previewImage"' in html
    assert 'id="voiceSample"' in html
    assert 'id="fileAudio"' not in html
    assert "URL.createObjectURL(file)" in html
    assert "Fresh task per clip" in html
    assert "4af4fdf6-a371-4a73-b02d-fdbf116186d5" not in html
    assert "Seedance 2.5" in html
    assert "Hailuo 3.0" in html
    assert "Workflow API" in html
    assert "Generation route" in html
    assert "Recreate Video 1 as one continuous take at 1.0x speed" in html
    assert "Image 1 provides the whole person" in html
    assert "Everything that is not the person comes from Video 1" in html
    assert "stays in sync with Video 1's original dialogue" in html
    assert "Audio 1" not in html
    assert "woman from Image 1" not in html
    assert "Confirm & generate" not in html
    assert "Original reference" not in html
    assert "Source vs clone QC" not in html
    assert "RUNWAYML_API_SECRET" not in html
    assert "ripple.device.v1" in html
    assert "ripple.history.v1" in html
    assert "X-Ripple-Device" in html
    assert "No generations on this device yet." in html
    assert "Generation failed." in html
    assert "Retry clip" in html
    assert "Retry unavailable" in html
    assert "nonRetryableFailure" in html
    assert "Provider blocked this clip." in html
    assert "playback-failed" in html
    assert "events?device=" in html
    assert "ripple.pending.v1" in html
    assert "Reconnecting" in html


def test_health_payload_exposes_render_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RENDER_GIT_COMMIT", "abc123")

    assert _health_payload() == {"status": "ok", "commit": "abc123"}


def test_ripple_review_only_polls_live_states() -> None:
    html = render_pipeline_html()

    assert "syncUpdates(data.state,forceLive)" in html
    live_states = (
        "const live=forceLive||"
        "['RUNNING','STITCHING','DRAFT','CONFIRMED'].includes(state)"
    )
    assert live_states in html
    assert "if(!live){stopUpdates();return}" in html
    assert "function renderRunning(data)" in html
    assert "function renderReview(data)" in html
    assert "original_url" not in html


def test_phase_e_server_serves_ui_and_empty_job_index(tmp_path: Path) -> None:
    try:
        server = create_pipeline_server(
            PipelineStudioConfig(port=0),
            output_root=tmp_path / "jobs",
            cache_dir=tmp_path / "cache",
        )
    except PermissionError:
        pytest.skip("sandbox does not permit binding a loopback socket")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(base + "/", timeout=5) as response:
            html = response.read().decode()
        request = urllib.request.Request(
            base + "/api/jobs", headers={"X-Ripple-Device": "a" * 32}
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            jobs = json.loads(response.read())

        assert response.status == 200
        assert "Ripple — video recreation" in html
        assert "Make it ripple." in html
        assert jobs == {"jobs": []}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_pipeline_server_serializes_memory_intensive_analysis(tmp_path: Path) -> None:
    try:
        server = create_pipeline_server(
            PipelineStudioConfig(port=0),
            output_root=tmp_path / "jobs",
            cache_dir=tmp_path / "cache",
        )
    except PermissionError:
        pytest.skip("sandbox does not permit binding a loopback socket")
    try:
        assert server.analysis_lock.acquire(blocking=False)
        assert not server.analysis_lock.acquire(blocking=False)
        server.analysis_lock.release()
    finally:
        server.server_close()


def test_preparation_session_survives_server_restart(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    session_id = "a" * 32
    state: dict[str, object] = {
        "status": "ANALYZING",
        "owner_hash": "b" * 64,
        "video": str(root / "staging" / session_id / "video.mp4"),
        "request": {"model": "seedance2", "auto_run": True},
    }
    _persist_session(root, session_id, state)

    assert _load_sessions(root) == {session_id: state}
    try:
        server = create_pipeline_server(
            PipelineStudioConfig(port=0),
            output_root=root,
            cache_dir=tmp_path / "cache",
        )
    except PermissionError:
        pytest.skip("sandbox does not permit binding a loopback socket")
    try:
        assert server.sessions[session_id] == state
    finally:
        server.server_close()


def test_detection_runs_in_secret_scrubbed_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    output = tmp_path / "detection"
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        output.mkdir(parents=True)
        (output / "predictions.json").write_text('{"segments": []}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setenv("RUNWAYML_API_SECRET", "must-not-enter-worker")
    monkeypatch.setattr(subprocess, "run", fake_run)

    predictions = _isolated_detection(video, output, tmp_path / "cache", timeout_sec=30)

    assert predictions == output / "predictions.json"
    assert "RUNWAYML_API_SECRET" not in captured["environment"]
    assert captured["command"][:3] == [captured["command"][0], "-m", "cutdetect"]


def test_hosted_server_health_and_optional_password(tmp_path: Path) -> None:
    try:
        server = create_pipeline_server(
            PipelineStudioConfig(port=0, access_password="invite-only"),
            output_root=tmp_path / "jobs",
            cache_dir=tmp_path / "cache",
        )
    except PermissionError:
        pytest.skip("sandbox does not permit binding a loopback socket")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(base + "/healthz", timeout=5) as response:
            assert json.loads(response.read()) == {"status": "ok"}

        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(base + "/", timeout=5)
        assert denied.value.code == 401

        credentials = base64.b64encode(b"ripple:invite-only").decode()
        request = urllib.request.Request(
            base + "/", headers={"Authorization": f"Basic {credentials}"}
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert "Ripple — video recreation" in response.read().decode()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
