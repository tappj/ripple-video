from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from cutdetect.pipeline.app import (
    PipelineStudioConfig,
    _health_payload,
    create_pipeline_server,
    render_pipeline_html,
)


def test_ripple_interface_has_streamlined_generation_flow() -> None:
    html = render_pipeline_html()

    assert "Ripple — video recreation" in html
    assert ">ripple</div>" in html
    assert "Source video" in html
    assert "Target face" in html
    assert "Target voice" in html
    assert "Optional · keep source audio" in html
    assert "consent" in html
    assert "Generate clips" in html
    assert "auto_run:true" in html
    assert "/api/jobs/${currentJob}/stitch" in html
    assert "model:model.value" in html
    assert "Seedance 2 · Router" in html
    assert '<select id="model">' in html
    assert '<select id="ratio">' in html
    assert '<select id="resolution">' in html
    assert 'id="previewVideo"' in html
    assert 'id="previewImage"' in html
    assert 'id="previewAudio"' in html
    assert "URL.createObjectURL(file)" in html
    assert "fresh task per clip" in html
    assert "Hailuo 3" in html
    assert "Direct API" in html
    assert "Generation route" in html
    assert "Replace the person in Video 1 with the person from Image 1" in html
    assert "woman from Image 1" not in html
    assert "Confirm & generate" not in html
    assert "Original reference" not in html
    assert "Source vs clone QC" not in html
    assert "RUNWAYML_API_SECRET" not in html


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
        with urllib.request.urlopen(base + "/api/jobs", timeout=5) as response:
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
