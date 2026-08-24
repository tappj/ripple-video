from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cutdetect.pipeline.audio_pipeline import (
    RUNWAY_PRESET_VOICES,
    RunwayAudioProcessor,
    validate_voice_preset,
    voice_audio_credit_cost,
)
from cutdetect.pipeline.runway_client import JsonlCallLogger, PipelineError
from cutdetect.pipeline.storage import LocalDiskStorage


class FakeSucceeded:
    status = "SUCCEEDED"

    def __init__(self, output: str) -> None:
        self.output = [output]


def _processor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    duration_sec: float,
) -> tuple[RunwayAudioProcessor, list[tuple[str, object]]]:
    calls: list[tuple[str, object]] = []

    class Uploads:
        def create_ephemeral(self, *, file: Path) -> SimpleNamespace:
            calls.append(("upload", file.name))
            return SimpleNamespace(uri=f"runway://{file.stem}")

    class Isolation:
        def create(self, **kwargs: object) -> SimpleNamespace:
            calls.append(("isolation", kwargs))
            return SimpleNamespace(id="isolation-task")

    class Speech:
        def create(self, **kwargs: object) -> SimpleNamespace:
            calls.append(("speech", kwargs))
            return SimpleNamespace(id="speech-task")

    class Tasks:
        def retrieve(self, task_id: str) -> FakeSucceeded:
            calls.append(("poll", task_id))
            return FakeSucceeded(f"https://outputs/{task_id}.mp3")

    class FakeRunway:
        def __init__(self, **_kwargs: object) -> None:
            self.uploads = Uploads()
            self.voice_isolation = Isolation()
            self.speech_to_speech = Speech()
            self.tasks = Tasks()

    monkeypatch.setattr("cutdetect.pipeline.audio_pipeline.RunwayML", FakeRunway)
    monkeypatch.setattr("cutdetect.pipeline.audio_pipeline.Succeeded", FakeSucceeded)
    monkeypatch.setattr(
        "cutdetect.pipeline.audio_pipeline.probe_video",
        lambda _path: SimpleNamespace(duration_sec=duration_sec),
    )

    def fake_extract(_source: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"source audio")
        return destination

    monkeypatch.setattr("cutdetect.pipeline.audio_pipeline.extract_audio_track", fake_extract)
    processor = RunwayAudioProcessor(
        api_key="key_test",
        logger=JsonlCallLogger(tmp_path / "calls.jsonl"),
        storage=LocalDiskStorage(tmp_path),
    )

    def fake_download(_url: str, key: str) -> Path:
        destination = tmp_path / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"generated audio")
        return destination

    monkeypatch.setattr(processor, "_download", fake_download)
    return processor, calls


def test_voice_catalog_and_credit_estimate_are_validated() -> None:
    assert len(RUNWAY_PRESET_VOICES) == 49
    assert validate_voice_preset("Maya") == "Maya"
    assert validate_voice_preset("") is None
    assert voice_audio_credit_cost(60) == 30
    with pytest.raises(PipelineError, match="unsupported Runway voice preset"):
        validate_voice_preset("not-a-real-voice")


def test_full_source_is_isolated_then_converted_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor, calls = _processor(tmp_path, monkeypatch, duration_sec=60)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    first = processor.convert_source_voice(source, preset_id="Maya", job_id="job")
    second = processor.convert_source_voice(source, preset_id="Maya", job_id="job")

    assert first == second
    assert first.read_bytes() == b"generated audio"
    assert [name for name, _payload in calls].count("isolation") == 1
    assert [name for name, _payload in calls].count("speech") == 1
    speech = next(payload for name, payload in calls if name == "speech")
    assert isinstance(speech, dict)
    assert speech["media"] == {"type": "audio", "uri": "runway://isolated"}
    assert speech["voice"] == {"type": "runway-preset", "preset_id": "Maya"}
    assert speech["remove_background_noise"] is False


def test_short_source_skips_unsupported_isolation_but_still_removes_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor, calls = _processor(tmp_path, monkeypatch, duration_sec=4)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    processor.convert_source_voice(source, preset_id="Rachel", job_id="short")

    assert not any(name == "isolation" for name, _payload in calls)
    speech = next(payload for name, payload in calls if name == "speech")
    assert isinstance(speech, dict)
    assert speech["media"] == {"type": "audio", "uri": "runway://source"}
    assert speech["remove_background_noise"] is True
