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
    speech_failures: int = 0,
    speech_failure_status: int = 500,
    transcription_api_key: str = "",
) -> tuple[RunwayAudioProcessor, list[tuple[str, object]]]:
    calls: list[tuple[str, object]] = []
    remaining_speech_failures = speech_failures

    class FakeStatusError(Exception):
        status_code = speech_failure_status

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
            nonlocal remaining_speech_failures
            calls.append(("speech", kwargs))
            if remaining_speech_failures:
                remaining_speech_failures -= 1
                raise FakeStatusError("provider unavailable")
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
    monkeypatch.setattr("cutdetect.pipeline.audio_pipeline.time.sleep", lambda _delay: None)
    monkeypatch.setattr(
        "cutdetect.pipeline.audio_pipeline.random.uniform", lambda _start, _end: 0
    )

    expected_duration = duration_sec

    def fake_fit(source: Path, destination: Path, *, duration_sec: float) -> Path:
        assert duration_sec in {expected_duration, 4.6}
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        return destination

    monkeypatch.setattr("cutdetect.pipeline.audio_pipeline.fit_audio_duration", fake_fit)
    processor = RunwayAudioProcessor(
        api_key="key_test",
        logger=JsonlCallLogger(tmp_path / "calls.jsonl"),
        storage=LocalDiskStorage(tmp_path),
        transcription_api_key=transcription_api_key,
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


def test_clip_is_isolated_then_converted_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor, calls = _processor(tmp_path, monkeypatch, duration_sec=60)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    first = processor.convert_clip_voice(
        source, preset_id="Maya", job_id="job", segment_index=2
    )
    second = processor.convert_clip_voice(
        source, preset_id="Maya", job_id="job", segment_index=2
    )

    assert first == second
    assert first.read_bytes() == b"generated audio"
    assert [name for name, _payload in calls].count("isolation") == 1
    assert [name for name, _payload in calls].count("speech") == 1
    speech = next(payload for name, payload in calls if name == "speech")
    assert isinstance(speech, dict)
    assert speech["media"] == {"type": "audio", "uri": "runway://isolated"}
    assert speech["voice"] == {"type": "runway-preset", "preset_id": "Maya"}
    assert speech["remove_background_noise"] is False


def test_isolated_source_transcript_is_cached_per_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor, _calls = _processor(
        tmp_path,
        monkeypatch,
        duration_sec=9,
        transcription_api_key="eleven-key",
    )
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    processor.convert_clip_voice(
        source, preset_id="Maya", job_id="scripted", segment_index=3
    )
    transcription_inputs: list[Path] = []

    def fake_transcribe(audio: Path) -> str:
        transcription_inputs.append(audio)
        return "This is the exact clip script."

    monkeypatch.setattr(processor, "_request_transcript", fake_transcribe)
    first = processor.transcribe_clip(source, job_id="scripted", segment_index=3)
    second = processor.transcribe_clip(source, job_id="scripted", segment_index=3)

    assert first == second == "This is the exact clip script."
    assert [path.name for path in transcription_inputs] == ["isolated.mp3"]
    assert (
        tmp_path / "jobs/scripted/segments/3/audio/transcript.txt"
    ).read_text(encoding="utf-8") == "This is the exact clip script.\n"


def test_transcription_outage_does_not_fail_voice_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor, _calls = _processor(
        tmp_path,
        monkeypatch,
        duration_sec=7,
        transcription_api_key="eleven-key",
    )
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    processor.convert_clip_voice(
        source, preset_id="Maya", job_id="fallback", segment_index=0
    )
    attempts = 0

    class TemporaryFailure(Exception):
        status_code = 500

    def fail_transcription(_audio: Path) -> str:
        nonlocal attempts
        attempts += 1
        raise TemporaryFailure("unavailable")

    monkeypatch.setattr(processor, "_request_transcript", fail_transcription)

    assert processor.transcribe_clip(source, job_id="fallback", segment_index=0) is None
    assert attempts == 4


def test_short_clip_is_padded_and_still_runs_voice_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor, calls = _processor(tmp_path, monkeypatch, duration_sec=4)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    processor.convert_clip_voice(
        source, preset_id="Rachel", job_id="short", segment_index=0
    )

    assert any(name == "isolation" for name, _payload in calls)
    speech = next(payload for name, payload in calls if name == "speech")
    assert isinstance(speech, dict)
    assert speech["media"] == {"type": "audio", "uri": "runway://isolated"}
    assert speech["remove_background_noise"] is False


def test_transient_speech_submission_retries_without_repeating_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor, calls = _processor(
        tmp_path,
        monkeypatch,
        duration_sec=8,
        speech_failures=2,
    )
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    processor.convert_clip_voice(
        source, preset_id="Maya", job_id="retry", segment_index=1
    )

    assert [name for name, _payload in calls].count("isolation") == 1
    assert [name for name, _payload in calls].count("speech") == 3


def test_persistent_speech_submission_stops_after_bounded_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor, calls = _processor(
        tmp_path,
        monkeypatch,
        duration_sec=8,
        speech_failures=4,
    )
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    with pytest.raises(
        PipelineError,
        match="speech-to-speech submission failed for audio clip 2 after 4 attempts",
    ):
        processor.convert_clip_voice(
            source, preset_id="Maya", job_id="persistent", segment_index=1
        )

    assert [name for name, _payload in calls].count("isolation") == 1
    assert [name for name, _payload in calls].count("speech") == 4


def test_nonretryable_speech_submission_fails_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor, calls = _processor(
        tmp_path,
        monkeypatch,
        duration_sec=8,
        speech_failures=4,
        speech_failure_status=400,
    )
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    with pytest.raises(
        PipelineError,
        match="speech-to-speech submission failed for audio clip 1 after 1 attempt",
    ):
        processor.convert_clip_voice(
            source, preset_id="Maya", job_id="invalid", segment_index=0
        )

    assert [name for name, _payload in calls].count("speech") == 1
