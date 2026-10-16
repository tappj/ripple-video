from __future__ import annotations

import shutil
import wave
from pathlib import Path

import pytest

from cutdetect.pipeline.media import RAW_VOICE_MAX_SEC, validate_raw_voice_sample
from cutdetect.pipeline.runway_client import PipelineError


def _write_silent_wav(path: Path, duration_sec: float) -> None:
    sample_rate = 8_000
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(b"\0\0" * round(sample_rate * duration_sec))


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is not installed")
def test_raw_voice_sample_accepts_eleven_seconds_and_rejects_longer_audio(
    tmp_path: Path,
) -> None:
    accepted = tmp_path / "accepted.wav"
    rejected = tmp_path / "rejected.wav"
    _write_silent_wav(accepted, RAW_VOICE_MAX_SEC)
    _write_silent_wav(rejected, RAW_VOICE_MAX_SEC + 0.1)

    assert validate_raw_voice_sample(accepted) == pytest.approx(RAW_VOICE_MAX_SEC, abs=0.001)
    with pytest.raises(PipelineError, match="11 seconds or shorter"):
        validate_raw_voice_sample(rejected)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is not installed")
def test_raw_voice_sample_rejects_an_undecodable_file(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.wav"
    invalid.write_bytes(b"not audio")

    with pytest.raises(PipelineError, match="could not read the uploaded audio"):
        validate_raw_voice_sample(invalid)
