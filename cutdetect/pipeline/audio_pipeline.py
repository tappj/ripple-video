"""Durable Runway/ElevenLabs voice mastering for Ripple source soundtracks."""

from __future__ import annotations

import json
import math
import random
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypeVar, cast

from runwayml import APIConnectionError, APITimeoutError, RunwayML
from runwayml.types.speech_to_speech_create_params import Voice
from runwayml.types.task_retrieve_response import Failed, Succeeded
from runwayml.types.text_to_speech_create_params import ElevenMultilingualV2Voice

from cutdetect.ingest import probe_video
from cutdetect.pipeline.media import extract_audio_track, fit_audio_duration
from cutdetect.pipeline.runway_client import JsonlCallLogger, PipelineError, public_failure_code
from cutdetect.pipeline.storage import LocalDiskStorage

RUNWAY_PRESET_VOICES = (
    "Maya", "Arjun", "Serene", "Bernard", "Billy", "Mark", "Clint", "Mabel",
    "Chad", "Leslie", "Eleanor", "Elias", "Elliot", "Grungle", "Brodie",
    "Sandra", "Kirk", "Kylie", "Lara", "Lisa", "Malachi", "Marlene", "Martin",
    "Miriam", "Monster", "Paula", "Pip", "Rusty", "Ragnar", "Xylar", "Maggie",
    "Jack", "Katie", "Noah", "James", "Rina", "Ella", "Mariah", "Frank",
    "Claudia", "Niki", "Vincent", "Kendrick", "Myrna", "Tom", "Wanda",
    "Benjamin", "Kiana", "Rachel",
)
RunwayPresetVoice = Literal[
    "Maya", "Arjun", "Serene", "Bernard", "Billy", "Mark", "Clint", "Mabel",
    "Chad", "Leslie", "Eleanor", "Elias", "Elliot", "Grungle", "Brodie",
    "Sandra", "Kirk", "Kylie", "Lara", "Lisa", "Malachi", "Marlene", "Martin",
    "Miriam", "Monster", "Paula", "Pip", "Rusty", "Ragnar", "Xylar", "Maggie",
    "Jack", "Katie", "Noah", "James", "Rina", "Ella", "Mariah", "Frank",
    "Claudia", "Niki", "Vincent", "Kendrick", "Myrna", "Tom", "Wanda",
    "Benjamin", "Kiana", "Rachel",
]

VOICE_ISOLATION_MIN_SEC = 4.6
SPEECH_TO_SPEECH_MAX_SEC = 300.0
VOICE_PREVIEW_TEXT = "This is your selected voice for Ripple."
AUDIO_SUBMISSION_MAX_ATTEMPTS = 4
AUDIO_RETRY_BASE_DELAY_SEC = 1.0
ELEVENLABS_TRANSCRIPTION_URL = "https://api.elevenlabs.io/v1/speech-to-text"
TRANSCRIPTION_MAX_ATTEMPTS = 4
TRANSCRIPTION_RETRY_BASE_DELAY_SEC = 1.0
MAX_CLIP_TRANSCRIPT_CHARS = 1000
_Result = TypeVar("_Result")


class _TranscriptionRequestError(Exception):
    """HTTP-shaped failure used by the bounded Scribe retry loop."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def validate_voice_preset(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    if value not in RUNWAY_PRESET_VOICES:
        raise PipelineError(f"unsupported Runway voice preset: {value}")
    return value


def voice_audio_credit_cost(duration_sec: float) -> int:
    """Estimate isolation plus speech-to-speech credits for one independent clip."""
    if duration_sec <= 0:
        return 0
    billable_duration = max(duration_sec, VOICE_ISOLATION_MIN_SEC)
    return math.ceil(billable_duration / 6) + math.ceil(billable_duration / 3)


class RunwayAudioProcessor:
    """Resume-safe two-stage voice conversion using direct Runway audio tasks."""

    def __init__(
        self,
        *,
        api_key: str,
        logger: JsonlCallLogger,
        storage: LocalDiskStorage,
        transcription_api_key: str = "",
    ) -> None:
        if not api_key:
            raise PipelineError("RUNWAYML_API_SECRET is not set")
        self._client = RunwayML(api_key=api_key, max_retries=0)
        self._logger = logger
        self._storage = storage
        self._transcription_api_key = transcription_api_key.strip()

    def _call_with_retry(
        self,
        operation: str,
        call: Callable[[], _Result],
        *,
        segment_index: int,
    ) -> _Result:
        """Retry transient provider submission failures without repeating completed stages."""
        for attempt in range(1, AUDIO_SUBMISSION_MAX_ATTEMPTS + 1):
            try:
                return call()
            except Exception as error:
                status = getattr(error, "status_code", None)
                retryable = (
                    isinstance(error, (APIConnectionError, APITimeoutError))
                    or status in {408, 409, 429}
                    or (isinstance(status, int) and status >= 500)
                )
                self._logger.write(
                    "runway.audio.submission_failed",
                    operation=operation,
                    segment_index=segment_index,
                    attempt=attempt,
                    max_attempts=AUDIO_SUBMISSION_MAX_ATTEMPTS,
                    status_code=status,
                    retryable=retryable,
                    error_type=type(error).__name__,
                )
                if not retryable or attempt == AUDIO_SUBMISSION_MAX_ATTEMPTS:
                    raise PipelineError(
                        f"Runway {operation} failed for audio clip {segment_index + 1} "
                        f"after {attempt} attempt{'s' if attempt != 1 else ''}: {error}"
                    ) from error
                base_delay = AUDIO_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
                delay = base_delay + random.uniform(0, base_delay * 0.5)
                self._logger.write(
                    "runway.audio.submission_retry_scheduled",
                    operation=operation,
                    segment_index=segment_index,
                    next_attempt=attempt + 1,
                    delay_sec=delay,
                )
                time.sleep(delay)
        raise AssertionError("audio retry loop exhausted without returning or raising")

    def _state(self, key: str) -> dict[str, object]:
        path = self._storage.path(key)
        if not path.is_file():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}

    def _save_state(self, key: str, state: dict[str, object]) -> None:
        self._storage.write_json(key, state)

    def _upload(self, path: Path, role: str) -> str:
        started = time.monotonic()
        try:
            result = self._client.uploads.create_ephemeral(file=path)
        except Exception as error:
            raise PipelineError(f"Runway audio upload failed for {role}: {error}") from error
        self._logger.write(
            "runway.audio.uploaded",
            role=role,
            size_bytes=path.stat().st_size,
            latency_sec=time.monotonic() - started,
        )
        return result.uri

    def _wait(self, task_id: str, *, timeout_sec: float = 3600.0) -> str:
        started = time.monotonic()
        while True:
            if time.monotonic() - started > timeout_sec:
                raise PipelineError(f"Runway audio task {task_id} timed out")
            try:
                task = self._client.tasks.retrieve(task_id)
            except Exception as error:
                self._logger.write(
                    "runway.audio.poll_failed", task_id=task_id, error_type=type(error).__name__
                )
                time.sleep(5)
                continue
            self._logger.write("runway.audio.polled", task_id=task_id, status=task.status)
            if isinstance(task, Failed):
                code = public_failure_code(task.failure_code)
                raise PipelineError(f"Runway audio processing failed ({code})")
            if isinstance(task, Succeeded):
                if not task.output:
                    raise PipelineError("Runway audio task succeeded without output")
                return task.output[0]
            if task.status == "CANCELLED":
                raise PipelineError("Runway audio processing was cancelled")
            time.sleep(5)

    def _download(self, url: str, key: str) -> Path:
        return self._storage.download_https(url, key)

    def _request_transcript(self, audio: Path) -> str:
        """Transcribe one isolated source track with verbatim Scribe settings."""
        boundary = f"ripple-{uuid.uuid4().hex}"
        content_type = "audio/mp4" if audio.suffix.lower() == ".m4a" else "audio/mpeg"
        chunks: list[bytes] = []

        def field(name: str, value: str) -> None:
            chunks.extend(
                (
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                    value.encode(),
                    b"\r\n",
                )
            )

        field("model_id", "scribe_v2")
        field("tag_audio_events", "false")
        field("num_speakers", "1")
        field("timestamps_granularity", "none")
        field("diarize", "false")
        field("no_verbatim", "false")
        field("temperature", "0")
        chunks.extend(
            (
                f"--{boundary}\r\n".encode(),
                (
                    'Content-Disposition: form-data; name="file"; '
                    f'filename="{audio.name}"\r\n'
                ).encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                audio.read_bytes(),
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            )
        )
        request = urllib.request.Request(
            ELEVENLABS_TRANSCRIPTION_URL,
            data=b"".join(chunks),
            headers={
                "xi-api-key": self._transcription_api_key,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "ripple-video/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload: object = json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise _TranscriptionRequestError(
                f"ElevenLabs transcription returned HTTP {error.code}",
                status_code=error.code,
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise _TranscriptionRequestError(
                "ElevenLabs transcription could not be reached"
            ) from error
        except json.JSONDecodeError as error:
            raise _TranscriptionRequestError(
                "ElevenLabs transcription returned an invalid response"
            ) from error
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise _TranscriptionRequestError(
                "ElevenLabs transcription returned no transcript"
            )
        return " ".join(payload["text"].split())

    def transcribe_clip(
        self,
        clip_video: Path,
        *,
        job_id: str,
        segment_index: int,
    ) -> str | None:
        """Return a cached per-clip script without making transcription a hard dependency."""
        base = f"jobs/{job_id}/segments/{segment_index}/audio"
        transcript = self._storage.path(f"{base}/transcript.txt")
        if transcript.is_file():
            cached = " ".join(transcript.read_text(encoding="utf-8").split())
            return cached or None
        if not self._transcription_api_key:
            self._logger.write(
                "elevenlabs.transcription.skipped",
                segment_index=segment_index,
                reason="ELEVENLABS_API_KEY is not set",
            )
            return None

        # The isolated source is the ground-truth wording. Transcribing the converted
        # voice could reinforce a word that speech-to-speech happened to distort.
        isolated = self._storage.path(f"{base}/isolated.mp3")
        source_audio = self._storage.path(f"{base}/source.m4a")
        transcription_input = isolated if isolated.is_file() else source_audio
        if not transcription_input.is_file():
            extract_audio_track(clip_video, source_audio)
            transcription_input = source_audio

        for attempt in range(1, TRANSCRIPTION_MAX_ATTEMPTS + 1):
            started = time.monotonic()
            try:
                text = self._request_transcript(transcription_input)
            except Exception as error:
                status = getattr(error, "status_code", None)
                retryable = status in {408, 409, 429} or (
                    isinstance(status, int) and status >= 500
                )
                self._logger.write(
                    "elevenlabs.transcription.failed",
                    segment_index=segment_index,
                    attempt=attempt,
                    max_attempts=TRANSCRIPTION_MAX_ATTEMPTS,
                    status_code=status,
                    retryable=retryable,
                    error_type=type(error).__name__,
                    latency_sec=time.monotonic() - started,
                )
                if not retryable or attempt == TRANSCRIPTION_MAX_ATTEMPTS:
                    self._logger.write(
                        "elevenlabs.transcription.skipped",
                        segment_index=segment_index,
                        reason="transcription remained unavailable",
                    )
                    return None
                base_delay = TRANSCRIPTION_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
                time.sleep(base_delay + random.uniform(0, base_delay * 0.5))
                continue
            if not text:
                self._logger.write(
                    "elevenlabs.transcription.skipped",
                    segment_index=segment_index,
                    reason="isolated clip contains no detected speech",
                )
                return None
            if len(text) > MAX_CLIP_TRANSCRIPT_CHARS:
                self._logger.write(
                    "elevenlabs.transcription.skipped",
                    segment_index=segment_index,
                    reason="transcript is implausibly long for one Ripple clip",
                    character_count=len(text),
                )
                return None
            transcript.write_text(text + "\n", encoding="utf-8")
            self._logger.write(
                "elevenlabs.transcription.complete",
                segment_index=segment_index,
                character_count=len(text),
                latency_sec=time.monotonic() - started,
            )
            return text
        return None

    def convert_clip_voice(
        self,
        clip_video: Path,
        *,
        preset_id: str,
        job_id: str,
        segment_index: int,
    ) -> Path:
        """Recreate the audio workflow independently for one already-cut clip."""
        preset = validate_voice_preset(preset_id)
        if preset is None:
            raise PipelineError("a voice preset is required for audio conversion")
        duration = probe_video(clip_video).duration_sec
        if duration > SPEECH_TO_SPEECH_MAX_SEC + 1e-6:
            raise PipelineError(
                "preset voice conversion currently supports clips up to 5 minutes"
            )
        base = f"jobs/{job_id}/segments/{segment_index}/audio"
        state_key = f"{base}/state.json"
        output_key = f"{base}/voice.m4a"
        output = self._storage.path(output_key)
        if output.is_file():
            return output
        state = self._state(state_key)
        source_audio = self._storage.path(f"{base}/source.m4a")
        if not source_audio.is_file():
            extract_audio_track(clip_video, source_audio)
        isolation_input = source_audio
        if duration < VOICE_ISOLATION_MIN_SEC:
            isolation_input = self._storage.path(f"{base}/isolation_input.m4a")
            if not isolation_input.is_file():
                fit_audio_duration(
                    source_audio,
                    isolation_input,
                    duration_sec=VOICE_ISOLATION_MIN_SEC,
                )
        source_uri = str(state.get("source_uri", "")) or self._upload(
            isolation_input, f"segment_{segment_index}_soundtrack"
        )
        state.update(source_uri=source_uri, preset_id=preset, duration_sec=duration)
        self._save_state(state_key, state)

        isolation_task_id = str(state.get("isolation_task_id", ""))
        if not isolation_task_id:
            try:
                created = self._call_with_retry(
                    "voice isolation submission",
                    lambda: self._client.voice_isolation.create(
                        model="eleven_voice_isolation", audio_uri=source_uri
                    ),
                    segment_index=segment_index,
                )
            except Exception as error:
                raise PipelineError(
                    f"Runway voice isolation submission failed: {error}"
                ) from error
            isolation_task_id = created.id
            state["isolation_task_id"] = isolation_task_id
            self._save_state(state_key, state)
            self._logger.write("runway.audio.isolation_submitted", task_id=isolation_task_id)
        isolated = self._storage.path(f"{base}/isolated.mp3")
        if not isolated.is_file():
            isolated_url = self._wait(isolation_task_id)
            self._download(isolated_url, f"{base}/isolated.mp3")
        isolated_uri = str(state.get("isolated_uri", "")) or self._upload(
            isolated, f"segment_{segment_index}_isolated_voice"
        )
        state["isolated_uri"] = isolated_uri
        self._save_state(state_key, state)

        speech_task_id = str(state.get("speech_task_id", ""))
        if not speech_task_id:
            voice = cast(Voice, {"type": "runway-preset", "preset_id": preset})
            try:
                created = self._call_with_retry(
                    "speech-to-speech submission",
                    lambda: self._client.speech_to_speech.create(
                        model="eleven_multilingual_sts_v2",
                        media={"type": "audio", "uri": isolated_uri},
                        voice=voice,
                        remove_background_noise=False,
                    ),
                    segment_index=segment_index,
                )
            except Exception as error:
                raise PipelineError(
                    f"Runway speech-to-speech submission failed: {error}"
                ) from error
            speech_task_id = created.id
            state["speech_task_id"] = speech_task_id
            self._save_state(state_key, state)
            self._logger.write(
                "runway.audio.speech_to_speech_submitted",
                task_id=speech_task_id,
                preset_id=preset,
            )
        output_url = self._wait(speech_task_id)
        raw_output = self._storage.path(f"{base}/voice_raw.mp3")
        if not raw_output.is_file():
            self._download(output_url, f"{base}/voice_raw.mp3")
        result = fit_audio_duration(raw_output, output, duration_sec=duration)
        state["complete"] = True
        self._save_state(state_key, state)
        return result

    def voice_preview(self, preset_id: str) -> Path | None:
        """Submit/poll a one-credit preset preview; return None while it is running."""
        preset = validate_voice_preset(preset_id)
        if preset is None:
            raise PipelineError("a voice preset is required")
        safe = preset.lower()
        base = f"voice_previews/{safe}"
        output_key = f"{base}.mp3"
        output = self._storage.path(output_key)
        if output.is_file():
            return output
        state_key = f"{base}.json"
        state = self._state(state_key)
        task_id = str(state.get("task_id", ""))
        if not task_id:
            voice = cast(
                ElevenMultilingualV2Voice,
                {"type": "runway-preset", "preset_id": preset},
            )
            try:
                created = self._client.text_to_speech.create(
                    model="eleven_multilingual_v2",
                    prompt_text=VOICE_PREVIEW_TEXT,
                    voice=voice,
                )
            except Exception as error:
                raise PipelineError(f"Runway voice preview submission failed: {error}") from error
            self._save_state(state_key, {"task_id": created.id, "preset_id": preset})
            self._logger.write(
                "runway.audio.preview_submitted", task_id=created.id, preset_id=preset
            )
            return None
        task = self._client.tasks.retrieve(task_id)
        if isinstance(task, Failed):
            raise PipelineError(
                f"Runway voice preview failed ({public_failure_code(task.failure_code)})"
            )
        if isinstance(task, Succeeded):
            if not task.output:
                raise PipelineError("Runway voice preview succeeded without output")
            return self._download(task.output[0], output_key)
        return None
