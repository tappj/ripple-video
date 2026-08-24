"""Durable Runway/ElevenLabs voice mastering for Ripple source soundtracks."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Literal, cast

from runwayml import RunwayML
from runwayml.types.speech_to_speech_create_params import Voice
from runwayml.types.task_retrieve_response import Failed, Succeeded
from runwayml.types.text_to_speech_create_params import ElevenMultilingualV2Voice

from cutdetect.ingest import probe_video
from cutdetect.pipeline.media import extract_audio_track
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


def validate_voice_preset(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    if value not in RUNWAY_PRESET_VOICES:
        raise PipelineError(f"unsupported Runway voice preset: {value}")
    return value


def voice_audio_credit_cost(duration_sec: float) -> int:
    """Estimate isolation plus speech-to-speech credits for one source track."""
    isolation = math.ceil(duration_sec / 6) if duration_sec >= VOICE_ISOLATION_MIN_SEC else 0
    return isolation + math.ceil(duration_sec / 3)


class RunwayAudioProcessor:
    """Resume-safe two-stage voice conversion using direct Runway audio tasks."""

    def __init__(
        self,
        *,
        api_key: str,
        logger: JsonlCallLogger,
        storage: LocalDiskStorage,
    ) -> None:
        if not api_key:
            raise PipelineError("RUNWAYML_API_SECRET is not set")
        self._client = RunwayML(api_key=api_key, max_retries=0)
        self._logger = logger
        self._storage = storage

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

    def convert_source_voice(
        self,
        source_video: Path,
        *,
        preset_id: str,
        job_id: str,
    ) -> Path:
        """Create or resume one continuous replacement-voice master track."""
        preset = validate_voice_preset(preset_id)
        if preset is None:
            raise PipelineError("a voice preset is required for audio conversion")
        duration = probe_video(source_video).duration_sec
        if duration > SPEECH_TO_SPEECH_MAX_SEC + 1e-6:
            raise PipelineError(
                "preset voice conversion currently supports source videos up to 5 minutes"
            )
        base = f"jobs/{job_id}/audio"
        state_key = f"{base}/state.json"
        output_key = f"{base}/voice_master.mp3"
        output = self._storage.path(output_key)
        if output.is_file():
            return output
        state = self._state(state_key)
        source_audio = self._storage.path(f"{base}/source.m4a")
        if not source_audio.is_file():
            extract_audio_track(source_video, source_audio)
        source_uri = str(state.get("source_uri", "")) or self._upload(
            source_audio, "source_soundtrack"
        )
        state.update(source_uri=source_uri, preset_id=preset, duration_sec=duration)
        self._save_state(state_key, state)

        speech_input_uri = source_uri
        remove_background_noise = True
        if duration >= VOICE_ISOLATION_MIN_SEC:
            isolation_task_id = str(state.get("isolation_task_id", ""))
            if not isolation_task_id:
                try:
                    created = self._client.voice_isolation.create(
                        model="eleven_voice_isolation", audio_uri=source_uri
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
                isolated, "isolated_voice"
            )
            state["isolated_uri"] = isolated_uri
            self._save_state(state_key, state)
            speech_input_uri = isolated_uri
            remove_background_noise = False

        speech_task_id = str(state.get("speech_task_id", ""))
        if not speech_task_id:
            voice = cast(Voice, {"type": "runway-preset", "preset_id": preset})
            try:
                created = self._client.speech_to_speech.create(
                    model="eleven_multilingual_sts_v2",
                    media={"type": "audio", "uri": speech_input_uri},
                    voice=voice,
                    remove_background_noise=remove_background_noise,
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
        result = self._download(output_url, output_key)
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
