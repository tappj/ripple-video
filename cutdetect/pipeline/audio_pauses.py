"""Find speech-friendly split points without loading video frames into memory."""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class AudioPauseSelector:
    """Select silence first, then the lowest-energy nearby audio moment."""

    sample_rate: int
    power: npt.NDArray[np.float32]
    silence_centers_sec: tuple[float, ...]
    search_radius_sec: float = 2.5

    def __call__(
        self, ideal_sec: float, lower_sec: float, upper_sec: float
    ) -> tuple[float, str]:
        search_lower = max(lower_sec, ideal_sec - self.search_radius_sec)
        search_upper = min(upper_sec, ideal_sec + self.search_radius_sec)
        pauses = tuple(
            value
            for value in self.silence_centers_sec
            if search_lower <= value <= search_upper
        )
        if pauses:
            return min(pauses, key=lambda value: abs(value - ideal_sec)), "audio_silence"

        start = max(0, math.ceil(search_lower * self.sample_rate))
        stop = min(len(self.power), math.floor(search_upper * self.sample_rate) + 1)
        if stop > start:
            offset = int(np.argmin(self.power[start:stop]))
            return (start + offset) / self.sample_rate, "audio_low_energy"
        return min(max(ideal_sec, lower_sec), upper_sec), "timed_fallback"


def load_audio_pause_selector(video: str | Path) -> AudioPauseSelector | None:
    """Decode a lightweight mono envelope and locate sentence-like pauses."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        return None
    sample_rate = 200
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-i",
            str(Path(video).expanduser().resolve()),
            "-map",
            "0:a:0?",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0 or not completed.stdout:
        return None
    samples = np.frombuffer(completed.stdout, dtype="<f4")
    if not len(samples):
        return None

    window_samples = max(1, round(0.08 * sample_rate))
    squared = np.square(samples, dtype=np.float32)
    kernel = np.full(window_samples, 1.0 / window_samples, dtype=np.float32)
    power = np.convolve(squared, kernel, mode="same").astype(np.float32, copy=False)
    quiet = power <= np.float32(10 ** (-38.0 / 10.0))
    minimum_quiet_samples = max(1, round(0.18 * sample_rate))
    transitions = np.diff(np.pad(quiet.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    centers = [
        (int(start) + int(stop) - 1) / (2.0 * sample_rate)
        for start, stop in zip(starts, stops, strict=True)
        if stop - start >= minimum_quiet_samples
    ]
    return AudioPauseSelector(sample_rate, power, tuple(centers))
