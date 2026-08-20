from __future__ import annotations

import numpy as np

from cutdetect.pipeline.audio_pauses import AudioPauseSelector


def test_audio_pause_selector_prefers_silence_near_ideal_boundary() -> None:
    selector = AudioPauseSelector(
        sample_rate=10,
        power=np.ones(200, dtype=np.float32),
        silence_centers_sec=(7.0, 10.4, 13.0),
    )

    split_sec, kind = selector(10.0, 4.0, 15.0)

    assert split_sec == 10.4
    assert kind == "audio_silence"


def test_audio_pause_selector_uses_lowest_energy_when_silence_is_unavailable() -> None:
    power = np.ones(200, dtype=np.float32)
    power[103] = 0.01
    selector = AudioPauseSelector(sample_rate=10, power=power, silence_centers_sec=())

    split_sec, kind = selector(10.0, 4.0, 15.0)

    assert split_sec == 10.3
    assert kind == "audio_low_energy"
