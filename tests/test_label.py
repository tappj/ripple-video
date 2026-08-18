from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from cutdetect.config import LabelConfig
from cutdetect.ingest import VideoContext
from cutdetect.label import render_labeler_html


def test_labeler_contains_required_sweep_review_and_keyboard_controls(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    context = VideoContext(
        source_path=video,
        working_video_path=video,
        artifact_dir=tmp_path / "cache",
        audio_path=None,
        cache_key="test",
        duration_sec=1.0,
        fps=Fraction(30, 1),
        frame_count=30,
        width=1080,
        height=1920,
        was_vfr=False,
        has_audio=False,
        video_codec="h264",
        audio_codec=None,
        source_rotation_deg=0,
        working_timestamps_sec=tuple(index / 30 for index in range(30)),
        original_timestamps_sec=tuple(index / 30 for index in range(30)),
    )

    html = render_labeler_html(context, LabelConfig())

    assert "Sweep" in html
    assert "Review" in html
    assert "waveform" in html
    assert "key === 'y'" in html
    assert "key === 'n'" in html
    assert "key === 's'" in html
    assert "META.reviewRadius" in html
