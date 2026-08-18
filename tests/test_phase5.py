import json
from pathlib import Path

import numpy as np
import pytest

from cutdetect.detect import sensitivity_config
from cutdetect.report import render_debug_report
from cutdetect.studio import render_studio_html


def test_sensitivity_moves_threshold_toward_high_recall() -> None:
    precise = sensitivity_config(0.0)
    recall = sensitivity_config(1.0)

    assert precise.tau_high > recall.tau_high
    assert recall.tau_low < precise.tau_low
    with pytest.raises(ValueError):
        sensitivity_config(1.01)


def test_studio_is_a_functional_local_drop_interface() -> None:
    html = render_studio_html(0.5)

    assert 'id="drop"' in html
    assert 'id="sensitivity"' in html
    assert "/api/process" in html
    assert "Download all clips" in html
    assert "No uploads leave this machine" in html


def test_debug_report_embeds_traces_and_diagnostics(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            {
                "video": {"frame_count": 3, "fps": 30.0},
                "cuts": [
                    {
                        "frame": 1,
                        "time_sec": 1 / 30,
                        "confidence": 0.9,
                        "agreement_count": 4,
                    }
                ],
                "diagnostics": {"face_detection_rate": 1.0},
            }
        )
    )
    normalized = tmp_path / "normalized.npz"
    np.savez(normalized, fused=np.asarray([0.9, 0.1]))
    thumbnails = tmp_path / "thumbs"
    thumbnails.mkdir()

    html = render_debug_report(predictions, normalized, thumbnails)

    assert "Signal timeline" in html
    assert "face_detection_rate" in html
    assert "0.9" in html
