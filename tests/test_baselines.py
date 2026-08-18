from pathlib import Path

from cutdetect.baselines import BaselineRun, render_baseline_pr_svg, threshold_peaks


def _run(precision: float, recall: float) -> BaselineRun:
    return BaselineRun(
        detector="content",
        params={"threshold": 1.0},
        cut_frames=(),
        metrics_by_tolerance={
            "2": {
                "precision": precision,
                "recall": recall,
                "f1": 0.0,
            }
        },
    )


def test_threshold_peaks_collapses_contiguous_scores_to_strongest_frame() -> None:
    scores = [0.0, 0.6, 0.9, 0.7, 0.0, 0.8]

    assert threshold_peaks(scores, 0.5, 1) == (2, 5)


def test_threshold_peaks_applies_high_score_nms_and_offset() -> None:
    scores = [0.0, 0.8, 0.0, 0.9, 0.0, 0.0]

    assert threshold_peaks(scores, 0.5, 4, frame_offset=1) == (4,)


def test_threshold_peaks_honors_eligibility_and_drops_shift_past_end() -> None:
    scores = [0.0, 0.8, 0.0, 0.9]
    eligibility = [True, False, True, True]

    assert threshold_peaks(scores, 0.5, 1, eligibility=eligibility, frame_offset=1) == ()


def test_render_baseline_pr_svg_contains_curve_and_tolerance(tmp_path: Path) -> None:
    svg = render_baseline_pr_svg({"content": [_run(0.8, 0.5), _run(0.7, 0.8)]}, 2)

    output = tmp_path / "curve.svg"
    output.write_text(svg, encoding="utf-8")
    assert "<polyline" in output.read_text(encoding="utf-8")
    assert "P/R at +/-2 frames" in svg
