from __future__ import annotations

import json
from pathlib import Path

import pytest

from cutdetect.cli import main
from cutdetect.evaluation import (
    GroundTruthLabel,
    Prediction,
    optimal_matches,
    score_predictions,
    shot_length_distribution,
    signal_ablation,
)


def test_optimal_matching_avoids_a_greedy_assignment_trap() -> None:
    predictions = [Prediction(frame=2, confidence=1.0), Prediction(frame=4, confidence=1.0)]
    labels = [GroundTruthLabel(frame=1, time=0.1), GroundTruthLabel(frame=3, time=0.3)]

    matches = optimal_matches(predictions, labels, tolerance_frames=1)

    assert len(matches) == 2
    assert {(match.prediction_index, match.label_index) for match in matches} == {(0, 0), (1, 1)}


@pytest.mark.parametrize("tolerance", [1, 2, 3])
def test_dummy_detector_reports_zero_recall(tolerance: int) -> None:
    labels = [GroundTruthLabel(frame=10, time=1.0)]

    metrics = score_predictions([], labels, tolerance)

    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.false_negatives == 1


def test_eval_cli_writes_json_and_svg_for_dummy_detector(tmp_path: Path) -> None:
    labels = tmp_path / "labels.json"
    predictions = tmp_path / "predictions.json"
    output = tmp_path / "evaluation"
    labels.write_text(
        json.dumps([{"frame": 10, "time": 1.0, "confidence": "certain", "type": "hard"}]),
        encoding="utf-8",
    )
    predictions.write_text('{"cuts": []}', encoding="utf-8")

    exit_code = main(
        [
            "eval",
            "--labels",
            str(labels),
            "--predictions",
            str(predictions),
            "--output-dir",
            str(output),
        ]
    )

    result = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert result["metrics_by_tolerance"]["2"]["recall"] == 0.0
    assert "<svg" in (output / "pr_curve.svg").read_text(encoding="utf-8")


def test_shot_length_distribution_includes_head_and_tail() -> None:
    labels = [GroundTruthLabel(frame=30, time=1.0), GroundTruthLabel(frame=90, time=3.0)]

    distribution = shot_length_distribution(labels, frame_count=150, fps=30.0)

    assert distribution["lengths_frames"] == [30, 60, 60]
    assert distribution["median_sec"] == 2.0


def test_signal_ablation_renormalizes_available_weights() -> None:
    labels = [GroundTruthLabel(frame=1, time=0.1)]
    scores = {"useful": [0.0, 1.0, 0.0], "noise": [0.0, 0.0, 0.0]}

    result = signal_ablation(
        scores,
        {"useful": 1.0, "noise": 1.0},
        labels,
        threshold=0.4,
        tolerance_frames=0,
    )

    assert result["all"] == 1.0
    assert result["without_useful"] == 0.0
