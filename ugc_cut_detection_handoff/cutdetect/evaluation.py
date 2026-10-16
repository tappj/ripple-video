# Ruff measures embedded SVG source lines as Python; preserving each SVG element
# on one line keeps the generated artifact readable.
# ruff: noqa: E501, RUF001
"""Ground-truth matching and detector evaluation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast

from cutdetect.config import EvaluationConfig

LabelConfidence = Literal["certain", "unsure"]


@dataclass(frozen=True, slots=True)
class GroundTruthLabel:
    """A human-labeled cut boundary."""

    frame: int
    time: float
    confidence: LabelConfidence = "certain"
    type: str = "hard"


@dataclass(frozen=True, slots=True)
class Prediction:
    """A predicted cut boundary and score."""

    frame: int
    confidence: float


@dataclass(frozen=True, slots=True)
class Match:
    """One prediction-to-label assignment."""

    prediction_index: int
    label_index: int
    distance_frames: int


@dataclass(frozen=True, slots=True)
class Metrics:
    """Precision/recall result at one frame tolerance."""

    tolerance_frames: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    matches: tuple[Match, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        value = asdict(self)
        value["matches"] = [asdict(match) for match in self.matches]
        return cast(dict[str, object], value)


def _load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_labels(path: str | Path, *, include_unsure: bool = False) -> list[GroundTruthLabel]:
    """Load a labels array or an object containing a ``labels`` array."""
    root = _load_json(path)
    raw_labels = root.get("labels", []) if isinstance(root, dict) else root
    if not isinstance(raw_labels, list):
        raise ValueError("labels JSON must be an array or contain a labels array")
    labels: list[GroundTruthLabel] = []
    for item in raw_labels:
        if not isinstance(item, dict):
            raise ValueError("each label must be an object")
        confidence = str(item.get("confidence", "certain"))
        if confidence not in {"certain", "unsure"}:
            raise ValueError(f"invalid label confidence: {confidence}")
        if confidence == "unsure" and not include_unsure:
            continue
        raw_time = item.get("time")
        if raw_time is None:
            raw_time = item.get("time_sec", 0.0)
        labels.append(
            GroundTruthLabel(
                frame=int(item["frame"]),
                time=float(cast(float | int | str, raw_time)),
                confidence=cast(LabelConfidence, confidence),
                type=str(item.get("type", "hard")),
            )
        )
    return sorted(labels, key=lambda label: label.frame)


def load_predictions(path: str | Path) -> list[Prediction]:
    """Load a prediction array or the stable output contract's ``cuts`` array."""
    root = _load_json(path)
    raw_predictions = root.get("cuts", []) if isinstance(root, dict) else root
    if not isinstance(raw_predictions, list):
        raise ValueError("predictions JSON must be an array or contain a cuts array")
    predictions: list[Prediction] = []
    for item in raw_predictions:
        if not isinstance(item, dict):
            raise ValueError("each prediction must be an object")
        predictions.append(
            Prediction(
                frame=int(item["frame"]),
                confidence=float(cast(float | int | str, item.get("confidence", 1.0))),
            )
        )
    return sorted(predictions, key=lambda prediction: prediction.frame)


def optimal_matches(
    predictions: Sequence[Prediction],
    labels: Sequence[GroundTruthLabel],
    tolerance_frames: int,
) -> tuple[Match, ...]:
    """Find a maximum-cardinality, minimum-distance bipartite matching.

    Sorted one-dimensional boundaries admit an order-preserving optimum. The
    dynamic program therefore has the same optimum as a general assignment
    solver while avoiding a heavyweight runtime dependency.
    """
    if tolerance_frames < 0:
        raise ValueError("tolerance_frames must be non-negative")
    ordered_predictions = sorted(enumerate(predictions), key=lambda item: item[1].frame)
    ordered_labels = sorted(enumerate(labels), key=lambda item: item[1].frame)
    # State is (match count, negative distance), so ordinary tuple comparison
    # maximizes cardinality first and minimizes total error second.
    states = [
        [(0, 0) for _ in range(len(ordered_labels) + 1)]
        for _ in range(len(ordered_predictions) + 1)
    ]
    actions = [
        ["" for _ in range(len(ordered_labels) + 1)] for _ in range(len(ordered_predictions) + 1)
    ]
    for prediction_pos in range(1, len(ordered_predictions) + 1):
        actions[prediction_pos][0] = "prediction"
    for label_pos in range(1, len(ordered_labels) + 1):
        actions[0][label_pos] = "label"

    for prediction_pos in range(1, len(ordered_predictions) + 1):
        for label_pos in range(1, len(ordered_labels) + 1):
            candidates = [
                (states[prediction_pos - 1][label_pos], "prediction"),
                (states[prediction_pos][label_pos - 1], "label"),
            ]
            distance = abs(
                ordered_predictions[prediction_pos - 1][1].frame
                - ordered_labels[label_pos - 1][1].frame
            )
            if distance <= tolerance_frames:
                previous = states[prediction_pos - 1][label_pos - 1]
                candidates.append(((previous[0] + 1, previous[1] - distance), "match"))
            state, action = max(
                candidates, key=lambda candidate: (candidate[0], candidate[1] == "match")
            )
            states[prediction_pos][label_pos] = state
            actions[prediction_pos][label_pos] = action

    matches: list[Match] = []
    prediction_pos = len(ordered_predictions)
    label_pos = len(ordered_labels)
    while prediction_pos > 0 or label_pos > 0:
        action = actions[prediction_pos][label_pos]
        if action == "match":
            prediction_index, prediction = ordered_predictions[prediction_pos - 1]
            label_index, label = ordered_labels[label_pos - 1]
            matches.append(
                Match(
                    prediction_index=prediction_index,
                    label_index=label_index,
                    distance_frames=abs(prediction.frame - label.frame),
                )
            )
            prediction_pos -= 1
            label_pos -= 1
        elif action == "prediction":
            prediction_pos -= 1
        else:
            label_pos -= 1
    return tuple(reversed(matches))


def score_predictions(
    predictions: Sequence[Prediction],
    labels: Sequence[GroundTruthLabel],
    tolerance_frames: int,
) -> Metrics:
    """Compute precision, recall, and F1 using optimal matching."""
    matches = optimal_matches(predictions, labels, tolerance_frames)
    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(labels) - true_positives
    precision = true_positives / len(predictions) if predictions else 0.0
    recall = true_positives / len(labels) if labels else 1.0
    denominator = precision + recall
    f1 = 2.0 * precision * recall / denominator if denominator else 0.0
    return Metrics(
        tolerance_frames=tolerance_frames,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        matches=matches,
    )


def precision_recall_curve(
    predictions: Sequence[Prediction],
    labels: Sequence[GroundTruthLabel],
    tolerance_frames: int,
    threshold_steps: int,
) -> list[dict[str, float | int]]:
    """Sweep confidence from zero to one and return P/R points."""
    if threshold_steps < 2:
        raise ValueError("threshold_steps must be at least 2")
    curve: list[dict[str, float | int]] = []
    for index in range(threshold_steps):
        threshold = index / (threshold_steps - 1)
        selected = [prediction for prediction in predictions if prediction.confidence >= threshold]
        metrics = score_predictions(selected, labels, tolerance_frames)
        curve.append(
            {
                "threshold": threshold,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "predictions": len(selected),
            }
        )
    return curve


def shot_length_distribution(
    labels: Sequence[GroundTruthLabel], frame_count: int, fps: float
) -> dict[str, float | int | list[int]]:
    """Summarize segment lengths induced by labeled boundaries."""
    if frame_count < 0 or fps <= 0.0:
        raise ValueError("frame_count and fps must be positive")
    boundaries = [0, *(label.frame for label in labels), frame_count]
    lengths = [right - left for left, right in pairwise(boundaries)]
    ordered = sorted(lengths)

    def percentile(fraction: float) -> float:
        if not ordered:
            return 0.0
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "cut_count": len(labels),
        "shot_count": len(lengths),
        "lengths_frames": lengths,
        "min_frames": min(lengths, default=0),
        "p25_frames": percentile(0.25),
        "median_frames": percentile(0.5),
        "p75_frames": percentile(0.75),
        "max_frames": max(lengths, default=0),
        "min_sec": min(lengths, default=0) / fps,
        "median_sec": percentile(0.5) / fps,
        "max_sec": max(lengths, default=0) / fps,
    }


def signal_ablation(
    signal_scores: Mapping[str, Sequence[float]],
    weights: Mapping[str, float],
    labels: Sequence[GroundTruthLabel],
    *,
    threshold: float,
    tolerance_frames: int,
) -> dict[str, float]:
    """Return F1 for full fusion and for fusion with each signal removed."""
    lengths = {len(values) for values in signal_scores.values()}
    if len(lengths) > 1:
        raise ValueError("all signal score arrays must have equal length")
    boundary_count = next(iter(lengths), 0)

    def fused_predictions(excluded: str | None) -> list[Prediction]:
        result: list[Prediction] = []
        for boundary in range(boundary_count):
            numerator = 0.0
            denominator = 0.0
            for name, values in signal_scores.items():
                if name == excluded:
                    continue
                value = values[boundary]
                weight = weights.get(name, 0.0)
                if not math.isnan(value) and weight > 0.0:
                    numerator += weight * value
                    denominator += weight
            confidence = numerator / denominator if denominator else 0.0
            if confidence >= threshold:
                result.append(Prediction(frame=boundary, confidence=confidence))
        return result

    outcomes = {"all": score_predictions(fused_predictions(None), labels, tolerance_frames).f1}
    for signal in sorted(signal_scores):
        outcomes[f"without_{signal}"] = score_predictions(
            fused_predictions(signal), labels, tolerance_frames
        ).f1
    return outcomes


def render_pr_curve_svg(curve: Sequence[Mapping[str, float | int]]) -> str:
    """Render a dependency-free SVG precision-recall curve."""
    width, height, margin = 720, 480, 54
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    points = " ".join(
        f"{margin + float(item['recall']) * plot_width:.2f},"
        f"{height - margin - float(item['precision']) * plot_height:.2f}"
        for item in curve
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#111827"/>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#94a3b8"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#94a3b8"/>
<polyline points="{points}" fill="none" stroke="#38bdf8" stroke-width="3"/>
<text x="{width / 2}" y="{height - 12}" text-anchor="middle" fill="#e2e8f0">Recall</text>
<text x="16" y="{height / 2}" text-anchor="middle" transform="rotate(-90 16 {height / 2})" fill="#e2e8f0">Precision</text>
<text x="{margin}" y="{margin - 18}" fill="#f8fafc" font-size="18">Precision–recall curve</text>
</svg>"""


def evaluate_files(
    labels_path: str | Path,
    predictions_path: str | Path,
    config: EvaluationConfig | None = None,
) -> dict[str, object]:
    """Evaluate prediction JSON against label JSON at all configured tolerances."""
    settings = config or EvaluationConfig()
    labels = load_labels(labels_path, include_unsure=settings.include_unsure_labels)
    predictions = load_predictions(predictions_path)
    metrics = {
        str(tolerance): score_predictions(predictions, labels, tolerance).to_dict()
        for tolerance in settings.tolerances_frames
    }
    curve = precision_recall_curve(
        predictions,
        labels,
        settings.primary_tolerance_frames,
        settings.threshold_steps,
    )
    return {
        "label_count": len(labels),
        "prediction_count": len(predictions),
        "metrics_by_tolerance": metrics,
        "pr_curve_tolerance_frames": settings.primary_tolerance_frames,
        "pr_curve": curve,
    }
