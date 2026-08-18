"""Phase 3 signal computation, tuning, detection, and checkpoint artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from html import escape
from itertools import pairwise
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from cutdetect.config import (
    DEFAULT_SIGNAL_WEIGHTS,
    EvaluationConfig,
    FeatureConfig,
    FusionConfig,
    IngestConfig,
    SignalConfig,
    TuningConfig,
)
from cutdetect.evaluation import GroundTruthLabel, Prediction, load_labels, score_predictions
from cutdetect.features import extract_features, load_feature_metadata
from cutdetect.fusion import (
    FusionResult,
    fuse_normalized,
    normalize_and_fuse,
    pick_boundaries,
    squash_zscores,
)
from cutdetect.ingest import VideoContext, ingest_video
from cutdetect.signals import AUDIO_SIGNALS, SignalBundle, compute_signal_bundle


@dataclass(frozen=True, slots=True)
class TunedFusion:
    """Best same-clip Phase 3 search result."""

    config: FusionConfig
    weights: dict[str, float]
    metrics: dict[str, object]
    evaluated_candidates: int


@dataclass(frozen=True, slots=True)
class DetectionRun:
    """Paths and primary metrics emitted by a Phase 3 run."""

    predictions_path: Path
    signals_path: Path
    normalized_signals_path: Path
    trace_path: Path
    ablation_path: Path
    tuning_path: Path | None
    evaluation_path: Path | None
    cut_count: int
    metrics_by_tolerance: dict[str, dict[str, object]] | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible CLI summary."""
        return {
            "predictions": str(self.predictions_path),
            "signals": str(self.signals_path),
            "normalized_signals": str(self.normalized_signals_path),
            "signal_traces": str(self.trace_path),
            "ablation": str(self.ablation_path),
            "tuning": str(self.tuning_path) if self.tuning_path else None,
            "evaluation": str(self.evaluation_path) if self.evaluation_path else None,
            "cut_count": self.cut_count,
            "metrics_by_tolerance": self.metrics_by_tolerance,
        }


def sensitivity_config(sensitivity: float, config: FusionConfig | None = None) -> FusionConfig:
    """Map the public 0-1 sensitivity knob onto the tuned threshold range."""
    if not 0.0 <= sensitivity <= 1.0:
        raise ValueError("sensitivity must be between 0 and 1")
    base = config or FusionConfig()
    threshold = 0.827_692_307_692_307_7 - 0.36 * sensitivity
    return replace(base, tau_high=threshold, tau_low=threshold * 0.85)


def _predictions(boundaries: Sequence[int], fused: npt.NDArray[np.float64]) -> list[Prediction]:
    return [
        Prediction(frame=boundary + 1, confidence=float(fused[boundary])) for boundary in boundaries
    ]


def _metric_key(
    predictions: Sequence[Prediction], labels: Sequence[GroundTruthLabel], tolerance: int
) -> tuple[float, float, float, int]:
    metrics = score_predictions(predictions, labels, tolerance)
    return metrics.f1, metrics.recall, metrics.precision, -len(predictions)


def tune_fusion(
    raw_signals: Mapping[str, npt.NDArray[np.float64]],
    labels: Sequence[GroundTruthLabel],
    fps: float,
    *,
    initial_config: FusionConfig | None = None,
    initial_weights: Mapping[str, float] | None = None,
    tuning: TuningConfig | None = None,
    tolerance_frames: int = 2,
) -> TunedFusion:
    """Run deterministic random weight search with a threshold/low-ratio grid."""
    base = initial_config or FusionConfig()
    weights_base = dict(initial_weights or DEFAULT_SIGNAL_WEIGHTS)
    settings = tuning or TuningConfig()
    if not labels:
        raise ValueError("tuning requires at least one ground-truth label")
    rng = np.random.default_rng(settings.random_seed)
    normalized_by_window: dict[float, dict[str, npt.NDArray[np.float64]]] = {}
    for window in settings.window_seconds:
        candidate = replace(base, local_window_sec=window)
        normalized_by_window[window] = normalize_and_fuse(
            raw_signals,
            weights_base,
            fps,
            candidate,
            audio_signals=AUDIO_SIGNALS,
        ).normalized
    thresholds = np.linspace(settings.tau_high_min, settings.tau_high_max, settings.threshold_steps)
    low_ratios = np.linspace(
        settings.tau_low_ratio_min,
        settings.tau_low_ratio_max,
        settings.tau_low_ratio_steps,
    )
    best_key = (-1.0, -1.0, -1.0, -(10**9), -1, -1, -math.inf)
    best_config = base
    best_weights = weights_base
    best_metrics: dict[str, object] = {}
    evaluated = 0
    weight_names = tuple(weights_base)
    for trial in range(settings.trials):
        if trial == 0:
            trial_weights = weights_base.copy()
            window = base.local_window_sec
            agreement_count = base.agreement_count
            min_shot_frames = base.min_shot_frames
        else:
            trial_weights = {
                name: float(value * math.exp(float(rng.normal(0.0, settings.weight_log_sigma))))
                for name, value in weights_base.items()
            }
            window = float(rng.choice(settings.window_seconds))
            agreement_count = int(rng.choice(settings.agreement_counts))
            min_shot_frames = int(rng.choice(settings.min_shot_frames))
        normalized = normalized_by_window[window]
        fused, agreement, _available = fuse_normalized(normalized, trial_weights)
        weight_distance = sum(
            abs(math.log(trial_weights[name] / weights_base[name])) for name in weight_names
        )
        for threshold in thresholds:
            for low_ratio in low_ratios:
                candidate = replace(
                    base,
                    local_window_sec=window,
                    tau_high=float(threshold),
                    tau_low=float(threshold * low_ratio),
                    agreement_count=agreement_count,
                    min_shot_frames=min_shot_frames,
                )
                boundaries = pick_boundaries(fused, agreement, candidate)
                predictions = _predictions(boundaries, fused)
                metrics = score_predictions(predictions, labels, tolerance_frames)
                key = (
                    metrics.f1,
                    metrics.recall,
                    metrics.precision,
                    -len(predictions),
                    candidate.agreement_count,
                    candidate.min_shot_frames,
                    -weight_distance,
                )
                evaluated += 1
                if key > best_key:
                    best_key = key
                    best_config = candidate
                    best_weights = trial_weights
                    best_metrics = metrics.to_dict()
    return TunedFusion(best_config, best_weights, best_metrics, evaluated)


def signal_ablation(
    normalized: Mapping[str, npt.NDArray[np.float64]],
    weights: Mapping[str, float],
    config: FusionConfig,
    labels: Sequence[GroundTruthLabel],
    tolerance_frames: int,
) -> dict[str, dict[str, float | int]]:
    """Score full fusion and each leave-one-signal-out configuration."""

    def outcome(excluded: str | None) -> dict[str, float | int]:
        selected = {name: values for name, values in normalized.items() if name != excluded}
        fused, agreement, _available = fuse_normalized(selected, weights)
        boundaries = pick_boundaries(fused, agreement, config)
        metrics = score_predictions(_predictions(boundaries, fused), labels, tolerance_frames)
        return {
            "f1": metrics.f1,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "predictions": len(boundaries),
        }

    results = {"all": outcome(None)}
    results.update({f"without_{name}": outcome(name) for name in sorted(normalized)})
    full_f1 = float(results["all"]["f1"])
    for name in sorted(normalized):
        results[f"without_{name}"]["f1_drop"] = full_f1 - float(results[f"without_{name}"]["f1"])
    return results


def render_signal_traces_svg(
    normalized: Mapping[str, npt.NDArray[np.float64]],
    fused: npt.NDArray[np.float64],
    labels: Sequence[GroundTruthLabel],
) -> str:
    """Render every normalized trace in aligned stacked rows with ground truth."""
    traces = {"FUSED": fused, **dict(sorted(normalized.items()))}
    width, left, right, row_height, top = 1200, 190, 24, 88, 48
    height = top + len(traces) * row_height + 34
    boundary_count = len(fused)
    plot_width = width - left - right
    label_boundaries = [label.frame - 1 for label in labels]
    rows: list[str] = []
    for row, (name, values) in enumerate(traces.items()):
        y_top = top + row * row_height
        y_bottom = y_top + row_height - 18
        point_items: list[str] = []
        for index, value in enumerate(values):
            numeric = float(value)
            amplitude = numeric if math.isfinite(numeric) else 0.0
            x_position = left + index * plot_width / max(1, boundary_count - 1)
            y_position = y_bottom - amplitude * (row_height - 24)
            point_items.append(f"{x_position:.2f},{y_position:.2f}")
        points = " ".join(point_items)
        color = "#fbbf24" if name == "FUSED" else "#38bdf8"
        markers = "".join(
            f'<line x1="{left + boundary * plot_width / max(1, boundary_count - 1):.2f}" '
            f'y1="{y_top}" x2="{left + boundary * plot_width / max(1, boundary_count - 1):.2f}" '
            f'y2="{y_bottom}" stroke="#fb7185" stroke-width="1" opacity="0.7"/>'
            for boundary in label_boundaries
        )
        rows.append(
            f'<text x="10" y="{y_top + 24}" fill="#e2e8f0">{escape(name)}</text>'
            f'<line x1="{left}" y1="{y_bottom}" x2="{width - right}" y2="{y_bottom}" '
            f'stroke="#334155"/>{markers}<polyline points="{points}" fill="none" '
            f'stroke="{color}" stroke-width="1.4"/>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#0f172a"/>'
        '<text x="10" y="28" fill="#f8fafc" font-size="18">Normalized boundary signals '
        "(pink = ground truth)</text>"
        f"{''.join(rows)}</svg>\n"
    )


def _discover_features(context: VideoContext) -> Path | None:
    candidates = sorted(
        context.artifact_dir.glob("features-v*.npz"), key=lambda path: path.stat().st_mtime
    )
    for candidate in reversed(candidates):
        try:
            metadata = load_feature_metadata(candidate)
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
        if (
            metadata.get("video_content_sha256") == context.cache_key
            and metadata.get("frame_count") == context.frame_count
        ):
            return candidate
    return None


def _load_words(path: Path | None) -> list[Mapping[str, object]] | None:
    if path is None:
        return None
    root: object = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(root, dict):
        root = root.get("words", [])
    if not isinstance(root, list) or not all(isinstance(item, dict) for item in root):
        raise ValueError("word timestamps must be a JSON array or contain a words array")
    return cast(list[Mapping[str, object]], root)


def _audio_offset(boundary: int, result: FusionResult, config: FusionConfig) -> int | None:
    available = [name for name in AUDIO_SIGNALS if name in result.zscores]
    if not available:
        return None
    start = max(0, boundary - config.audio_association_radius_frames)
    stop = min(len(result.fused), boundary + config.audio_association_radius_frames + 1)
    score = np.zeros(stop - start, dtype=np.float64)
    evidence = np.zeros(stop - start, dtype=np.bool_)
    for name in available:
        values = squash_zscores(result.zscores[name][start:stop], config.z_max)
        finite = np.isfinite(values)
        score[finite] += values[finite]
        evidence |= finite
    if not evidence.any():
        return None
    score[~evidence] = -1.0
    return int(start + np.argmax(score) - boundary)


def _segments(context: VideoContext, cut_frames: Sequence[int]) -> list[dict[str, float | int]]:
    boundaries = [0, *cut_frames, context.frame_count]
    result: list[dict[str, float | int]] = []
    for index, (start, end) in enumerate(pairwise(boundaries)):
        start_sec = (
            context.source_time_for_frame(start)
            if start < context.frame_count
            else context.duration_sec
        )
        end_sec = (
            context.source_time_for_frame(end)
            if end < context.frame_count
            else context.duration_sec
        )
        result.append(
            {
                "index": index,
                "start_frame": start,
                "end_frame": end,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": end_sec - start_sec,
            }
        )
    return result


def _prediction_contract(
    context: VideoContext,
    bundle: SignalBundle,
    fusion: FusionResult,
    config: FusionConfig,
    signal_config: SignalConfig,
    weights: Mapping[str, float],
    sensitivity: float,
) -> dict[str, object]:
    cut_frames = [boundary + 1 for boundary in fusion.cut_boundaries]
    cuts: list[dict[str, object]] = []
    for boundary, frame in zip(fusion.cut_boundaries, cut_frames, strict=True):
        scale_score = fusion.normalized.get("scale_delta", np.zeros(len(fusion.fused)))[boundary]
        raw_scale = bundle.raw["scale_delta"][boundary]
        cut_type = (
            "zoom_disguised"
            if math.isfinite(scale_score)
            and scale_score > 0.5
            and raw_scale >= signal_config.zoom_scale_log_threshold
            else "hard"
        )
        cuts.append(
            {
                "frame": frame,
                "time_sec": context.source_time_for_frame(frame),
                "confidence": float(fusion.fused[boundary]),
                "cut_type": cut_type,
                "span": None,
                "audio_offset_frames": _audio_offset(boundary, fusion, config),
                "agreement_count": int(fusion.agreement[boundary]),
                "signals": {
                    name: float(values[boundary])
                    for name, values in fusion.zscores.items()
                    if math.isfinite(float(values[boundary]))
                },
            }
        )
    availability = {
        name: float(np.mean(np.isfinite(values))) for name, values in bundle.raw.items()
    }
    return {
        "schema_version": "1.0",
        "tool_version": "0.1.0",
        "video": {
            "path": str(context.source_path),
            "duration_sec": context.duration_sec,
            "fps": float(context.fps),
            "frame_count": context.frame_count,
            "width": context.width,
            "height": context.height,
            "was_vfr": context.was_vfr,
            "has_audio": context.has_audio,
        },
        "cuts": cuts,
        "segments": _segments(context, cut_frames),
        "diagnostics": {
            "face_detection_rate": bundle.face_detection_rate,
            "signal_availability": availability,
            "signals_available": [name for name, rate in availability.items() if rate > 0.0],
            "signals_disabled": sorted(bundle.disabled_reasons),
            "disable_reasons": bundle.disabled_reasons,
            "captions_region_masked": bundle.captions_region_masked,
        },
        "params": {
            "sensitivity": sensitivity,
            "fusion": asdict(config),
            "weights": dict(weights),
        },
    }


def run_detection(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    features_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    words_path: str | Path | None = None,
    model_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    signal_config: SignalConfig | None = None,
    fusion_config: FusionConfig | None = None,
    tuning_config: TuningConfig | None = None,
    sensitivity: float = 0.5,
) -> DetectionRun:
    """Run Phase 3 end-to-end and write contract, traces, tuning, and ablation."""
    eval_config = EvaluationConfig()
    context = ingest_video(
        video_path,
        IngestConfig(cache_dir=Path(cache_dir) if cache_dir is not None else None),
    )
    if features_path is None:
        feature_file = _discover_features(context)
        if feature_file is None:
            feature_file = extract_features(
                video_path,
                cache_dir=cache_dir,
                model_path=model_path,
                config=FeatureConfig(),
            ).feature_path
    else:
        feature_file = Path(features_path).expanduser().resolve()
    if not feature_file.is_file():
        raise FileNotFoundError(feature_file)
    words = _load_words(Path(words_path) if words_path is not None else None)
    signal_settings = signal_config or SignalConfig()
    bundle = compute_signal_bundle(
        feature_file,
        context.working_video_path,
        config=signal_settings,
        words=words,
    )
    weights = dict(DEFAULT_SIGNAL_WEIGHTS)
    settings = sensitivity_config(sensitivity, fusion_config)
    labels: list[GroundTruthLabel] = []
    tuned: TunedFusion | None = None
    if labels_path is not None:
        labels = load_labels(labels_path)
        tuned = tune_fusion(
            bundle.raw,
            labels,
            float(context.fps),
            initial_config=settings,
            initial_weights=weights,
            tuning=tuning_config,
            tolerance_frames=eval_config.primary_tolerance_frames,
        )
        settings = tuned.config
        weights = tuned.weights
    fusion = normalize_and_fuse(
        bundle.raw,
        weights,
        float(context.fps),
        settings,
        audio_signals=AUDIO_SIGNALS,
    )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    predictions_path = destination / "predictions.json"
    signals_path = destination / "signals_raw.npz"
    normalized_path = destination / "signals_normalized.npz"
    trace_path = destination / "signal_traces.svg"
    ablation_path = destination / "ablation.json"
    tuning_path = destination / "tuning.json" if tuned else None
    evaluation_path = destination / "evaluation.json" if labels else None
    predictions_path.write_text(
        json.dumps(
            _prediction_contract(
                context,
                bundle,
                fusion,
                settings,
                signal_settings,
                weights,
                sensitivity,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(signals_path, **bundle.raw, **bundle.auxiliary)
    np.savez_compressed(
        normalized_path,
        **fusion.normalized,
        fused=fusion.fused,
        agreement=fusion.agreement,
        available_weight=fusion.available_weight,
    )
    trace_path.write_text(render_signal_traces_svg(fusion.normalized, fusion.fused, labels))
    metrics_by_tolerance: dict[str, dict[str, object]] | None = None
    if labels:
        predictions = _predictions(fusion.cut_boundaries, fusion.fused)
        metrics_by_tolerance = {
            str(tolerance): score_predictions(predictions, labels, tolerance).to_dict()
            for tolerance in eval_config.tolerances_frames
        }
        ablation = signal_ablation(
            fusion.normalized,
            weights,
            settings,
            labels,
            eval_config.primary_tolerance_frames,
        )
        ablation_path.write_text(json.dumps(ablation, indent=2) + "\n", encoding="utf-8")
        assert evaluation_path is not None
        evaluation_path.write_text(
            json.dumps({"metrics_by_tolerance": metrics_by_tolerance}, indent=2) + "\n",
            encoding="utf-8",
        )
        assert tuned is not None and tuning_path is not None
        tuning_path.write_text(
            json.dumps(
                {
                    "warning": "parameters tuned and evaluated on the same reference clip",
                    "evaluated_candidates": tuned.evaluated_candidates,
                    "best_metrics_at_2_frames": tuned.metrics,
                    "fusion": asdict(tuned.config),
                    "weights": tuned.weights,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        ablation_path.write_text(
            json.dumps({"status": "labels not supplied"}, indent=2) + "\n", encoding="utf-8"
        )
    return DetectionRun(
        predictions_path,
        signals_path,
        normalized_path,
        trace_path,
        ablation_path,
        tuning_path,
        evaluation_path,
        len(fusion.cut_boundaries),
        metrics_by_tolerance,
    )


def detect_cuts(
    video_path: str | Path,
    sensitivity: float = 0.5,
    *,
    output_dir: str | Path | None = None,
    features_path: str | Path | None = None,
    model_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, object]:
    """Detect cuts through the public API and return the stable JSON contract."""
    context = ingest_video(
        video_path,
        IngestConfig(cache_dir=Path(cache_dir) if cache_dir is not None else None),
    )
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else context.artifact_dir / "detections" / f"sensitivity-{round(sensitivity * 1000):04d}"
    )
    run = run_detection(
        video_path,
        destination,
        features_path=features_path,
        model_path=model_path,
        cache_dir=cache_dir,
        sensitivity=sensitivity,
    )
    root: object = json.loads(run.predictions_path.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        raise TypeError("prediction contract must be a JSON object")
    return cast(dict[str, object], root)
