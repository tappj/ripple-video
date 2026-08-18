"""Phase 1 off-the-shelf detector baselines."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from cutdetect.config import BaselineConfig, EvaluationConfig, IngestConfig
from cutdetect.evaluation import GroundTruthLabel, Prediction, load_labels, score_predictions
from cutdetect.ingest import VideoContext, ingest_video

ScoreVector = Sequence[float] | npt.NDArray[np.float64]
EligibilityVector = Sequence[bool] | npt.NDArray[np.bool_]


class BaselineError(RuntimeError):
    """Raised when a baseline dependency or inference pass fails."""


@dataclass(frozen=True, slots=True)
class BaselineRun:
    """One detector configuration and its resulting cut frames/metrics."""

    detector: str
    params: dict[str, float | int | str]
    cut_frames: tuple[int, ...]
    metrics_by_tolerance: dict[str, dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return cast(dict[str, object], asdict(self))


def _linspace(start: float, stop: float, steps: int) -> npt.NDArray[np.float64]:
    if steps < 2:
        raise ValueError("threshold sweep requires at least two steps")
    return np.linspace(start, stop, steps, dtype=np.float64)


def _geomspace(start: float, stop: float, steps: int) -> npt.NDArray[np.float64]:
    if start <= 0.0 or stop <= 0.0 or steps < 2:
        raise ValueError("log threshold sweep requires positive bounds and at least two steps")
    return np.geomspace(start, stop, steps, dtype=np.float64)


def threshold_peaks(
    scores: ScoreVector,
    threshold: float,
    min_separation: int,
    *,
    eligibility: EligibilityVector | None = None,
    frame_offset: int = 0,
) -> tuple[int, ...]:
    """Threshold a trace, collapse plateaus to peaks, then apply temporal NMS."""
    if min_separation < 1:
        raise ValueError("min_separation must be positive")
    if eligibility is not None and len(eligibility) != len(scores):
        raise ValueError("eligibility and scores must have equal length")
    candidates = [
        index
        for index, score in enumerate(scores)
        if math.isfinite(score)
        and score >= threshold
        and (eligibility is None or eligibility[index])
    ]
    groups: list[list[int]] = []
    for candidate in candidates:
        if not groups or candidate > groups[-1][-1] + 1:
            groups.append([candidate])
        else:
            groups[-1].append(candidate)
    peaks = [max(group, key=lambda index: scores[index]) for group in groups]
    # Highest-score NMS avoids allowing a weaker caption-animation peak to
    # suppress a stronger cut that happens within the separation window.
    selected: list[int] = []
    for peak in sorted(peaks, key=lambda index: scores[index], reverse=True):
        shifted = peak + frame_offset
        if shifted < 0 or shifted >= len(scores):
            continue
        if all(abs(shifted - accepted) >= min_separation for accepted in selected):
            selected.append(shifted)
    return tuple(sorted(selected))


def _score_run(
    detector: str,
    params: dict[str, float | int | str],
    cuts: Sequence[int],
    scores: ScoreVector,
    labels: Sequence[GroundTruthLabel],
    evaluation: EvaluationConfig,
    *,
    score_offset: int = 0,
) -> BaselineRun:
    predictions = [
        Prediction(frame=frame, confidence=float(scores[frame - score_offset])) for frame in cuts
    ]
    metrics = {
        str(tolerance): score_predictions(predictions, labels, tolerance).to_dict()
        for tolerance in evaluation.tolerances_frames
    }
    return BaselineRun(
        detector=detector,
        params=params,
        cut_frames=tuple(cuts),
        metrics_by_tolerance=metrics,
    )


def extract_scenedetect_scores(
    context: VideoContext, config: BaselineConfig
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Decode once with PySceneDetect and cache content/adaptive frame metrics."""
    cache_path = context.artifact_dir / "baseline_scenedetect_scores.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            content = np.asarray(archive["content_val"], dtype=np.float64)
            adaptive = np.asarray(archive["adaptive_ratio"], dtype=np.float64)
        if len(content) == context.frame_count and len(adaptive) == context.frame_count:
            return content, adaptive

    try:
        scenedetect = importlib.import_module("scenedetect")
        detectors = importlib.import_module("scenedetect.detectors")
    except ModuleNotFoundError as error:
        raise BaselineError("install the 'baselines' extra to run PySceneDetect") from error
    stats = scenedetect.StatsManager()
    manager = scenedetect.SceneManager(stats_manager=stats)
    manager.add_detector(
        detectors.AdaptiveDetector(
            adaptive_threshold=float("inf"),
            min_scene_len=1,
            window_width=config.adaptive_window_width,
            min_content_val=0.0,
        )
    )
    video = scenedetect.open_video(str(context.working_video_path))
    processed = int(manager.detect_scenes(video=video))
    if processed != context.frame_count:
        raise BaselineError(
            f"PySceneDetect decoded {processed} frames; expected {context.frame_count}"
        )
    adaptive_key = f"adaptive_ratio (w={config.adaptive_window_width})"
    content = np.full(context.frame_count, np.nan, dtype=np.float64)
    adaptive = np.full(context.frame_count, np.nan, dtype=np.float64)
    for frame in range(context.frame_count):
        raw_content, raw_adaptive = stats.get_metrics(frame, ["content_val", adaptive_key])
        if raw_content is not None:
            content[frame] = float(raw_content)
        if raw_adaptive is not None:
            adaptive[frame] = float(raw_adaptive)
    np.savez_compressed(cache_path, content_val=content, adaptive_ratio=adaptive)
    return content, adaptive


def extract_transnet_scores(
    context: VideoContext, config: BaselineConfig
) -> npt.NDArray[np.float64]:
    """Run bundled TransNetV2 weights on CPU and cache raw frame probabilities."""
    cache_path = context.artifact_dir / "baseline_transnet_scores.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            probabilities = np.asarray(archive["single_frame_probability"], dtype=np.float64)
        if len(probabilities) == context.frame_count:
            return probabilities
    try:
        module = importlib.import_module("transnetv2_pytorch")
    except ModuleNotFoundError as error:
        raise BaselineError("install the 'baselines' extra to run TransNetV2") from error
    model = module.TransNetV2(device=config.transnet_device)
    _frames, single_frame, _all_frames = model.predict_video(
        str(context.working_video_path), quiet=True
    )
    probabilities = np.asarray(single_frame.cpu().detach().numpy(), dtype=np.float64).reshape(-1)
    if len(probabilities) != context.frame_count:
        raise BaselineError(
            f"TransNetV2 decoded {len(probabilities)} frames; expected {context.frame_count}"
        )
    np.savez_compressed(cache_path, single_frame_probability=probabilities)
    return probabilities


def sweep_content(
    scores: npt.NDArray[np.float64],
    labels: Sequence[GroundTruthLabel],
    config: BaselineConfig,
    evaluation: EvaluationConfig,
) -> list[BaselineRun]:
    """Sweep ContentDetector's raw HSV threshold."""
    runs: list[BaselineRun] = []
    for threshold in _linspace(
        config.content_threshold_min,
        config.content_threshold_max,
        config.content_threshold_steps,
    ):
        cuts = threshold_peaks(scores, float(threshold), config.min_shot_frames)
        runs.append(
            _score_run(
                "content",
                {"threshold": float(threshold), "min_scene_len": config.min_shot_frames},
                cuts,
                scores,
                labels,
                evaluation,
            )
        )
    return runs


def sweep_adaptive(
    content: npt.NDArray[np.float64],
    adaptive: npt.NDArray[np.float64],
    labels: Sequence[GroundTruthLabel],
    config: BaselineConfig,
    evaluation: EvaluationConfig,
) -> list[BaselineRun]:
    """Sweep AdaptiveDetector's ratio and minimum-content thresholds."""
    runs: list[BaselineRun] = []
    ratio_thresholds = _linspace(
        config.adaptive_threshold_min,
        config.adaptive_threshold_max,
        config.adaptive_threshold_steps,
    )
    content_thresholds = _linspace(
        config.adaptive_min_content_min,
        config.adaptive_min_content_max,
        config.adaptive_min_content_steps,
    )
    for min_content in content_thresholds:
        eligible = np.isfinite(content) & (content >= min_content)
        for threshold in ratio_thresholds:
            cuts = threshold_peaks(
                adaptive,
                float(threshold),
                config.min_shot_frames,
                eligibility=eligible,
            )
            runs.append(
                _score_run(
                    "adaptive",
                    {
                        "adaptive_threshold": float(threshold),
                        "min_content_val": float(min_content),
                        "window_width": config.adaptive_window_width,
                        "min_scene_len": config.min_shot_frames,
                    },
                    cuts,
                    adaptive,
                    labels,
                    evaluation,
                )
            )
    return runs


def sweep_transnet(
    probabilities: npt.NDArray[np.float64],
    labels: Sequence[GroundTruthLabel],
    config: BaselineConfig,
    evaluation: EvaluationConfig,
) -> list[BaselineRun]:
    """Sweep raw TransNet single-frame transition probabilities."""
    runs: list[BaselineRun] = []
    for threshold in _geomspace(
        config.transnet_threshold_min,
        config.transnet_threshold_max,
        config.transnet_threshold_steps,
    ):
        # TransNet scores transition frame i; cutdetect reports the first frame
        # after that boundary, i + 1.
        cuts = threshold_peaks(
            probabilities,
            float(threshold),
            config.min_shot_frames,
            frame_offset=1,
        )
        runs.append(
            _score_run(
                "transnet",
                {"threshold": float(threshold), "device": config.transnet_device},
                cuts,
                probabilities,
                labels,
                evaluation,
                score_offset=1,
            )
        )
    return runs


def _best_run(runs: Sequence[BaselineRun], tolerance: int) -> BaselineRun:
    return max(
        runs,
        key=lambda run: (
            _metric(run, tolerance, "f1"),
            _metric(run, tolerance, "recall"),
            _metric(run, tolerance, "precision"),
            -len(run.cut_frames),
        ),
    )


def _metric(run: BaselineRun, tolerance: int, name: str) -> float:
    value = run.metrics_by_tolerance[str(tolerance)][name]
    if not isinstance(value, int | float):
        raise TypeError(f"baseline metric {name!r} is not numeric")
    return float(value)


def _prediction_contract(
    detector: str,
    run: BaselineRun,
    scores: ScoreVector,
    context: VideoContext,
    *,
    score_offset: int = 0,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "tool_version": "0.1.0",
        "baseline": detector,
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
        "cuts": [
            {
                "frame": frame,
                "time_sec": context.source_time_for_frame(frame),
                "confidence": float(scores[frame - score_offset]),
            }
            for frame in run.cut_frames
        ],
        "params": run.params,
    }


def _pr_envelope(runs: Sequence[BaselineRun], tolerance: int) -> list[tuple[float, float]]:
    points = sorted(
        {
            (
                _metric(run, tolerance, "recall"),
                _metric(run, tolerance, "precision"),
            )
            for run in runs
        }
    )
    envelope: list[tuple[float, float]] = []
    best_precision = 0.0
    for recall, precision in reversed(points):
        best_precision = max(best_precision, precision)
        envelope.append((recall, best_precision))
    return list(reversed(envelope))


def render_baseline_pr_svg(
    runs_by_detector: Mapping[str, Sequence[BaselineRun]], tolerance: int
) -> str:
    """Render combined Phase 1 precision-recall envelopes as SVG."""
    width, height, margin = 760, 500, 58
    plot_width, plot_height = width - 2 * margin, height - 2 * margin
    colors = {"adaptive": "#38bdf8", "content": "#fb7185", "transnet": "#a3e635"}
    lines: list[str] = []
    for detector, runs in runs_by_detector.items():
        points = " ".join(
            f"{margin + recall * plot_width:.2f},{height - margin - precision * plot_height:.2f}"
            for recall, precision in _pr_envelope(runs, tolerance)
        )
        color = colors.get(detector, "#e2e8f0")
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    legend = " ".join(
        f'<text x="{margin + index * 180}" y="30" '
        f'fill="{colors.get(name, "#e2e8f0")}">{name}</text>'
        for index, name in enumerate(runs_by_detector)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#111827"/>\n'
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" '
        'y2="{height - margin}" stroke="#94a3b8"/>\n'
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" '
        'stroke="#94a3b8"/>\n'
        f"{''.join(lines)}\n{legend}\n"
        f'<text x="{width / 2}" y="{height - 14}" text-anchor="middle" '
        'fill="#e2e8f0">Recall</text>\n'
        f'<text x="18" y="{height / 2}" text-anchor="middle" '
        f'transform="rotate(-90 18 {height / 2})" fill="#e2e8f0">Precision</text>\n'
        f'<text x="{margin}" y="52" fill="#f8fafc">P/R at +/-{tolerance} frames</text>\n'
        "</svg>\n"
    )


def run_baselines(
    video_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    config: BaselineConfig | None = None,
    evaluation: EvaluationConfig | None = None,
) -> dict[str, object]:
    """Extract, sweep, score, and persist all Phase 1 baselines."""
    settings = config or BaselineConfig()
    eval_settings = evaluation or EvaluationConfig()
    context = ingest_video(
        video_path,
        IngestConfig(cache_dir=Path(cache_dir) if cache_dir is not None else None),
    )
    labels = load_labels(labels_path, include_unsure=eval_settings.include_unsure_labels)
    content, adaptive = extract_scenedetect_scores(context, settings)
    runs_by_detector: dict[str, list[BaselineRun]] = {
        "adaptive": sweep_adaptive(content, adaptive, labels, settings, eval_settings),
        "content": sweep_content(content, labels, settings, eval_settings),
    }
    score_by_detector: dict[str, tuple[ScoreVector, int]] = {
        "adaptive": (adaptive, 0),
        "content": (content, 0),
    }
    if settings.include_transnet:
        transnet = extract_transnet_scores(context, settings)
        runs_by_detector["transnet"] = sweep_transnet(transnet, labels, settings, eval_settings)
        score_by_detector["transnet"] = (transnet, 1)

    best_by_tolerance: dict[str, dict[str, object]] = {}
    for tolerance in eval_settings.tolerances_frames:
        best_by_tolerance[str(tolerance)] = {
            detector: _best_run(runs, tolerance).to_dict()
            for detector, runs in runs_by_detector.items()
        }

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    result = {
        "video": context.to_dict(include_timestamps=False),
        "labels": str(Path(labels_path).expanduser().resolve()),
        "config": asdict(settings),
        "best_by_tolerance": best_by_tolerance,
        "sweeps": {
            detector: [run.to_dict() for run in runs] for detector, runs in runs_by_detector.items()
        },
    }
    (destination / "baseline_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "pr_curves.svg").write_text(
        render_baseline_pr_svg(runs_by_detector, eval_settings.primary_tolerance_frames),
        encoding="utf-8",
    )
    primary = str(eval_settings.primary_tolerance_frames)
    for detector, runs in runs_by_detector.items():
        best = _best_run(runs, eval_settings.primary_tolerance_frames)
        scores, offset = score_by_detector[detector]
        contract = _prediction_contract(detector, best, scores, context, score_offset=offset)
        (destination / f"predictions_{detector}.json").write_text(
            json.dumps(contract, indent=2) + "\n", encoding="utf-8"
        )
    return {
        "output_dir": str(destination),
        "primary_tolerance_frames": eval_settings.primary_tolerance_frames,
        "best_at_primary_tolerance": best_by_tolerance[primary],
    }
