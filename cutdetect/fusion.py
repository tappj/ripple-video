"""Local robust normalization, multimodal fusion, and boundary peak selection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cutdetect.config import FusionConfig


@dataclass(frozen=True, slots=True)
class FusionResult:
    """Normalized signals and their fused boundary trace."""

    zscores: dict[str, npt.NDArray[np.float64]]
    normalized: dict[str, npt.NDArray[np.float64]]
    fused: npt.NDArray[np.float64]
    agreement: npt.NDArray[np.int16]
    available_weight: npt.NDArray[np.float64]
    cut_boundaries: tuple[int, ...]


def local_robust_zscore(
    values: npt.ArrayLike,
    window_radius: int,
    center_exclusion: int,
    *,
    scale_constant: float = 1.4826,
    epsilon: float = 0.000_001,
) -> npt.NDArray[np.float64]:
    """Normalize each value against its local median/MAD, excluding its center."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1:
        raise ValueError("signal values must be one-dimensional")
    if window_radius < 1 or center_exclusion < 0:
        raise ValueError("window radius must be positive and center exclusion nonnegative")
    result = np.full(len(source), np.nan, dtype=np.float64)
    for index, value in enumerate(source):
        if not math.isfinite(float(value)):
            continue
        start = max(0, index - window_radius)
        stop = min(len(source), index + window_radius + 1)
        left_stop = max(start, index - center_exclusion)
        right_start = min(stop, index + center_exclusion + 1)
        neighborhood = np.concatenate((source[start:left_stop], source[right_start:stop]))
        neighborhood = neighborhood[np.isfinite(neighborhood)]
        if not len(neighborhood):
            continue
        median = float(np.median(neighborhood))
        mad = float(np.median(np.abs(neighborhood - median)))
        result[index] = (float(value) - median) / (scale_constant * mad + epsilon)
    return result


def squash_zscores(values: npt.ArrayLike, z_max: float) -> npt.NDArray[np.float64]:
    """Keep positive deviations and map them into [0, 1], preserving NaNs."""
    if z_max <= 0.0:
        raise ValueError("z_max must be positive")
    source = np.asarray(values, dtype=np.float64)
    return np.asarray(
        np.where(np.isfinite(source), np.clip(source, 0.0, z_max) / z_max, np.nan),
        dtype=np.float64,
    )


def dilate_signal(values: npt.ArrayLike, radius: int) -> npt.NDArray[np.float64]:
    """Associate nearby evidence by a NaN-aware temporal maximum."""
    source = np.asarray(values, dtype=np.float64)
    if radius < 0:
        raise ValueError("dilation radius must be nonnegative")
    if radius == 0:
        return source.copy()
    result = np.full(len(source), np.nan, dtype=np.float64)
    for index in range(len(source)):
        window = source[max(0, index - radius) : index + radius + 1]
        finite = window[np.isfinite(window)]
        if len(finite):
            result[index] = float(np.max(finite))
    return result


def fuse_normalized(
    normalized: Mapping[str, npt.NDArray[np.float64]],
    weights: Mapping[str, float],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int16], npt.NDArray[np.float64]]:
    """Fuse signals while renormalizing weights over values available per boundary."""
    lengths = {len(values) for values in normalized.values()}
    if len(lengths) > 1:
        raise ValueError("all normalized signal arrays must have equal length")
    boundary_count = next(iter(lengths), 0)
    numerator = np.zeros(boundary_count, dtype=np.float64)
    denominator = np.zeros(boundary_count, dtype=np.float64)
    agreement = np.zeros(boundary_count, dtype=np.int16)
    for name, values in normalized.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0.0:
            continue
        finite = np.isfinite(values)
        numerator[finite] += weight * values[finite]
        denominator[finite] += weight
        agreement[finite & (values > 0.5)] += 1
    fused = np.divide(
        numerator,
        denominator,
        out=np.zeros(boundary_count, dtype=np.float64),
        where=denominator > 0.0,
    )
    return fused, agreement, denominator


def pick_boundaries(
    fused: npt.ArrayLike,
    agreement: npt.ArrayLike,
    config: FusionConfig,
) -> tuple[int, ...]:
    """Apply the high/low agreement gate, contiguous peak collapse, and NMS."""
    scores = np.asarray(fused, dtype=np.float64)
    counts = np.asarray(agreement, dtype=np.int16)
    if len(scores) != len(counts):
        raise ValueError("fused and agreement arrays must have equal length")
    accepted = (scores >= config.tau_high) & (counts >= config.agreement_count)
    accepted |= (scores >= config.tau_low) & (counts >= config.agreement_count + 1)
    candidates = np.flatnonzero(accepted).tolist()
    groups: list[list[int]] = []
    for candidate in candidates:
        if not groups or candidate > groups[-1][-1] + 1:
            groups.append([candidate])
        else:
            groups[-1].append(candidate)
    peaks = [max(group, key=lambda item: scores[item]) for group in groups]
    selected: list[int] = []
    for peak in sorted(peaks, key=lambda item: scores[item], reverse=True):
        if all(abs(peak - prior) >= config.min_shot_frames for prior in selected):
            selected.append(peak)
    return tuple(sorted(selected))


def normalize_and_fuse(
    raw_signals: Mapping[str, npt.NDArray[np.float64]],
    weights: Mapping[str, float],
    fps: float,
    config: FusionConfig,
    *,
    audio_signals: set[str] | frozenset[str] = frozenset(),
) -> FusionResult:
    """Execute the Phase 3 normalization/fusion specification exactly."""
    window_radius = max(1, round(config.local_window_sec * fps))
    zscores = {
        name: local_robust_zscore(
            values,
            window_radius,
            config.center_exclusion_samples,
            scale_constant=config.robust_scale_constant,
            epsilon=config.epsilon,
        )
        for name, values in raw_signals.items()
    }
    normalized = {name: squash_zscores(values, config.z_max) for name, values in zscores.items()}
    associated = {
        name: (
            dilate_signal(values, config.audio_association_radius_frames)
            if name in audio_signals
            else values
        )
        for name, values in normalized.items()
    }
    fused, agreement, available_weight = fuse_normalized(associated, weights)
    cuts = pick_boundaries(fused, agreement, config)
    return FusionResult(zscores, associated, fused, agreement, available_weight, cuts)
