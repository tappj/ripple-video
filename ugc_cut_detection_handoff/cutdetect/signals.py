"""Pure Phase 3 boundary-signal functions and cache orchestration."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from cutdetect.config import SignalConfig

AUDIO_SIGNALS = frozenset(
    {
        "spectral_flux",
        "noise_floor_step",
        "lpc_residual_spike",
        "f0_discontinuity",
        "mfcc_delta",
    }
)


@dataclass(frozen=True, slots=True)
class SignalBundle:
    """Raw fusion signals plus diagnostic-only supporting traces."""

    raw: dict[str, npt.NDArray[np.float64]]
    auxiliary: dict[str, npt.NDArray[np.float64]]
    disabled_reasons: dict[str, str]
    captions_region_masked: bool
    face_detection_rate: float


def _nan_boundaries(frame_count: int) -> npt.NDArray[np.float64]:
    return np.full(max(0, frame_count - 1), np.nan, dtype=np.float64)


def pose_vectors(
    matrices: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.float64]:
    """Decompose facial transforms into roll/pitch/yaw vectors in radians."""
    result = np.full((len(matrices), 3), np.nan, dtype=np.float64)
    for index, matrix in enumerate(matrices):
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            continue
        rotation = np.asarray(matrix[:3, :3], dtype=np.float64)
        horizontal = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
        if horizontal > 1e-8:
            roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
            pitch = math.atan2(float(-rotation[2, 0]), horizontal)
            yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        else:
            roll = math.atan2(float(-rotation[1, 2]), float(rotation[1, 1]))
            pitch = math.atan2(float(-rotation[2, 0]), horizontal)
            yaw = 0.0
        result[index] = (roll, pitch, yaw)
    return np.asarray(result, dtype=np.float64)


def _acceleration(
    vectors: npt.NDArray[np.floating[Any]], availability: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    result = _nan_boundaries(len(vectors))
    if len(vectors) < 3:
        return result
    velocity = np.diff(np.asarray(vectors, dtype=np.float64), axis=0)
    result[1:] = np.linalg.norm(velocity[1:] - velocity[:-1], axis=1)
    valid = availability[:-2] & availability[1:-1] & availability[2:]
    result[1:][~valid] = np.nan
    return np.asarray(result, dtype=np.float64)


def pose_accel(
    matrices: npt.NDArray[np.floating[Any]], availability: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """Magnitude of the second difference of rigid head pose."""
    return _acceleration(pose_vectors(matrices), availability)


def centroid_accel(
    centroids: npt.NDArray[np.floating[Any]],
    scales: npt.NDArray[np.floating[Any]],
    availability: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Second difference of centroid motion normalized by face scale."""
    points = np.asarray(centroids, dtype=np.float64)
    face_scales = np.asarray(scales, dtype=np.float64)
    velocity = np.diff(points, axis=0) / np.maximum(
        ((face_scales[:-1] + face_scales[1:]) / 2.0)[:, None],
        1e-6,
    )
    result = _nan_boundaries(len(points))
    if len(points) >= 3:
        result[1:] = np.linalg.norm(velocity[1:] - velocity[:-1], axis=1)
        valid = availability[:-2] & availability[1:-1] & availability[2:]
        result[1:][~valid] = np.nan
    return np.asarray(result, dtype=np.float64)


def scale_delta(
    scales: npt.NDArray[np.floating[Any]], availability: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """Absolute first difference of log face scale."""
    source = np.asarray(scales, dtype=np.float64)
    result = np.abs(np.diff(np.log(np.maximum(source, 1e-6))))
    result[~(availability[:-1] & availability[1:])] = np.nan
    return np.asarray(result, dtype=np.float64)


def _procrustes_pair(left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]) -> float:
    left_centered = left - left.mean(axis=0)
    right_centered = right - right.mean(axis=0)
    left_norm = float(np.linalg.norm(left_centered))
    right_norm = float(np.linalg.norm(right_centered))
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        return math.nan
    left_unit = left_centered / left_norm
    right_unit = right_centered / right_norm
    u, _singular, vt = np.linalg.svd(right_unit.T @ left_unit, full_matrices=False)
    rotation = u @ vt
    aligned = right_unit @ rotation
    return float(np.sqrt(np.mean(np.square(left_unit - aligned))))


def procrustes_residual(
    landmarks: npt.NDArray[np.floating[Any]], availability: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """Similarity-invariant non-rigid landmark change per boundary."""
    result = _nan_boundaries(len(landmarks))
    valid = availability[:-1] & availability[1:]
    for boundary in np.flatnonzero(valid):
        result[boundary] = _procrustes_pair(
            np.asarray(landmarks[boundary, :, :2], dtype=np.float64),
            np.asarray(landmarks[boundary + 1, :, :2], dtype=np.float64),
        )
    return result


def blendshape_delta(
    blendshapes: npt.NDArray[np.floating[Any]],
    names: Sequence[str],
    availability: npt.NDArray[np.bool_],
    emphasis: float,
) -> npt.NDArray[np.float64]:
    """Weighted expression/viseme change, emphasizing mouth and blink controls."""
    weights = np.ones(len(names), dtype=np.float64)
    for index, name in enumerate(names):
        lowered = name.lower()
        if lowered == "jawopen" or lowered.startswith("mouth") or lowered.startswith("eyeblink"):
            weights[index] = emphasis
    differences = np.diff(np.asarray(blendshapes, dtype=np.float64), axis=0) * weights
    result = np.linalg.norm(differences, axis=1) / math.sqrt(max(1, len(weights)))
    result[~(availability[:-1] & availability[1:])] = np.nan
    return np.asarray(result, dtype=np.float64)


def _similarity_affine(
    source: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64] | None:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = float(np.square(source_centered).sum())
    if variance <= 1e-9:
        return None
    u, singular, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = u @ vt
    scale = float(singular.sum() / variance)
    translation = target_mean - scale * (source_mean @ rotation)
    affine = np.empty((2, 3), dtype=np.float64)
    affine[:, :2] = (scale * rotation).T
    affine[:, 2] = translation
    return affine


def stabilized_residual(
    grayscale: npt.NDArray[np.uint8],
    landmarks: npt.NDArray[np.floating[Any]],
    bboxes: npt.NDArray[np.floating[Any]],
    availability: npt.NDArray[np.bool_],
    source_size: tuple[int, int],
    config: SignalConfig,
) -> npt.NDArray[np.float64]:
    """Gradient residual after similarity-stabilizing the face/torso crop."""
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError("opencv is required for stabilized residuals") from error
    frame_count, height, width = grayscale.shape
    source_width, source_height = source_size
    ratio = np.asarray([width / source_width, height / source_height], dtype=np.float64)
    result = _nan_boundaries(frame_count)
    for boundary in np.flatnonzero(availability[:-1] & availability[1:]):
        target_points = np.asarray(landmarks[boundary, :, :2], dtype=np.float64) * ratio
        source_points = np.asarray(landmarks[boundary + 1, :, :2], dtype=np.float64) * ratio
        affine = _similarity_affine(source_points, target_points)
        if affine is None:
            continue
        warped = cv2.warpAffine(
            grayscale[boundary + 1],
            affine,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        box = np.asarray(bboxes[boundary], dtype=np.float64)
        box[[0, 2]] *= width / source_width
        box[[1, 3]] *= height / source_height
        face_width = max(1.0, float(box[2] - box[0]))
        face_height = max(1.0, float(box[3] - box[1]))
        center_x = float((box[0] + box[2]) / 2.0)
        left = max(0, round(center_x - face_width * config.stabilized_crop_width_scale / 2))
        right = min(width, round(center_x + face_width * config.stabilized_crop_width_scale / 2))
        top = max(0, round(float(box[1]) - face_height * config.stabilized_crop_top_scale))
        bottom = min(
            height, round(float(box[3]) + face_height * config.stabilized_crop_bottom_scale)
        )
        if right <= left or bottom <= top:
            continue
        first = grayscale[boundary, top:bottom, left:right].astype(np.float32)
        second = warped[top:bottom, left:right].astype(np.float32)
        first_gradient = cv2.Sobel(first, cv2.CV_32F, 1, 0) + cv2.Sobel(first, cv2.CV_32F, 0, 1)
        second_gradient = cv2.Sobel(second, cv2.CV_32F, 1, 0) + cv2.Sobel(second, cv2.CV_32F, 0, 1)
        denominator = float(np.mean(np.abs(first_gradient)) + 1e-6)
        result[boundary] = float(np.mean(np.abs(first_gradient - second_gradient)) / denominator)
    return result


def flow_incoherence(
    flow: npt.NDArray[np.floating[Any]], config: SignalConfig
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]:
    """Fit global affine flow and return residual incoherence plus diagnostics."""
    boundary_count, height, width, _channels = flow.shape
    y_grid, x_grid = np.mgrid[
        0 : height : config.flow_sample_stride, 0 : width : config.flow_sample_stride
    ]
    design = np.column_stack((x_grid.ravel(), y_grid.ravel(), np.ones(x_grid.size)))
    incoherence = np.empty(boundary_count, dtype=np.float64)
    magnitude = np.empty(boundary_count, dtype=np.float64)
    inlier_ratio = np.empty(boundary_count, dtype=np.float64)
    for boundary in range(boundary_count):
        sampled = np.asarray(
            flow[boundary, :: config.flow_sample_stride, :: config.flow_sample_stride],
            dtype=np.float64,
        ).reshape(-1, 2)
        coefficients, _residuals, _rank, _singular = np.linalg.lstsq(design, sampled, rcond=None)
        residual = np.linalg.norm(sampled - design @ coefficients, axis=1)
        residual_median = float(np.median(residual))
        residual_mad = float(np.median(np.abs(residual - residual_median)))
        threshold = residual_median + config.flow_inlier_mad_scale * 1.4826 * residual_mad
        ratio = float(np.mean(residual <= threshold))
        magnitude[boundary] = float(np.median(np.linalg.norm(sampled, axis=1)))
        inlier_ratio[boundary] = ratio
        incoherence[boundary] = residual_median * (1.0 + (1.0 - ratio))
    return incoherence, {"flow_magnitude": magnitude, "flow_inlier_ratio": inlier_ratio}


def background_delta(
    grayscale: npt.NDArray[np.uint8],
    bboxes: npt.NDArray[np.floating[Any]],
    source_size: tuple[int, int],
    bins: int,
    config: SignalConfig,
) -> tuple[npt.NDArray[np.float64], bool]:
    """Histogram/edge delta outside the subject and persistent animated regions."""
    frame_count, height, width = grayscale.shape
    source_width, source_height = source_size
    # Compute temporal variance online. Casting the complete frame cube to
    # float32 previously created a second full-video allocation.
    temporal_mean = np.zeros((height, width), dtype=np.float64)
    temporal_m2 = np.zeros((height, width), dtype=np.float64)
    for index, frame in enumerate(grayscale, start=1):
        values = frame.astype(np.float64)
        delta = values - temporal_mean
        temporal_mean += delta / index
        temporal_m2 += delta * (values - temporal_mean)
    temporal_variance = np.sqrt(temporal_m2 / max(1, frame_count))
    variance_limit = float(np.percentile(temporal_variance, config.background_variance_percentile))
    stable = temporal_variance <= variance_limit
    bottom_dynamic_fraction = float(np.mean(~stable[height // 2 :]))
    captions_masked = bottom_dynamic_fraction > 0.01
    result = np.full(frame_count - 1, np.nan, dtype=np.float64)
    previous_gradient: npt.NDArray[np.float32] | None = None
    for boundary in range(frame_count - 1):
        if previous_gradient is None:
            previous_gy, previous_gx = np.gradient(grayscale[boundary].astype(np.float32))
            previous_gradient = np.hypot(previous_gx, previous_gy)
        next_gy, next_gx = np.gradient(grayscale[boundary + 1].astype(np.float32))
        next_gradient = np.hypot(next_gx, next_gy)
        union = np.nanmean(bboxes[boundary : boundary + 2], axis=0)
        if not np.isfinite(union).all():
            previous_gradient = next_gradient
            continue
        box = np.asarray(union, dtype=np.float64)
        box[[0, 2]] *= width / source_width
        box[[1, 3]] *= height / source_height
        face_width = max(1.0, float(box[2] - box[0]))
        face_height = max(1.0, float(box[3] - box[1]))
        center_x = float((box[0] + box[2]) / 2.0)
        left = max(0, round(center_x - face_width * config.background_subject_width_scale / 2))
        right = min(width, round(center_x + face_width * config.background_subject_width_scale / 2))
        top = max(0, round(float(box[1]) - face_height * config.background_subject_top_scale))
        mask = stable.copy()
        mask[top:, left:right] = False
        if np.count_nonzero(mask) < bins:
            previous_gradient = next_gradient
            continue
        first = grayscale[boundary][mask]
        second = grayscale[boundary + 1][mask]
        first_hist, _ = np.histogram(first, bins=bins, range=(0, 256), density=True)
        second_hist, _ = np.histogram(second, bins=bins, range=(0, 256), density=True)
        histogram_change = float(np.abs(first_hist - second_hist).sum())
        edge_change = float(
            np.mean(np.abs(previous_gradient[mask] - next_gradient[mask])) / 255.0
        )
        result[boundary] = histogram_change + edge_change
        previous_gradient = next_gradient
    return result, captions_masked


def _pool_fine_signal(
    values: npt.NDArray[np.floating[Any]],
    times: npt.NDArray[np.floating[Any]],
    boundary_times: npt.NDArray[np.floating[Any]],
    radius_sec: float,
) -> npt.NDArray[np.float64]:
    source = np.asarray(values, dtype=np.float64)
    source_times = np.asarray(times, dtype=np.float64)
    result = np.full(len(boundary_times), np.nan, dtype=np.float64)
    for index, boundary_time in enumerate(boundary_times):
        left = int(np.searchsorted(source_times, boundary_time - radius_sec, side="left"))
        right = int(np.searchsorted(source_times, boundary_time + radius_sec, side="right"))
        finite = source[left:right][np.isfinite(source[left:right])]
        if len(finite):
            result[index] = float(np.max(finite))
    return result


def spectral_flux(
    fine_flux: npt.NDArray[np.floating[Any]],
    audio_times: npt.NDArray[np.floating[Any]],
    boundary_times: npt.NDArray[np.floating[Any]],
    config: SignalConfig,
) -> npt.NDArray[np.float64]:
    """Take the maximum response across 5/20/50 ms audio resolutions."""
    source = np.asarray(fine_flux, dtype=np.float64)
    multiscale: list[npt.NDArray[np.float64]] = []
    for hops in config.audio_multiscale_hops:
        if hops <= 1:
            multiscale.append(source)
        else:
            kernel = np.ones(hops, dtype=np.float64) / hops
            multiscale.append(np.convolve(source, kernel, mode="same"))
    combined = np.max(np.stack(multiscale), axis=0)
    return _pool_fine_signal(
        combined, audio_times, boundary_times, config.audio_boundary_radius_sec
    )


def noise_floor_step(
    high_band_rms: npt.NDArray[np.floating[Any]],
    audio_times: npt.NDArray[np.floating[Any]],
    boundary_times: npt.NDArray[np.floating[Any]],
    config: SignalConfig,
) -> npt.NDArray[np.float64]:
    """Absolute log-ratio of the high-band rolling 10th percentile on each side."""
    values = np.asarray(high_band_rms, dtype=np.float64)
    times = np.asarray(audio_times, dtype=np.float64)
    result = np.full(len(boundary_times), np.nan, dtype=np.float64)
    for index, boundary_time in enumerate(boundary_times):
        left_start = int(np.searchsorted(times, boundary_time - config.noise_floor_context_sec))
        center = int(np.searchsorted(times, boundary_time))
        right_stop = int(np.searchsorted(times, boundary_time + config.noise_floor_context_sec))
        if center <= left_start or right_stop <= center:
            continue
        left = float(np.percentile(values[left_start:center], 10))
        right = float(np.percentile(values[center:right_stop], 10))
        result[index] = abs(math.log((right + 1e-9) / (left + 1e-9)))
    return result


def lpc_residual_spike(
    waveform: npt.NDArray[np.floating[Any]],
    sample_rate: int,
    boundary_times: npt.NDArray[np.floating[Any]],
    config: SignalConfig,
) -> npt.NDArray[np.float64]:
    """Fit a left-context autoregressor and measure error immediately after boundary."""
    source = np.asarray(waveform, dtype=np.float64)
    context_samples = max(config.lpc_order + 2, round(config.lpc_context_sec * sample_rate))
    evaluation_samples = max(1, round(config.lpc_evaluation_sec * sample_rate))
    result = np.full(len(boundary_times), np.nan, dtype=np.float64)
    for index, boundary_time in enumerate(boundary_times):
        center = round(float(boundary_time) * sample_rate)
        if center < context_samples or center + evaluation_samples >= len(source):
            continue
        training = source[center - context_samples : center]
        correlation = np.correlate(training, training, mode="full")[context_samples - 1 :]
        toeplitz = correlation[
            np.abs(np.subtract.outer(np.arange(config.lpc_order), np.arange(config.lpc_order)))
        ]
        target = correlation[1 : config.lpc_order + 1]
        try:
            coefficients = np.linalg.solve(toeplitz + np.eye(config.lpc_order) * 1e-8, target)
        except np.linalg.LinAlgError:
            continue
        extended = source[center - config.lpc_order : center + evaluation_samples]
        errors = []
        for offset in range(config.lpc_order, len(extended)):
            history = extended[offset - config.lpc_order : offset][::-1]
            errors.append(float(extended[offset] - coefficients @ history))
        baseline = float(np.sqrt(np.mean(np.square(training))) + 1e-8)
        result[index] = float(np.sqrt(np.mean(np.square(errors))) / baseline)
    return result


def _side_medians(
    values: npt.NDArray[np.floating[Any]],
    times: npt.NDArray[np.floating[Any]],
    boundary_times: npt.NDArray[np.floating[Any]],
    context_sec: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    source = np.asarray(values, dtype=np.float64)
    source_times = np.asarray(times, dtype=np.float64)
    left = np.full(len(boundary_times), np.nan, dtype=np.float64)
    right = np.full(len(boundary_times), np.nan, dtype=np.float64)
    for index, boundary_time in enumerate(boundary_times):
        start = int(np.searchsorted(source_times, boundary_time - context_sec))
        center = int(np.searchsorted(source_times, boundary_time))
        stop = int(np.searchsorted(source_times, boundary_time + context_sec))
        left_values = source[start:center]
        right_values = source[center:stop]
        left_values = left_values[np.isfinite(left_values)]
        right_values = right_values[np.isfinite(right_values)]
        if len(left_values) and len(right_values):
            left[index] = float(np.median(left_values))
            right[index] = float(np.median(right_values))
    return left, right


def f0_discontinuity(
    f0_hz: npt.NDArray[np.floating[Any]],
    audio_times: npt.NDArray[np.floating[Any]],
    boundary_times: npt.NDArray[np.floating[Any]],
    config: SignalConfig,
) -> npt.NDArray[np.float64]:
    """Log pitch ratio across boundaries when both sides are voiced."""
    left, right = _side_medians(f0_hz, audio_times, boundary_times, config.f0_context_sec)
    return np.asarray(np.abs(np.log(right / left)), dtype=np.float64)


def mfcc_delta(
    mfcc: npt.NDArray[np.floating[Any]],
    audio_times: npt.NDArray[np.floating[Any]],
    boundary_times: npt.NDArray[np.floating[Any]],
    config: SignalConfig,
) -> npt.NDArray[np.float64]:
    """L2 distance between short MFCC means on either side of each boundary."""
    values = np.asarray(mfcc, dtype=np.float64)
    times = np.asarray(audio_times, dtype=np.float64)
    result = np.full(len(boundary_times), np.nan, dtype=np.float64)
    for index, boundary_time in enumerate(boundary_times):
        start = int(np.searchsorted(times, boundary_time - config.mfcc_context_sec))
        center = int(np.searchsorted(times, boundary_time))
        stop = int(np.searchsorted(times, boundary_time + config.mfcc_context_sec))
        if center > start and stop > center:
            result[index] = float(
                np.linalg.norm(values[start:center].mean(axis=0) - values[center:stop].mean(axis=0))
            )
    return result


def content_value(
    artifact_dir: Path,
    histograms: npt.NDArray[np.floating[Any]],
) -> tuple[npt.NDArray[np.float64], str | None]:
    """Load PySceneDetect content values, falling back to cached HSV histogram deltas."""
    path = artifact_dir / "baseline_scenedetect_scores.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            values = np.asarray(archive["content_val"], dtype=np.float64)
        if len(values) == len(histograms):
            return values[1:], None
    fallback = np.abs(np.diff(np.asarray(histograms, dtype=np.float64), axis=0)).sum(axis=1)
    return fallback, "PySceneDetect cache unavailable; using HSV histogram delta"


def transnet_probability(
    artifact_dir: Path, frame_count: int
) -> tuple[npt.NDArray[np.float64], str | None]:
    """Load raw TransNet probabilities in boundary coordinates."""
    path = artifact_dir / "baseline_transnet_scores.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            values = np.asarray(archive["single_frame_probability"], dtype=np.float64)
        if len(values) == frame_count:
            return values[:-1], None
    return _nan_boundaries(frame_count), "TransNetV2 score cache unavailable"


def iframe_prior(
    video_path: Path, frame_count: int, config: SignalConfig
) -> tuple[npt.NDArray[np.float64], str | None]:
    """Build an I-frame/packet-size prior and disable it for fixed-GOP encodes."""
    executable = shutil.which("ffprobe")
    if executable is None:
        return _nan_boundaries(frame_count), "ffprobe unavailable"
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=pict_type,pkt_size",
            "-of",
            "json",
            str(video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return _nan_boundaries(frame_count), "ffprobe frame query failed"
    root = json.loads(completed.stdout)
    items = root.get("frames", [])
    types = [str(item.get("pict_type", "")) for item in items[:frame_count]]
    sizes = np.asarray([float(item.get("pkt_size", 0.0)) for item in items[:frame_count]])
    if len(types) != frame_count:
        return _nan_boundaries(frame_count), "ffprobe frame count mismatch"
    iframe_indices = np.flatnonzero(np.asarray(types) == "I")
    if len(iframe_indices) >= 3:
        intervals = np.diff(iframe_indices)
        coefficient = float(np.std(intervals) / max(float(np.mean(intervals)), 1.0))
        if coefficient <= config.fixed_gop_cv_threshold:
            return _nan_boundaries(frame_count), "fixed GOP detected; I-frame prior uninformative"
    size_scale = float(np.median(sizes) + 1e-6)
    values = (np.asarray(types[1:]) == "I").astype(np.float64) * np.log1p(sizes[1:] / size_scale)
    return values, None


def word_gap_anomaly(
    words: Sequence[Mapping[str, object]], boundary_times: npt.NDArray[np.floating[Any]]
) -> npt.NDArray[np.float64]:
    """Score boundaries close to abnormally short inter-word gaps."""
    parsed = sorted(
        (
            (
                float(cast(float | int | str, item["start"])),
                float(cast(float | int | str, item["end"])),
            )
            for item in words
            if "start" in item and "end" in item
        ),
        key=lambda item: item[0],
    )
    result = np.full(len(boundary_times), np.nan, dtype=np.float64)
    if len(parsed) < 3:
        return result
    pairs = list(pairwise(parsed))
    gaps = np.asarray([right[0] - left[1] for left, right in pairs])
    median = float(np.median(gaps))
    mad = float(np.median(np.abs(gaps - median)) + 1e-6)
    for (left, right), gap in zip(pairs, gaps, strict=True):
        midpoint = (left[1] + right[0]) / 2.0
        boundary = int(np.argmin(np.abs(np.asarray(boundary_times) - midpoint)))
        result[boundary] = max(0.0, (median - float(gap)) / (1.4826 * mad))
    return result


def compute_signal_bundle(
    feature_path: str | Path,
    video_path: str | Path,
    *,
    config: SignalConfig | None = None,
    words: Sequence[Mapping[str, object]] | None = None,
) -> SignalBundle:
    """Compute all Phase 3 signals from one versioned Phase 2 archive."""
    settings = config or SignalConfig()
    feature_file = Path(feature_path).expanduser().resolve()
    video_file = Path(video_path).expanduser().resolve()
    disabled: dict[str, str] = {}
    with np.load(feature_file, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        frame_times = np.asarray(archive["frame_times_sec"], dtype=np.float64)
        boundary_times = frame_times[1:]
        frame_count = len(frame_times)
        source_size = (int(metadata["source_width"]), int(metadata["source_height"]))

        # Large compressed arrays are deliberately loaded and reduced one at a
        # time. Keeping grayscale and float32 dense flow resident together was
        # enough to exceed a 512 MiB hosted instance on ordinary clips.
        if "flow_incoherence" in archive.files:
            flow_score = np.asarray(archive["flow_incoherence"], dtype=np.float64)
            flow_auxiliary = {
                "flow_magnitude": np.asarray(archive["flow_magnitude"], dtype=np.float64),
                "flow_inlier_ratio": np.asarray(
                    archive["flow_inlier_ratio"], dtype=np.float64
                ),
            }
        else:
            flow = np.asarray(archive["dense_flow"], dtype=np.float16)
            flow_score, flow_auxiliary = flow_incoherence(flow, settings)
            del flow

        histograms = np.asarray(archive["hsv_histograms"], dtype=np.float32)
        histogram_bins = int(histograms.shape[1] // 3)
        content, content_reason = content_value(feature_file.parent, histograms)
        del histograms
        if content_reason:
            disabled["content_val"] = content_reason

        availability = np.asarray(archive["face_available"], dtype=np.bool_)
        face_detection_rate = float(np.mean(availability)) if len(availability) else 0.0

        matrices = np.asarray(archive["facial_transformation_matrices"], dtype=np.float32)
        pose_score = pose_accel(matrices, availability)
        del matrices

        scales = np.asarray(archive["face_scale_px"], dtype=np.float32)
        centroids = np.asarray(archive["face_centroid_px"], dtype=np.float32)
        centroid_score = centroid_accel(centroids, scales, availability)
        scale_score = scale_delta(scales, availability)
        del centroids, scales

        landmarks = np.asarray(archive["face_landmarks_px"], dtype=np.float32)
        procrustes_score = procrustes_residual(landmarks, availability)
        if "stabilized_residual" in archive.files and "background_delta" in archive.files:
            stabilized_score = np.asarray(archive["stabilized_residual"], dtype=np.float64)
            background_score = np.asarray(archive["background_delta"], dtype=np.float64)
            captions_masked = bool(archive["captions_region_masked"].item())
            del landmarks
        else:
            bboxes = np.asarray(archive["face_bbox_px"], dtype=np.float32)
            grayscale = np.asarray(archive["grayscale"], dtype=np.uint8)
            stabilized_score = stabilized_residual(
                grayscale, landmarks, bboxes, availability, source_size, settings
            )
            background_score, captions_masked = background_delta(
                grayscale, bboxes, source_size, histogram_bins, settings
            )
            del grayscale, landmarks, bboxes

        blendshapes = np.asarray(archive["face_blendshapes"], dtype=np.float32)
        names = [str(value) for value in archive["face_blendshape_names"].tolist()]
        blendshape_score = blendshape_delta(
            blendshapes, names, availability, settings.blendshape_emphasis
        )
        del blendshapes

        audio_times = np.asarray(archive["audio_times_sec"], dtype=np.float64)
        sample_rate = int(archive["audio_sample_rate"].item())
        if sample_rate <= 0 or not len(audio_times):
            audio_values = {name: _nan_boundaries(frame_count) for name in AUDIO_SIGNALS}
            for name in AUDIO_SIGNALS:
                disabled[name] = "audio unavailable"
        else:
            fine_flux = np.asarray(archive["audio_spectral_flux"], dtype=np.float32)
            spectral_score = spectral_flux(fine_flux, audio_times, boundary_times, settings)
            del fine_flux
            high_band = np.asarray(archive["audio_high_band_rms"], dtype=np.float32)
            noise_score = noise_floor_step(high_band, audio_times, boundary_times, settings)
            del high_band
            waveform = np.asarray(archive["audio_waveform"], dtype=np.float32)
            lpc_score = lpc_residual_spike(waveform, sample_rate, boundary_times, settings)
            del waveform
            f0_hz = np.asarray(archive["audio_f0_hz"], dtype=np.float32)
            f0_score = f0_discontinuity(f0_hz, audio_times, boundary_times, settings)
            del f0_hz
            mfcc = np.asarray(archive["audio_mfcc"], dtype=np.float32)
            mfcc_score = mfcc_delta(mfcc, audio_times, boundary_times, settings)
            del mfcc
            audio_values = {
                "spectral_flux": spectral_score,
                "noise_floor_step": noise_score,
                "lpc_residual_spike": lpc_score,
                "f0_discontinuity": f0_score,
                "mfcc_delta": mfcc_score,
            }

    transnet, transnet_reason = transnet_probability(feature_file.parent, frame_count)
    if transnet_reason:
        disabled["transnet_prob"] = transnet_reason
    iframe, iframe_reason = iframe_prior(video_file, frame_count, settings)
    if iframe_reason:
        disabled["iframe_prior"] = iframe_reason
    if words is None:
        word_gap = _nan_boundaries(frame_count)
        disabled["word_gap_anomaly"] = "word timestamps not provided"
    else:
        word_gap = word_gap_anomaly(words, boundary_times)
    raw = {
        "pose_accel": pose_score,
        "centroid_accel": centroid_score,
        "scale_delta": scale_score,
        "procrustes_residual": procrustes_score,
        "blendshape_delta": blendshape_score,
        "stabilized_residual": stabilized_score,
        "flow_incoherence": flow_score,
        "background_delta": background_score,
        "content_val": content,
        "transnet_prob": transnet,
        **audio_values,
        "iframe_prior": iframe,
        "word_gap_anomaly": word_gap,
    }
    for name, values in raw.items():
        if not np.isfinite(values).any() and name not in disabled:
            disabled[name] = "signal unavailable for every boundary"
    return SignalBundle(
        raw,
        flow_auxiliary,
        disabled,
        captions_masked,
        face_detection_rate,
    )
