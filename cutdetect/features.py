"""Cached, single-pass Phase 2 visual and audio feature extraction."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import time
import wave
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from cutdetect.config import FeatureConfig, IngestConfig
from cutdetect.ingest import VideoContext, ingest_video, iter_rgb_frames

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


class FeatureError(RuntimeError):
    """Raised when Phase 2 dependencies or feature extraction fail."""


@dataclass(frozen=True, slots=True)
class FeatureExtractionResult:
    """Paths and diagnostics returned by the feature extractor."""

    feature_path: Path
    overlay_path: Path
    frame_count: int
    face_count: int
    face_detection_rate: float
    elapsed_sec: float
    cache_hit: bool
    extractor_version: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible extraction summary."""
        value = asdict(self)
        value["feature_path"] = str(self.feature_path)
        value["overlay_path"] = str(self.overlay_path)
        return cast(dict[str, object], value)


def _optional_module(name: str, extra: str = "features") -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        raise FeatureError(f"install the {extra!r} extra to extract features") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_face_model(context: VideoContext, configured: Path | None) -> Path:
    """Resolve the Face Landmarker bundle without silently downloading at runtime."""
    candidates = []
    if configured is not None:
        candidates.append(configured.expanduser())
    candidates.extend(
        [
            Path.cwd() / ".cutdetect" / "models" / "face_landmarker.task",
            context.artifact_dir.parent / "models" / "face_landmarker.task",
            Path.home() / ".cache" / "cutdetect" / "models" / "face_landmarker.task",
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FeatureError(
        "Face Landmarker model not found; pass --model after downloading "
        f"{FACE_LANDMARKER_MODEL_URL}"
    )


def _cache_paths(
    context: VideoContext, config: FeatureConfig, model_path: Path
) -> tuple[Path, Path]:
    raw_config = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(config).items()
    }
    identity = json.dumps(
        {
            "video_content_sha256": context.cache_key,
            "extractor_version": config.extractor_version,
            "config": raw_config,
            "model_sha256": _sha256(model_path),
        },
        sort_keys=True,
    )
    fingerprint = hashlib.sha256(identity.encode()).hexdigest()[:12]
    stem = f"features-v{config.extractor_version}-{fingerprint}"
    return context.artifact_dir / f"{stem}.npz", context.artifact_dir / f"{stem}-overlay.mp4"


def _read_cached_result(
    feature_path: Path, overlay_path: Path, config: FeatureConfig
) -> FeatureExtractionResult | None:
    if not feature_path.is_file() or not overlay_path.is_file():
        return None
    try:
        with np.load(feature_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            availability = np.asarray(archive["face_available"], dtype=np.bool_)
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None
    return FeatureExtractionResult(
        feature_path=feature_path,
        overlay_path=overlay_path,
        frame_count=len(availability),
        face_count=int(np.count_nonzero(availability)),
        face_detection_rate=float(np.mean(availability)) if len(availability) else 0.0,
        elapsed_sec=float(metadata.get("elapsed_sec", 0.0)),
        cache_hit=True,
        extractor_version=config.extractor_version,
    )


def _frame_grid(context: VideoContext, width: int) -> tuple[int, int]:
    if width < 16:
        raise ValueError("downscale_width_px must be at least 16")
    height = max(16, round(context.height * width / context.width))
    return width, height


def _new_writer(cv2: Any, path: Path, context: VideoContext, config: FeatureConfig) -> Any:
    if len(config.overlay_codec) != 4:
        raise ValueError("overlay_codec must contain exactly four characters")
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*config.overlay_codec),
        float(context.fps),
        (context.width, context.height),
    )
    if not writer.isOpened():
        raise FeatureError(f"could not create overlay video: {path}")
    return writer


def _draw_overlay(
    cv2: Any,
    rgb: npt.NDArray[np.uint8],
    landmarks: npt.NDArray[np.float32] | None,
    matrix: npt.NDArray[np.float32] | None,
    bbox: npt.NDArray[np.float32] | None,
    config: FeatureConfig,
) -> npt.NDArray[np.uint8]:
    bgr = cast(npt.NDArray[np.uint8], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if landmarks is None or bbox is None:
        cv2.putText(
            bgr,
            "FACE NOT DETECTED",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return bgr
    for x, y, _z in landmarks[:: config.overlay_landmark_stride]:
        cv2.circle(bgr, (round(float(x)), round(float(y))), 1, (80, 255, 120), -1)
    left, top, right, bottom = (round(float(value)) for value in bbox)
    cv2.rectangle(bgr, (left, top), (right, bottom), (255, 190, 30), 2)
    if matrix is not None:
        origin = np.array([(left + right) / 2.0, (top + bottom) / 2.0], dtype=np.float32)
        colors = ((0, 0, 255), (0, 255, 0), (255, 80, 0))
        rotation = matrix[:3, :3]
        for axis, color in enumerate(colors):
            direction = np.array([rotation[0, axis], -rotation[1, axis]], dtype=np.float32)
            endpoint = origin + direction * config.overlay_axis_length_px
            cv2.line(
                bgr,
                tuple(np.rint(origin).astype(int)),
                tuple(np.rint(endpoint).astype(int)),
                color,
                3,
                cv2.LINE_AA,
            )
    return bgr


def _face_values(
    result: Any, width: int, height: int
) -> (
    tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        tuple[str, ...],
        npt.NDArray[np.float32],
        float,
        npt.NDArray[np.float32],
    ]
    | None
):
    if not result.face_landmarks:
        return None
    normalized = result.face_landmarks[0]
    landmarks = np.asarray(
        [
            (
                float(point.x) * width,
                float(point.y) * height,
                float(point.z) * width,
            )
            for point in normalized
        ],
        dtype=np.float32,
    )
    matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32)
    categories = result.face_blendshapes[0]
    names = tuple(str(category.category_name) for category in categories)
    blendshapes = np.asarray([float(category.score) for category in categories], dtype=np.float32)
    minimum = landmarks[:, :2].min(axis=0)
    maximum = landmarks[:, :2].max(axis=0)
    bbox = np.asarray([minimum[0], minimum[1], maximum[0], maximum[1]], dtype=np.float32)
    centroid = landmarks[:, :2].mean(axis=0, dtype=np.float64).astype(np.float32)
    # MediaPipe indices 33 and 263 are the outer eye corners.
    face_scale = float(np.linalg.norm(landmarks[33, :2] - landmarks[263, :2]))
    return landmarks, matrix, blendshapes, names, bbox, face_scale, centroid


def _visual_features(
    context: VideoContext,
    model_path: Path,
    overlay_path: Path,
    config: FeatureConfig,
) -> dict[str, npt.NDArray[Any]]:
    cv2 = _optional_module("cv2")
    # MediaPipe imports matplotlib; provide a persistent writable cache so the
    # import does not rebuild font metadata on every CLI invocation.
    mpl_cache = context.artifact_dir.parent / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    mp = _optional_module("mediapipe")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path), delegate=mp.tasks.BaseOptions.Delegate.CPU
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=config.min_face_detection_confidence,
        min_face_presence_confidence=config.min_face_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    small_width, small_height = _frame_grid(context, config.downscale_width_px)
    face_width, face_height = _frame_grid(context, config.face_input_width_px)
    frame_times = np.full(context.frame_count, np.nan, dtype=np.float64)
    available = np.zeros(context.frame_count, dtype=np.bool_)
    pose = np.full((context.frame_count, 4, 4), np.nan, dtype=np.float32)
    bbox = np.full((context.frame_count, 4), np.nan, dtype=np.float32)
    scale = np.full(context.frame_count, np.nan, dtype=np.float32)
    centroid = np.full((context.frame_count, 2), np.nan, dtype=np.float32)
    grayscale = np.empty((context.frame_count, small_height, small_width), dtype=np.uint8)
    histogram = np.empty((context.frame_count, config.hsv_histogram_bins * 3), dtype=np.float32)
    flow = np.empty(
        (max(0, context.frame_count - 1), small_height, small_width, 2), dtype=np.float16
    )
    landmark_items: list[npt.NDArray[np.float32] | None] = []
    blendshape_items: list[npt.NDArray[np.float32] | None] = []
    blendshape_names: tuple[str, ...] = ()
    previous_gray: npt.NDArray[np.uint8] | None = None
    writer = _new_writer(cv2, overlay_path, context, config)
    decoded_count = 0
    previous_timestamp_ms = -1
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            for index, timestamp, rgb in iter_rgb_frames(context):
                if index >= context.frame_count:
                    raise FeatureError("decoder returned more frames than the ingest context")
                decoded_count += 1
                frame_times[index] = timestamp
                small = cv2.resize(rgb, (small_width, small_height), interpolation=cv2.INTER_AREA)
                gray = cast(npt.NDArray[np.uint8], cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
                grayscale[index] = gray
                hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
                for channel in range(3):
                    values = cv2.calcHist(
                        [hsv], [channel], None, [config.hsv_histogram_bins], [0, 256]
                    ).reshape(-1)
                    values /= max(float(values.sum()), 1.0)
                    start = channel * config.hsv_histogram_bins
                    histogram[index, start : start + config.hsv_histogram_bins] = values
                if previous_gray is not None:
                    dense = cv2.calcOpticalFlowFarneback(
                        previous_gray,
                        gray,
                        None,
                        config.flow_pyramid_scale,
                        config.flow_levels,
                        config.flow_window_size,
                        config.flow_iterations,
                        config.flow_poly_neighborhood,
                        config.flow_poly_sigma,
                        0,
                    )
                    flow[index - 1] = dense.astype(np.float16)
                previous_gray = gray
                timestamp_ms = max(previous_timestamp_ms + 1, round(timestamp * 1000.0))
                previous_timestamp_ms = timestamp_ms
                face_rgb = cv2.resize(rgb, (face_width, face_height), interpolation=cv2.INTER_AREA)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(face_rgb)
                )
                result = landmarker.detect_for_video(image, timestamp_ms)
                values = _face_values(result, context.width, context.height)
                if values is None:
                    landmark_items.append(None)
                    blendshape_items.append(None)
                    writer.write(_draw_overlay(cv2, rgb, None, None, None, config))
                    continue
                landmarks, matrix, blendshapes, names, box, face_scale, center = values
                available[index] = True
                pose[index] = matrix
                bbox[index] = box
                scale[index] = face_scale
                centroid[index] = center
                landmark_items.append(landmarks)
                blendshape_items.append(blendshapes)
                if not blendshape_names:
                    blendshape_names = names
                writer.write(_draw_overlay(cv2, rgb, landmarks, matrix, box, config))
    finally:
        writer.release()
    if decoded_count != context.frame_count:
        raise FeatureError(f"decoded {decoded_count} frames; expected {context.frame_count}")
    landmark_count = next(
        (len(item) for item in landmark_items if item is not None),
        0,
    )
    blendshape_count = next(
        (len(item) for item in blendshape_items if item is not None),
        0,
    )
    landmarks_array = np.full((context.frame_count, landmark_count, 3), np.nan, dtype=np.float32)
    blendshapes_array = np.full((context.frame_count, blendshape_count), np.nan, dtype=np.float32)
    for index, item in enumerate(landmark_items):
        if item is not None:
            landmarks_array[index] = item
    for index, item in enumerate(blendshape_items):
        if item is not None:
            blendshapes_array[index] = item
    return {
        "frame_times_sec": frame_times,
        "face_available": available,
        "face_landmarks_px": landmarks_array,
        "facial_transformation_matrices": pose,
        "face_blendshapes": blendshapes_array,
        "face_blendshape_names": np.asarray(blendshape_names, dtype="U64"),
        "face_bbox_px": bbox,
        "face_scale_px": scale,
        "face_centroid_px": centroid,
        "grayscale": grayscale,
        "hsv_histograms": histogram,
        "dense_flow": flow,
    }


def _mel_filterbank(
    sample_rate: int, fft_size: int, bands: int, min_hz: float, max_hz: float
) -> npt.NDArray[np.float32]:
    if not 0.0 <= min_hz < max_hz <= sample_rate / 2:
        raise ValueError("mel frequency range must lie within Nyquist")

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (math.pow(10.0, mel / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(min_hz), hz_to_mel(max_hz), bands + 2)
    frequencies = np.asarray([mel_to_hz(value) for value in mel_points])
    bins = np.floor((fft_size + 1) * frequencies / sample_rate).astype(int)
    bins = np.clip(bins, 0, fft_size // 2)
    filters = np.zeros((bands, fft_size // 2 + 1), dtype=np.float32)
    for band in range(bands):
        left, center, right = (int(value) for value in bins[band : band + 3])
        if center <= left:
            center = min(left + 1, fft_size // 2)
        if right <= center:
            right = min(center + 1, fft_size // 2)
        if center > left:
            filters[band, left:center] = np.linspace(0.0, 1.0, center - left, endpoint=False)
        if right > center:
            filters[band, center:right] = np.linspace(1.0, 0.0, right - center, endpoint=False)
    return filters


def _read_pcm16(path: Path) -> tuple[int, npt.NDArray[np.float32]]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise FeatureError("ingested audio must be mono 16-bit PCM")
        sample_rate = handle.getframerate()
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return sample_rate, samples.astype(np.float32) / 32768.0


def _audio_features(context: VideoContext, config: FeatureConfig) -> dict[str, npt.NDArray[Any]]:
    if context.audio_path is None:
        return {
            "audio_sample_rate": np.asarray(0, dtype=np.int32),
            "audio_waveform": np.empty(0, dtype=np.float32),
            "audio_times_sec": np.empty(0, dtype=np.float64),
            "audio_log_mel": np.empty((0, config.audio_mel_bands), dtype=np.float32),
            "audio_mfcc": np.empty((0, config.audio_mfcc_count), dtype=np.float32),
            "audio_spectral_flux": np.empty(0, dtype=np.float32),
            "audio_rms": np.empty(0, dtype=np.float32),
            "audio_high_band_rms": np.empty(0, dtype=np.float32),
            "audio_highpass_rms": np.empty(0, dtype=np.float32),
            "audio_f0_hz": np.empty(0, dtype=np.float32),
            "audio_voicing": np.empty(0, dtype=np.float32),
        }
    scipy_fft = _optional_module("scipy.fft")
    sample_rate, waveform = _read_pcm16(context.audio_path)
    hop = max(1, round(sample_rate * config.audio_hop_ms / 1000.0))
    window_size = max(hop, round(sample_rate * config.audio_window_ms / 1000.0))
    if config.audio_fft_size < window_size:
        raise ValueError("audio_fft_size must be at least the audio window size")
    padded = np.pad(waveform, (window_size // 2, window_size // 2))
    frame_count = 1 + (len(padded) - window_size) // hop
    frames = np.lib.stride_tricks.sliding_window_view(padded, window_size)[::hop][:frame_count]
    window = np.hanning(window_size).astype(np.float32)
    power = np.empty((frame_count, config.audio_fft_size // 2 + 1), dtype=np.float32)
    f0 = np.full(frame_count, np.nan, dtype=np.float32)
    voicing = np.zeros(frame_count, dtype=np.float32)
    block_size = 512
    min_lag = max(1, round(sample_rate / config.audio_f0_max_hz))
    max_lag = min(window_size - 1, round(sample_rate / config.audio_f0_min_hz))
    for start in range(0, frame_count, block_size):
        stop = min(start + block_size, frame_count)
        spectrum = scipy_fft.rfft(
            frames[start:stop] * window,
            n=config.audio_fft_size,
            axis=1,
        )
        block_power = np.abs(spectrum) ** 2
        power[start:stop] = block_power.astype(np.float32)
        autocorrelation = scipy_fft.irfft(block_power, n=config.audio_fft_size, axis=1)
        region = autocorrelation[:, min_lag : max_lag + 1]
        lags = np.argmax(region, axis=1) + min_lag
        rows = np.arange(stop - start)
        strengths = region[rows, lags - min_lag] / np.maximum(autocorrelation[:, 0], 1e-12)
        voicing[start:stop] = strengths.astype(np.float32)
        voiced = strengths >= config.audio_voicing_threshold
        f0[start:stop][voiced] = sample_rate / lags[voiced]
    mel_filters = _mel_filterbank(
        sample_rate,
        config.audio_fft_size,
        config.audio_mel_bands,
        config.audio_mel_min_hz,
        min(config.audio_mel_max_hz, sample_rate / 2),
    )
    log_mel = np.log1p(power @ mel_filters.T).astype(np.float32)
    mfcc = scipy_fft.dct(log_mel, type=2, norm="ortho", axis=1)[
        :, : config.audio_mfcc_count
    ].astype(np.float32)
    spectral_flux = np.zeros(frame_count, dtype=np.float32)
    spectral_flux[1:] = np.sqrt(np.square(np.maximum(log_mel[1:] - log_mel[:-1], 0.0)).sum(axis=1))
    rms = np.sqrt(np.mean(np.square(frames), axis=1)).astype(np.float32)
    highpass_rms = np.sqrt(np.mean(np.square(np.diff(frames, axis=1)), axis=1)).astype(np.float32)
    frequencies = np.fft.rfftfreq(config.audio_fft_size, 1.0 / sample_rate)
    high_band = (frequencies >= config.audio_high_band_min_hz) & (
        frequencies <= config.audio_high_band_max_hz
    )
    high_band_rms = np.sqrt(np.mean(power[:, high_band], axis=1)).astype(np.float32)
    times = np.arange(frame_count, dtype=np.float64) * hop / sample_rate
    return {
        "audio_sample_rate": np.asarray(sample_rate, dtype=np.int32),
        "audio_waveform": waveform.astype(np.float32),
        "audio_times_sec": times,
        "audio_log_mel": log_mel,
        "audio_mfcc": mfcc,
        "audio_spectral_flux": spectral_flux,
        "audio_rms": rms,
        "audio_high_band_rms": high_band_rms,
        "audio_highpass_rms": highpass_rms,
        "audio_f0_hz": f0,
        "audio_voicing": voicing,
    }


def _save_archive(
    path: Path,
    arrays: Mapping[str, npt.NDArray[Any]],
    metadata: Mapping[str, object],
) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, metadata=np.asarray(json.dumps(metadata)), **arrays)
    temporary.replace(path)


def extract_features(
    video_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    config: FeatureConfig | None = None,
    model_path: str | Path | None = None,
    force: bool = False,
) -> FeatureExtractionResult:
    """Extract and cache all Phase 2 features plus a tracking overlay."""
    settings = config or FeatureConfig()
    if model_path is not None:
        settings = replace(settings, model_path=Path(model_path))
    context = ingest_video(
        video_path,
        IngestConfig(cache_dir=Path(cache_dir) if cache_dir is not None else None),
    )
    model = resolve_face_model(context, settings.model_path)
    feature_path, overlay_path = _cache_paths(context, settings, model)
    if not force:
        cached = _read_cached_result(feature_path, overlay_path, settings)
        if cached is not None:
            return cached
    started = time.perf_counter()
    visual = _visual_features(context, model, overlay_path, settings)
    audio = _audio_features(context, settings)
    elapsed = time.perf_counter() - started
    availability = np.asarray(visual["face_available"], dtype=np.bool_)
    metadata: dict[str, object] = {
        "schema_version": "1.0",
        "extractor_version": settings.extractor_version,
        "video_content_sha256": context.cache_key,
        "model_sha256": _sha256(model),
        "model_path": str(model),
        "frame_count": context.frame_count,
        "fps": float(context.fps),
        "source_width": context.width,
        "source_height": context.height,
        "face_count": int(np.count_nonzero(availability)),
        "face_detection_rate": float(np.mean(availability)) if len(availability) else 0.0,
        "elapsed_sec": elapsed,
        "config": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in asdict(settings).items()
        },
    }
    _save_archive(feature_path, {**visual, **audio}, metadata)
    return FeatureExtractionResult(
        feature_path=feature_path,
        overlay_path=overlay_path,
        frame_count=context.frame_count,
        face_count=int(np.count_nonzero(availability)),
        face_detection_rate=float(np.mean(availability)) if len(availability) else 0.0,
        elapsed_sec=elapsed,
        cache_hit=False,
        extractor_version=settings.extractor_version,
    )


def load_feature_metadata(path: str | Path) -> dict[str, object]:
    """Load only an archive's versioned JSON metadata."""
    with np.load(Path(path), allow_pickle=False) as archive:
        raw = json.loads(str(archive["metadata"].item()))
    if not isinstance(raw, dict):
        raise ValueError("feature metadata must be a JSON object")
    return cast(dict[str, object], raw)
