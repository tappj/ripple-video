"""Central configuration for all tunable cutdetect parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class IngestConfig:
    """Settings used while probing and normalizing source media."""

    # A frame delta may differ from the median by this relative fraction and
    # still be considered constant-rate timestamp jitter.
    pts_relative_tolerance: float = 0.02
    # This absolute delta tolerance protects very-high-fps inputs from floating
    # point noise in ffprobe's decimal timestamps.
    pts_absolute_tolerance_sec: float = 0.000_05
    # Rate metadata can contain harmless rounding differences (e.g. 29.97).
    rate_relative_tolerance: float = 0.001
    # CFR normalization preserves useful detail without creating huge cache files.
    normalization_crf: int = 18
    normalization_preset: str = "veryfast"
    # All downstream audio analysis operates at this canonical rate and layout.
    audio_sample_rate: int = 48_000
    audio_channels: int = 1
    # Hash reads use bounded chunks instead of loading a video into memory.
    hash_chunk_bytes: int = 1024 * 1024
    cache_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class LabelConfig:
    """Settings for the local ground-truth labeling UI."""

    host: str = "127.0.0.1"
    port: int = 8765
    thumbnail_width_px: int = 64
    thumbnail_jpeg_quality: int = 4
    review_radius_frames: int = 3
    waveform_window_sec: float = 1.0
    waveform_points: int = 256
    browser_open_delay_sec: float = 0.25
    max_label_payload_bytes: int = 1024 * 1024


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Ground-truth matching and precision-recall settings."""

    tolerances_frames: tuple[int, ...] = (1, 2, 3)
    primary_tolerance_frames: int = 2
    threshold_steps: int = 101
    include_unsure_labels: bool = False


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Phase 1 baseline extraction and threshold-sweep settings."""

    # The sample's shortest labeled shot is 32 frames. Half that duration
    # preserves a conservative margin while suppressing duplicate peaks.
    min_shot_frames: int = 15
    adaptive_window_width: int = 2
    adaptive_threshold_min: float = 0.5
    adaptive_threshold_max: float = 8.0
    adaptive_threshold_steps: int = 76
    adaptive_min_content_min: float = 0.5
    adaptive_min_content_max: float = 35.0
    adaptive_min_content_steps: int = 36
    content_threshold_min: float = 0.5
    content_threshold_max: float = 35.0
    content_threshold_steps: int = 70
    # TransNet under-fires on subtle same-background cuts, so sweep its small
    # probabilities logarithmically rather than only near the default 0.5.
    transnet_threshold_min: float = 0.000_01
    transnet_threshold_max: float = 0.9
    transnet_threshold_steps: int = 100
    transnet_device: str = "cpu"
    include_transnet: bool = True


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Phase 2 visual/audio feature extraction settings."""

    extractor_version: str = "1.1"
    # Landmarker input is reduced for CPU throughput; outputs remain normalized
    # and are projected back into source-pixel coordinates.
    face_input_width_px: int = 360
    # Flow/grayscale frames retain the source aspect ratio at this width.
    downscale_width_px: int = 96
    hsv_histogram_bins: int = 16
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # Dense Farneback flow settings, chosen for the small cached frame grid.
    flow_pyramid_scale: float = 0.5
    flow_levels: int = 3
    flow_window_size: int = 15
    flow_iterations: int = 3
    flow_poly_neighborhood: int = 5
    flow_poly_sigma: float = 1.2
    # Fine-grid audio representation used by Phase 3 boundary signals.
    audio_hop_ms: float = 5.0
    audio_window_ms: float = 25.0
    audio_fft_size: int = 2048
    audio_mel_bands: int = 40
    audio_mfcc_count: int = 13
    audio_mel_min_hz: float = 50.0
    audio_mel_max_hz: float = 12_000.0
    audio_high_band_min_hz: float = 6_000.0
    audio_high_band_max_hz: float = 12_000.0
    audio_f0_min_hz: float = 80.0
    audio_f0_max_hz: float = 400.0
    audio_voicing_threshold: float = 0.25
    overlay_landmark_stride: int = 2
    overlay_axis_length_px: int = 80
    overlay_codec: str = "mp4v"
    model_path: Path | None = None


@dataclass(frozen=True, slots=True)
class SignalConfig:
    """Phase 3 raw boundary-signal settings."""

    audio_multiscale_hops: tuple[int, ...] = (1, 4, 10)
    audio_boundary_radius_sec: float = 0.02
    noise_floor_context_sec: float = 0.20
    f0_context_sec: float = 0.04
    mfcc_context_sec: float = 0.02
    lpc_order: int = 12
    lpc_context_sec: float = 0.02
    lpc_evaluation_sec: float = 0.005
    blendshape_emphasis: float = 2.0
    flow_sample_stride: int = 4
    flow_inlier_mad_scale: float = 2.5
    background_variance_percentile: float = 85.0
    background_subject_width_scale: float = 1.8
    background_subject_top_scale: float = 0.25
    zoom_scale_log_threshold: float = 0.05
    stabilized_crop_width_scale: float = 1.8
    stabilized_crop_top_scale: float = 0.35
    stabilized_crop_bottom_scale: float = 2.5
    fixed_gop_cv_threshold: float = 0.05


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Exact local normalization, agreement, and peak-selection settings."""

    local_window_sec: float = 1.5
    center_exclusion_samples: int = 2
    robust_scale_constant: float = 1.4826
    epsilon: float = 0.000_001
    z_max: float = 8.0
    tau_high: float = 0.647_692_307_692_307_7
    tau_low: float = 0.550_538_461_538_461_5
    agreement_count: int = 3
    min_shot_frames: int = 15
    audio_association_radius_frames: int = 3


@dataclass(frozen=True, slots=True)
class TuningConfig:
    """Deterministic Phase 3 random-search ranges."""

    random_seed: int = 7
    trials: int = 100
    weight_log_sigma: float = 0.45
    window_seconds: tuple[float, ...] = (1.5, 2.0, 2.5)
    agreement_counts: tuple[int, ...] = (2, 3)
    min_shot_frames: tuple[int, ...] = (6, 10, 15)
    tau_high_min: float = 0.08
    tau_high_max: float = 0.90
    threshold_steps: int = 27
    tau_low_ratio_min: float = 0.55
    tau_low_ratio_max: float = 0.85
    tau_low_ratio_steps: int = 3


@dataclass(frozen=True, slots=True)
class ExportConfig:
    """Frame-accurate clip-export settings."""

    video_codec: str = "libx264"
    video_preset: str = "veryfast"
    video_crf: int = 18
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    filename_digits: int = 2
    ffmpeg_threads: int = 1
    ffmpeg_timeout_sec: float = 600.0


@dataclass(frozen=True, slots=True)
class StudioConfig:
    """Local drag-and-drop application settings."""

    host: str = "127.0.0.1"
    port: int = 8787
    browser_open_delay_sec: float = 0.35
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024
    upload_chunk_bytes: int = 1024 * 1024
    default_sensitivity: float = 0.5


DEFAULT_SIGNAL_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("pose_accel", 1.224_356_672_133_238_4),
    ("procrustes_residual", 0.359_802_648_344_462_4),
    ("centroid_accel", 1.222_108_594_890_204_6),
    ("scale_delta", 0.482_397_016_810_393_54),
    ("blendshape_delta", 0.421_720_792_748_830_84),
    ("stabilized_residual", 1.165_038_586_767_977_1),
    ("flow_incoherence", 0.954_901_534_580_772_3),
    ("background_delta", 0.466_672_166_725_525_14),
    ("content_val", 0.481_794_406_463_980_4),
    ("transnet_prob", 0.849_861_330_138_697_4),
    ("spectral_flux", 1.391_237_753_779_803_5),
    ("noise_floor_step", 1.225_865_436_630_107_6),
    ("lpc_residual_spike", 0.478_110_866_184_596_6),
    ("f0_discontinuity", 0.764_412_218_503_446_1),
    ("mfcc_delta", 0.538_473_488_906_569_1),
    ("iframe_prior", 0.207_975_298_551_237_95),
    ("word_gap_anomaly", 0.132_112_550_534_526_4),
)
