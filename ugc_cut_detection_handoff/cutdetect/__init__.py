"""Talking-head jump-cut detection."""

from cutdetect.detect import DetectionRun, detect_cuts, run_detection
from cutdetect.export import ExportResult, split_video
from cutdetect.features import FeatureExtractionResult, extract_features
from cutdetect.ingest import VideoContext, ingest_video, probe_video

__all__ = [
    "DetectionRun",
    "ExportResult",
    "FeatureExtractionResult",
    "VideoContext",
    "detect_cuts",
    "extract_features",
    "ingest_video",
    "probe_video",
    "run_detection",
    "split_video",
]
__version__ = "0.1.0"
