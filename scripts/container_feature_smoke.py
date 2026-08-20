"""Exercise Ripple's native feature runtime while building its container."""

from __future__ import annotations

import ctypes
import importlib
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    """Load native graphics libraries and run one CPU FaceLandmarker inference."""
    model_path = Path(sys.argv[1])
    ctypes.CDLL("libEGL.so.1")
    ctypes.CDLL("libGLESv2.so.2")

    cv2 = importlib.import_module("cv2")
    if not cv2.getBuildInformation():
        raise RuntimeError("OpenCV did not report build information")
    mp = importlib.import_module("mediapipe")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.zeros((64, 64, 3), dtype=np.uint8),
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        landmarker.detect_for_video(image, 0)


if __name__ == "__main__":
    main()
