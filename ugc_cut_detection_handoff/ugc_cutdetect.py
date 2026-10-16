"""Small standalone CLI for the transferable detection and split core."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cutdetect.detect import run_detection
from cutdetect.export import split_from_predictions


def main() -> None:
    parser = argparse.ArgumentParser(prog="ugc-cutdetect")
    commands = parser.add_subparsers(dest="command", required=True)

    detect = commands.add_parser("detect", help="write cut predictions and diagnostics")
    detect.add_argument("video", type=Path)
    detect.add_argument("--output-dir", type=Path, required=True)
    detect.add_argument("--model", type=Path, help="MediaPipe face_landmarker.task")
    detect.add_argument("--cache-dir", type=Path)
    detect.add_argument("--features", type=Path, help="reuse an existing feature archive")
    detect.add_argument("--words", type=Path, help="optional word timestamp JSON")
    detect.add_argument("--sensitivity", type=float, default=0.5)

    split = commands.add_parser("split", help="export frame-accurate clips")
    split.add_argument("video", type=Path)
    split.add_argument("--predictions", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--cache-dir", type=Path)

    args = parser.parse_args()
    if args.command == "detect":
        result = run_detection(
            args.video,
            args.output_dir,
            model_path=args.model,
            cache_dir=args.cache_dir,
            features_path=args.features,
            words_path=args.words,
            sensitivity=args.sensitivity,
        )
    else:
        result = split_from_predictions(
            args.video,
            args.predictions,
            args.output_dir,
            cache_dir=args.cache_dir,
        )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
