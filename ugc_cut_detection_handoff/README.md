# Cut detection handoff for a UGC / TikTok video editor

This is a portable copy of the cut detection and frame-accurate splitting core from Ripple, prepared for another AI agent to integrate into a UGC editor. The source files are copied without algorithm changes. The new `ugc_cutdetect.py` and this package metadata give the copied core a standalone CLI with no Runway or recreation pipeline dependency.

## Start here

1. Use Python 3.11 and install FFmpeg so `ffmpeg` and `ffprobe` are on `PATH`.
2. From this directory run `python3.11 -m venv .venv`, activate it, then run `python -m pip install -e '.[features,dev]'`.
3. Download the MediaPipe Face Landmarker `face_landmarker.task` from the official model URL shown in `cutdetect/features.py` and supply it with `--model`. The model binary is intentionally absent from this archive.
4. Run `ugc-cutdetect detect input.mp4 --model /path/to/face_landmarker.task --output-dir detection --cache-dir cache`.
5. Read `detection/predictions.json`, review boundaries in the new editor, and optionally run `ugc-cutdetect split input.mp4 --predictions detection/predictions.json --output-dir clips --cache-dir cache`.
6. Run `python -m pytest` for the included core regression tests. They use synthetic inputs, not private media.

`python ugc_cutdetect.py ...` also works from this directory after dependencies are installed. `--sensitivity` ranges from 0 to 1; larger values lower the detection threshold. The default is 0.5. The CLI prints the artifact paths as JSON.

## What is inside

| File | Role |
| --- | --- |
| `cutdetect/ingest.py` | Probe source media, normalize variable frame rate, preserve mapping to original timestamps, extract mono audio, and cache by content hash. |
| `cutdetect/features.py` | Extract facial landmarks, motion, background/color, and audio features. Requires OpenCV, MediaPipe, SciPy, and the separate Face Landmarker model. |
| `cutdetect/signals.py` | Turn frame and audio features into candidate boundary evidence, with diagnostics for unavailable signals. |
| `cutdetect/fusion.py` | Locally normalize signals, combine available evidence, select peaks, and enforce minimum shot length. |
| `cutdetect/detect.py` | Orchestrate detection and write the prediction contract and traces. |
| `cutdetect/export.py` | Validate cut frames and re-encode exact half-open frame ranges to clips. |
| `cutdetect/config.py` | All defaults, including signal weights and thresholds. |
| `cutdetect/evaluation.py` | Label loading and frame-tolerance scoring used by optional tuning. |
| `ugc_cutdetect.py` | Focused detect/split CLI for this handoff. |
| `tests/` | Core ingest, feature, signal, fusion, and export regression tests. |

The original studio UI, AI generation/recreation pipeline, deployment configuration, model binaries, videos, caches, output clips, and credentials are outside this archive.

## Prediction contract and editor integration

`predictions.json` has `schema_version: "1.0"`. Its useful fields are:

- `video`: source path, FPS, dimensions, frame count, duration, audio and variable-rate flags.
- `cuts[]`: each cut has `frame`, `time_sec`, `confidence`, `cut_type`, `audio_offset_frames`, `agreement_count`, and per-signal evidence. `frame` is the first frame of the next segment; it is never frame 0.
- `segments[]`: complete ordered coverage as `[start_frame, end_frame)` ranges, with source times. An `end_frame` is exclusive.
- `diagnostics`: face detection rate, available and disabled signals, reasons, and caption masking status.
- `params`: sensitivity and exact fusion weights/settings used for the run.

For a timeline, store frame indices as the authority and render seconds only for display. For variable-frame-rate sources, keep the ingest mapping to original presentation timestamps; a simple `frame / fps` conversion is only valid on the normalized working timeline. Give editors a way to move, add, and remove suggested cuts. Call `split_video(video_path, approved_cut_frames, output_dir)` if your UI has an edited boundary list, or `split_from_predictions(...)` for the raw result. Ensure the prediction source and split source match: `split_from_predictions` validates boundary ranges but does not verify video identity.

Python API example:

```python
from cutdetect.detect import run_detection
from cutdetect.export import split_video

run = run_detection(
    "input.mp4", "detection", model_path="face_landmarker.task", sensitivity=0.5
)
# After reviewing detection/predictions.json:
result = split_video("input.mp4", [120, 252], "clips")
```

Detection writes `predictions.json`, raw and normalized signal `.npz` files, `signal_traces.svg`, and `ablation.json`. Labels are optional; passing them enables same-clip tuning and evaluation. Treat same-clip scores as debugging evidence, not a generalization estimate.

## Adapting to UGC footage

The current weights and minimum shot length were tuned for talking-head jump cuts. They can miss rapid montage cuts, B-roll changes, picture-in-picture edits, animated text, dissolves, and cuts under heavy effects; they can also respond to flashes or camera motion. Preserve this detector as one source of candidate cuts, then benchmark on your own vertical UGC sample with hand-labeled boundaries. Measure precision and recall separately for talking head, B-roll/product shots, overlay-heavy sections, and transitions. Retune `FusionConfig.min_shot_frames`, signal weights, and thresholds from that labeled set; add a scene-change path for footage without a visible face. Do not treat a word pause or visual cut alone as an instruction to remove content: pacing, semantic relevance, and final edit decisions belong to the editor layer.

The face model and FFmpeg are external requirements. There are no API keys or paid model calls in this package. The license in `LICENSE.md` must travel with copied source; review third-party model terms separately.
