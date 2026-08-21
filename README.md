# cutdetect

Frame-accurate jump-cut detection and clip export for talking-head video. `cutdetect`
combines face-motion discontinuities, stabilized pixels, optical flow, scene scores,
TransNetV2, and audio-splice evidence behind one sensitivity control.

## Cut Room interface

Install the working environment and official MediaPipe model, then open the local app:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[features,baselines,dev]'
mkdir -p .cutdetect/models
curl -L https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
  -o .cutdetect/models/face_landmarker.task
.venv/bin/cutdetect studio
```

Open `http://127.0.0.1:8787/`, drop an MP4/MOV/M4V/WEBM file, then download the
individual clips or a ZIP. Processing remains on the local machine. Thirteen edit
points yield fourteen clips because the boundaries divide the entire source timeline.

For the existing local installation, start it at any time with:

```bash
cd "/Users/jadontapp/Developer/Cut Detection"
.venv311/bin/cutdetect studio
```

Stop the server with `Ctrl+C`; run the same command to restart it.

## Secrets

Local secrets belong in `.env`, which is excluded from source control and must have
owner-only permissions. The CLI automatically reads this file from the directory where
it is started, without executing it as shell code. Copy the variable names from
`.env.example`, then add secret values only to `.env`.

```dotenv
RUNWAYML_API_SECRET=
RUNWAY_SEEDANCE2_WORKFLOW_ID=f28115cf-16bd-453f-9f3c-e766982951a4
RUNWAY_SEEDANCE25_WORKFLOW_ID=4af4fdf6-a371-4a73-b02d-fdbf116186d5
RUNWAY_HAILUO3_WORKFLOW_ID=9172f9ee-e4e9-4a25-92e1-29779d698556
```

Never commit `.env`, paste keys into documentation, or send them through chat. Exported
environment variables take precedence over values in `.env`.

## Runway regeneration pipeline

Phase A includes a no-charge preflight and a cost-capped single-segment proof. First,
preview the exact request duration, output ratio, internal cuts, and estimated credits:

```bash
.venv311/bin/cutdetect pipeline gen-one source.mp4 face.jpg voice.wav \
  --start 4 --end 12 --model seedance2 --dry-run
```

When the estimate is acceptable, use a hard credit ceiling. The 8-second 720p example
currently estimates 288 credits; the command refuses to upload or generate when the
estimate exceeds the ceiling.

```bash
.venv311/bin/cutdetect pipeline gen-one source.mp4 face.jpg voice.wav \
  --start 4 --end 12 --model seedance2 --max-credits 288
```

The output, immutable raw clip, job manifest, and JSONL API-call log are retained under
`.cutdetect/pipeline/jobs/<job_id>/`.

Plan boundary-preserving generation batches with:

```bash
.venv311/bin/cutdetect pipeline plan "Sample video.mp4" \
  --predictions eval/phase3/predictions.json --max-group-segments 4
```

Each generated clip contains **complete adjacent** cut sections and is at least 4 seconds.
The planner prefers clips no longer than 10 seconds and one to four source sections. If
that cannot absorb a short section, it may merge additional adjacent sections into a clip
up to the model-safe 15-second ceiling. Internal hard cuts remain in the source media sent
to the workflow: the planner never trims inside a section, removes a boundary, or adds a
transition. Set `--max-group-segments 1` for explicit no-grouping mode when every
individual section already satisfies the duration bounds.

An uninterrupted visual section over 15 seconds is subdivided near 10-second intervals.
The splitter chooses a detected silence first, then the lowest-energy nearby audio moment.
Only media with no decodable source audio uses the balanced time point as a last-resort
fallback. Short sections are merged with adjacent sections whenever the combined clip fits
within 15 seconds. If a sub-4-second remainder is mathematically impossible to merge, the
generation request is padded to the model minimum and the result is trimmed back to the
exact source duration.

## Durable parallel generation

Prepare a Phase C job locally before spending credits:

```bash
.venv311/bin/cutdetect pipeline prepare-job "Sample video.mp4" \
  "target-face sample 2.jpeg" "target-voice sample.mp3" \
  --predictions eval/phase3/predictions.json --consent
```

Preparation exports the exact `(4, 10]` second groups, writes the grouping manifest, and creates
the resumable SQLite job. It makes no API calls. Inspect it at any time with:

```bash
.venv311/bin/cutdetect pipeline job-status JOB_ID
```

Running is an explicit paid action. Inspect `estimated_credits` from `prepare-job`, then use
that exact value as the initial ceiling. A higher ceiling is required if automatic retries
should be allowed:

```bash
.venv311/bin/cutdetect pipeline run-job JOB_ID --max-credits ESTIMATED_CREDITS
```

New Hailuo 3, Seedance 2.0, and Seedance 2.5 jobs each use a dedicated published Runway Workflow.
Every Workflow is fixed to 9:16 and a 15-second maximum; Ripple sends each source group as an
independent invocation and trims the result to the source group's exact duration. The worker uploads
the face and voice once, gives every clip a separate source upload and a brand-new invocation,
submits every eligible clip before polling, saves outputs immediately, and records state transitions
in SQLite. Re-running the same command resumes existing invocation IDs instead of rebilling completed
clips. Use `--once` to advance one worker cycle and exit. Older Workflow/direct/router jobs remain
resumable.

## Review gate

After generation reaches `REVIEW`, inspect the complete gate:

```bash
.venv311/bin/cutdetect pipeline review-status JOB_ID
.venv311/bin/cutdetect pipeline review-suggest JOB_ID 0
.venv311/bin/cutdetect pipeline review-trim JOB_ID 0
.venv311/bin/cutdetect pipeline review-approve JOB_ID 0
```

`review-trim` uses the conservative silence-plus-low-motion suggestion unless an explicit
`--end-frame` is supplied. It writes `output_final.mp4` while keeping `output_raw.mp4`
unchanged. `review-approve-all` refuses to proceed if any clip is still running or failed.
An edited regeneration is queued separately and requires an explicitly increased ceiling:

```bash
.venv311/bin/cutdetect pipeline review-regenerate JOB_ID 0 \
  --prompt "EDITED PROMPT" --max-credits NEW_CEILING
```

The next `run-job` invocation submits only that queued segment and preserves the previous
raw attempt. Stitching remains locked until `review-status` reports `can_stitch: true`.

## Complete local platform

Start the Phase E interface with:

```bash
.venv311/bin/cutdetect pipeline studio
```

It opens at `http://127.0.0.1:8790/`. Stop it with `Ctrl+C`; running the same command
resumes from the durable local database. The interface covers reference uploads, cut
detection and grouping, editable versioned prompts, model/output settings, exact credit
confirmation, live task progress, source/output review, trim and approval, regeneration,
validated stitching, and final/QC downloads. The Runway key remains server-side in the
ignored `.env` file and is never included in browser responses.

## Host Ripple

Ripple now ships with a production container, a `/healthz` endpoint, environment-driven storage
paths, a configurable upload ceiling, and optional shared-password protection. The included
`render.yaml` deploys a free Render beta; it deliberately generates an access password so anonymous
visitors cannot spend the owner's Runway credits. Sign in with username `ripple` and the generated
password from the Render environment settings.

The free instance is suitable only for short beta tests: it has 512 MB RAM, sleeps after inactivity,
and loses uploaded inputs, SQLite jobs, and output videos whenever the instance restarts or sleeps.
Runway generation is also billed separately from hosting. See [DEPLOYMENT.md](DEPLOYMENT.md) for the
exact launch steps, router setup, free-tier limitations, and paid persistent options.

## Library API

```python
from cutdetect import detect_cuts, split_video

result = detect_cuts("video.mp4", sensitivity=0.5)
cut_frames = [cut["frame"] for cut in result["cuts"]]
export = split_video("video.mp4", cut_frames, "clips")
```

Sensitivity ranges from `0.0` (higher precision) to `1.0` (higher recall). The default
`0.5` favors recall while retaining the tuned reference operating point.

## CLI

```bash
cutdetect ingest video.mp4
cutdetect extract video.mp4 --model face_landmarker.task
cutdetect detect video.mp4 --sensitivity 0.5 --output-dir detection
cutdetect split video.mp4 --predictions detection/predictions.json --output-dir clips
cutdetect report video.mp4 --predictions detection/predictions.json \
  --normalized-signals detection/signals_normalized.npz --output report.html
cutdetect label video.mp4
cutdetect eval --labels labels.json --predictions detection/predictions.json
```

Adding `--labels labels.json` to `detect` runs the deterministic tuning search,
evaluation, trace plot, and leave-one-signal-out ablation. Phase 4 synthetic generation
is intentionally deferred.

## Prediction contract

The versioned JSON output contains source metadata, detected cuts, editor-ready
segments, diagnostics, and all effective parameters:

```json
{
  "schema_version": "1.0",
  "tool_version": "0.1.0",
  "video": {
    "path": "video.mp4",
    "duration_sec": 46.2,
    "fps": 30.0,
    "frame_count": 1386,
    "width": 720,
    "height": 1280,
    "was_vfr": false,
    "has_audio": true
  },
  "cuts": [{
    "frame": 88,
    "time_sec": 2.933333,
    "confidence": 0.76,
    "cut_type": "hard",
    "span": null,
    "audio_offset_frames": 1,
    "agreement_count": 11,
    "signals": {"pose_accel": 8.2}
  }],
  "segments": [{
    "index": 0,
    "start_frame": 0,
    "end_frame": 88,
    "start_sec": 0.0,
    "end_sec": 2.933333,
    "duration_sec": 2.933333
  }],
  "diagnostics": {
    "face_detection_rate": 1.0,
    "signals_available": ["pose_accel"],
    "signals_disabled": ["iframe_prior"],
    "disable_reasons": {"iframe_prior": "fixed GOP detected; I-frame prior uninformative"},
    "captions_region_masked": true
  },
  "params": {"sensitivity": 0.5, "fusion": {}, "weights": {}}
}
```

Frame `n` means the first frame after the edit. `segments` use half-open frame ranges:
`start_frame` is included and `end_frame` is excluded.
