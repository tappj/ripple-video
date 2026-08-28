# Ripple

A platform where you can clone UGC videos, swap the character, voicing, or products
and generate clips up to a minute+ long. Uses a cut detection tool to break up your
video into clips and generate each individually. Review and approve each clip to be
stitched back together. 

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

Open `http://127.0.0.1:8787/`

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
ELEVENLABS_API_KEY=
RUNWAY_SEEDANCE2_WORKFLOW_ID=fecc0662-a1e1-4941-b84a-76df0bde1e4f
RUNWAY_SEEDANCE25_WORKFLOW_ID=4af4fdf6-a371-4a73-b02d-fdbf116186d5
RUNWAY_HAILUO3_WORKFLOW_ID=55cd8a57-dd96-4401-b9d0-0ab506130f77
RUNWAY_PRODUCT_CLONE_WORKFLOW_ID=0b9a4bd0-27a2-4ef7-a2d3-ba1d89a8a0d0
```

Never commit `.env`, paste keys into documentation, or send them through chat. Exported
environment variables take precedence over values in `.env`.

`ELEVENLABS_API_KEY` enables per-clip Scribe v2 transcription. Ripple transcribes the
isolated original speech, caches it with that clip, and appends it to only the matching
video prompt. If the key or transcription service is unavailable, generation continues
with the transformed audio and the established prompt.

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
