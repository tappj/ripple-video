# Ripple

Ripple is a local-first tool for finding jump cuts in talking-head videos, splitting the source
into clips, and reviewing optional AI-assisted recreations in one browser interface.

![Ripple's empty project screen](docs/images/ripple-studio.png)

> [!IMPORTANT]
> AI generation uses paid third-party APIs. Detection and splitting stay local, but files sent for
> generation leave your computer. Use only media you have permission to process.

## Quick start

You need Python 3.11, FFmpeg, and a terminal. On macOS, install FFmpeg with
`brew install ffmpeg`; on Ubuntu/Debian, use `sudo apt install ffmpeg`.

```bash
git clone https://github.com/tappj/ripple-video.git
cd ripple-video
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[features]'
mkdir -p .cutdetect/models
curl -L https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
  -o .cutdetect/models/face_landmarker.task
cutdetect pipeline studio
```

Open <http://127.0.0.1:8790>, add your video and reference image, then follow the on-screen steps.
Press `Ctrl+C` in the terminal to stop Ripple. Local inputs and outputs are stored under
`.cutdetect/`, which Git ignores.

## Optional API setup

Cut detection works without API keys. AI recreation requires your own Runway account; optional
transcription uses ElevenLabs.

```bash
cp .env.example .env
chmod 600 .env
```

Add values only to `.env`. Never commit that file, screenshots containing credentials, or media
you do not have permission to share. Provider usage may cost money, so review the estimate before
confirming a generation job.

## What Ripple does

- Detects talking-head jump cuts with visual, motion, scene, and audio signals.
- Splits source video without dropping the frames between edits.
- Keeps raw generations immutable while you review, trim, approve, and assemble clips.
- Stores jobs locally so interrupted work can resume.
- Offers a CLI for detection, splitting, labeling, evaluation, and reports.

Run `cutdetect --help` or `cutdetect pipeline --help` for the command-line options.

## Development

```bash
python -m pip install -e '.[features,baselines,dev]'
pytest
ruff check .
mypy cutdetect
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Deployment notes are in
[DEPLOYMENT.md](DEPLOYMENT.md), and security issues should follow [SECURITY.md](SECURITY.md).

## Safety and privacy

- Get consent from anyone whose face, voice, or performance you use.
- Do not use Ripple for impersonation, fraud, harassment, or misleading content.
- Treat generated media as synthetic and disclose it where viewers could be confused.
- Keep `.env`, `.cutdetect/`, outputs, recordings, and source media out of Git.
- Use a spending limit and authentication before exposing a hosted instance.

## License

[MIT](LICENSE). Third-party models and services have their own terms and are not covered by this
license.
