# Deployment

Ripple is safest as a local application. Hosting it exposes uploads, generation controls, and a
paid API credential, so do not publish an unprotected instance.

## Render Blueprint

1. Fork the repository and create a Render Blueprint from `render.yaml`.
2. Add `RUNWAYML_API_SECRET` in Render's Environment settings.
3. Add `ELEVENLABS_API_KEY` only if you want transcription.
4. Set every `RUNWAY_*_WORKFLOW_ID` or router ID required by your own Runway setup.
5. Let Render generate `RIPPLE_ACCESS_PASSWORD`, then retrieve it from the service environment.
6. Verify `/healthz` returns `{"status": "ok"}` before uploading non-sensitive test media.

The login username is `ripple`; the password is the generated `RIPPLE_ACCESS_PASSWORD` value.

## Required precautions

- Never put API keys in GitHub, `render.yaml`, URLs, browser code, screenshots, or support logs.
- Keep password protection enabled while the server can spend API credits.
- Configure provider-side spending limits and use only test media you are allowed to upload.
- Use persistent storage for real workloads. Without it, restarts may erase uploads, job state,
  and generated files.
- Add individual accounts, quotas, abuse controls, object storage, and a managed job queue before
  serving untrusted public traffic.

## Configuration

Copy `.env.example` for the full list. The container uses `/data/ripple` for jobs and `/data/cache`
for cached analysis; `PORT` defaults to `10000` when the host does not supply one.

The included Docker image contains FFmpeg and downloads MediaPipe's Face Landmarker model during
the build. Build locally with `docker build -t ripple .` if you want to verify the image first.
