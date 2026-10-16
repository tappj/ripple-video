# Deployment

Ripple can run locally or as a public demo. Hosting exposes uploads and paid generation controls;
the server applies the limits described in [PUBLIC_DEMO_LIMITS.md](PUBLIC_DEMO_LIMITS.md).

## Render Blueprint

1. Fork the repository and create a Render Blueprint from `render.yaml`.
2. Add `RUNWAYML_API_SECRET` in Render's Environment settings.
3. Optional direct transcription requires both `ELEVENLABS_API_KEY` and
   `RIPPLE_ENABLE_TRANSCRIPTION=true`; it is disabled by default and billed separately.
4. Set every `RUNWAY_*_WORKFLOW_ID` or router ID required by your own Runway setup.
5. Leave `RIPPLE_ACCESS_PASSWORD` unset for public access, or set it for a private instance.
6. Verify `/healthz` returns `{"status": "ok"}` before uploading non-sensitive test media.

If password protection is configured, the login username is `ripple` and the password is the
`RIPPLE_ACCESS_PASSWORD` value.

## Required precautions

- Never put API keys in GitHub, `render.yaml`, URLs, browser code, screenshots, or support logs.
- Keep Runway automatic refill disabled for a demo funded by a fixed prepaid balance.
- Configure provider-side spending limits and use only test media you are allowed to upload.
- Use persistent storage for real workloads. Without it, restarts may erase uploads, job state,
  and generated files.
- The demo limits are per server process. A larger public service needs individual accounts,
  durable quotas, object storage, and a shared job queue.

## Configuration

Copy `.env.example` for the full list. The container uses `/data/ripple` for jobs and `/data/cache`
for cached analysis; `PORT` defaults to `10000` when the host does not supply one.

The included Docker image contains FFmpeg and downloads MediaPipe's Face Landmarker model during
the build. Build locally with `docker build -t ripple .` if you want to verify the image first.
