# Ripple hosting

## Recommended first launch: Render free beta

The repository is ready to deploy as a Docker web service. Render is the simplest current
zero-hosting-cost option, but the free service is a beta environment rather than production:

- 512 MB RAM and 0.1 CPU.
- It sleeps after 15 minutes without inbound traffic and can take about a minute to wake.
- Its filesystem is temporary. Uploads, SQLite state, outputs, and unfinished jobs disappear on a
  sleep, restart, or deployment.
- Runway API generations are never free just because the web host is free.

Official limitations: <https://render.com/docs/free> and
<https://render.com/docs/compute-plans>.

The image includes Python 3.11, FFmpeg, OpenCV, MediaPipe, and the official 3.6 MB Face Landmarker
model. The build context excludes every local secret, source/target media file, generated output,
cache, and virtual environment.

## One-time Runway setup

Publish one dedicated Runway Workflow for each selectable model. Each graph accepts the source
video, target face, optional target voice, and prompt, then exposes its generated video output.

| UI choice | Published Workflow ID |
|---|---|
| Seedance 2.0 | `f28115cf-16bd-453f-9f3c-e766982951a4` |
| Seedance 2.5 | `4af4fdf6-a371-4a73-b02d-fdbf116186d5` |
| Hailuo 3.0 | `9172f9ee-e4e9-4a25-92e1-29779d698556` |

All three graphs use 9:16 output and a 15-second maximum. Ripple overrides the exposed prompt for
every independent clip and trims each returned video to the source clip's exact duration.

## Account-side launch steps

These steps require the owner's browser login, so they cannot be completed by an unauthenticated
local coding session:

1. Authenticate GitHub from this machine with `gh auth login -h github.com`.
2. From this directory, push the already-prepared `main` commit to a private repository:

   ```bash
   gh repo create ripple-video --private --source=. --remote=origin --push
   ```

3. Create or sign in to a Render account, choose **New → Blueprint**, connect the private
   `ripple-video` repository, and approve the detected `render.yaml`.
4. When Render prompts for `RUNWAYML_API_SECRET`, paste it into the secret field. Never put the key
   in GitHub, `render.yaml`, a URL, or browser code.
5. Wait for the deploy to become healthy, then open the assigned `onrender.com` URL.
6. In the Render service's Environment page, reveal/copy the generated
   `RIPPLE_ACCESS_PASSWORD`. The browser login is:

   - Username: `ripple`
   - Password: the generated value

7. Verify `https://YOUR-SERVICE.onrender.com/healthz` returns `{"status": "ok"}`, then test with a
   small, non-sensitive video before inviting anyone.

The shared password can be changed at any time. Removing `RIPPLE_ACCESS_PASSWORD` makes the entire
app public, including its job list and generation controls. Do not do that while the server uses
your Runway key unless per-user authentication, quotas, abuse controls, and a Runway spending limit
are in place.

## Runtime configuration

| Variable | Default in container | Purpose |
|---|---|---|
| `RUNWAYML_API_SECRET` | none | Server-only Runway developer API credential |
| `RUNWAY_SEEDANCE2_WORKFLOW_ID` | published ID above | Seedance 2.0 Workflow |
| `RUNWAY_SEEDANCE25_WORKFLOW_ID` | published ID above | Seedance 2.5 Workflow |
| `RUNWAY_HAILUO3_WORKFLOW_ID` | published ID above | Hailuo 3.0 Workflow |
| `RUNWAY_PRODUCT_CLONE_WORKFLOW_ID` | published ID above | UGC product-clone Hailuo 3 Workflow |
| `RIPPLE_ACCESS_PASSWORD` | none | Optional shared beta password |
| `RIPPLE_MAX_UPLOAD_MIB` | `256` | Maximum size of each uploaded asset |
| `RIPPLE_DATA_DIR` | `/data/ripple` | Jobs, SQLite, uploads, and output media |
| `RIPPLE_CACHE_DIR` | `/data/cache` | Detection/feature cache |
| `PORT` | `10000` | Host-assigned HTTP port |

## Paid options for durable use

The existing Docker image can move without application changes:

1. **Railway Hobby** is the lowest-friction persistent option. The current plan costs $5/month and
   includes $5 of usage; additional RAM, CPU, storage, and egress are usage-billed. Configure at
   least 2 GB RAM and attach a volume mounted at `/data`. Official pricing:
   <https://docs.railway.com/pricing/plans>.
2. **Render Standard plus persistent disk** keeps the current Blueprint workflow. Change the
   service plan from Free to Standard (2 GB RAM, 1 CPU), attach at least a 10 GB disk at `/data`,
   and keep the two `RIPPLE_*_DIR` values unchanged. Official compute and disk documentation:
   <https://render.com/docs/compute-plans> and <https://render.com/docs/disks>.
3. **Hugging Face PRO Docker Space** costs $9/month for the required account plan; CPU Basic then
   has no hourly compute charge and supplies 2 CPU, 16 GB RAM, and 50 GB temporary disk. It still
   sleeps when unused and does not provide durable job storage, so it is better for demos than
   production. Official pricing and limits: <https://huggingface.co/pricing> and
   <https://huggingface.co/docs/hub/spaces-overview>.

For real public traffic, use persistent object storage for uploaded/generated video, move job state
to a managed database/queue, add individual accounts and per-user credit quotas, and run web and
generation workers separately. The current single-container mode is intentionally a protected beta.
