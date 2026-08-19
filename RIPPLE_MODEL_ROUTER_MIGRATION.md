# Ripple generation-route migration

Updated: 2026-08-19

## Required Runway configuration

Ripple exposes an explicit model selector. Seedance 2 is available in Runway's Model Router
catalog, so its deterministic user choice uses one single-model router. Hailuo 3 is available to
the organization and direct video-to-video API but does not appear in the router catalog; Hailuo
therefore stays pinned through its direct API endpoint.

| UI choice | Name | Config ID | Eligible-model mode | Models | Optimize for |
|---|---|---|---|---|---|
| Seedance 2 | `Ripple — Seedance 2` | `ripple-seedance-2` | Allow list | `seedance2` only | Quality |

Use these descriptions:

- `Ripple production video routing pinned to Seedance 2 for independent video + face + voice recreations.`

Leave the router-level maximum video credits unset. Ripple calculates every clip's model-specific
cost and enforces a hard total job ceiling before paid submissions. A fixed per-generation router
cap would incorrectly reject valid high-resolution or longer clips.

The config ID above works without more local changes. If a different immutable config ID is used,
set it in `.env`:

```dotenv
RUNWAY_SEEDANCE_ROUTER_CONFIG_ID=
```

## Architecture change

New Seedance jobs use Runway's `POST /v1/generate/video` Model Router endpoint. New Hailuo jobs use
the direct `hailuo3` video-to-video contract with source video, face image, and optional voice
reference. The old published Seedance Workflow and previously saved routes remain resumable.

Before any paid Seedance routed task, Ripple:

1. Confirms the selected config exists and is allow-listed to exactly the selected model.
2. Uploads the original face and optional voice references once for the job.
3. Uploads each original source clip separately.
4. Sends a no-charge router dry run with the exact live payload.
5. Verifies the resolved model and current credit estimate.
6. Creates a brand-new Runway task for that clip.

Hailuo skips router validation but still uploads each source clip separately and creates a new
direct task for every clip. It never feeds a generated output into a later task.

Every clip task receives only its own original source section, the original target references, and
the current prompt. A generated clip is never used as the source for another clip or retry. Task
IDs, attempts, inputs, and raw outputs remain separate and durable.

## Parallel generation

Ripple already submits every eligible clip before polling any of them. It has no one-at-a-time
client queue, so additional tasks can enter Runway as `THROTTLED` and begin automatically as the
organization has capacity.

The live organization currently reports a concurrency limit of **1** and a rolling limit of **50**
daily generations for both `seedance2` and `hailuo3`. All video models share the organization's
video concurrency pool; changing routes or using two API keys in the same organization does not
increase it.

Runway's current self-serve thresholds are:

| Tier | Concurrent video generations | Qualifying API credit purchases |
|---|---:|---:|
| 1 | 1 | Default |
| 2 | 3 | $50 |
| 3 | 5 | $100 |
| 4 | 10 | $1,000 |
| 5 | 20 | $5,000 |

Credits must be purchased in the same Runway developer/API organization, not the regular Runway web
application. If this organization has already met the purchase threshold but still reports one
concurrent generation, submit a limits exception from the developer portal Usage page with the
organization ID, qualifying receipts, and the live limit response.

## Compatibility note

Earlier direct Seedance 2 probes with the same consented video, image, and audio failed provider
preprocessing while the published Workflow succeeded. The routed path is therefore guarded by a
dry run and preserves the old Workflow implementation for existing jobs. A dry run validates model
eligibility and cost, but the first real Seedance routed generation is still the definitive content
preprocessing test.
