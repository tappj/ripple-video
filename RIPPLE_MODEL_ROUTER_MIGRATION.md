# Ripple Model Router migration

Date: 2026-08-16

## Required Runway configuration

Ripple exposes an explicit model selector. Because a Model Router normally chooses a model for
the caller, deterministic user selection requires one single-model router per choice.

| UI choice | Name | Config ID | Eligible-model mode | Models | Optimize for |
|---|---|---|---|---|---|
| Seedance 2 | `Ripple — Seedance 2` | `ripple-seedance-2` | Allow list | `seedance2` only | Quality |
| Hailuo 3 | `Ripple — Hailuo 3` | `ripple-hailuo-3` | Allow list | `hailuo3` only | Quality |

Use these descriptions:

- Seedance: `Ripple production video routing pinned to Seedance 2 for independent video + face + voice recreations.`
- Hailuo: `Ripple production video routing pinned to Hailuo 3 for independent video + face + voice recreations.`

Leave the router-level maximum video credits unset. Ripple calculates every clip's model-specific
cost and enforces a hard total job ceiling before paid submissions. A fixed per-generation router
cap would incorrectly reject valid high-resolution or longer clips.

The config IDs above work without more local changes. If different immutable config IDs are used,
set them in `.env`:

```dotenv
RUNWAY_SEEDANCE_ROUTER_CONFIG_ID=
RUNWAY_HAILUO_ROUTER_CONFIG_ID=
```

## Architecture change

New Ripple jobs now use Runway's `POST /v1/generate/video` Model Router endpoint for both model
choices. The old published Seedance Workflow and direct Hailuo route remain in the code only so
already-created jobs can resume safely.

Before any paid routed task, Ripple now:

1. Confirms the selected config exists and is allow-listed to exactly the selected model.
2. Uploads the original face and optional voice references once for the job.
3. Uploads each original source clip separately.
4. Sends a no-charge router dry run with the exact live payload.
5. Verifies the resolved model and current credit estimate.
6. Creates a brand-new Runway task for that clip.

Every clip task receives only its own original source section, the original target references, and
the current prompt. A generated clip is never used as the source for another clip or retry. Task
IDs, attempts, inputs, and raw outputs remain separate and durable.

## Parallel generation

Ripple already submits every eligible clip before polling any of them. It has no one-at-a-time
client queue, so additional tasks can enter Runway as `THROTTLED` and begin automatically as the
organization has capacity.

The live organization currently reports a concurrency limit of **1** and a rolling limit of **50**
daily generations for both `seedance2` and `hailuo3`. All video models share the organization's
video concurrency pool; using two routers or two API keys in the same organization does not increase
it.

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
