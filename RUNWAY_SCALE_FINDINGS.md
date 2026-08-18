# Runway Seedance scale findings

Date: 2026-08-13

## Bottom line

The three-input UGC clone works through the published **Clone UGC Talking videos**
Workflow, but the same consented media is rejected by every tested public direct Seedance
route at the provider preprocessing layer. Prompt wording does not change that outcome.

The organization currently reports a hard limit of **1 concurrent video generation** and
**50 video generations per rolling day** for `seedance2`, `seedance2_fast`, and
`seedance2_mini`. Extra submissions are accepted as `THROTTLED` and Runway queues them.
No prompt, endpoint, workflow copy, or second API key in the same organization increases
that limit.

## Evidence

| Path | Inputs | Result |
|---|---|---|
| Published Workflow, Seedance 2 | video + face + voice + prompt | Succeeded; convincing clone |
| Direct Hailuo 3 | video + face + voice + prompt | Succeeded; convincing clone |
| Direct Seedance 2 text-to-video references | video + face + voice + prompt | `INPUT_PREPROCESSING.SAFETY.THIRD_PARTY` |
| Direct Seedance 2 video-to-video, source only | video | Same failure |
| Direct Seedance 2 video-to-video, neutral three-input prompt | video + face + voice | Same failure |
| Direct Seedance 2 Mini video-to-video | source only and three-input variants | Same failure |
| Direct Seedance 2 Fast video-to-video | video + face + voice | Same failure |
| Direct Seedance 2 Fast text-to-video references | video + face + voice | Same failure |

The final three probes were submitted together. Runway exposed overflow as `THROTTLED`;
all three then failed preprocessing. The API balance stayed at 11,831 credits, so these
three failed probes consumed no credits.

Detailed report:
`.cutdetect/pipeline/scale_probes/b8aad31e37f647c7acaee1b1f72cafae/report.json`

## Working production route

1. Keep the published Workflow as the Seedance execution route.
2. Change its fixed Seedance duration from 7 seconds to **8 seconds** and republish.
3. Group the 14 hard-cut sections into seven complete, boundary-aligned source clips:
   5.467s, 7.400s, 6.167s, 6.667s, 5.567s, 7.500s, and 7.433s.
4. Pad each Workflow input to 8 seconds with neighboring source footage, then trim each
   generated result back to its exact group boundary before review.
5. Submit all seven invocations immediately. Let Runway hold overflow as `THROTTLED`;
   never add a client-side one-at-a-time queue.

Seven 8-second 720p Seedance 2 calls cost an estimated **2,016 credits ($20.16)**.

## Getting 3–6 jobs to execute simultaneously

The live organization payload must report at least 3 concurrent generations. Runway's
documented self-serve tiers are 3 concurrent after $50 of qualifying API credit purchases
and 5 after $100. Six simultaneous jobs requires the next self-serve tier (10 concurrent
after $1,000) or a custom limits exception. This organization still reports Tier 1 despite
its current balance.

Check that the purchases were made inside the developer/API organization
`cb65dfcf-02ec-4efd-8ae9-4e03dffd071e`, not the regular Runway web app. If they were,
send Runway support the organization ID, purchase receipts, and the live limit response and
request a tier correction through the Usage page's limits exception form. Web-app credits
and API-organization credits are separate.
