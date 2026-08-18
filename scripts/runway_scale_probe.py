"""Submit minimal three-reference Seedance probes together and record queue behavior."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from runwayml import RunwayML
from runwayml.types.task_retrieve_response import Failed, Succeeded

from cutdetect.environment import load_env_file
from cutdetect.pipeline.gen_one import _slice_source
from cutdetect.pipeline.storage import LocalDiskStorage

SOURCE = Path("outputs/phase_c_grouped/jobs/12b1f12e24af475f8369ba061f1fe768/segments/0/input.mp4")
FACE = Path("outputs/phase_c_grouped/jobs/12b1f12e24af475f8369ba061f1fe768/refs/target_face.jpeg")
VOICE = Path("target-voice sample.mp3")
ROOT = Path(".cutdetect/pipeline/scale_probes")

CONSENT_PROMPT = (
    "Create a natural UGC talking-head clip using the supplied media from consenting adult "
    "performers. Keep the input video's exact camera, background, timing, cuts, gestures, "
    "facial performance, spoken words, and pacing. Render the performer as the adult woman "
    "in Image 1 and use Audio 1 only as the voice reference. No new dialogue or scene changes."
)
WORKFLOW_PROMPT = (
    "Use Video 1 as the reference video. Preserve the original video's camera angle, framing, "
    "background, timing, body movements, gestures, facial expressions, and pacing. Replace the "
    "person in Video 1 with the woman from Image 1. Use Audio 1 only as the voice reference; "
    "keep the exact spoken words, timing, and pacing from Video 1."
)


@dataclass
class Probe:
    name: str
    endpoint: str
    model: str
    prompt: str
    estimated_credits: int
    task_id: str | None = None
    terminal_status: str | None = None
    failure_code: str | None = None
    output_path: str | None = None


def main() -> None:
    load_env_file()
    api_key = os.environ.get("RUNWAYML_API_SECRET", "")
    if not api_key:
        raise RuntimeError("RUNWAYML_API_SECRET is not set")
    for asset in (SOURCE, FACE, VOICE):
        if not asset.is_file():
            raise FileNotFoundError(asset)

    run_id = uuid.uuid4().hex
    storage = LocalDiskStorage(ROOT / run_id)
    source_4s = storage.path("inputs/source_4s.mp4")
    _slice_source(SOURCE, source_4s, start_sec=0.0, end_sec=4.0, cache_dir=Path(".cutdetect/cache"))

    client = RunwayML(api_key=api_key, max_retries=0)
    org_before = client.organization.retrieve()
    source_uri = client.uploads.create_ephemeral(file=source_4s).uri
    face_uri = client.uploads.create_ephemeral(file=FACE).uri
    voice_uri = client.uploads.create_ephemeral(file=VOICE).uri
    images = [{"uri": face_uri}]
    audios = [{"type": "audio", "uri": voice_uri}]
    videos = [{"type": "video", "uri": source_uri}]

    probes = [
        Probe(
            "mini_video_to_video_three_inputs",
            "/v1/video_to_video",
            "seedance2_mini",
            CONSENT_PROMPT,
            64,
        ),
        Probe(
            "fast_video_to_video_three_inputs",
            "/v1/video_to_video",
            "seedance2_fast",
            WORKFLOW_PROMPT,
            116,
        ),
        Probe(
            "fast_text_to_video_three_references",
            "/v1/text_to_video",
            "seedance2_fast",
            CONSENT_PROMPT,
            116,
        ),
    ]

    # Intentionally submit every task before polling. This is Runway's documented
    # scaling pattern: overflow remains provider-side in THROTTLED state.
    for probe in probes:
        if probe.endpoint == "/v1/video_to_video":
            created = client.video_to_video.create(  # type: ignore[call-overload]
                model=probe.model,
                prompt_video=source_uri,
                prompt_text=probe.prompt,
                ratio="720:1280",
                duration=4,
                audio=True,
                references=images,
                reference_audio=audios,
            )
        else:
            created = client.text_to_video.create(  # type: ignore[call-overload]
                model=probe.model,
                prompt_text=probe.prompt,
                ratio="720:1280",
                duration=4,
                audio=True,
                references=images,
                reference_videos=videos,
                reference_audio=audios,
            )
        probe.task_id = created.id

    timeline: list[dict[str, object]] = []
    deadline = time.monotonic() + 30 * 60
    while any(probe.terminal_status is None for probe in probes):
        if time.monotonic() > deadline:
            break
        snapshot: dict[str, str] = {}
        for probe in probes:
            if probe.terminal_status is not None or probe.task_id is None:
                continue
            state = client.tasks.retrieve(probe.task_id)
            snapshot[probe.name] = state.status
            if isinstance(state, Failed):
                probe.terminal_status = "FAILED"
                probe.failure_code = state.failure_code
            elif isinstance(state, Succeeded):
                probe.terminal_status = "SUCCEEDED"
                if state.output:
                    path = storage.download_https(state.output[0], f"outputs/{probe.name}.mp4")
                    probe.output_path = str(path)
            elif state.status in {"CANCELLED", "CANCELED"}:
                probe.terminal_status = state.status
        timeline.append({"elapsed_sec": len(timeline) * 5, "statuses": snapshot})
        if any(probe.terminal_status is None for probe in probes):
            time.sleep(5)

    org_after = client.organization.retrieve()
    limit = org_before.tier.models["seedance2"]
    report = {
        "run_id": run_id,
        "organization": {
            "credit_balance_before": org_before.credit_balance,
            "credit_balance_after": org_after.credit_balance,
            "seedance_max_concurrent_generations": limit.max_concurrent_generations,
            "seedance_max_daily_generations": limit.max_daily_generations,
        },
        "estimated_credit_ceiling": sum(probe.estimated_credits for probe in probes),
        "probes": [asdict(probe) for probe in probes],
        "timeline": timeline,
    }
    report_path = storage.path("report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report_path": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
