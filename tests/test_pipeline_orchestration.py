from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from cutdetect.pipeline.capabilities import credit_cost
from cutdetect.pipeline.grouping import AtomicSegment, group_atomic_segments
from cutdetect.pipeline.orchestration import (
    DIRECT_API_ROUTE,
    JobState,
    PhaseCStore,
    PhaseCWorker,
    Retry,
    SegmentState,
    format_sse_events,
)
from cutdetect.pipeline.runway_client import GenerationPoll, GenerationRequest
from cutdetect.pipeline.workflow_client import PRODUCT_CLONE_WORKFLOW, SEEDANCE25_WORKFLOW

SEGMENT_CREDITS = 102


class FakeGateway:
    def __init__(
        self,
        outcomes: Callable[[int], list[GenerationPoll]],
        output_root: Path,
    ) -> None:
        self.outcomes = outcomes
        self.output_root = output_root
        self.actions: list[str] = []
        self.uploads: list[str] = []
        self.submissions: list[str] = []
        self.polls: dict[str, list[GenerationPoll]] = {}

    def upload(self, path: Path, *, role: str) -> str:
        self.actions.append(f"upload:{role}")
        self.uploads.append(role)
        return f"runway://{role}"

    def submit(self, request: GenerationRequest) -> str:
        invocation_id = f"invocation-{len(self.submissions)}"
        self.actions.append(f"submit:{invocation_id}")
        self.submissions.append(invocation_id)
        self.polls[invocation_id] = self.outcomes(len(self.submissions) - 1)
        assert request.reference_video.startswith("runway://segment_")
        assert request.reference_image == "runway://target_face"
        assert request.reference_audio == "runway://target_voice"
        assert request.prompt_text
        assert request.model == "hailuo3"
        assert request.duration == 5
        assert request.ratio == "9:16"
        assert request.resolution == "768P"
        return invocation_id

    def poll(self, invocation_id: str) -> GenerationPoll:
        self.actions.append(f"poll:{invocation_id}")
        outcomes = self.polls[invocation_id]
        return outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]

    def download(self, url: str, destination_key: str) -> Path:
        self.actions.append(f"download:{destination_key}")
        destination = self.output_root / destination_key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(url.encode())
        return destination


def _create_job(
    store: PhaseCStore,
    root: Path,
    count: int,
    *,
    job_id: str = "test-job",
    owner_device_hash: str | None = None,
    model_id: str = "hailuo3",
    route_id: str = DIRECT_API_ROUTE,
    resolution: str = "768P",
    product: bool = False,
) -> str:
    source = root / f"{job_id}-source.mp4"
    face = root / f"{job_id}-face.jpg"
    voice = root / f"{job_id}-voice.mp3"
    product_image = root / f"{job_id}-product.jpg"
    for path in (source, face, voice, product_image):
        path.write_bytes(b"test")
    segments = tuple(
        AtomicSegment(
            index=index,
            start_frame=index * 150,
            end_frame=(index + 1) * 150,
            start_sec=float(index * 5),
            end_sec=float((index + 1) * 5),
            duration_sec=5.0,
        )
        for index in range(count)
    )
    grouping = group_atomic_segments(
        segments, model_id=model_id, target_sec=5, max_group_segments=1
    )
    inputs = []
    outputs = []
    for index in range(count):
        path = root / "jobs" / job_id / "segments" / str(index) / "input.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"segment")
        inputs.append(path)
        outputs.append(f"jobs/{job_id}/segments/{index}/output_raw.mp4")
    store.create_job(
        job_id=job_id,
        source_path=source,
        target_face_path=face,
        target_voice_path=voice,
        target_product_path=product_image if product else None,
        prompt="test prompt",
        grouping=grouping,
        input_paths=inputs,
        output_keys=outputs,
        model_id=model_id,
        ratio="9:16",
        resolution=resolution,
        route_id=route_id,
        owner_device_hash=owner_device_hash,
    )
    return job_id


def test_workflow_jobs_request_fixed_maximum_then_trim_to_source_duration(
    tmp_path: Path,
) -> None:
    store = PhaseCStore(tmp_path / "jobs.sqlite3")
    job_id = _create_job(
        store,
        tmp_path,
        1,
        model_id="seedance2_5",
        route_id=SEEDANCE25_WORKFLOW.route_id,
        resolution="720p",
    )

    job = store.job(job_id)
    segment = store.segments(job_id)[0]
    assert job.route_id == SEEDANCE25_WORKFLOW.route_id
    assert segment.duration_sec == 5
    assert segment.requested_duration_sec == 15
    assert segment.estimated_credits == credit_cost(
        "seedance2_5", 15, "720p", reference_video_duration_s=5
    )


def test_product_clone_workflow_is_a_distinct_hailuo_route(tmp_path: Path) -> None:
    store = PhaseCStore(tmp_path / "jobs.sqlite3")
    job_id = _create_job(
        store,
        tmp_path,
        1,
        model_id="hailuo3",
        route_id=PRODUCT_CLONE_WORKFLOW.route_id,
        resolution="768P",
        product=True,
    )

    job = store.job(job_id)
    segment = store.segments(job_id)[0]
    assert job.target_product_path is not None
    assert job.route_id == PRODUCT_CLONE_WORKFLOW.route_id
    assert segment.requested_duration_sec == 15
    store.close()


def test_jobs_are_isolated_by_device_identity(tmp_path: Path) -> None:
    store = PhaseCStore(tmp_path / "jobs.sqlite3")
    first_hash = "a" * 64
    second_hash = "b" * 64
    first_job = _create_job(
        store,
        tmp_path,
        1,
        job_id="a" * 32,
        owner_device_hash=first_hash,
    )
    second_job = _create_job(
        store,
        tmp_path,
        1,
        job_id="b" * 32,
        owner_device_hash=second_hash,
    )

    assert [job.id for job in store.jobs_for_device(first_hash)] == [first_job]
    assert [job.id for job in store.jobs_for_device(second_hash)] == [second_job]
    assert store.job_owned_by(first_job, first_hash)
    assert not store.job_owned_by(first_job, second_hash)
    store.close()


def test_submits_every_segment_before_polling_and_reuses_references(tmp_path: Path) -> None:
    store = PhaseCStore(tmp_path / "jobs.sqlite3")
    job_id = _create_job(store, tmp_path, 3)
    store.confirm(job_id, 3 * SEGMENT_CREDITS)
    gateway = FakeGateway(
        lambda _index: [GenerationPoll("SUCCEEDED", output_urls=("https://output",))],
        tmp_path,
    )

    job = PhaseCWorker(store=store, gateway=gateway).run_once(job_id)

    assert job.state == JobState.REVIEW
    assert gateway.uploads.count("target_face") == 1
    assert gateway.uploads.count("target_voice") == 1
    assert len(gateway.submissions) == 3
    first_poll = next(
        index for index, action in enumerate(gateway.actions) if action.startswith("poll:")
    )
    last_submit = max(
        index for index, action in enumerate(gateway.actions) if action.startswith("submit:")
    )
    assert last_submit < first_poll
    assert all(segment.state == SegmentState.READY_FOR_REVIEW for segment in store.segments(job_id))
    assert store.job(job_id).submitted_credits == 3 * SEGMENT_CREDITS

    event_text = "".join(format_sse_events(store.events_since(job_id)))
    assert "event: segment.submitted" in event_text
    assert "event: job.state" in event_text
    store.close()


def test_preset_voice_is_mastered_once_and_sent_before_video_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = PhaseCStore(tmp_path / "jobs.sqlite3")
    source = tmp_path / "source.mp4"
    face = tmp_path / "face.jpg"
    segment_path = tmp_path / "segment.mp4"
    for path in (source, face, segment_path):
        path.write_bytes(b"test")
    grouping = group_atomic_segments(
        (
            AtomicSegment(
                index=0,
                start_frame=0,
                end_frame=150,
                start_sec=0,
                end_sec=5,
                duration_sec=5,
            ),
        ),
        model_id="hailuo3",
        target_sec=5,
        max_group_segments=1,
    )
    store.create_job(
        job_id="preset-job",
        source_path=source,
        target_face_path=face,
        target_voice_path=None,
        prompt="test prompt",
        grouping=grouping,
        input_paths=(segment_path,),
        output_keys=("jobs/preset-job/segments/0/output_raw.mp4",),
        model_id="hailuo3",
        ratio="9:16",
        resolution="768P",
        voice_preset_id="Maya",
    )
    store.confirm("preset-job", 999)

    class VoiceProcessor:
        calls = 0

        def convert_source_voice(
            self, source_video: Path, *, preset_id: str, job_id: str
        ) -> Path:
            self.calls += 1
            assert source_video == source
            assert preset_id == "Maya"
            track = tmp_path / "voice-master.mp3"
            track.write_bytes(b"voice")
            return track

    class Gateway:
        def upload(self, _path: Path, *, role: str) -> str:
            return f"runway://{role}"

        def submit(self, request: GenerationRequest) -> str:
            assert request.reference_audio == "runway://segment_0_voice"
            assert "Audio 1 is the final dialogue track" in request.prompt_text
            return "video-task"

        def poll(self, _task_id: str) -> GenerationPoll:
            return GenerationPoll("SUCCEEDED", output_urls=("https://video",))

        def download(self, _url: str, destination_key: str) -> Path:
            destination = tmp_path / destination_key
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"provider video")
            return destination

    def fake_slice(
        _audio: Path, destination: Path, *, start_sec: float, duration_sec: float
    ) -> Path:
        assert start_sec == 0
        assert duration_sec == 5
        destination.write_bytes(b"clip voice")
        return destination

    def fake_trim(_generated: Path, duration_sec: float, destination: Path) -> Path:
        assert duration_sec == 5
        destination.write_bytes(b"final video")
        return destination

    monkeypatch.setattr("cutdetect.pipeline.orchestration.slice_audio_track", fake_slice)
    monkeypatch.setattr("cutdetect.pipeline.orchestration.trim_generated_duration", fake_trim)
    processor = VoiceProcessor()
    job = PhaseCWorker(
        store=store,
        gateway=Gateway(),
        audio_processor=processor,
    ).run_once("preset-job")

    assert job.state == JobState.REVIEW
    assert processor.calls == 1
    assert store.job("preset-job").voice_preset_id == "Maya"
    assert store.job("preset-job").audio_state == "READY"
    store.close()


def test_reopens_database_and_resumes_existing_invocations(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    store = PhaseCStore(database)
    job_id = _create_job(store, tmp_path, 2)
    store.confirm(job_id, 2 * SEGMENT_CREDITS)
    gateway = FakeGateway(
        lambda _index: [
            GenerationPoll("RUNNING", progress=0.2),
            GenerationPoll("SUCCEEDED", output_urls=("https://output",)),
        ],
        tmp_path,
    )

    first = PhaseCWorker(store=store, gateway=gateway).run_once(job_id)
    assert first.state == JobState.RUNNING
    assert len(gateway.submissions) == 2
    store.close()

    reopened = PhaseCStore(database)
    second = PhaseCWorker(store=reopened, gateway=gateway).run_once(job_id)

    assert second.state == JobState.REVIEW
    assert len(gateway.submissions) == 2
    assert gateway.uploads.count("target_face") == 1
    assert all(
        segment.state == SegmentState.READY_FOR_REVIEW for segment in reopened.segments(job_id)
    )
    reopened.close()


def test_failed_clip_retries_without_resubmitting_successes(tmp_path: Path) -> None:
    store = PhaseCStore(tmp_path / "jobs.sqlite3")
    job_id = _create_job(store, tmp_path, 2)
    store.confirm(job_id, 3 * SEGMENT_CREDITS)

    def outcomes(submission_index: int) -> list[GenerationPoll]:
        if submission_index == 0:
            return [
                GenerationPoll(
                    "FAILED",
                    failure_code="INTERNAL.BAD_OUTPUT.01",
                    failure_message="bad output",
                )
            ]
        return [GenerationPoll("SUCCEEDED", output_urls=("https://output",))]

    gateway = FakeGateway(outcomes, tmp_path)
    immediate_retry = {
        "INTERNAL.BAD_OUTPUT": Retry(2, 0),
        None: Retry(0, 0),
    }
    worker = PhaseCWorker(store=store, gateway=gateway, retry_policy=immediate_retry)

    first = worker.run_once(job_id)
    first_segments = store.segments(job_id)
    assert first.state == JobState.RUNNING
    assert first_segments[0].state == SegmentState.PENDING
    assert first_segments[1].state == SegmentState.READY_FOR_REVIEW

    second = worker.run_once(job_id)
    final_segments = store.segments(job_id)
    assert second.state == JobState.REVIEW
    assert final_segments[0].attempt_count == 2
    assert final_segments[1].attempt_count == 1
    assert gateway.uploads.count("segment_1") == 1
    assert gateway.uploads.count("segment_0") == 1
    assert store.job(job_id).submitted_credits == 3 * SEGMENT_CREDITS
    store.close()


def test_confirmation_does_not_impose_an_internal_credit_ceiling(tmp_path: Path) -> None:
    store = PhaseCStore(tmp_path / "jobs.sqlite3")
    job_id = _create_job(store, tmp_path, 2)

    confirmed = store.confirm(job_id, 0)

    assert confirmed.state == JobState.CONFIRMED
    assert confirmed.max_credits is None
    assert store.has_submission_budget(job_id, 0) is True
    store.close()
