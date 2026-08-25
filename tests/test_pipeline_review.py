from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import av
import pytest

from cutdetect.pipeline.grouping import AtomicSegment, group_atomic_segments
from cutdetect.pipeline.orchestration import JobState, PhaseCStore, SegmentState
from cutdetect.pipeline.review import ReviewService, inspect_media, prepare_review_proxy
from cutdetect.pipeline.runway_client import PipelineError
from cutdetect.pipeline.stitch import stitch_job
from cutdetect.pipeline.storage import LocalDiskStorage


def _media_fixture(path: Path, *, pixel_format: str = "yuv420p") -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for media review tests")
    completed = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x284:r=30:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1.2",
            "-filter_complex",
            "[1:a]apad=pad_dur=0.8[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-t",
            "2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            pixel_format,
            "-c:a",
            "aac",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _review_job(tmp_path: Path) -> tuple[PhaseCStore, LocalDiskStorage, str, Path]:
    storage = LocalDiskStorage(tmp_path / "review")
    store = PhaseCStore(storage.path("orchestration.sqlite3"))
    source = tmp_path / "source.mp4"
    _media_fixture(source)
    face = tmp_path / "face.jpeg"
    voice = tmp_path / "voice.mp3"
    face.write_bytes(b"face")
    voice.write_bytes(b"voice")
    segment = AtomicSegment(0, 0, 150, 0.0, 5.0, 5.0)
    grouping = group_atomic_segments(
        (segment,), model_id="hailuo3", target_sec=5, max_group_segments=1
    )
    raw_key = "jobs/review-job/segments/0/output_raw.mp4"
    input_path = storage.copy_in(source, "jobs/review-job/segments/0/input.mp4")
    store.create_job(
        job_id="review-job",
        source_path=source,
        target_face_path=face,
        target_voice_path=voice,
        prompt="keep the performance",
        grouping=grouping,
        input_paths=(input_path,),
        output_keys=(raw_key,),
        model_id="hailuo3",
        ratio="9:16",
        resolution="768P",
    )
    store.confirm("review-job", store.job("review-job").estimated_credits)
    store.mark_submitted("review-job", 0, "task-one")
    raw = storage.copy_in(source, raw_key)
    store.set_segment_state("review-job", 0, SegmentState.READY_FOR_REVIEW)
    store.set_job_state("review-job", JobState.REVIEW)
    return store, storage, "review-job", raw


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_suggest_trim_approve_and_stitch_gate_are_reversible(tmp_path: Path) -> None:
    store, storage, job_id, raw = _review_job(tmp_path)
    review = ReviewService(store=store, storage=storage)
    original_hash = _sha256(raw)

    suggestion = review.suggest(job_id, 0)
    assert suggestion.end_frame < suggestion.original_end_frame
    assert suggestion.trailing_silence_start_sec == pytest.approx(1.2, abs=0.08)

    trimmed = review.trim(
        job_id,
        0,
        start_frame=suggestion.start_frame,
        end_frame=suggestion.end_frame,
    )
    assert trimmed.state == SegmentState.READY_FOR_REVIEW
    assert trimmed.final_output_key is not None
    assert _sha256(raw) == original_hash
    final_path = storage.path(trimmed.final_output_key)
    assert inspect_media(final_path).duration_sec == pytest.approx(1.2, abs=0.08)

    approved = review.approve(job_id, 0)
    assert approved.state == SegmentState.APPROVED
    snapshot = review.snapshot(job_id)
    assert snapshot["can_stitch"] is True
    assert snapshot["approved_count"] == 1
    assert snapshot["custom_voice_fix"] == {
        "available": False,
        "reason": review.custom_voice_fix_reason,
    }
    store.close()


def test_review_proxy_is_browser_safe_without_touching_raw(tmp_path: Path) -> None:
    raw = tmp_path / "output_raw.mp4"
    _media_fixture(raw, pixel_format="yuv444p")
    original_hash = _sha256(raw)

    proxy = prepare_review_proxy(raw)

    proxy_bytes = proxy.read_bytes()
    assert proxy.name == "output_raw_review_h264.mp4"
    assert proxy_bytes.find(b"moov") < proxy_bytes.find(b"mdat")
    assert _sha256(raw) == original_hash
    assert inspect_media(proxy).duration_sec == pytest.approx(inspect_media(raw).duration_sec)
    with av.open(str(proxy)) as container:
        video = next(stream for stream in container.streams if stream.type == "video")
        audio = next(stream for stream in container.streams if stream.type == "audio")
        assert video.codec_context.name == "h264"
        assert video.codec_context.pix_fmt == "yuv420p"
        assert audio.codec_context.name == "aac"


def test_review_proxy_reports_corrupt_provider_output(tmp_path: Path) -> None:
    raw = tmp_path / "output_raw.mp4"
    raw.write_bytes(b"not a playable media file")

    with pytest.raises(PipelineError, match="inspect provider output for playback"):
        prepare_review_proxy(raw)


def test_paid_regeneration_is_disabled(tmp_path: Path) -> None:
    store, storage, job_id, raw = _review_job(tmp_path)
    review = ReviewService(store=store, storage=storage)
    review.approve(job_id, 0)

    with pytest.raises(PipelineError, match="paid regeneration is disabled"):
        review.regenerate(
            job_id,
            0,
            prompt="edited prompt",
            max_credits=0,
        )
    assert raw.is_file()
    assert store.job(job_id).state == JobState.REVIEW
    assert store.job(job_id).max_credits == store.job(job_id).estimated_credits
    assert review.snapshot(job_id)["can_stitch"] is True
    store.close()


def test_stitch_normalizes_validates_and_publishes_qc(tmp_path: Path) -> None:
    store, storage, job_id, _raw = _review_job(tmp_path)
    review = ReviewService(store=store, storage=storage)
    review.approve(job_id, 0)

    result = stitch_job(store, storage, job_id)

    assert result.final_path.is_file()
    assert result.qc_path.is_file()
    assert result.validation.valid
    assert result.validation.pixel_format == "yuv420p"
    assert result.validation.sample_aspect_ratio == "1:1"
    assert store.job(job_id).state == JobState.COMPLETE
    assert store.job(job_id).final_output_key == "jobs/review-job/final.mp4"
    store.close()
