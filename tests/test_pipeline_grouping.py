import json
from pathlib import Path

import pytest

from cutdetect.pipeline.grouping import group_atomic_segments, load_atomic_segments
from cutdetect.pipeline.runway_client import PipelineError


def _write_predictions(path: Path, durations: list[float]) -> Path:
    segments = []
    start_frame = 0
    start_sec = 0.0
    for index, duration in enumerate(durations):
        frame_count = round(duration * 30)
        end_frame = start_frame + frame_count
        end_sec = start_sec + duration
        segments.append(
            {
                "index": index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": duration,
            }
        )
        start_frame = end_frame
        start_sec = end_sec
    path.write_text(json.dumps({"segments": segments}), encoding="utf-8")
    return path


def test_groups_only_complete_segments_and_retains_every_hard_cut(tmp_path: Path) -> None:
    segments = load_atomic_segments(
        _write_predictions(tmp_path / "predictions.json", [2.9, 2.5, 4.3, 1.8, 1.4, 6.2])
    )
    plan = group_atomic_segments(segments, target_sec=9.0, max_group_segments=4)

    flattened = [index for group in plan.groups for index in group.segment_indices]
    retained = [frame for group in plan.groups for frame in group.hard_cut_frames]
    group_edges = [group.end_frame for group in plan.groups[:-1]]

    assert flattened == list(range(len(segments)))
    assert sorted([*retained, *group_edges]) == [segment.end_frame for segment in segments[:-1]]
    assert all(1 <= len(group.segment_indices) <= 4 for group in plan.groups)
    assert all(4.0 < group.duration_sec <= 10.0 for group in plan.groups)
    assert plan.preserve_hard_boundaries
    assert plan.generation_strategy == "generate_group_with_internal_hard_cuts"
    assert plan.total_generation_requests == len(plan.groups)
    assert all(group.request_count == 1 for group in plan.groups)


def test_one_segment_limit_is_explicit_no_merge_mode(tmp_path: Path) -> None:
    segments = load_atomic_segments(
        _write_predictions(tmp_path / "predictions.json", [4.1, 5.0, 6.0, 7.0])
    )
    plan = group_atomic_segments(segments, target_sec=7.0, max_group_segments=1)

    assert [group.segment_indices for group in plan.groups] == [(0,), (1,), (2,), (3,)]
    assert all(not group.hard_cut_frames for group in plan.groups)
    assert plan.total_groups == 4


def test_group_limit_cannot_exceed_four(tmp_path: Path) -> None:
    segments = load_atomic_segments(_write_predictions(tmp_path / "predictions.json", [5.0, 5.0]))

    with pytest.raises(PipelineError, match="between 1 and 4"):
        group_atomic_segments(segments, max_group_segments=5)


def test_exactly_four_seconds_does_not_satisfy_strict_minimum(tmp_path: Path) -> None:
    segments = load_atomic_segments(_write_predictions(tmp_path / "predictions.json", [4.0]))

    with pytest.raises(PipelineError, match="longer than 4s"):
        group_atomic_segments(segments, target_sec=6.0, max_group_segments=1)


def test_sample_partition_is_six_generation_clips_between_four_and_ten(
    tmp_path: Path,
) -> None:
    durations = [
        2.933333,
        2.533334,
        4.266666,
        1.766667,
        1.366667,
        6.166666,
        2.666667,
        4.0,
        5.566667,
        2.5,
        5.0,
        4.666666,
        1.7,
        1.066667,
    ]
    segments = load_atomic_segments(_write_predictions(tmp_path / "predictions.json", durations))

    plan = group_atomic_segments(segments, target_sec=9.0, max_group_segments=4)

    assert [group.segment_indices for group in plan.groups] == [
        (0, 1),
        (2, 3, 4),
        (5, 6),
        (7, 8),
        (9, 10),
        (11, 12, 13),
    ]
    assert plan.total_generation_requests == 6
    assert all(4.0 < group.duration_sec <= 10.0 for group in plan.groups)
    internal = [frame for group in plan.groups for frame in group.hard_cut_frames]
    group_edges = [group.end_frame for group in plan.groups[:-1]]
    assert sorted([*internal, *group_edges]) == [segment.end_frame for segment in segments[:-1]]


def test_noncontiguous_ranges_are_rejected(tmp_path: Path) -> None:
    path = _write_predictions(tmp_path / "predictions.json", [2.0, 3.0])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["segments"][1]["start_frame"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PipelineError, match="do not share a hard-cut frame"):
        load_atomic_segments(path)
