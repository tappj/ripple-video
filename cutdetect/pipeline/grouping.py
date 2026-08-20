"""Partition complete cut sections into model-safe generation clips."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import cast

from cutdetect.pipeline.capabilities import MODEL_CAPABILITIES
from cutdetect.pipeline.runway_client import PipelineError


@dataclass(frozen=True, slots=True)
class AtomicSegment:
    """One indivisible source range bounded by detected hard cuts."""

    index: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    duration_sec: float

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class SegmentGroup:
    """One paid generation clip containing complete adjacent cut sections."""

    index: int
    segment_indices: tuple[int, ...]
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    duration_sec: float
    hard_cut_frames: tuple[int, ...]
    hard_cut_times_sec: tuple[float, ...]
    hard_cut_offsets_sec: tuple[float, ...]
    source_segment_count: int
    request_count: int

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for name in (
            "segment_indices",
            "hard_cut_frames",
            "hard_cut_times_sec",
            "hard_cut_offsets_sec",
        ):
            value[name] = list(value[name])
        return cast(dict[str, object], value)


@dataclass(frozen=True, slots=True)
class GroupingPlan:
    """A complete plan that cannot erase a detected source boundary."""

    model_id: str
    target_sec: float
    min_group_sec: float
    max_group_sec: float
    max_group_segments: int
    group_cost_bias: float
    generation_strategy: str
    preserve_hard_boundaries: bool
    total_groups: int
    total_generation_requests: int
    segments: tuple[AtomicSegment, ...]
    groups: tuple[SegmentGroup, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "params": {
                "target_sec": self.target_sec,
                "min_group_sec": self.min_group_sec,
                "max_group_sec": self.max_group_sec,
                "max_group_segments": self.max_group_segments,
                "group_cost_bias": self.group_cost_bias,
            },
            "generation_strategy": self.generation_strategy,
            "preserve_hard_boundaries": self.preserve_hard_boundaries,
            "total_groups": self.total_groups,
            "total_generation_requests": self.total_generation_requests,
            "segments": [segment.to_dict() for segment in self.segments],
            "groups": [{**group.to_dict(), "generation_unit": True} for group in self.groups],
        }


def automatic_target(duration_sec: float) -> float:
    """Return the Phase B review-batch duration target."""
    if duration_sec < 30:
        return 6.0
    return 9.0


def _segment_from_json(raw: object, expected_index: int) -> AtomicSegment:
    if not isinstance(raw, dict):
        raise PipelineError(f"segment {expected_index} must be an object")
    required = ("index", "start_frame", "end_frame", "start_sec", "end_sec")
    if any(name not in raw for name in required):
        raise PipelineError(f"segment {expected_index} is missing required boundary fields")
    segment = AtomicSegment(
        index=int(raw["index"]),
        start_frame=int(raw["start_frame"]),
        end_frame=int(raw["end_frame"]),
        start_sec=float(raw["start_sec"]),
        end_sec=float(raw["end_sec"]),
        duration_sec=float(raw["end_sec"]) - float(raw["start_sec"]),
    )
    if segment.index != expected_index:
        raise PipelineError(
            f"segments must be ordered and zero-indexed; expected {expected_index}, "
            f"got {segment.index}"
        )
    if segment.end_frame <= segment.start_frame or segment.duration_sec <= 0:
        raise PipelineError(f"segment {segment.index} has an empty or reversed range")
    return segment


def load_atomic_segments(predictions: str | Path) -> tuple[AtomicSegment, ...]:
    """Load and validate the contiguous segment contract from Phase 1/3."""
    path = Path(predictions).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"predictions not found: {path}")
    root: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict) or not isinstance(root.get("segments"), list):
        raise PipelineError("prediction JSON must contain a segments array")
    raw_segments = cast(list[object], root["segments"])
    if not raw_segments:
        raise PipelineError("prediction JSON contains no segments")
    segments = tuple(_segment_from_json(raw, index) for index, raw in enumerate(raw_segments))
    for left, right in pairwise(segments):
        if left.end_frame != right.start_frame:
            raise PipelineError(
                f"segments {left.index} and {right.index} do not share a hard-cut frame"
            )
        if abs(left.end_sec - right.start_sec) > 1e-5:
            raise PipelineError(
                f"segments {left.index} and {right.index} do not share a hard-cut time"
            )
    return segments


def group_atomic_segments(
    segments: tuple[AtomicSegment, ...],
    *,
    model_id: str = "seedance2",
    target_sec: float | None = None,
    min_group_sec: float = 4.0,
    max_group_sec: float = 15.0,
    max_group_segments: int = 4,
    group_cost_bias: float = 4.0,
) -> GroupingPlan:
    """DP-partition adjacent sections without moving or removing hard boundaries.

    Clips prefer the original 4-10 second and four-section policy. When that
    cannot cover every complete section, the planner may extend a clip to the
    model-safe maximum and absorb additional adjacent short sections.
    """
    if model_id not in MODEL_CAPABILITIES:
        raise PipelineError(f"unsupported model: {model_id}")
    if not segments:
        raise PipelineError("at least one segment is required")
    if not 1 <= max_group_segments <= 4:
        raise PipelineError("max_group_segments must be between 1 and 4")
    if group_cost_bias < 0:
        raise PipelineError("group_cost_bias must be non-negative")
    if min_group_sec <= 0 or max_group_sec < min_group_sec:
        raise PipelineError("group duration bounds must satisfy 0 < min <= max")
    model_max_sec = MODEL_CAPABILITIES[model_id].max_duration_s
    if max_group_sec > model_max_sec + 1e-6:
        raise PipelineError(
            f"max_group_sec cannot exceed the {model_max_sec:g}s {model_id} model maximum"
        )
    duration_sec = segments[-1].end_sec - segments[0].start_sec
    selected_target = target_sec if target_sec is not None else automatic_target(duration_sec)
    if not min_group_sec <= selected_target <= max_group_sec:
        raise PipelineError(f"target_sec must be between {min_group_sec:g} and {max_group_sec:g}")

    prefix = [0.0]
    for segment in segments:
        prefix.append(prefix[-1] + segment.duration_sec)
    count = len(segments)

    def solve(max_duration_sec: float, section_limit: int) -> list[tuple[int, int]] | None:
        infinity = float("inf")
        best = [infinity] * (count + 1)
        previous: list[int | None] = [None] * (count + 1)
        best[0] = 0.0
        for end in range(1, count + 1):
            first = max(0, end - section_limit)
            for start in range(first, end):
                batch_duration = prefix[end] - prefix[start]
                if (
                    batch_duration < min_group_sec - 1e-6
                    or batch_duration > max_duration_sec + 1e-6
                ):
                    continue
                cost = best[start] + (batch_duration - selected_target) ** 2 + group_cost_bias
                if cost < best[end]:
                    best[end] = cost
                    previous[end] = start

        if previous[count] is None:
            return None
        result: list[tuple[int, int]] = []
        cursor = count
        while cursor:
            predecessor = previous[cursor]
            if predecessor is None:  # pragma: no cover - guarded by the full-plan check
                return None
            result.append((predecessor, cursor))
            cursor = predecessor
        result.reverse()
        return result

    preferred_max_sec = min(max_group_sec, max(10.0, min_group_sec))
    ranges = solve(preferred_max_sec, max_group_segments)
    if ranges is None:
        # A limit of one is an explicit no-merge mode. Otherwise, the section
        # count is a preference: many tiny cuts may safely share one <=15s clip.
        fallback_section_limit = 1 if max_group_segments == 1 else count
        ranges = solve(max_group_sec, fallback_section_limit)
    if ranges is None:
        oversized = tuple(
            segment.index for segment in segments if segment.duration_sec > max_group_sec + 1e-6
        )
        if oversized:
            raise PipelineError(
                f"source sections {oversized} exceed the {max_group_sec:g}s maximum; "
                "they cannot be split silently"
            )
        section_note = (
            " in explicit no-merge mode" if max_group_segments == 1 else ""
        )
        raise PipelineError(
            f"cannot partition these complete cut sections into clips at least "
            f"{min_group_sec:g}s and no longer than {max_group_sec:g}s{section_note}"
        )

    groups: list[SegmentGroup] = []
    for group_index, (start, end) in enumerate(ranges):
        members = segments[start:end]
        hard_cut_times = tuple(segment.end_sec for segment in members[:-1])
        groups.append(
            SegmentGroup(
                index=group_index,
                segment_indices=tuple(segment.index for segment in members),
                start_frame=members[0].start_frame,
                end_frame=members[-1].end_frame,
                start_sec=members[0].start_sec,
                end_sec=members[-1].end_sec,
                duration_sec=members[-1].end_sec - members[0].start_sec,
                hard_cut_frames=tuple(segment.end_frame for segment in members[:-1]),
                hard_cut_times_sec=hard_cut_times,
                hard_cut_offsets_sec=tuple(
                    cut_time - members[0].start_sec for cut_time in hard_cut_times
                ),
                source_segment_count=len(members),
                request_count=1,
            )
        )

    return GroupingPlan(
        model_id=model_id,
        target_sec=selected_target,
        min_group_sec=min_group_sec,
        max_group_sec=max_group_sec,
        max_group_segments=max_group_segments,
        group_cost_bias=group_cost_bias,
        generation_strategy="generate_group_with_internal_hard_cuts",
        preserve_hard_boundaries=True,
        total_groups=len(groups),
        total_generation_requests=len(groups),
        segments=segments,
        groups=tuple(groups),
    )


def plan_from_predictions(
    predictions: str | Path,
    *,
    model_id: str = "seedance2",
    target_sec: float | None = None,
    min_group_sec: float = 4.0,
    max_group_sec: float = 15.0,
    max_group_segments: int = 4,
    group_cost_bias: float = 4.0,
) -> GroupingPlan:
    """Build a boundary-preserving grouping directly from prediction JSON."""
    return group_atomic_segments(
        load_atomic_segments(predictions),
        model_id=model_id,
        target_sec=target_sec,
        min_group_sec=min_group_sec,
        max_group_sec=max_group_sec,
        max_group_segments=max_group_segments,
        group_cost_bias=group_cost_bias,
    )
