"""Partition complete cut sections into model-safe generation clips."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import cast

from cutdetect.pipeline.capabilities import MODEL_CAPABILITIES
from cutdetect.pipeline.runway_client import PipelineError


@dataclass(frozen=True, slots=True)
class AtomicSegment:
    """One source range bounded by a visual cut or an inserted audio pause."""

    index: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    duration_sec: float
    end_boundary_kind: str = "visual_cut"

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


def requires_cut_partition(duration_sec: float, *, limit_sec: float = 15.0) -> bool:
    """Return whether a source must be analyzed and partitioned for generation."""
    if duration_sec <= 0 or limit_sec <= 0:
        raise PipelineError("video duration and generation limit must be positive")
    return duration_sec > limit_sec + 1e-6


def whole_video_plan(
    *,
    model_id: str,
    frame_count: int,
    duration_sec: float,
    min_group_sec: float = 4.0,
    max_group_sec: float = 15.0,
    max_group_segments: int = 4,
    group_cost_bias: float = 4.0,
) -> GroupingPlan:
    """Create one generation unit for a source that already fits the model limit."""
    if model_id not in MODEL_CAPABILITIES:
        raise PipelineError(f"unsupported model: {model_id}")
    if frame_count <= 0 or duration_sec <= 0:
        raise PipelineError("whole-video generation requires non-empty media")
    if requires_cut_partition(duration_sec, limit_sec=max_group_sec):
        raise PipelineError(
            f"{duration_sec:g}s source exceeds the {max_group_sec:g}s single-generation limit"
        )
    segment = AtomicSegment(
        index=0,
        start_frame=0,
        end_frame=frame_count,
        start_sec=0.0,
        end_sec=duration_sec,
        duration_sec=duration_sec,
        end_boundary_kind="source_end",
    )
    group = SegmentGroup(
        index=0,
        segment_indices=(0,),
        start_frame=0,
        end_frame=frame_count,
        start_sec=0.0,
        end_sec=duration_sec,
        duration_sec=duration_sec,
        hard_cut_frames=(),
        hard_cut_times_sec=(),
        hard_cut_offsets_sec=(),
        source_segment_count=1,
        request_count=1,
    )
    return GroupingPlan(
        model_id=model_id,
        target_sec=min(max(duration_sec, min_group_sec), max_group_sec),
        min_group_sec=min_group_sec,
        max_group_sec=max_group_sec,
        max_group_segments=max_group_segments,
        group_cost_bias=group_cost_bias,
        generation_strategy="whole_source_under_generation_limit",
        preserve_hard_boundaries=True,
        total_groups=1,
        total_generation_requests=1,
        segments=(segment,),
        groups=(group,),
    )


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


SplitPointSelector = Callable[[float, float, float], tuple[float, str]]


def _split_long_segments(
    segments: tuple[AtomicSegment, ...],
    *,
    min_group_sec: float,
    max_group_sec: float,
    preferred_group_sec: float,
    split_selector: SplitPointSelector | None,
) -> tuple[AtomicSegment, ...]:
    """Subdivide visual sections over the model limit at audio-friendly points."""
    result: list[AtomicSegment] = []
    for segment in segments:
        if segment.duration_sec <= max_group_sec + 1e-6:
            result.append(segment)
            continue
        piece_count = max(2, math.ceil(segment.duration_sec / preferred_group_sec))
        ideal_piece_sec = segment.duration_sec / piece_count
        boundaries: list[tuple[int, float, str]] = []
        previous_sec = segment.start_sec
        for piece_index in range(1, piece_count):
            remaining_pieces = piece_count - piece_index
            ideal_sec = segment.start_sec + ideal_piece_sec * piece_index
            lower_sec = max(
                previous_sec + min_group_sec,
                segment.end_sec - remaining_pieces * max_group_sec,
            )
            upper_sec = min(
                previous_sec + max_group_sec,
                segment.end_sec - remaining_pieces * min_group_sec,
            )
            selected_sec, boundary_kind = (
                split_selector(ideal_sec, lower_sec, upper_sec)
                if split_selector is not None
                else (ideal_sec, "timed_fallback")
            )
            selected_sec = min(max(selected_sec, lower_sec), upper_sec)
            frame_span = segment.end_frame - segment.start_frame
            time_fraction = (selected_sec - segment.start_sec) / segment.duration_sec
            selected_frame = round(segment.start_frame + time_fraction * frame_span)
            minimum_frame = math.ceil(
                segment.start_frame
                + (lower_sec - segment.start_sec) / segment.duration_sec * frame_span
            )
            maximum_frame = math.floor(
                segment.start_frame
                + (upper_sec - segment.start_sec) / segment.duration_sec * frame_span
            )
            selected_frame = min(max(selected_frame, minimum_frame), maximum_frame)
            if selected_frame <= segment.start_frame or selected_frame >= segment.end_frame:
                raise PipelineError("source timing metadata cannot represent a safe audio split")
            selected_sec = segment.start_sec + (
                (selected_frame - segment.start_frame) / frame_span * segment.duration_sec
            )
            boundaries.append((selected_frame, selected_sec, boundary_kind))
            previous_sec = selected_sec

        starts = [(segment.start_frame, segment.start_sec, "")]
        ends = [*boundaries, (segment.end_frame, segment.end_sec, segment.end_boundary_kind)]
        for (start_frame, start_sec, _kind), (end_frame, end_sec, kind) in zip(
            starts + boundaries, ends, strict=True
        ):
            result.append(
                AtomicSegment(
                    index=len(result),
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    duration_sec=end_sec - start_sec,
                    end_boundary_kind=kind,
                )
            )

    return tuple(
        AtomicSegment(
            index=index,
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
            duration_sec=segment.duration_sec,
            end_boundary_kind=segment.end_boundary_kind,
        )
        for index, segment in enumerate(result)
    )


def group_atomic_segments(
    segments: tuple[AtomicSegment, ...],
    *,
    model_id: str = "seedance2",
    target_sec: float | None = None,
    min_group_sec: float = 4.0,
    max_group_sec: float = 15.0,
    max_group_segments: int = 4,
    group_cost_bias: float = 4.0,
    split_selector: SplitPointSelector | None = None,
) -> GroupingPlan:
    """DP-partition adjacent sections without moving or removing hard boundaries.

    Clips prefer the original 4-10 second and four-section policy. When that
    cannot cover every complete section, the planner may extend a clip to the
    model-safe maximum and absorb additional adjacent short sections. Visual
    sections beyond that maximum are divided at audio-selected pause points.
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
    preferred_max_sec = min(max_group_sec, max(10.0, min_group_sec))
    segments = _split_long_segments(
        segments,
        min_group_sec=min_group_sec,
        max_group_sec=max_group_sec,
        preferred_group_sec=preferred_max_sec,
        split_selector=split_selector,
    )
    duration_sec = segments[-1].end_sec - segments[0].start_sec
    selected_target = target_sec if target_sec is not None else automatic_target(duration_sec)
    if not min_group_sec <= selected_target <= max_group_sec:
        raise PipelineError(f"target_sec must be between {min_group_sec:g} and {max_group_sec:g}")

    prefix = [0.0]
    for segment in segments:
        prefix.append(prefix[-1] + segment.duration_sec)
    count = len(segments)

    def solve(
        minimum_duration_sec: float, max_duration_sec: float, section_limit: int
    ) -> list[tuple[int, int]] | None:
        infinity = float("inf")
        best = [infinity] * (count + 1)
        previous: list[int | None] = [None] * (count + 1)
        best[0] = 0.0
        for end in range(1, count + 1):
            first = max(0, end - section_limit)
            for start in range(first, end):
                batch_duration = prefix[end] - prefix[start]
                if (
                    batch_duration < minimum_duration_sec - 1e-6
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

    ranges = solve(min_group_sec, preferred_max_sec, max_group_segments)
    if ranges is None:
        # A limit of one is an explicit no-merge mode. Otherwise, the section
        # count is a preference: many tiny cuts may safely share one <=15s clip.
        fallback_section_limit = 1 if max_group_segments == 1 else count
        ranges = solve(min_group_sec, max_group_sec, fallback_section_limit)
    if ranges is None:
        # A source shorter than the model minimum, or a short section trapped
        # beside a full-length section, is padded at request time and trimmed
        # back to its exact source duration after generation.
        ranges = solve(0.0, max_group_sec, count)
    if ranges is None:
        raise PipelineError(
            "source timing metadata cannot be partitioned into positive model-safe clips"
        )

    groups: list[SegmentGroup] = []
    for group_index, (start, end) in enumerate(ranges):
        members = segments[start:end]
        visual_boundaries = tuple(
            segment for segment in members[:-1] if segment.end_boundary_kind == "visual_cut"
        )
        hard_cut_times = tuple(segment.end_sec for segment in visual_boundaries)
        groups.append(
            SegmentGroup(
                index=group_index,
                segment_indices=tuple(segment.index for segment in members),
                start_frame=members[0].start_frame,
                end_frame=members[-1].end_frame,
                start_sec=members[0].start_sec,
                end_sec=members[-1].end_sec,
                duration_sec=members[-1].end_sec - members[0].start_sec,
                hard_cut_frames=tuple(segment.end_frame for segment in visual_boundaries),
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
    source_video: str | Path | None = None,
) -> GroupingPlan:
    """Build a boundary-preserving grouping directly from prediction JSON."""
    segments = load_atomic_segments(predictions)
    split_selector = None
    if source_video is not None and any(
        segment.duration_sec > max_group_sec + 1e-6 for segment in segments
    ):
        from cutdetect.pipeline.audio_pauses import load_audio_pause_selector

        split_selector = load_audio_pause_selector(source_video)
    return group_atomic_segments(
        segments,
        model_id=model_id,
        target_sec=target_sec,
        min_group_sec=min_group_sec,
        max_group_sec=max_group_sec,
        max_group_segments=max_group_segments,
        group_cost_bias=group_cost_bias,
        split_selector=split_selector,
    )
