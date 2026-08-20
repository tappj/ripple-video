"""Durable Phase C orchestration for independent direct Runway generations."""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

from cutdetect.export import split_video
from cutdetect.ingest import IngestError, probe_video
from cutdetect.pipeline.capabilities import (
    MODEL_CAPABILITIES,
    ROUTER_ASPECT_RATIOS,
    ROUTER_RESOLUTIONS,
    credit_cost,
)
from cutdetect.pipeline.grouping import GroupingPlan, plan_from_predictions
from cutdetect.pipeline.media import preserve_source_audio, trim_generated_duration
from cutdetect.pipeline.runway_client import (
    GenerationPoll,
    GenerationRequest,
    PipelineError,
    RouterConfigurationError,
    RunwayReferenceModel,
    model_router_route,
)
from cutdetect.pipeline.storage import LocalDiskStorage
from cutdetect.pipeline.templates import UGC_CLONE_V1
from cutdetect.pipeline.workflow_client import (
    TALKING_WORKFLOW_ROUTE,
    WORKFLOW_CREDITS_PER_RUN,
    WORKFLOW_DURATION_SEC,
)

DIRECT_API_ROUTE = "direct-reference-v1"


def generation_route(model_id: str) -> str:
    """Use the router catalog where available and a pinned direct model otherwise."""
    if model_id == "seedance2":
        return model_router_route("seedance2")
    if model_id == "hailuo3":
        return DIRECT_API_ROUTE
    raise PipelineError(f"unsupported direct reference model: {model_id}")


class JobState(StrEnum):
    DRAFT = "DRAFT"
    ESTIMATING = "ESTIMATING"
    CONFIRMED = "CONFIRMED"
    RUNNING = "RUNNING"
    REVIEW = "REVIEW"
    STITCHING = "STITCHING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class SegmentState(StrEnum):
    PENDING = "PENDING"
    UPLOADING = "UPLOADING"
    SUBMITTED = "SUBMITTED"
    THROTTLED = "THROTTLED"
    RUNNING = "RUNNING"
    DOWNLOADING = "DOWNLOADING"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    APPROVED = "APPROVED"
    FAILED = "FAILED"
    REGENERATING = "REGENERATING"


@dataclass(frozen=True, slots=True)
class Retry:
    max_retries: int
    delay_sec: float
    hint: str | None = None


RETRY_POLICY: dict[str | None, Retry] = {
    "INTERNAL.BAD_OUTPUT": Retry(2, 5, "check for captions or watermarks"),
    "INPUT_PREPROCESSING.INTERNAL": Retry(3, 30),
    "THIRD_PARTY.UNAVAILABLE": Retry(2, 300),
    "INTERNAL": Retry(3, 30),
    None: Retry(3, 30),
}


class CreditLimitError(PipelineError):
    """Raised before a paid submission could exceed the explicit ceiling."""


class GenerationGateway(Protocol):
    """Narrow boundary implemented by Runway and deterministic test doubles."""

    def upload(self, path: Path, *, role: str) -> str: ...

    def submit(self, request: GenerationRequest) -> str: ...

    def poll(self, task_id: str) -> GenerationPoll: ...

    def download(self, url: str, destination_key: str) -> Path: ...


@dataclass(frozen=True, slots=True)
class JobRecord:
    id: str
    state: JobState
    source_path: Path
    target_face_path: Path
    target_voice_path: Path | None
    prompt: str
    prompt_template_id: str
    prompt_template_version: int
    consent_affirmed_at: datetime | None
    route_id: str
    model_id: str
    ratio: str
    resolution: str | None
    estimated_credits: int
    max_credits: int | None
    submitted_credits: int
    final_output_key: str | None
    qc_output_key: str | None
    completed_at: datetime | None
    target_face_uri: str | None
    target_voice_uri: str | None
    references_uploaded_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    job_id: str
    index: int
    group_index: int
    state: SegmentState
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    duration_sec: float
    requested_duration_sec: int
    estimated_credits: int
    hard_cut_offsets_sec: tuple[float, ...]
    actual_duration_sec: float | None
    input_path: Path
    output_key: str
    final_output_key: str | None
    trim_start_frame: int | None
    trim_end_frame: int | None
    approved_at: datetime | None
    prompt_override: str | None
    source_uri: str | None
    source_uploaded_at: datetime | None
    invocation_id: str | None
    attempt_count: int
    retry_count: int
    next_attempt_at: datetime | None
    failure_code: str | None
    failure_message: str | None
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class PreparedJob:
    job: JobRecord
    grouping_path: Path
    database_path: Path
    job_directory: Path
    generation_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job.id,
            "state": self.job.state.value,
            "route_id": self.job.route_id,
            "model_id": self.job.model_id,
            "ratio": self.job.ratio,
            "resolution": self.job.resolution,
            "segment_count": self.generation_count,
            "estimated_credits": self.job.estimated_credits,
            "audio_mode": "reference" if self.job.target_voice_path else "source",
            "grouping_path": str(self.grouping_path),
            "database_path": str(self.database_path),
            "job_directory": str(self.job_directory),
        }


def _now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


class PhaseCStore:
    """SQLite-backed job state that can be reopened after process restarts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> PhaseCStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                source_path TEXT NOT NULL,
                target_face_path TEXT NOT NULL,
                target_voice_path TEXT NOT NULL,
                prompt TEXT NOT NULL,
                prompt_template_id TEXT NOT NULL DEFAULT 'ugc_clone_v1',
                prompt_template_version INTEGER NOT NULL DEFAULT 1,
                consent_affirmed_at TEXT,
                workflow_id TEXT NOT NULL,
                model_id TEXT NOT NULL DEFAULT 'hailuo3',
                ratio TEXT NOT NULL DEFAULT '9:16',
                resolution TEXT,
                estimated_credits INTEGER NOT NULL,
                max_credits INTEGER,
                submitted_credits INTEGER NOT NULL DEFAULT 0,
                final_output_key TEXT,
                qc_output_key TEXT,
                completed_at TEXT,
                target_face_uri TEXT,
                target_voice_uri TEXT,
                references_uploaded_at TEXT,
                owner_device_hash TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS segments (
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                segment_index INTEGER NOT NULL,
                group_index INTEGER NOT NULL,
                state TEXT NOT NULL,
                start_frame INTEGER NOT NULL,
                end_frame INTEGER NOT NULL,
                start_sec REAL NOT NULL,
                end_sec REAL NOT NULL,
                duration_sec REAL NOT NULL,
                requested_duration_sec INTEGER NOT NULL DEFAULT 10,
                estimated_credits INTEGER NOT NULL DEFAULT 0,
                hard_cut_offsets_json TEXT NOT NULL DEFAULT '[]',
                actual_duration_sec REAL,
                input_path TEXT NOT NULL,
                output_key TEXT NOT NULL,
                final_output_key TEXT,
                trim_start_frame INTEGER,
                trim_end_frame INTEGER,
                approved_at TEXT,
                prompt_override TEXT,
                source_uri TEXT,
                source_uploaded_at TEXT,
                invocation_id TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                retry_count INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT,
                failure_code TEXT,
                failure_message TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (job_id, segment_index)
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                segment_index INTEGER,
                event TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_segments_job_state
                ON segments(job_id, state);
            CREATE INDEX IF NOT EXISTS idx_events_job_id
                ON events(job_id, id);
            """
        )
        self._ensure_column("jobs", "model_id", "TEXT NOT NULL DEFAULT 'hailuo3'")
        self._ensure_column("jobs", "ratio", "TEXT NOT NULL DEFAULT '9:16'")
        self._ensure_column("jobs", "resolution", "TEXT")
        self._ensure_column("jobs", "prompt_template_id", "TEXT NOT NULL DEFAULT 'ugc_clone_v1'")
        self._ensure_column("jobs", "prompt_template_version", "INTEGER NOT NULL DEFAULT 1")
        self._ensure_column("jobs", "consent_affirmed_at", "TEXT")
        self._ensure_column("jobs", "final_output_key", "TEXT")
        self._ensure_column("jobs", "qc_output_key", "TEXT")
        self._ensure_column("jobs", "completed_at", "TEXT")
        self._ensure_column("jobs", "owner_device_hash", "TEXT")
        self._ensure_column("segments", "requested_duration_sec", "INTEGER NOT NULL DEFAULT 10")
        self._ensure_column("segments", "estimated_credits", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("segments", "hard_cut_offsets_json", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("segments", "actual_duration_sec", "REAL")
        self._ensure_column("segments", "final_output_key", "TEXT")
        self._ensure_column("segments", "trim_start_frame", "INTEGER")
        self._ensure_column("segments", "trim_end_frame", "INTEGER")
        self._ensure_column("segments", "approved_at", "TEXT")
        self._ensure_column("segments", "prompt_override", "TEXT")
        self._connection.commit()

    def _ensure_column(self, table: str, name: str, declaration: str) -> None:
        columns = {
            str(row["name"])
            for row in self._connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if name not in columns:
            self._connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")

    def _event(
        self,
        job_id: str,
        event: str,
        *,
        segment_index: int | None = None,
        payload: object | None = None,
        at: datetime | None = None,
    ) -> None:
        self._connection.execute(
            "INSERT INTO events(job_id, segment_index, event, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                job_id,
                segment_index,
                event,
                json.dumps(payload if payload is not None else {}, separators=(",", ":")),
                _timestamp(at or _now()),
            ),
        )

    def create_job(
        self,
        *,
        job_id: str,
        source_path: Path,
        target_face_path: Path,
        target_voice_path: Path | None,
        prompt: str,
        grouping: GroupingPlan,
        input_paths: Sequence[Path],
        output_keys: Sequence[str],
        model_id: str,
        ratio: str,
        resolution: str | None,
        route_id: str = DIRECT_API_ROUTE,
        prompt_template_id: str = UGC_CLONE_V1.id,
        prompt_template_version: int = UGC_CLONE_V1.version,
        consent_affirmed: bool = False,
        owner_device_hash: str | None = None,
    ) -> JobRecord:
        if len(input_paths) != len(grouping.groups) or len(output_keys) != len(grouping.groups):
            raise ValueError("every generation group needs one input and output path")
        if grouping.model_id != model_id:
            raise ValueError("grouping and generation model must match")
        if model_id not in MODEL_CAPABILITIES:
            raise PipelineError(f"unsupported direct reference model: {model_id}")
        is_router_route = route_id.startswith("router:")
        if route_id not in {DIRECT_API_ROUTE, TALKING_WORKFLOW_ROUTE} and not is_router_route:
            raise PipelineError(f"unsupported generation route: {route_id}")
        if route_id == TALKING_WORKFLOW_ROUTE and model_id != "seedance2":
            raise PipelineError("the published talking-video Workflow requires Seedance 2")
        caps = MODEL_CAPABILITIES[model_id]
        allowed_ratios = ROUTER_ASPECT_RATIOS if is_router_route else caps.supported_ratios
        allowed_resolutions = (
            ROUTER_RESOLUTIONS[model_id] if is_router_route else caps.supported_resolutions
        )
        if ratio not in allowed_ratios:
            raise PipelineError(f"unsupported {model_id} ratio: {ratio}")
        if resolution is not None and resolution not in allowed_resolutions:
            raise PipelineError(f"unsupported {model_id} resolution: {resolution}")
        if is_router_route and resolution is None:
            raise PipelineError("Model Router jobs require a resolution")
        request_specs: list[tuple[int, int]] = []
        for group in grouping.groups:
            requested_duration = (
                WORKFLOW_DURATION_SEC
                if route_id == TALKING_WORKFLOW_ROUTE
                else math.ceil(group.duration_sec - 1e-6)
            )
            if not caps.min_duration_s <= requested_duration <= caps.max_duration_s:
                raise PipelineError(
                    f"group {group.index} requests {requested_duration}s outside the "
                    f"{caps.min_duration_s:g}-{caps.max_duration_s:g}s {model_id} range"
                )
            segment_credits = (
                WORKFLOW_CREDITS_PER_RUN
                if route_id == TALKING_WORKFLOW_ROUTE
                else credit_cost(
                    model_id,
                    requested_duration,
                    resolution or ratio,
                    reference_video_duration_s=group.duration_sec,
                )
            )
            request_specs.append((requested_duration, segment_credits))
        created = _now()
        estimated = sum(cost for _duration, cost in request_specs)
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO jobs(
                    id, state, source_path, target_face_path, target_voice_path,
                    prompt, prompt_template_id, prompt_template_version, consent_affirmed_at,
                    workflow_id, model_id, ratio, resolution,
                    estimated_credits, owner_device_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    JobState.DRAFT.value,
                    str(source_path),
                    str(target_face_path),
                    str(target_voice_path) if target_voice_path is not None else "",
                    prompt,
                    prompt_template_id,
                    prompt_template_version,
                    _timestamp(created) if consent_affirmed else None,
                    route_id,
                    model_id,
                    ratio,
                    resolution,
                    estimated,
                    owner_device_hash,
                    _timestamp(created),
                    _timestamp(created),
                ),
            )
            for group, input_path, output_key, request_spec in zip(
                grouping.groups, input_paths, output_keys, request_specs, strict=True
            ):
                requested_duration, segment_credits = request_spec
                self._connection.execute(
                    """
                    INSERT INTO segments(
                        job_id, segment_index, group_index, state, start_frame,
                        end_frame, start_sec, end_sec, duration_sec,
                        requested_duration_sec, estimated_credits, hard_cut_offsets_json,
                        input_path, output_key, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        group.index,
                        group.index,
                        SegmentState.PENDING.value,
                        group.start_frame,
                        group.end_frame,
                        group.start_sec,
                        group.end_sec,
                        group.duration_sec,
                        requested_duration,
                        segment_credits,
                        json.dumps(group.hard_cut_offsets_sec, separators=(",", ":")),
                        str(input_path),
                        output_key,
                        _timestamp(created),
                    ),
                )
            self._event(
                job_id,
                "job.created",
                payload={
                    "state": JobState.DRAFT.value,
                    "generation_groups": len(grouping.groups),
                    "estimated_credits": estimated,
                    "model_id": model_id,
                    "route_id": route_id,
                    "consent_affirmed": consent_affirmed,
                    "audio_mode": "reference" if target_voice_path else "source",
                    "prompt_template": {
                        "id": prompt_template_id,
                        "version": prompt_template_version,
                    },
                },
                at=created,
            )
        return self.job(job_id)

    def _job_from_row(self, row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            id=str(row["id"]),
            state=JobState(str(row["state"])),
            source_path=Path(str(row["source_path"])),
            target_face_path=Path(str(row["target_face_path"])),
            target_voice_path=(
                Path(str(row["target_voice_path"])) if str(row["target_voice_path"]) else None
            ),
            prompt=str(row["prompt"]),
            prompt_template_id=str(row["prompt_template_id"]),
            prompt_template_version=int(row["prompt_template_version"]),
            consent_affirmed_at=_datetime(cast(str | None, row["consent_affirmed_at"])),
            route_id=str(row["workflow_id"]),
            model_id=str(row["model_id"]),
            ratio=str(row["ratio"]),
            resolution=cast(str | None, row["resolution"]),
            estimated_credits=int(row["estimated_credits"]),
            max_credits=int(row["max_credits"]) if row["max_credits"] is not None else None,
            submitted_credits=int(row["submitted_credits"]),
            final_output_key=cast(str | None, row["final_output_key"]),
            qc_output_key=cast(str | None, row["qc_output_key"]),
            completed_at=_datetime(cast(str | None, row["completed_at"])),
            target_face_uri=cast(str | None, row["target_face_uri"]),
            target_voice_uri=cast(str | None, row["target_voice_uri"]),
            references_uploaded_at=_datetime(cast(str | None, row["references_uploaded_at"])),
            created_at=cast(datetime, _datetime(str(row["created_at"]))),
            updated_at=cast(datetime, _datetime(str(row["updated_at"]))),
        )

    def job(self, job_id: str) -> JobRecord:
        row = self._connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise PipelineError(f"unknown Phase C job: {job_id}")
        return self._job_from_row(row)

    def jobs(self) -> tuple[JobRecord, ...]:
        rows = self._connection.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return tuple(self._job_from_row(row) for row in rows)

    def jobs_for_device(self, owner_device_hash: str) -> tuple[JobRecord, ...]:
        """Return only jobs created by one browser-scoped device identity."""
        rows = self._connection.execute(
            "SELECT * FROM jobs WHERE owner_device_hash = ? ORDER BY created_at DESC",
            (owner_device_hash,),
        ).fetchall()
        return tuple(self._job_from_row(row) for row in rows)

    def job_owned_by(self, job_id: str, owner_device_hash: str) -> bool:
        """Check ownership without exposing the stored identity hash."""
        row = self._connection.execute(
            "SELECT 1 FROM jobs WHERE id = ? AND owner_device_hash = ?",
            (job_id, owner_device_hash),
        ).fetchone()
        return row is not None

    def _segment_from_row(self, row: sqlite3.Row) -> SegmentRecord:
        return SegmentRecord(
            job_id=str(row["job_id"]),
            index=int(row["segment_index"]),
            group_index=int(row["group_index"]),
            state=SegmentState(str(row["state"])),
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
            start_sec=float(row["start_sec"]),
            end_sec=float(row["end_sec"]),
            duration_sec=float(row["duration_sec"]),
            requested_duration_sec=int(row["requested_duration_sec"]),
            estimated_credits=int(row["estimated_credits"]),
            hard_cut_offsets_sec=tuple(
                float(value) for value in json.loads(str(row["hard_cut_offsets_json"]))
            ),
            actual_duration_sec=(
                float(row["actual_duration_sec"])
                if row["actual_duration_sec"] is not None
                else None
            ),
            input_path=Path(str(row["input_path"])),
            output_key=str(row["output_key"]),
            final_output_key=cast(str | None, row["final_output_key"]),
            trim_start_frame=(
                int(row["trim_start_frame"]) if row["trim_start_frame"] is not None else None
            ),
            trim_end_frame=(
                int(row["trim_end_frame"]) if row["trim_end_frame"] is not None else None
            ),
            approved_at=_datetime(cast(str | None, row["approved_at"])),
            prompt_override=cast(str | None, row["prompt_override"]),
            source_uri=cast(str | None, row["source_uri"]),
            source_uploaded_at=_datetime(cast(str | None, row["source_uploaded_at"])),
            invocation_id=cast(str | None, row["invocation_id"]),
            attempt_count=int(row["attempt_count"]),
            retry_count=int(row["retry_count"]),
            next_attempt_at=_datetime(cast(str | None, row["next_attempt_at"])),
            failure_code=cast(str | None, row["failure_code"]),
            failure_message=cast(str | None, row["failure_message"]),
            updated_at=cast(datetime, _datetime(str(row["updated_at"]))),
        )

    def segments(self, job_id: str) -> tuple[SegmentRecord, ...]:
        rows = self._connection.execute(
            "SELECT * FROM segments WHERE job_id = ? ORDER BY segment_index", (job_id,)
        ).fetchall()
        return tuple(self._segment_from_row(row) for row in rows)

    def confirm(self, job_id: str, max_credits: int) -> JobRecord:
        job = self.job(job_id)
        if job.state not in {JobState.DRAFT, JobState.ESTIMATING, JobState.CONFIRMED}:
            raise PipelineError(f"job {job_id} cannot be confirmed from {job.state.value}")
        if max_credits < job.estimated_credits:
            raise CreditLimitError(
                f"job requires an initial ceiling of at least {job.estimated_credits} credits; "
                f"received {max_credits}"
            )
        changed = _now()
        with self._connection:
            self._connection.execute(
                "UPDATE jobs SET state = ?, max_credits = ?, updated_at = ? WHERE id = ?",
                (JobState.CONFIRMED.value, max_credits, _timestamp(changed), job_id),
            )
            self._event(
                job_id,
                "job.confirmed",
                payload={"max_credits": max_credits},
                at=changed,
            )
        return self.job(job_id)

    def raise_credit_ceiling(self, job_id: str, max_credits: int) -> JobRecord:
        """Raise, but never lower, the locked ceiling for an explicit review action."""
        job = self.job(job_id)
        current = job.max_credits or 0
        if max_credits < current:
            raise CreditLimitError(
                f"credit ceiling cannot be lowered from {current} to {max_credits}"
            )
        changed = _now()
        with self._connection:
            self._connection.execute(
                "UPDATE jobs SET max_credits = ?, updated_at = ? WHERE id = ?",
                (max_credits, _timestamp(changed), job_id),
            )
            self._event(
                job_id,
                "job.credit_ceiling_raised",
                payload={"previous": current, "max_credits": max_credits},
                at=changed,
            )
        return self.job(job_id)

    def request_regeneration(
        self,
        job_id: str,
        index: int,
        *,
        prompt: str,
        max_credits: int,
    ) -> SegmentRecord:
        """Unlock one reviewed segment and create an immutable new output attempt."""
        if not prompt.strip():
            raise PipelineError("regeneration prompt must not be empty")
        segment = self.segments(job_id)[index]
        if segment.state not in {
            SegmentState.READY_FOR_REVIEW,
            SegmentState.APPROVED,
            SegmentState.FAILED,
        }:
            raise PipelineError(f"segment {index} cannot regenerate from {segment.state.value}")
        job = self.job(job_id)
        required_ceiling = job.submitted_credits + segment.estimated_credits
        if required_ceiling > max_credits:
            raise CreditLimitError(
                f"regenerating segment {index} requires a ceiling of at least "
                f"{required_ceiling} credits"
            )
        self.raise_credit_ceiling(job_id, max_credits)
        previous_key = Path(segment.output_key)
        next_attempt = segment.attempt_count + 1
        output_key = str(previous_key.with_name(f"output_raw_attempt_{next_attempt:02d}.mp4"))
        changed = _now()
        updated = self.set_segment_state(
            job_id,
            index,
            SegmentState.PENDING,
            output_key=output_key,
            final_output_key=None,
            trim_start_frame=None,
            trim_end_frame=None,
            approved_at=None,
            prompt_override=prompt,
            invocation_id=None,
            retry_count=0,
            next_attempt_at=None,
            failure_code=None,
            failure_message=None,
            actual_duration_sec=None,
        )
        self.set_job_state(job_id, JobState.RUNNING)
        self.record_event(
            job_id,
            "segment.regeneration_requested",
            segment_index=index,
            payload={
                "output_key": output_key,
                "incremental_credits": segment.estimated_credits,
                "requested_at": _timestamp(changed),
            },
        )
        return updated

    def set_job_state(self, job_id: str, state: JobState) -> JobRecord:
        changed = _now()
        with self._connection:
            self._connection.execute(
                "UPDATE jobs SET state = ?, updated_at = ? WHERE id = ?",
                (state.value, _timestamp(changed), job_id),
            )
            self._event(
                job_id,
                "job.state",
                payload={"state": state.value},
                at=changed,
            )
        return self.job(job_id)

    def mark_complete(
        self,
        job_id: str,
        *,
        final_output_key: str,
        qc_output_key: str,
    ) -> JobRecord:
        """Atomically publish validated final and QC artifacts."""
        changed = _now()
        with self._connection:
            self._connection.execute(
                """
                UPDATE jobs SET state = ?, final_output_key = ?, qc_output_key = ?,
                    completed_at = ?, updated_at = ? WHERE id = ?
                """,
                (
                    JobState.COMPLETE.value,
                    final_output_key,
                    qc_output_key,
                    _timestamp(changed),
                    _timestamp(changed),
                    job_id,
                ),
            )
            self._event(
                job_id,
                "job.completed",
                payload={
                    "final_output_key": final_output_key,
                    "qc_output_key": qc_output_key,
                },
                at=changed,
            )
        return self.job(job_id)

    def update_references(
        self,
        job_id: str,
        *,
        target_face_uri: str,
        target_voice_uri: str | None,
        uploaded_at: datetime,
    ) -> None:
        with self._connection:
            self._connection.execute(
                """
                UPDATE jobs SET target_face_uri = ?, target_voice_uri = ?,
                    references_uploaded_at = ?, updated_at = ? WHERE id = ?
                """,
                (
                    target_face_uri,
                    target_voice_uri,
                    _timestamp(uploaded_at),
                    _timestamp(uploaded_at),
                    job_id,
                ),
            )
            self._event(job_id, "job.references_uploaded", at=uploaded_at)

    def set_segment_state(
        self,
        job_id: str,
        index: int,
        state: SegmentState,
        **updates: object,
    ) -> SegmentRecord:
        allowed = {
            "source_uri",
            "source_uploaded_at",
            "invocation_id",
            "attempt_count",
            "retry_count",
            "next_attempt_at",
            "failure_code",
            "failure_message",
            "actual_duration_sec",
            "output_key",
            "final_output_key",
            "trim_start_frame",
            "trim_end_frame",
            "approved_at",
            "prompt_override",
        }
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(f"unsupported segment fields: {sorted(unknown)}")
        event_updates = {
            name: _timestamp(value) if isinstance(value, datetime) else value
            for name, value in updates.items()
        }
        converted = dict(event_updates)
        converted["state"] = state.value
        changed = _now()
        converted["updated_at"] = _timestamp(changed)
        assignments = ", ".join(f"{name} = ?" for name in converted)
        values = [*converted.values(), job_id, index]
        with self._connection:
            self._connection.execute(
                f"UPDATE segments SET {assignments} WHERE job_id = ? AND segment_index = ?",
                values,
            )
            self._event(
                job_id,
                "segment.state",
                segment_index=index,
                payload={"state": state.value, **event_updates},
                at=changed,
            )
        return self.segments(job_id)[index]

    def has_submission_budget(self, job_id: str, index: int) -> bool:
        job = self.job(job_id)
        segment = self.segments(job_id)[index]
        return (
            job.max_credits is not None
            and job.submitted_credits + segment.estimated_credits <= job.max_credits
        )

    def mark_submitted(self, job_id: str, index: int, invocation_id: str) -> SegmentRecord:
        if not self.has_submission_budget(job_id, index):
            raise CreditLimitError(f"job {job_id} has reached its credit ceiling")
        segment = self.segments(job_id)[index]
        changed = _now()
        with self._connection:
            self._connection.execute(
                """
                UPDATE segments SET state = ?, invocation_id = ?, attempt_count = ?,
                    next_attempt_at = NULL, failure_code = NULL, failure_message = NULL,
                    updated_at = ? WHERE job_id = ? AND segment_index = ?
                """,
                (
                    SegmentState.SUBMITTED.value,
                    invocation_id,
                    segment.attempt_count + 1,
                    _timestamp(changed),
                    job_id,
                    index,
                ),
            )
            self._connection.execute(
                "UPDATE jobs SET submitted_credits = submitted_credits + ?, "
                "updated_at = ? WHERE id = ?",
                (segment.estimated_credits, _timestamp(changed), job_id),
            )
            self._event(
                job_id,
                "segment.submitted",
                segment_index=index,
                payload={
                    "invocation_id": invocation_id,
                    "estimated_credits": segment.estimated_credits,
                    "attempt": segment.attempt_count + 1,
                    "independent_task": True,
                },
                at=changed,
            )
        return self.segments(job_id)[index]

    def record_event(
        self,
        job_id: str,
        event: str,
        *,
        segment_index: int | None = None,
        payload: object | None = None,
    ) -> None:
        with self._connection:
            self._event(
                job_id,
                event,
                segment_index=segment_index,
                payload=payload,
            )

    def events_since(self, job_id: str, last_id: int = 0) -> tuple[dict[str, object], ...]:
        rows = self._connection.execute(
            "SELECT * FROM events WHERE job_id = ? AND id > ? ORDER BY id",
            (job_id, last_id),
        ).fetchall()
        return tuple(
            {
                "id": int(row["id"]),
                "job_id": str(row["job_id"]),
                "segment_index": row["segment_index"],
                "event": str(row["event"]),
                "payload": json.loads(str(row["payload"])),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        )


def _retry_for(code: str | None, policy: dict[str | None, Retry]) -> Retry:
    if code is not None and (
        code.startswith(("SAFETY.", "ASSET.INVALID"))
        or ".SAFETY." in code
        or code.endswith(".SAFETY")
    ):
        return Retry(0, 0)
    if code is not None:
        for prefix in (
            "INTERNAL.BAD_OUTPUT",
            "INPUT_PREPROCESSING.INTERNAL",
            "THIRD_PARTY.UNAVAILABLE",
            "INTERNAL",
        ):
            if code.startswith(prefix) and prefix in policy:
                return policy[prefix]
    return policy.get(code, policy.get(None, Retry(0, 0)))


def _fresh(uploaded_at: datetime | None, *, now: datetime) -> bool:
    return uploaded_at is not None and now - uploaded_at < timedelta(hours=20)


class PhaseCWorker:
    """Submit all pending sections, then poll every active invocation once."""

    def __init__(
        self,
        *,
        store: PhaseCStore,
        gateway: GenerationGateway,
        retry_policy: dict[str | None, Retry] | None = None,
        now: Callable[[], datetime] = _now,
    ) -> None:
        self.store = store
        self.gateway = gateway
        self.retry_policy = retry_policy or RETRY_POLICY
        self.now = now

    def _ensure_references(self, job: JobRecord) -> JobRecord:
        current = self.now()
        if (
            job.target_face_uri
            and (job.target_voice_path is None or job.target_voice_uri)
            and _fresh(job.references_uploaded_at, now=current)
        ):
            return job
        face_uri = self.gateway.upload(job.target_face_path, role="target_face")
        voice_uri = (
            self.gateway.upload(job.target_voice_path, role="target_voice")
            if job.target_voice_path is not None
            else None
        )
        self.store.update_references(
            job.id,
            target_face_uri=face_uri,
            target_voice_uri=voice_uri,
            uploaded_at=current,
        )
        return self.store.job(job.id)

    def _fail_or_retry(
        self,
        segment: SegmentRecord,
        *,
        failure_code: str | None,
        failure_message: str,
    ) -> None:
        rule = _retry_for(failure_code, self.retry_policy)
        if segment.retry_count < rule.max_retries:
            next_attempt = self.now() + timedelta(seconds=rule.delay_sec)
            self.store.set_segment_state(
                segment.job_id,
                segment.index,
                SegmentState.PENDING,
                invocation_id=None,
                retry_count=segment.retry_count + 1,
                next_attempt_at=next_attempt,
                failure_code=failure_code,
                failure_message=failure_message,
            )
            return
        self.store.set_segment_state(
            segment.job_id,
            segment.index,
            SegmentState.FAILED,
            failure_code=failure_code,
            failure_message=failure_message,
            next_attempt_at=None,
        )

    def _submit(self, job: JobRecord, segment: SegmentRecord) -> None:
        if not self.store.has_submission_budget(job.id, segment.index):
            self.store.set_segment_state(
                job.id,
                segment.index,
                SegmentState.FAILED,
                failure_code="CREDIT_LIMIT",
                failure_message="submission would exceed the explicit job credit ceiling",
            )
            return
        self.store.set_segment_state(job.id, segment.index, SegmentState.UPLOADING)
        current = self.now()
        try:
            if segment.source_uri and _fresh(segment.source_uploaded_at, now=current):
                source_uri = segment.source_uri
            else:
                source_uri = self.gateway.upload(
                    segment.input_path,
                    role=f"segment_{segment.index}",
                )
                self.store.set_segment_state(
                    job.id,
                    segment.index,
                    SegmentState.UPLOADING,
                    source_uri=source_uri,
                    source_uploaded_at=current,
                )
            refreshed_job = self.store.job(job.id)
            if not refreshed_job.target_face_uri:
                raise PipelineError("face reference upload is unavailable")
            if job.model_id not in {"seedance2", "hailuo3"}:
                raise PipelineError(f"unsupported direct reference model: {job.model_id}")
            # Every cut receives a new Runway task with only its original source
            # section and current prompt. No generated output is ever fed into a
            # later request, including retries.
            request = GenerationRequest(
                reference_video=source_uri,
                reference_image=refreshed_job.target_face_uri,
                reference_audio=refreshed_job.target_voice_uri,
                prompt_text=segment.prompt_override or job.prompt,
                duration=segment.requested_duration_sec,
                ratio=job.ratio,
                reference_video_duration_sec=segment.duration_sec,
                model=cast(RunwayReferenceModel, job.model_id),
                resolution=job.resolution,
            )
            if request.estimated_credits != segment.estimated_credits:
                raise PipelineError("stored segment cost no longer matches the request")
            task_id = self.gateway.submit(request)
            self.store.mark_submitted(job.id, segment.index, task_id)
        except RouterConfigurationError as error:
            self.store.set_segment_state(
                job.id,
                segment.index,
                SegmentState.FAILED,
                failure_code="ROUTER.CONFIGURATION",
                failure_message=str(error),
                next_attempt_at=None,
            )
        except PipelineError as error:
            refreshed = self.store.segments(job.id)[segment.index]
            self._fail_or_retry(
                refreshed,
                failure_code=None,
                failure_message=str(error),
            )

    def _poll(self, segment: SegmentRecord) -> None:
        if segment.invocation_id is None:
            self._fail_or_retry(
                segment,
                failure_code="STATE.INVALID",
                failure_message="active segment has no Runway task ID",
            )
            return
        try:
            result = self.gateway.poll(segment.invocation_id)
        except PipelineError as error:
            self.store.record_event(
                segment.job_id,
                "segment.poll_error",
                segment_index=segment.index,
                payload={"message": str(error)},
            )
            return
        if result.status == "THROTTLED":
            self.store.set_segment_state(segment.job_id, segment.index, SegmentState.THROTTLED)
            return
        if result.status == "RUNNING":
            self.store.set_segment_state(segment.job_id, segment.index, SegmentState.RUNNING)
            return
        if result.status in {"FAILED", "CANCELLED"}:
            self._fail_or_retry(
                segment,
                failure_code=result.failure_code or result.status,
                failure_message=result.failure_message or result.status.lower(),
            )
            return
        if result.status != "SUCCEEDED":
            return
        if not result.output_urls:
            self._fail_or_retry(
                segment,
                failure_code="GENERATION.EMPTY_OUTPUT",
                failure_message="generation succeeded without an output URL",
            )
            return
        self.store.set_segment_state(segment.job_id, segment.index, SegmentState.DOWNLOADING)
        try:
            job = self.store.job(segment.job_id)
            if job.route_id == TALKING_WORKFLOW_ROUTE:
                output_key = Path(segment.output_key)
                provider_key = str(output_key.with_name(f"{output_key.stem}_provider.mp4"))
                provider_path = self.gateway.download(result.output_urls[0], provider_key)
                destination = provider_path.with_name(output_key.name)
                output_path = (
                    trim_generated_duration(provider_path, segment.duration_sec, destination)
                    if job.target_voice_path is not None
                    else preserve_source_audio(provider_path, segment.input_path, destination)
                )
            elif job.target_voice_path is None:
                output_key = Path(segment.output_key)
                provider_key = str(output_key.with_name(f"{output_key.stem}_provider.mp4"))
                provider_path = self.gateway.download(result.output_urls[0], provider_key)
                output_path = preserve_source_audio(
                    provider_path, segment.input_path, provider_path.with_name(output_key.name)
                )
            else:
                output_path = self.gateway.download(result.output_urls[0], segment.output_key)
        except Exception as error:
            self.store.record_event(
                segment.job_id,
                "segment.download_error",
                segment_index=segment.index,
                payload={"message": str(error)},
            )
            return
        try:
            actual_duration = probe_video(output_path).duration_sec
        except (IngestError, OSError):
            actual_duration = None
        self.store.set_segment_state(
            segment.job_id,
            segment.index,
            SegmentState.READY_FOR_REVIEW,
            failure_code=None,
            failure_message=None,
            actual_duration_sec=actual_duration,
        )

    def run_once(self, job_id: str) -> JobRecord:
        """Advance a durable job without sleeping."""
        job = self.store.job(job_id)
        if job.state == JobState.CONFIRMED:
            job = self.store.set_job_state(job_id, JobState.RUNNING)
        if job.state not in {JobState.RUNNING, JobState.REVIEW}:
            raise PipelineError(f"job {job_id} cannot run from {job.state.value}")
        if job.state == JobState.REVIEW:
            return job

        segments = self.store.segments(job_id)
        current = self.now()
        pending = tuple(
            segment
            for segment in segments
            if segment.state == SegmentState.PENDING
            and (segment.next_attempt_at is None or segment.next_attempt_at <= current)
        )
        if pending:
            try:
                job = self._ensure_references(job)
            except PipelineError as error:
                self.store.record_event(
                    job_id,
                    "job.reference_upload_error",
                    payload={"message": str(error)},
                )
                return self.store.job(job_id)
            # Submission happens for every eligible segment before any task is polled.
            for segment in pending:
                self._submit(job, segment)

        active_states = {
            SegmentState.SUBMITTED,
            SegmentState.THROTTLED,
            SegmentState.RUNNING,
            SegmentState.DOWNLOADING,
        }
        for segment in self.store.segments(job_id):
            if segment.state in active_states:
                self._poll(segment)

        finished = self.store.segments(job_id)
        terminal = {SegmentState.READY_FOR_REVIEW, SegmentState.FAILED}
        if finished and all(segment.state in terminal for segment in finished):
            return self.store.set_job_state(job_id, JobState.REVIEW)
        return self.store.job(job_id)

    def run_until_review(
        self,
        job_id: str,
        *,
        poll_interval_sec: float = 5.0,
        timeout_sec: float = 7200.0,
    ) -> JobRecord:
        """Run or resume until every segment is ready or permanently failed."""
        started = time.monotonic()
        while True:
            job = self.run_once(job_id)
            if job.state == JobState.REVIEW:
                return job
            if time.monotonic() - started >= timeout_sec:
                raise PipelineError(f"Phase C job {job_id} timed out after {timeout_sec:.0f}s")
            time.sleep(poll_interval_sec)


def _validate_file(path: str | Path, role: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} not found: {resolved}")
    return resolved


def prepare_phase_c_job(
    video: str | Path,
    image: str | Path,
    audio: str | Path | None,
    *,
    predictions: str | Path = "eval/phase3/predictions.json",
    output_root: str | Path = ".cutdetect/pipeline_phase_c",
    cache_dir: str | Path | None = ".cutdetect/cache",
    target_sec: float | None = None,
    min_group_sec: float = 4.0,
    max_group_sec: float = 15.0,
    max_group_segments: int = 4,
    model_id: str = "seedance2",
    ratio: str | None = None,
    resolution: str | None = None,
    prompt: str = UGC_CLONE_V1.body,
    prompt_template_id: str = UGC_CLONE_V1.id,
    prompt_template_version: int = UGC_CLONE_V1.version,
    consent_affirmed: bool = False,
    owner_device_hash: str | None = None,
) -> PreparedJob:
    """Create a fully local, no-charge Phase C job ready for confirmation."""
    source = _validate_file(video, "source video")
    face = _validate_file(image, "target face")
    voice = _validate_file(audio, "target voice") if audio is not None else None
    prediction_path = _validate_file(predictions, "predictions")
    source_probe = probe_video(source)
    if voice is None and not source_probe.has_audio:
        raise PipelineError("the source video has no audio; add an optional voice reference")
    if not prompt.strip():
        raise PipelineError("prompt must not be empty")
    if not consent_affirmed:
        raise PipelineError("confirm that you have permission to use the supplied reference media")
    if model_id not in MODEL_CAPABILITIES:
        raise PipelineError(f"unsupported direct reference model: {model_id}")
    selected_ratio = ratio or "9:16"
    selected_resolution = resolution or ("768P" if model_id == "hailuo3" else "720p")
    route_id = generation_route(model_id)
    grouping = plan_from_predictions(
        prediction_path,
        model_id=model_id,
        target_sec=target_sec,
        min_group_sec=min_group_sec,
        max_group_sec=max_group_sec,
        max_group_segments=max_group_segments,
    )
    job_id = uuid.uuid4().hex
    storage = LocalDiskStorage(output_root)
    job_key = f"jobs/{job_id}"
    exported = split_video(
        source,
        tuple(group.end_frame for group in grouping.groups[:-1]),
        storage.path(f"{job_key}/source_segments"),
        cache_dir=cache_dir,
    )
    input_paths = tuple(
        storage.copy_in(clip.path, f"{job_key}/segments/{index}/input.mp4")
        for index, clip in enumerate(exported.clips)
    )
    output_keys = tuple(
        f"{job_key}/segments/{index}/output_raw.mp4" for index in range(len(grouping.groups))
    )
    stored_face = storage.copy_in(face, f"{job_key}/refs/target_face{face.suffix.lower()}")
    stored_voice = (
        storage.copy_in(voice, f"{job_key}/refs/target_voice{voice.suffix.lower()}")
        if voice is not None
        else None
    )
    grouping_path = storage.write_json(f"{job_key}/grouping.json", grouping.to_dict())
    database_path = storage.path("orchestration.sqlite3")
    with PhaseCStore(database_path) as store:
        job = store.create_job(
            job_id=job_id,
            source_path=source,
            target_face_path=stored_face,
            target_voice_path=stored_voice,
            prompt=prompt,
            grouping=grouping,
            input_paths=input_paths,
            output_keys=output_keys,
            model_id=model_id,
            ratio=selected_ratio,
            resolution=selected_resolution,
            route_id=route_id,
            prompt_template_id=prompt_template_id,
            prompt_template_version=prompt_template_version,
            consent_affirmed=consent_affirmed,
            owner_device_hash=owner_device_hash,
        )
    manifest = {
        "job_id": job.id,
        "state": job.state.value,
        "route_id": job.route_id,
        "model_id": job.model_id,
        "ratio": job.ratio,
        "resolution": job.resolution,
        "prompt_template": {
            "id": job.prompt_template_id,
            "version": job.prompt_template_version,
        },
        "consent_affirmed_at": _timestamp(job.consent_affirmed_at),
        "estimated_credits": job.estimated_credits,
        "audio_mode": "reference" if stored_voice else "source",
        "grouping": str(grouping_path),
        "database": str(database_path),
    }
    storage.write_json(f"{job_key}/job.json", manifest)
    return PreparedJob(
        job=job,
        grouping_path=grouping_path,
        database_path=database_path,
        job_directory=storage.path(job_key),
        generation_count=len(grouping.groups),
    )


def format_sse_events(events: Sequence[dict[str, object]]) -> Iterator[str]:
    """Format stored lifecycle events for a future UI's SSE response."""
    for event in events:
        yield (
            f"id: {event['id']}\n"
            f"event: {event['event']}\n"
            f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
        )


def job_status(store: PhaseCStore, job_id: str) -> dict[str, object]:
    """Return a compact CLI/UI snapshot without exposing signed upload URIs."""
    job = store.job(job_id)
    segments = store.segments(job_id)
    counts: dict[str, int] = {}
    for segment in segments:
        counts[segment.state.value] = counts.get(segment.state.value, 0) + 1
    return {
        "job_id": job.id,
        "state": job.state.value,
        "route_id": job.route_id,
        "model_id": job.model_id,
        "ratio": job.ratio,
        "resolution": job.resolution,
        "prompt_template": {
            "id": job.prompt_template_id,
            "version": job.prompt_template_version,
        },
        "consent_affirmed_at": _timestamp(job.consent_affirmed_at),
        "estimated_credits": job.estimated_credits,
        "max_credits": job.max_credits,
        "submitted_credits": job.submitted_credits,
        "audio_mode": "reference" if job.target_voice_path else "source",
        "final_output_key": job.final_output_key,
        "qc_output_key": job.qc_output_key,
        "completed_at": _timestamp(job.completed_at),
        "segment_states": counts,
        "segments": [
            {
                **asdict(segment),
                "state": segment.state.value,
                "input_path": str(segment.input_path),
                "source_uri": "stored" if segment.source_uri else None,
                "source_uploaded_at": _timestamp(segment.source_uploaded_at),
                "next_attempt_at": _timestamp(segment.next_attempt_at),
                "updated_at": _timestamp(segment.updated_at),
            }
            for segment in segments
        ],
    }
