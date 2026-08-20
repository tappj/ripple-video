"""Command-line interface for cutdetect."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np

from cutdetect.baselines import BaselineError, run_baselines
from cutdetect.config import (
    BaselineConfig,
    EvaluationConfig,
    IngestConfig,
    LabelConfig,
    StudioConfig,
)
from cutdetect.detect import run_detection
from cutdetect.environment import EnvironmentFileError, load_env_file
from cutdetect.evaluation import (
    evaluate_files,
    load_labels,
    render_pr_curve_svg,
    shot_length_distribution,
    signal_ablation,
)
from cutdetect.export import ExportError, split_from_predictions
from cutdetect.features import FeatureError, extract_features
from cutdetect.ingest import IngestError, ingest_video
from cutdetect.label import label_video, prepare_thumbnails
from cutdetect.pipeline.runway_client import PipelineError
from cutdetect.report import write_debug_report
from cutdetect.studio import run_studio


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cutdetect", description="Talking-head jump-cut detection"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    label_defaults = LabelConfig()

    ingest = subcommands.add_parser("ingest", help="probe and normalize a video")
    ingest.add_argument("video", type=Path)
    ingest.add_argument("--cache-dir", type=Path)
    ingest.add_argument("--output", type=Path, help="write a copy of context JSON here")

    label = subcommands.add_parser("label", help="open the ground-truth labeling UI")
    label.add_argument("video", type=Path)
    label.add_argument("--labels", type=Path, default=Path("labels.json"))
    label.add_argument("--cache-dir", type=Path)
    label.add_argument("--host", default=label_defaults.host)
    label.add_argument("--port", type=int, default=label_defaults.port)
    label.add_argument("--no-open", action="store_true", help="do not open a browser automatically")

    evaluate = subcommands.add_parser("eval", help="score predictions against labels")
    evaluate.add_argument("--labels", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, default=Path("eval"))
    evaluate.add_argument("--include-unsure", action="store_true")
    evaluate.add_argument("--frame-count", type=int)
    evaluate.add_argument("--fps", type=float)
    evaluate.add_argument("--signal-scores", type=Path, help="optional Phase 3 boundary-signal NPZ")
    evaluate.add_argument("--weights", type=Path, help="JSON object of signal weights")
    evaluate.add_argument("--fusion-threshold", type=float, default=0.35)

    baseline = subcommands.add_parser(
        "baseline", help="run and score the Phase 1 detector baselines"
    )
    baseline.add_argument("video", type=Path)
    baseline.add_argument("--labels", type=Path, required=True)
    baseline.add_argument("--output-dir", type=Path, default=Path("eval/baselines"))
    baseline.add_argument("--cache-dir", type=Path)
    baseline.add_argument(
        "--skip-transnet", action="store_true", help="run only the PySceneDetect baselines"
    )

    extract = subcommands.add_parser("extract", help="extract and cache Phase 2 features")
    extract.add_argument("video", type=Path)
    extract.add_argument("--cache-dir", type=Path)
    extract.add_argument("--model", type=Path, help="MediaPipe face_landmarker.task bundle")
    extract.add_argument("--force", action="store_true", help="ignore an existing feature cache")

    detect = subcommands.add_parser("detect", help="run Phase 3 multimodal cut detection")
    detect.add_argument("video", type=Path)
    detect.add_argument("--features", type=Path, help="existing Phase 2 feature archive")
    detect.add_argument("--labels", type=Path, help="ground truth used for Phase 3 tuning")
    detect.add_argument("--words", type=Path, help="optional word timestamp JSON")
    detect.add_argument("--model", type=Path, help="MediaPipe model if extraction is required")
    detect.add_argument("--cache-dir", type=Path)
    detect.add_argument("--output-dir", type=Path, default=Path("eval/phase3"))
    detect.add_argument("--sensitivity", type=float, default=0.5)

    split = subcommands.add_parser("split", help="export clips from prediction JSON")
    split.add_argument("video", type=Path)
    split.add_argument("--predictions", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, default=Path("clips"))
    split.add_argument("--cache-dir", type=Path)

    report = subcommands.add_parser("report", help="create a self-contained debug report")
    report.add_argument("video", type=Path)
    report.add_argument("--predictions", type=Path, required=True)
    report.add_argument("--normalized-signals", type=Path, required=True)
    report.add_argument("--labels", type=Path, help="optional ground-truth JSON")
    report.add_argument("--output", type=Path, default=Path("cutdetect-report.html"))
    report.add_argument("--cache-dir", type=Path)

    studio_defaults = StudioConfig()
    studio = subcommands.add_parser("studio", help="open the drag-and-drop Cut Room")
    studio.add_argument("--host", default=studio_defaults.host)
    studio.add_argument("--port", type=int, default=studio_defaults.port)
    studio.add_argument("--cache-dir", type=Path, default=Path(".cutdetect/cache"))
    studio.add_argument("--jobs-dir", type=Path, default=Path(".cutdetect/studio/jobs"))
    studio.add_argument("--no-open", action="store_true", help="do not open a browser")

    pipeline = subcommands.add_parser("pipeline", help="Runway regeneration pipeline")
    pipeline_commands = pipeline.add_subparsers(dest="pipeline_command", required=True)
    gen_one = pipeline_commands.add_parser("gen-one", help="run the Phase A single-clip proof")
    gen_one.add_argument("video", type=Path)
    gen_one.add_argument("image", type=Path)
    gen_one.add_argument("audio", type=Path)
    gen_one.add_argument("--start", type=float, required=True)
    gen_one.add_argument("--end", type=float, required=True)
    gen_one.add_argument("--model", choices=("seedance2", "hailuo3"), default="seedance2")
    gen_one.add_argument("--ratio", help="output size/aspect ratio; defaults to source orientation")
    gen_one.add_argument("--prompt", help="override the ugc_clone_v1 prompt")
    gen_one.add_argument(
        "--max-credits",
        type=int,
        help="required for a paid call; request is refused above this hard ceiling",
    )
    gen_one.add_argument(
        "--dry-run", action="store_true", help="validate and estimate; no API calls"
    )
    gen_one.add_argument("--output-root", type=Path, default=Path(".cutdetect/pipeline"))
    gen_one.add_argument("--cache-dir", type=Path, default=Path(".cutdetect/cache"))
    gen_one.add_argument("--predictions", type=Path, default=Path("eval/phase3/predictions.json"))

    plan = pipeline_commands.add_parser(
        "plan", help="group complete cut sections without erasing hard boundaries"
    )
    plan.add_argument("video", type=Path)
    plan.add_argument("--predictions", type=Path, default=Path("eval/phase3/predictions.json"))
    plan.add_argument("--model", choices=("seedance2", "hailuo3"), default="seedance2")
    plan.add_argument("--target", type=float, help="generation clip duration target in seconds")
    plan.add_argument("--min-group-sec", type=float, default=4.0)
    plan.add_argument(
        "--max-group-sec",
        type=float,
        default=15.0,
        help="hard model-safe ceiling; clips remain at or below 10s when possible",
    )
    plan.add_argument(
        "--max-group-segments",
        type=int,
        choices=(1, 2, 3, 4),
        default=4,
        help="preferred complete sections per clip; 1 explicitly disables grouping",
    )
    plan.add_argument("--output", type=Path, help="also save the JSON plan here")

    prepare_job = pipeline_commands.add_parser(
        "prepare-job", help="prepare a durable Phase C job without API calls"
    )
    prepare_job.add_argument("video", type=Path)
    prepare_job.add_argument("image", type=Path)
    prepare_job.add_argument(
        "audio",
        type=Path,
        nargs="?",
        help="optional voice reference; omit to preserve the source video's audio",
    )
    prepare_job.add_argument(
        "--predictions", type=Path, default=Path("eval/phase3/predictions.json")
    )
    prepare_job.add_argument(
        "--output-root", type=Path, default=Path(".cutdetect/pipeline_phase_c")
    )
    prepare_job.add_argument("--cache-dir", type=Path, default=Path(".cutdetect/cache"))
    prepare_job.add_argument("--target", type=float)
    prepare_job.add_argument("--min-group-sec", type=float, default=4.0)
    prepare_job.add_argument(
        "--max-group-sec",
        type=float,
        default=15.0,
        help="hard model-safe ceiling; clips remain at or below 10s when possible",
    )
    prepare_job.add_argument("--max-group-segments", type=int, choices=(1, 2, 3, 4), default=4)
    prepare_job.add_argument("--model", choices=("seedance2", "hailuo3"), default="seedance2")
    prepare_job.add_argument("--ratio", help="routed output aspect ratio; defaults to 9:16")
    prepare_job.add_argument(
        "--resolution", help="routed output resolution; defaults to 720p or Hailuo 768P"
    )
    prepare_job.add_argument("--prompt")
    prepare_job.add_argument(
        "--consent",
        action="store_true",
        help="affirm permission to use the supplied video, likeness, and voice",
    )

    run_job = pipeline_commands.add_parser(
        "run-job", help="confirm and run or resume a prepared Phase C job"
    )
    run_job.add_argument("job_id")
    run_job.add_argument("--max-credits", type=int, required=True)
    run_job.add_argument("--output-root", type=Path, default=Path(".cutdetect/pipeline_phase_c"))
    run_job.add_argument("--once", action="store_true", help="advance once and exit")
    run_job.add_argument("--poll-interval", type=float, default=5.0)

    status = pipeline_commands.add_parser("job-status", help="print a durable Phase C job snapshot")
    status.add_argument("job_id")
    status.add_argument("--output-root", type=Path, default=Path(".cutdetect/pipeline_phase_c"))

    pipeline_studio = pipeline_commands.add_parser(
        "studio", help="open the complete generation, review, and stitch interface"
    )
    pipeline_studio.add_argument("--host", default="127.0.0.1")
    pipeline_studio.add_argument("--port", type=int, default=8790)
    pipeline_studio.add_argument(
        "--output-root", type=Path, default=Path(".cutdetect/pipeline_studio")
    )
    pipeline_studio.add_argument("--cache-dir", type=Path, default=Path(".cutdetect/cache"))
    pipeline_studio.add_argument("--no-open", action="store_true")

    review_status = pipeline_commands.add_parser(
        "review-status", help="show the Phase D review gate and stitch readiness"
    )
    review_status.add_argument("job_id")
    review_status.add_argument(
        "--output-root", type=Path, default=Path(".cutdetect/pipeline_phase_c")
    )

    review_suggest = pipeline_commands.add_parser(
        "review-suggest", help="suggest a silence-plus-low-motion tail trim"
    )
    review_suggest.add_argument("job_id")
    review_suggest.add_argument("segment_index", type=int)
    review_suggest.add_argument(
        "--output-root", type=Path, default=Path(".cutdetect/pipeline_phase_c")
    )

    review_trim = pipeline_commands.add_parser(
        "review-trim", help="write a reversible frame-accurate output_final clip"
    )
    review_trim.add_argument("job_id")
    review_trim.add_argument("segment_index", type=int)
    review_trim.add_argument("--start-frame", type=int, default=0)
    review_trim.add_argument("--end-frame", type=int)
    review_trim.add_argument(
        "--output-root", type=Path, default=Path(".cutdetect/pipeline_phase_c")
    )

    review_approve = pipeline_commands.add_parser(
        "review-approve", help="approve and lock one reviewed clip"
    )
    review_approve.add_argument("job_id")
    review_approve.add_argument("segment_index", type=int)
    review_approve.add_argument(
        "--output-root", type=Path, default=Path(".cutdetect/pipeline_phase_c")
    )

    review_approve_all = pipeline_commands.add_parser(
        "review-approve-all", help="approve all clips only when every clip is reviewable"
    )
    review_approve_all.add_argument("job_id")
    review_approve_all.add_argument(
        "--output-root", type=Path, default=Path(".cutdetect/pipeline_phase_c")
    )

    review_regenerate = pipeline_commands.add_parser(
        "review-regenerate", help="queue one immutable paid regeneration attempt"
    )
    review_regenerate.add_argument("job_id")
    review_regenerate.add_argument("segment_index", type=int)
    review_regenerate.add_argument("--prompt", required=True)
    review_regenerate.add_argument("--max-credits", type=int, required=True)
    review_regenerate.add_argument(
        "--output-root", type=Path, default=Path(".cutdetect/pipeline_phase_c")
    )

    stitch = pipeline_commands.add_parser(
        "stitch-job", help="normalize, validate, and stitch a fully approved job"
    )
    stitch.add_argument("job_id")
    stitch.add_argument("--output-root", type=Path, default=Path(".cutdetect/pipeline_phase_c"))

    probe = pipeline_commands.add_parser(
        "probe-capabilities", help="run five minimal paid Runway diagnostic probes"
    )
    probe.add_argument("video", type=Path)
    probe.add_argument("image", type=Path)
    probe.add_argument("audio", type=Path)
    probe.add_argument("--max-credits", type=int, required=True)
    probe.add_argument("--output-root", type=Path, default=Path(".cutdetect/pipeline/probes"))
    probe.add_argument("--cache-dir", type=Path, default=Path(".cutdetect/cache"))
    return parser


def _ingest(args: argparse.Namespace) -> int:
    config = IngestConfig(cache_dir=args.cache_dir)
    context = ingest_video(args.video, config)
    serialized = json.dumps(context.to_dict(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


def _label(args: argparse.Namespace) -> int:
    label_video(
        args.video,
        labels_path=args.labels,
        ingest_config=IngestConfig(cache_dir=args.cache_dir),
        label_config=LabelConfig(host=args.host, port=args.port),
        open_browser=not args.no_open,
    )
    return 0


def _ablation(args: argparse.Namespace, config: EvaluationConfig) -> dict[str, float] | None:
    if args.signal_scores is None and args.weights is None:
        return None
    if args.signal_scores is None or args.weights is None:
        raise ValueError("--signal-scores and --weights must be provided together")
    raw_weights = json.loads(args.weights.read_text(encoding="utf-8"))
    if not isinstance(raw_weights, dict):
        raise ValueError("weights JSON must be an object")
    weights = {str(name): float(value) for name, value in raw_weights.items()}
    with np.load(args.signal_scores, allow_pickle=False) as archive:
        scores = {name: archive[name].astype(float).tolist() for name in archive.files}
    labels = load_labels(args.labels, include_unsure=config.include_unsure_labels)
    return signal_ablation(
        scores,
        weights,
        labels,
        threshold=args.fusion_threshold,
        tolerance_frames=config.primary_tolerance_frames,
    )


def _evaluate(args: argparse.Namespace) -> int:
    config = EvaluationConfig(include_unsure_labels=args.include_unsure)
    result = evaluate_files(args.labels, args.predictions, config)
    result["signal_ablation_f1"] = _ablation(args, config)
    if args.frame_count is not None or args.fps is not None:
        if args.frame_count is None or args.fps is None:
            raise ValueError("--frame-count and --fps must be provided together")
        result["shot_length_distribution"] = shot_length_distribution(
            load_labels(args.labels, include_unsure=config.include_unsure_labels),
            args.frame_count,
            args.fps,
        )
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "evaluation.json"
    curve_path = output_dir / "pr_curve.svg"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    curve = result["pr_curve"]
    if not isinstance(curve, list):
        raise TypeError("evaluation curve has unexpected type")
    curve_path.write_text(render_pr_curve_svg(curve), encoding="utf-8")
    summary = {
        "evaluation": str(result_path.resolve()),
        "pr_curve": str(curve_path.resolve()),
        "label_count": result["label_count"],
        "prediction_count": result["prediction_count"],
        "metrics_by_tolerance": result["metrics_by_tolerance"],
    }
    print(json.dumps(summary, indent=2))
    return 0


def _baseline(args: argparse.Namespace) -> int:
    result = run_baselines(
        args.video,
        args.labels,
        args.output_dir,
        cache_dir=args.cache_dir,
        config=BaselineConfig(include_transnet=not args.skip_transnet),
    )
    print(json.dumps(result, indent=2))
    return 0


def _extract(args: argparse.Namespace) -> int:
    result = extract_features(
        args.video,
        cache_dir=args.cache_dir,
        model_path=args.model,
        force=args.force,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _detect(args: argparse.Namespace) -> int:
    result = run_detection(
        args.video,
        args.output_dir,
        features_path=args.features,
        labels_path=args.labels,
        words_path=args.words,
        model_path=args.model,
        cache_dir=args.cache_dir,
        sensitivity=args.sensitivity,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _split(args: argparse.Namespace) -> int:
    result = split_from_predictions(
        args.video,
        args.predictions,
        args.output_dir,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _report(args: argparse.Namespace) -> int:
    context = ingest_video(args.video, IngestConfig(cache_dir=args.cache_dir))
    thumbnails = prepare_thumbnails(context, LabelConfig())
    path = write_debug_report(
        args.predictions,
        args.normalized_signals,
        thumbnails,
        args.output,
        args.labels,
    )
    print(json.dumps({"report": str(path)}, indent=2))
    return 0


def _studio(args: argparse.Namespace) -> int:
    run_studio(
        config=StudioConfig(host=args.host, port=args.port),
        cache_dir=args.cache_dir,
        jobs_dir=args.jobs_dir,
        open_browser=not args.no_open,
    )
    return 0


def _pipeline(args: argparse.Namespace) -> int:
    if args.pipeline_command == "studio":
        from cutdetect.pipeline.app import PipelineStudioConfig, run_pipeline_studio

        run_pipeline_studio(
            config=PipelineStudioConfig(host=args.host, port=args.port),
            output_root=args.output_root,
            cache_dir=args.cache_dir,
            open_browser=not args.no_open,
        )
        return 0
    if args.pipeline_command == "prepare-job":
        from cutdetect.pipeline.orchestration import prepare_phase_c_job
        from cutdetect.pipeline.templates import UGC_CLONE_V1

        prepared = prepare_phase_c_job(
            args.video,
            args.image,
            args.audio,
            predictions=args.predictions,
            output_root=args.output_root,
            cache_dir=args.cache_dir,
            target_sec=args.target,
            min_group_sec=args.min_group_sec,
            max_group_sec=args.max_group_sec,
            max_group_segments=args.max_group_segments,
            model_id=args.model,
            ratio=args.ratio,
            resolution=args.resolution,
            prompt=args.prompt or UGC_CLONE_V1.body,
            consent_affirmed=args.consent,
        )
        print(json.dumps(prepared.to_dict(), indent=2))
        return 0
    if args.pipeline_command == "job-status":
        from cutdetect.pipeline.orchestration import PhaseCStore, job_status

        database = args.output_root.expanduser().resolve() / "orchestration.sqlite3"
        with PhaseCStore(database) as store:
            print(json.dumps(job_status(store, args.job_id), indent=2, default=str))
        return 0
    if args.pipeline_command.startswith("review-"):
        from cutdetect.pipeline.orchestration import PhaseCStore
        from cutdetect.pipeline.review import ReviewService
        from cutdetect.pipeline.storage import LocalDiskStorage

        storage = LocalDiskStorage(args.output_root)
        database = storage.path("orchestration.sqlite3")
        with PhaseCStore(database) as store:
            review = ReviewService(store=store, storage=storage)
            if args.pipeline_command == "review-status":
                payload: object = review.snapshot(args.job_id)
            elif args.pipeline_command == "review-suggest":
                payload = review.suggest(args.job_id, args.segment_index).to_dict()
            elif args.pipeline_command == "review-trim":
                end_frame = args.end_frame
                if end_frame is None:
                    end_frame = review.suggest(args.job_id, args.segment_index).end_frame
                payload = asdict(
                    review.trim(
                        args.job_id,
                        args.segment_index,
                        start_frame=args.start_frame,
                        end_frame=end_frame,
                    )
                )
            elif args.pipeline_command == "review-approve":
                payload = asdict(review.approve(args.job_id, args.segment_index))
            elif args.pipeline_command == "review-approve-all":
                payload = [asdict(segment) for segment in review.approve_all(args.job_id)]
            elif args.pipeline_command == "review-regenerate":
                payload = asdict(
                    review.regenerate(
                        args.job_id,
                        args.segment_index,
                        prompt=args.prompt,
                        max_credits=args.max_credits,
                    )
                )
            else:
                raise ValueError(f"unknown review command: {args.pipeline_command}")
            print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.pipeline_command == "stitch-job":
        from cutdetect.pipeline.orchestration import PhaseCStore
        from cutdetect.pipeline.stitch import stitch_job
        from cutdetect.pipeline.storage import LocalDiskStorage

        storage = LocalDiskStorage(args.output_root)
        with PhaseCStore(storage.path("orchestration.sqlite3")) as store:
            print(json.dumps(stitch_job(store, storage, args.job_id).to_dict(), indent=2))
        return 0
    if args.pipeline_command == "run-job":
        from cutdetect.pipeline.orchestration import (
            DIRECT_API_ROUTE,
            GenerationGateway,
            JobState,
            PhaseCStore,
            PhaseCWorker,
            job_status,
        )
        from cutdetect.pipeline.runway_client import (
            MODEL_ROUTER_ROUTE_PREFIX,
            JsonlCallLogger,
            RunwayDirectGateway,
            RunwayReferenceModel,
            RunwayRouterGateway,
            router_config_id_from_route,
        )
        from cutdetect.pipeline.storage import LocalDiskStorage
        from cutdetect.pipeline.workflow_client import (
            TALKING_WORKFLOW_ROUTE,
            RunwayWorkflowGateway,
        )

        root = args.output_root.expanduser().resolve()
        storage = LocalDiskStorage(root)
        database = storage.path("orchestration.sqlite3")
        with PhaseCStore(database) as store:
            job = store.job(args.job_id)
            if job.consent_affirmed_at is None:
                raise ValueError("job has no recorded reference-media permission affirmation")
            if job.state not in {JobState.DRAFT, JobState.ESTIMATING, JobState.CONFIRMED} and (
                job.max_credits != args.max_credits
            ):
                raise ValueError(
                    f"running job ceiling is locked at {job.max_credits}; "
                    f"received {args.max_credits}"
                )
            logger = JsonlCallLogger(storage.path(f"jobs/{args.job_id}/runway_calls.jsonl"))
            gateway: GenerationGateway
            if job.route_id == TALKING_WORKFLOW_ROUTE:
                gateway = RunwayWorkflowGateway(
                    api_key=os.environ.get("RUNWAYML_API_SECRET", ""),
                    logger=logger,
                    storage=storage,
                )
            elif job.route_id == DIRECT_API_ROUTE:
                gateway = RunwayDirectGateway(
                    api_key=os.environ.get("RUNWAYML_API_SECRET", ""),
                    logger=logger,
                    storage=storage,
                )
            elif job.route_id.startswith(MODEL_ROUTER_ROUTE_PREFIX):
                gateway = RunwayRouterGateway(
                    api_key=os.environ.get("RUNWAYML_API_SECRET", ""),
                    logger=logger,
                    storage=storage,
                    config_id=router_config_id_from_route(job.route_id),
                    expected_model=cast(RunwayReferenceModel, job.model_id),
                )
            else:
                raise ValueError(f"unsupported generation route: {job.route_id}")
            if job.state in {JobState.DRAFT, JobState.ESTIMATING, JobState.CONFIRMED}:
                job = store.confirm(args.job_id, args.max_credits)
            worker = PhaseCWorker(store=store, gateway=gateway)
            if args.once:
                worker.run_once(args.job_id)
            else:
                worker.run_until_review(
                    args.job_id,
                    poll_interval_sec=args.poll_interval,
                )
            print(json.dumps(job_status(store, args.job_id), indent=2, default=str))
        return 0
    if args.pipeline_command == "plan":
        from cutdetect.pipeline.grouping import plan_from_predictions

        if not args.video.expanduser().resolve().is_file():
            raise FileNotFoundError(f"video not found: {args.video.expanduser().resolve()}")
        grouping = plan_from_predictions(
            args.predictions,
            model_id=args.model,
            target_sec=args.target,
            min_group_sec=args.min_group_sec,
            max_group_sec=args.max_group_sec,
            max_group_segments=args.max_group_segments,
            source_video=args.video,
        )
        payload = grouping.to_dict()
        if args.output is not None:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0
    if args.pipeline_command == "gen-one":
        from cutdetect.pipeline.gen_one import generate_one, plan_one
        from cutdetect.pipeline.templates import UGC_CLONE_V1

        values = {
            "start_sec": args.start,
            "end_sec": args.end,
            "model": args.model,
            "ratio": args.ratio,
            "prompt": args.prompt or UGC_CLONE_V1.body,
            "predictions": args.predictions,
        }
        if args.dry_run:
            plan = plan_one(args.video, args.image, args.audio, **values)
            print(json.dumps({"dry_run": True, **plan.to_dict()}, indent=2))
            return 0
        if args.max_credits is None:
            raise ValueError("--max-credits is required unless --dry-run is used")
        generation_result = generate_one(
            args.video,
            args.image,
            args.audio,
            max_credits=args.max_credits,
            output_root=args.output_root,
            cache_dir=args.cache_dir,
            **values,
        )
        print(json.dumps(generation_result.to_dict(), indent=2))
        return 0
    if args.pipeline_command == "probe-capabilities":
        from cutdetect.pipeline.probe_suite import run_capability_probe_suite

        probe_result = run_capability_probe_suite(
            args.video,
            args.image,
            args.audio,
            max_credits=args.max_credits,
            output_root=args.output_root,
            cache_dir=args.cache_dir,
        )
        print(json.dumps(probe_result.to_dict(), indent=2))
        return 0
    raise ValueError(f"unknown pipeline command: {args.pipeline_command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return an exit status."""
    parser = _parser()
    try:
        load_env_file()
    except EnvironmentFileError as error:
        parser.error(str(error))
    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            return _ingest(args)
        if args.command == "label":
            return _label(args)
        if args.command == "eval":
            return _evaluate(args)
        if args.command == "baseline":
            return _baseline(args)
        if args.command == "extract":
            return _extract(args)
        if args.command == "detect":
            return _detect(args)
        if args.command == "split":
            return _split(args)
        if args.command == "report":
            return _report(args)
        if args.command == "studio":
            return _studio(args)
        if args.command == "pipeline":
            return _pipeline(args)
    except (
        BaselineError,
        ExportError,
        FeatureError,
        FileNotFoundError,
        IngestError,
        PipelineError,
        ValueError,
    ) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    sys.exit(main())
