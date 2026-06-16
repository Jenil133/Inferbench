"""Benchmark orchestration with immutable executions and atomic checkpoints."""

from __future__ import annotations

import json
import os
import platform
import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inferbench import __version__
from inferbench.analysis import analyze
from inferbench.config import Config
from inferbench.external import build_server_command, start_process, stop_process, tool_versions
from inferbench.http_backend import BackendError, HttpSettings, readiness, run_trial
from inferbench.models import BenchmarkReport, MatrixPoint, PointResult, RequestMetric, TrialMetric
from inferbench.nsight import extract_profile
from inferbench.persistence import (
    acquire_execution_lock,
    atomic_write_json,
    code_provenance,
    release_execution_lock,
    sha256_file,
    source_tree_digest,
    utc_now,
)
from inferbench.preflight import run_preflight
from inferbench.simulation import SIMULATION_NOTICE, simulate_trial
from inferbench.statistics import summarize_trials

Checkpoint = Callable[[PointResult], None]


def _execution_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:12]}"


def _environment(mode: str) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "tools": tool_versions() if mode == "hardware" else {"note": "not probed in synthetic mode"},
    }


def _point_from_dict(value: dict[str, Any]) -> PointResult:
    point = MatrixPoint(**value["point"])
    trials = []
    for trial_value in value["trials"]:
        requests = [RequestMetric(**item) for item in trial_value.pop("requests", [])]
        trials.append(TrialMetric(**trial_value, requests=requests))
    return PointResult(
        point=point, trials=trials, summary=value["summary"], status=value.get("status", "completed"),
        profile_status=value.get("profile_status", "disabled"),
        profile_artifact=value.get("profile_artifact"), profile_summary=value.get("profile_summary"),
        notes=value.get("notes", []),
    )


def _load_checkpoint(path: Path) -> PointResult:
    return _point_from_dict(json.loads(path.read_text(encoding="utf-8")))


def _source_root() -> Path:
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").is_file() else Path(__file__).resolve().parent


def _create_or_resume(config: Config, resume_dir: Path | None) -> tuple[str, Path, dict[str, Any], list[PointResult]]:
    fingerprint = config.fingerprint()
    if resume_dir is not None:
        execution_dir = resume_dir.expanduser().resolve()
        manifest_path = execution_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_fingerprint") != fingerprint:
            raise RuntimeError("resume refused: configuration fingerprint does not match manifest")
        current_source = source_tree_digest(_source_root())
        if manifest.get("source_tree", {}).get("sha256") != current_source["sha256"]:
            raise RuntimeError("resume refused: InferBench source/package digest does not match manifest")
        if manifest.get("status") == "completed":
            raise RuntimeError("resume refused: execution is already complete and immutable")
        expected = [point.point_id for point in config.points()]
        actual = [item["point_id"] for item in manifest.get("execution_plan", [])]
        if expected != actual:
            raise RuntimeError("resume refused: execution plan does not match manifest")
        completed = [
            _load_checkpoint(execution_dir / item["checkpoint"])
            for item in manifest.get("points", []) if item.get("status") == "completed"
        ]
        manifest["status"] = "running"
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest["resumed_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        return str(manifest["execution_id"]), execution_dir, manifest, completed
    execution_id = _execution_id()
    execution_dir = config.output_dir / execution_id
    execution_dir.mkdir(parents=True, exist_ok=False)
    source_copy = execution_dir / "config.source.json"
    source_copy.write_bytes(config.source.read_bytes())
    plan = [
        {"sequence": sequence, "point_id": point.point_id, "point": asdict(point)}
        for sequence, point in enumerate(config.points(), 1)
    ]
    project_root = _source_root()
    manifest = {
        "schema_version": "2.0",
        "execution_id": execution_id,
        "config_fingerprint": fingerprint,
        "status": "initializing",
        "created_at": utc_now(),
        "config_source_name": config.source.name,
        "config_source_sha256": sha256_file(source_copy),
        "inferbench_version": __version__,
        "code_provenance": code_provenance(project_root),
        "source_tree": source_tree_digest(project_root),
        "execution_plan": plan,
        "resume_count": 0,
        "points": [],
        "artifacts": [],
    }
    atomic_write_json(execution_dir / "manifest.json", manifest)
    return execution_id, execution_dir, manifest, []


def run(config: Config, resume_dir: Path | None = None) -> BenchmarkReport:
    execution_id, execution_dir, manifest, results = _create_or_resume(config, resume_dir)
    manifest_path = execution_dir / "manifest.json"
    lock_path = acquire_execution_lock(execution_dir)
    try:
        preflight = run_preflight(config, execution_dir)
        manifest["preflight"] = preflight
        manifest["status"] = "running"
        manifest["started_at"] = manifest.get("started_at", utc_now())
        atomic_write_json(manifest_path, manifest)
        if preflight["status"] == "failed":
            raise RuntimeError("hardware preflight failed: " + "; ".join(preflight["errors"]))
        completed_ids = {result.point.point_id for result in results}

        def checkpoint(result: PointResult) -> None:
            attempt = int(manifest.get("resume_count", 0)) + 1
            path = execution_dir / "points" / f"{result.point.point_id}.attempt-{attempt}.json"
            atomic_write_json(path, asdict(result))
            record = {
                "point_id": result.point.point_id,
                "attempt": attempt,
                "status": result.status,
                "checkpoint": str(path.relative_to(execution_dir)),
                "updated_at": utc_now(),
            }
            for index, item in enumerate(manifest["points"]):
                if item["point_id"] == result.point.point_id and item.get("attempt", 1) == attempt:
                    manifest["points"][index] = record
                    break
            else:
                manifest["points"].append(record)
            atomic_write_json(manifest_path, manifest)

        raw = config.raw
        seed = int(raw.get("seed", 2025))
        for point in config.points():
            if point.point_id in completed_ids:
                continue
            if config.mode == "simulate":
                result = _run_synthetic_point(config, point, seed, checkpoint)
            else:
                result = _run_hardware_point(config, point, execution_dir, checkpoint)
            results.append(result)
            checkpoint(result)
        synthetic = config.mode == "simulate"
        report = BenchmarkReport(
            schema_version="2.0",
            config_fingerprint=config.fingerprint(),
            execution_id=execution_id,
            execution_dir=str(execution_dir),
            mode=config.mode,
            synthetic=synthetic,
            synthetic_notice=SIMULATION_NOTICE if synthetic else None,
            model=str(raw["model"]),
            seed=seed,
            warmup_iterations=int(raw["warmup_iterations"]),
            measured_trials=int(raw["measured_trials"]),
            independent_repetitions=config.independent_repetitions,
            request_count=config.request_count,
            environment=_environment(config.mode),
            configuration=config.result_configuration(),
            execution_plan=manifest["execution_plan"],
            results=results,
            correlations=analyze(results, synthetic),
            methodology={
                "aggregation": (
                    "request p50/p99 are pooled across all successful measured requests with explicit "
                    "sample counts; per-trial percentile and throughput medians/IQR are retained"
                ),
                "failed_request_policy": "failures are retained and excluded from latency/TTFT percentiles and token throughput",
                "warmup_excluded": True,
                "request_concurrency": "fixed worker count independent of configured request_count",
                "ttft": "request start to first non-empty streamed text event",
                "latency": "request start through explicit SSE [DONE] completion",
                "throughput": "successful exact usage.completion_tokens divided by whole trial wall time",
                "nsight_association": (
                    "selected OS-runtime wait/blocking duration covers the complete server-process profile; "
                    "benchmark-side trial markers are outside that process tree, so no trial attribution or causal scheduling claim is made"
                ),
                "execution_order": raw.get("execution_order", "counterbalanced"),
            },
        )
        return report
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = utc_now()
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        atomic_write_json(manifest_path, manifest)
        raise
    finally:
        release_execution_lock(lock_path)


def _run_synthetic_point(config: Config, point: MatrixPoint, seed: int, checkpoint: Checkpoint) -> PointResult:
    raw = config.raw
    output_tokens = int(raw.get("output_tokens", 128))
    warmup_count = int(raw.get("warmup_request_count", point.concurrency))
    for iteration in range(int(raw["warmup_iterations"])):
        simulate_trial(point, -iteration - 1, seed, output_tokens, warmup_count)
    trials: list[TrialMetric] = []
    notes = [SIMULATION_NOTICE]
    if point.affinity == "pinned":
        notes.append("Synthetic affinity scenario applies a 0.60 jitter multiplier; it is not hardware evidence.")
    for trial in range(1, int(raw["measured_trials"]) + 1):
        trials.append(simulate_trial(point, trial, seed, output_tokens, config.request_count))
        checkpoint(PointResult(point, list(trials), summarize_trials(trials), status="partial", notes=notes))
    return PointResult(point, trials, summarize_trials(trials), notes=notes)


def _run_hardware_point(config: Config, point: MatrixPoint, execution_dir: Path, checkpoint: Checkpoint) -> PointResult:
    raw, server = config.raw, config.raw["server"]
    nsight = raw.get("nsight", {})
    profile_enabled = bool(nsight.get("enabled", True))
    profile_required = bool(nsight.get("required", True))
    attempt = len(list((execution_dir / "logs").glob(f"{point.point_id}.attempt-*.log"))) + 1
    profile_base = (
        execution_dir / "profiles" / f"{point.point_id}.attempt-{attempt}"
        if profile_enabled else None
    )
    command, profile_status = build_server_command(raw, point, profile_base)
    if profile_status == "missing_tool" and profile_required:
        raise RuntimeError("nsys is required but unavailable")
    log_path = execution_dir / "logs" / f"{point.point_id}.attempt-{attempt}.log"
    process, log = start_process(command, log_path)
    profile_path = Path(f"{profile_base}.nsys-rep") if profile_status == "capture_requested" else None
    api_key = os.environ.get(str(server.get("api_key_env", "INFERBENCH_API_KEY")))
    settings = HttpSettings(
        endpoint=str(server["endpoint"]),
        model=str(server.get("served_model_name", raw["model"])),
        prompt=str(raw.get("prompt", "Explain tensor parallelism in one sentence.")),
        max_tokens=int(raw.get("output_tokens", 128)),
        timeout_seconds=float(server.get("request_timeout_seconds", 120)),
        trial_timeout_seconds=float(server.get("trial_timeout_seconds", 900)),
        api_key=api_key,
        request_format=str(server.get("request_format", "completions")),
    )
    trials: list[TrialMetric] = []
    shutdown: dict[str, Any] | None = None
    notes = ["Hardware measurements from the configured streaming endpoint.", f"Server log: {log_path}"]
    try:
        readiness(
            str(server["readiness_url"]), float(server.get("startup_timeout_seconds", 300)),
            process=process, api_key=api_key,
            accepted_statuses=tuple(server.get("readiness_statuses", [200])), log_path=log_path,
        )
        warmup_count = int(raw.get("warmup_request_count", point.concurrency))
        for warmup in range(int(raw["warmup_iterations"])):
            warmup_result = run_trial(settings, point.concurrency, warmup_count, -warmup - 1, point.repetition, process)
            if not warmup_result.successful_requests:
                raise BackendError("warm-up produced no successful requests")
        for trial in range(1, int(raw["measured_trials"]) + 1):
            trials.append(run_trial(settings, point.concurrency, config.request_count, trial, point.repetition, process))
            checkpoint(PointResult(point, list(trials), summarize_trials(trials), status="partial", profile_status=profile_status, notes=notes))
    finally:
        shutdown = stop_process(process, log, float(server.get("shutdown_timeout_seconds", 20)))
    notes.append(f"Process shutdown: {shutdown}")
    profile_summary: dict[str, Any] | None = None
    if profile_status == "capture_requested" and profile_path and profile_path.exists():
        try:
            profile_summary = extract_profile(profile_path).to_dict()
            profile_status = "parsed"
            notes.append("Selected OS-runtime wait/blocking time is whole-profile and not trial-window isolated.")
        except RuntimeError as exc:
            profile_status = "extract_failed"
            if profile_required:
                raise
            notes.append(f"Best-effort Nsight extraction unavailable: {exc}")
    elif profile_status == "capture_requested":
        profile_status = "missing_artifact"
        message = f"Nsight profile artifact was not created: {profile_path}"
        if profile_required:
            raise RuntimeError(message)
        notes.append(message + "; best-effort profiling did not block the benchmark.")
    elif profile_status == "missing_tool":
        notes.append("nsys was absent; best-effort profiling was skipped without blocking measurements.")
    status = "completed" if all(trial.status == "completed" for trial in trials) else "partial"
    return PointResult(
        point, trials, summarize_trials(trials), status=status, profile_status=profile_status,
        profile_artifact=str(profile_path) if profile_path else None,
        profile_summary=profile_summary, notes=notes,
    )
