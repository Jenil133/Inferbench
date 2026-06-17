"""Atomic JSON, honest CSV contracts, and Markdown report generation."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from inferbench.models import BenchmarkReport
from inferbench.persistence import (
    acquire_execution_lock,
    atomic_write_json,
    atomic_write_text,
    hash_execution_artifacts,
    release_execution_lock,
    utc_now,
)

TRIAL_FIELDS = [
    "schema_version", "execution_id", "config_fingerprint", "point_id", "repetition",
    "concurrency", "tensor_parallel", "precision", "affinity", "trial", "status",
    "request_count", "successful_requests", "failed_requests", "wall_time_s",
    "throughput_successful_tokens_s", "ttft_success_p50_ms", "ttft_success_p99_ms",
    "latency_success_p50_ms", "latency_success_p99_ms", "profile_status", "synthetic",
]
REQUEST_FIELDS = [
    "schema_version", "execution_id", "config_fingerprint", "point_id", "repetition",
    "trial", "request_id", "sequence", "success", "started_at", "finished_at", "ttft_ms",
    "latency_ms", "prompt_tokens", "completion_tokens", "total_tokens", "finish_reason",
    "error_type", "error_message", "http_status", "response_id", "events", "synthetic",
]


def _csv_text(fields: list[str], rows: list[dict[str, object]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _write_report(report: BenchmarkReport, output_dir: Path | None = None) -> list[Path]:
    output_dir = Path(report.execution_dir) if output_dir is None else output_dir
    if output_dir.resolve() != Path(report.execution_dir).resolve():
        raise RuntimeError("reports must be written inside their immutable execution directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    if payload.get("schema_version") != "2.0" or not payload.get("execution_id"):
        raise RuntimeError("report does not satisfy InferBench report schema contract")
    json_path = output_dir / "report.json"
    trial_path = output_dir / "trials.csv"
    request_path = output_dir / "requests.csv"
    markdown_path = output_dir / "report.md"
    atomic_write_json(json_path, payload)
    trial_rows: list[dict[str, object]] = []
    request_rows: list[dict[str, object]] = []
    for result in report.results:
        point = result.point
        for trial in result.trials:
            trial_rows.append(
                {
                    "schema_version": report.schema_version, "execution_id": report.execution_id,
                    "config_fingerprint": report.config_fingerprint, "point_id": point.point_id,
                    "repetition": point.repetition, "concurrency": point.concurrency,
                    "tensor_parallel": point.tensor_parallel, "precision": point.precision,
                    "affinity": point.affinity, "trial": trial.trial, "status": trial.status,
                    "request_count": trial.request_count, "successful_requests": trial.successful_requests,
                    "failed_requests": trial.failed_requests, "wall_time_s": trial.wall_time_s,
                    "throughput_successful_tokens_s": trial.throughput_tokens_s,
                    "ttft_success_p50_ms": trial.ttft_p50_ms, "ttft_success_p99_ms": trial.ttft_p99_ms,
                    "latency_success_p50_ms": trial.latency_p50_ms,
                    "latency_success_p99_ms": trial.latency_p99_ms,
                    "profile_status": result.profile_status, "synthetic": str(report.synthetic).lower(),
                }
            )
            for request in trial.requests:
                request_rows.append(
                    {
                        "schema_version": report.schema_version, "execution_id": report.execution_id,
                        "config_fingerprint": report.config_fingerprint, "point_id": point.point_id,
                        "repetition": point.repetition, "trial": trial.trial,
                        "request_id": request.request_id, "sequence": request.sequence,
                        "success": str(request.success).lower(), "started_at": request.started_at,
                        "finished_at": request.finished_at, "ttft_ms": request.ttft_ms,
                        "latency_ms": request.latency_ms, "prompt_tokens": request.prompt_tokens,
                        "completion_tokens": request.completion_tokens, "total_tokens": request.total_tokens,
                        "finish_reason": request.finish_reason, "error_type": request.error_type,
                        "error_message": request.error_message, "http_status": request.http_status,
                        "response_id": request.response_id, "events": request.events,
                        "synthetic": str(report.synthetic).lower(),
                    }
                )
    atomic_write_text(trial_path, _csv_text(TRIAL_FIELDS, trial_rows))
    atomic_write_text(request_path, _csv_text(REQUEST_FIELDS, request_rows))
    atomic_write_text(markdown_path, _markdown(report))
    paths = [json_path, trial_path, request_path, markdown_path]
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "completed"
    manifest["completed_at"] = utc_now()
    manifest["artifacts"] = hash_execution_artifacts(output_dir)
    atomic_write_json(manifest_path, manifest)
    return paths


def _value(value: object, suffix: str = "") -> str:
    return "n/a" if value is None else f"{float(value):.2f}{suffix}"


def _markdown(report: BenchmarkReport) -> str:
    lines = ["# InferBench Report", ""]
    if report.synthetic_notice:
        lines.extend([f"> **{report.synthetic_notice}**", ""])
    lines.extend(
        [
            f"- Execution ID: `{report.execution_id}`",
            f"- Config fingerprint: `{report.config_fingerprint}`",
            f"- Schema: `{report.schema_version}`",
            f"- Mode: **{report.mode}**",
            f"- Requests per measured trial: {report.request_count}",
            f"- Independent server repetitions: {report.independent_repetitions}",
            "",
            "## Pooled request results and trial spread",
            "",
            "Pooled percentiles use successful requests and show their sample count. Trial p99 is shown as median (IQR). Failed requests remain in JSON and requests.csv.",
            "",
            "| Point repetition | Success / failed | Pooled latency p50 / p99 ms | Trial p99 median (IQR) ms | Throughput median (IQR) tok/s | Profile status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for result in report.results:
        summary = result.summary
        pooled = f"{_value(summary.get('pooled_latency_p50_ms'))} / {_value(summary.get('pooled_latency_p99_ms'))}"
        trial_p99 = f"{_value(summary.get('latency_p99_ms_trial_median'))} ({_value(summary.get('latency_p99_ms_trial_iqr'))})"
        throughput = f"{_value(summary.get('throughput_tokens_s_trial_median'))} ({_value(summary.get('throughput_tokens_s_trial_iqr'))})"
        lines.append(
            f"| `{result.point.point_id}` | {summary.get('pooled_successful_requests', 0)} / "
            f"{summary.get('pooled_failed_requests', 0)} | {pooled} (n={summary.get('pooled_latency_samples', 0)}) | "
            f"{trial_p99} | {throughput} | `{result.profile_status}` |"
        )
        for note in result.notes:
            lines.append(f"<!-- {result.point.point_id}: {note.replace('--', '—')} -->")
    profiles = [result for result in report.results if result.profile_summary]
    if profiles:
        lines.extend(["", "## Selected OS-runtime wait/blocking time", ""])
        lines.append("Whole-profile totals include startup, warm-up, and shutdown. They are not trial-window isolated and do not establish scheduling causation.")
        for result in profiles:
            value = result.profile_summary.get("selected_os_runtime_wait_ms") if result.profile_summary else None
            lines.append(f"- `{result.point.point_id}`: {_value(value)} ms (`{result.profile_artifact}`)")
    lines.extend(["", "## Analysis", ""])
    for item in report.correlations:
        if item["analysis"] == "p99_selected_os_runtime_wait_association":
            lines.append(f"- Exploratory selected-wait association: r=`{item['pearson_r']}`, n={item['samples']}. {item['interpretation']}")
        else:
            lines.append(
                f"- **{item['label']}** `{item['precision']}` concurrency={item['concurrency']} "
                f"tp={item['tensor_parallel']}: p99 IQR {item['unpinned_p99_iqr_ms']:.2f} → "
                f"{item['pinned_p99_iqr_ms']:.2f} ms across {item['independent_repetitions']} repetitions."
            )
    lines.extend(
        [
            "", "## Methodology limitations", "",
            "- Warm-ups are excluded; request errors are retained and excluded from success-only latency/throughput statistics.",
            "- Concurrency controls in-flight workers; request_count independently controls measured sample size.",
            "- Pinned/unpinned order is seeded and counterbalanced or randomized according to the saved execution plan.",
            "- InferBench does not claim that OS-runtime waiting caused latency. Open the timeline for investigation.",
            "- report.json and manifest.json are authoritative; CSV column names explicitly identify success-only metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: BenchmarkReport, output_dir: Path | None = None) -> list[Path]:
    """Finalize an execution under lock and persist finalization failures."""
    directory = Path(report.execution_dir) if output_dir is None else output_dir
    manifest_path = directory / "manifest.json"
    lock_path = acquire_execution_lock(directory)
    try:
        return _write_report(report, directory)
    except BaseException as exc:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "failed"
            manifest["failed_at"] = utc_now()
            manifest["failure"] = {
                "type": type(exc).__name__,
                "stage": "report_finalization",
                "message": str(exc)[:1000],
            }
            atomic_write_json(manifest_path, manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        raise
    finally:
        release_execution_lock(lock_path)
