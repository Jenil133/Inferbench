"""Typed domain models and stable serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class MatrixPoint:
    concurrency: int
    tensor_parallel: int
    precision: str
    affinity: str = "unpinned"
    repetition: int = 1

    @property
    def batch_size(self) -> int:
        """Compatibility alias for command templates and older consumers."""
        return self.concurrency

    @property
    def point_id(self) -> str:
        return (
            f"c{self.concurrency}-tp{self.tensor_parallel}-{self.precision}-"
            f"{self.affinity}-r{self.repetition}"
        )

    @property
    def comparison_id(self) -> str:
        return f"c{self.concurrency}-tp{self.tensor_parallel}-{self.precision}"


@dataclass(frozen=True)
class RequestMetric:
    request_id: str
    sequence: int
    success: bool
    started_at: str
    finished_at: str
    ttft_ms: float | None
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    http_status: int | None = None
    response_id: str | None = None
    events: int = 0


@dataclass(frozen=True)
class TrialMetric:
    trial: int
    repetition: int
    status: str
    concurrency: int
    request_count: int
    successful_requests: int
    failed_requests: int
    wall_time_s: float
    throughput_tokens_s: float | None
    ttft_p50_ms: float | None
    ttft_p99_ms: float | None
    latency_p50_ms: float | None
    latency_p99_ms: float | None
    selected_os_runtime_wait_ms: float | None = None
    requests: list[RequestMetric] = field(default_factory=list)


@dataclass
class PointResult:
    point: MatrixPoint
    trials: list[TrialMetric]
    summary: dict[str, Any]
    status: str = "completed"
    profile_status: str = "disabled"
    profile_artifact: str | None = None
    profile_summary: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    schema_version: str
    config_fingerprint: str
    execution_id: str
    execution_dir: str
    mode: str
    synthetic: bool
    synthetic_notice: str | None
    model: str
    seed: int
    warmup_iterations: int
    measured_trials: int
    independent_repetitions: int
    request_count: int
    environment: dict[str, Any]
    configuration: dict[str, Any]
    execution_plan: list[dict[str, Any]]
    results: list[PointResult]
    correlations: list[dict[str, Any]]
    methodology: dict[str, Any]

    @property
    def run_id(self) -> str:
        """Deprecated compatibility alias: this is the unique execution ID."""
        return self.execution_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
