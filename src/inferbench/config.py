"""Configuration loading, validation, fingerprinting, and execution planning."""

from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from inferbench.models import MatrixPoint


class ConfigError(ValueError):
    """Raised when a benchmark configuration is invalid."""


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    source: Path

    @property
    def mode(self) -> str:
        return str(self.raw.get("mode", "simulate"))

    @property
    def output_dir(self) -> Path:
        value = Path(str(self.raw.get("output_dir", "reports")))
        return (value if value.is_absolute() else self.source.parent / value).resolve()

    @property
    def concurrency_values(self) -> list[int]:
        matrix = self.raw["matrix"]
        return [int(value) for value in matrix.get("concurrency", matrix.get("batch_sizes", []))]

    @property
    def request_count(self) -> int:
        return int(self.raw.get("request_count", max(100, max(self.concurrency_values) * 10)))

    @property
    def independent_repetitions(self) -> int:
        return int(self.raw.get("independent_repetitions", 1))

    def result_configuration(self) -> dict[str, Any]:
        return {key: deepcopy(value) for key, value in self.raw.items() if key != "output_dir"}

    def fingerprint(self) -> str:
        canonical = json.dumps(self.result_configuration(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def points(self) -> list[MatrixPoint]:
        """Return a seeded, persisted-plan-ready order with affinity counterbalancing."""
        matrix = self.raw["matrix"]
        affinities = [str(value).lower() for value in matrix.get("affinity", ["unpinned"])]
        groups = list(
            product(
                self.concurrency_values,
                matrix["tensor_parallel"],
                matrix["precisions"],
            )
        )
        rng = random.Random(int(self.raw.get("execution_order_seed", self.raw.get("seed", 2025))))
        if self.raw.get("execution_order", "counterbalanced") == "randomized":
            rng.shuffle(groups)
        points: list[MatrixPoint] = []
        for repetition in range(1, self.independent_repetitions + 1):
            repetition_groups = list(groups)
            if self.raw.get("execution_order", "counterbalanced") == "randomized":
                rng.shuffle(repetition_groups)
            for concurrency, tp, precision in repetition_groups:
                ordered = list(affinities)
                if set(ordered) == {"pinned", "unpinned"} and repetition % 2 == 0:
                    ordered.reverse()
                if self.raw.get("execution_order") == "randomized":
                    rng.shuffle(ordered)
                points.extend(
                    MatrixPoint(int(concurrency), int(tp), str(precision).lower(), affinity, repetition)
                    for affinity in ordered
                )
        return points


def load_config(path: str | Path) -> Config:
    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot load {source}: {exc}") from exc
    _validate(raw)
    return Config(raw, source)


def with_output_dir(config: Config, output_dir: str | Path) -> Config:
    raw = deepcopy(config.raw)
    raw["output_dir"] = str(Path(output_dir).expanduser().resolve())
    return Config(raw, config.source)


def _positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be a positive integer")


def _positive_number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ConfigError(f"{name} must be a positive number")


def _http_url(value: Any, name: str) -> None:
    parsed = urlparse(str(value))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError(f"{name} must be an absolute HTTP(S) URL")


def _validate(raw: dict[str, Any]) -> None:
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be an object")
    required = {"model", "matrix", "warmup_iterations", "measured_trials"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ConfigError(f"missing required keys: {', '.join(missing)}")
    if not isinstance(raw["model"], str) or not raw["model"].strip():
        raise ConfigError("model must be a non-empty string")
    if raw.get("mode", "simulate") not in {"simulate", "hardware"}:
        raise ConfigError("mode must be 'simulate' or 'hardware'")
    matrix = raw["matrix"]
    if not isinstance(matrix, dict):
        raise ConfigError("matrix must be an object")
    if "concurrency" in matrix and "batch_sizes" in matrix:
        raise ConfigError("use matrix.concurrency; do not also specify deprecated matrix.batch_sizes")
    concurrency_key = "concurrency" if "concurrency" in matrix else "batch_sizes"
    for key in (concurrency_key, "tensor_parallel", "precisions"):
        if not isinstance(matrix.get(key), list) or not matrix[key]:
            raise ConfigError(f"matrix.{key} must be a non-empty list")
    for key in (concurrency_key, "tensor_parallel"):
        for value in matrix[key]:
            _positive_int(value, f"matrix.{key} value")
        if len(set(matrix[key])) != len(matrix[key]):
            raise ConfigError(f"matrix.{key} must not contain duplicates")
    max_concurrency = max(int(value) for value in matrix[concurrency_key])
    request_count = raw.get("request_count", max(100, max_concurrency * 10))
    _positive_int(request_count, "request_count")
    if request_count < max_concurrency:
        raise ConfigError("request_count must be at least the largest configured concurrency")
    for key in ("warmup_iterations", "measured_trials", "independent_repetitions"):
        _positive_int(raw.get(key, 1), key)
    _positive_int(raw.get("warmup_request_count", max_concurrency), "warmup_request_count")
    _positive_int(raw.get("output_tokens", 128), "output_tokens")
    order = raw.get("execution_order", "counterbalanced")
    if order not in {"counterbalanced", "randomized"}:
        raise ConfigError("execution_order must be 'counterbalanced' or 'randomized'")
    precisions = [str(item).lower() for item in matrix["precisions"]]
    if set(precisions) - {"fp8", "fp16"} or len(set(precisions)) != len(precisions):
        raise ConfigError("matrix.precisions must contain unique fp8/fp16 values")
    affinities = [str(item).lower() for item in matrix.get("affinity", ["unpinned"])]
    if set(affinities) - {"pinned", "unpinned"} or len(set(affinities)) != len(affinities):
        raise ConfigError("matrix.affinity must contain unique pinned/unpinned values")
    if raw.get("mode", "simulate") != "hardware":
        return
    server = raw.get("server")
    if not isinstance(server, dict):
        raise ConfigError("hardware mode requires a server object")
    command = server.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
        raise ConfigError("server.command must be a non-empty list of strings")
    for key in ("endpoint", "readiness_url"):
        if key not in server:
            raise ConfigError(f"server.{key} is required in hardware mode")
        _http_url(server[key], f"server.{key}")
    if server.get("request_format", "completions") not in {"completions", "chat"}:
        raise ConfigError("server.request_format must be 'completions' or 'chat'")
    if server.get("require_token_usage", True) is not True:
        raise ConfigError("server.require_token_usage cannot be false")
    for key, default in (
        ("startup_timeout_seconds", 300),
        ("request_timeout_seconds", 120),
        ("trial_timeout_seconds", 900),
        ("shutdown_timeout_seconds", 20),
    ):
        _positive_number(server.get(key, default), f"server.{key}")
    statuses = server.get("readiness_statuses", [200])
    if not isinstance(statuses, list) or not statuses or any(
        isinstance(x, bool) or not isinstance(x, int) or not 100 <= x <= 599 for x in statuses
    ):
        raise ConfigError("server.readiness_statuses must contain HTTP status integers")
    model_map = server.get("model_by_precision", {})
    if not isinstance(model_map, dict) or any(
        key not in {"fp8", "fp16"} or not isinstance(value, str) or not value
        for key, value in model_map.items()
    ):
        raise ConfigError("server.model_by_precision must map fp8/fp16 to non-empty paths")
    command_text = " ".join(command)
    if "{tp}" not in command_text:
        raise ConfigError("hardware server.command must include {tp}")
    missing_precision = set(precisions) - set(model_map)
    if missing_precision and "{precision}" not in command_text:
        raise ConfigError("each precision must use model_by_precision or {precision}")
    mapped = [model_map[item] for item in precisions if item in model_map]
    if len(precisions) > 1 and "{precision}" not in command_text and len(set(mapped)) < len(precisions):
        raise ConfigError("FP8 and FP16 points must select distinct artifacts")
    if "pinned" in affinities:
        affinity = server.get("affinity")
        if not isinstance(affinity, dict) or not str(affinity.get("cpus", "")).strip():
            raise ConfigError("pinned points require server.affinity.cpus")
    nsight = raw.get("nsight", {"enabled": True})
    if not isinstance(nsight, dict) or not isinstance(nsight.get("enabled", True), bool):
        raise ConfigError("nsight.enabled must be boolean")
    if not isinstance(nsight.get("required", True), bool):
        raise ConfigError("nsight.required must be boolean")
