"""Small dependency-free statistics helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one observation")
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return float(ordered[low])
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def median(values: Iterable[float]) -> float:
    return percentile(values, 0.5)


def spread(values: Iterable[float]) -> float:
    items = list(values)
    return percentile(items, 0.75) - percentile(items, 0.25)


def summarize_trials(trials: list[Any]) -> dict[str, Any]:
    """Preserve trial medians/IQR and add pooled request percentiles with sample counts."""
    result: dict[str, Any] = {}
    for name in ("throughput_tokens_s", "ttft_p50_ms", "ttft_p99_ms", "latency_p50_ms", "latency_p99_ms"):
        values = [float(value) for trial in trials if (value := getattr(trial, name)) is not None]
        result[f"{name}_trial_median"] = round(median(values), 3) if values else None
        result[f"{name}_trial_iqr"] = round(spread(values), 3) if values else None
        result[f"{name}_trial_samples"] = len(values)
    successful = [request for trial in trials for request in trial.requests if request.success]
    ttft = [float(request.ttft_ms) for request in successful if request.ttft_ms is not None]
    latency = [float(request.latency_ms) for request in successful]
    result.update(
        {
            "pooled_successful_requests": len(successful),
            "pooled_failed_requests": sum(trial.failed_requests for trial in trials),
            "pooled_ttft_samples": len(ttft),
            "pooled_latency_samples": len(latency),
            "pooled_ttft_p50_ms": round(percentile(ttft, 0.50), 3) if ttft else None,
            "pooled_ttft_p99_ms": round(percentile(ttft, 0.99), 3) if ttft else None,
            "pooled_latency_p50_ms": round(percentile(latency, 0.50), 3) if latency else None,
            "pooled_latency_p99_ms": round(percentile(latency, 0.99), 3) if latency else None,
        }
    )
    return result
