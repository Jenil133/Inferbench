"""Non-causal profile association and affinity comparison analysis."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from inferbench.models import PointResult
from inferbench.statistics import median, spread


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    x_mean, y_mean = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    denominator = math.sqrt(sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys))
    return round(numerator / denominator, 4) if denominator else None


def analyze(results: list[PointResult], synthetic: bool) -> list[dict[str, Any]]:
    pairs: list[tuple[float, float]] = []
    if synthetic:
        pairs = [
            (float(trial.selected_os_runtime_wait_ms), float(trial.latency_p99_ms))
            for result in results for trial in result.trials
            if trial.selected_os_runtime_wait_ms is not None and trial.latency_p99_ms is not None
        ]
        scope = "synthetic trial"
    else:
        for result in results:
            if result.profile_summary is None:
                continue
            wait = result.profile_summary.get("selected_os_runtime_wait_ms")
            latency = result.summary.get("latency_p99_ms_trial_median")
            if wait is not None and latency is not None:
                pairs.append((float(wait), float(latency)))
        scope = "point repetition (whole-profile selected wait time vs median trial p99)"
    records: list[dict[str, Any]] = [
        {
            "analysis": "p99_selected_os_runtime_wait_association",
            "pearson_r": _pearson([x for x, _ in pairs], [y for _, y in pairs]),
            "samples": len(pairs),
            "sample_scope": scope,
            "interpretation": (
                "Exploratory association only. The profile is not trial-window isolated and selected "
                "wait/blocking calls may be normal; no scheduling or latency causation is claimed."
            ),
            "synthetic": synthetic,
        }
    ]
    grouped: dict[tuple[int, int, str], dict[str, list[PointResult]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        point = result.point
        grouped[(point.concurrency, point.tensor_parallel, point.precision)][point.affinity].append(result)
    for key, affinity_results in grouped.items():
        if not {"pinned", "unpinned"}.issubset(affinity_results):
            continue
        before = [
            float(trial.latency_p99_ms) for result in affinity_results["unpinned"]
            for trial in result.trials if trial.latency_p99_ms is not None
        ]
        after = [
            float(trial.latency_p99_ms) for result in affinity_results["pinned"]
            for trial in result.trials if trial.latency_p99_ms is not None
        ]
        before_jitter, after_jitter = spread(before), spread(after)
        reduction = 0.0 if before_jitter == 0 else (before_jitter - after_jitter) / before_jitter * 100
        records.append(
            {
                "analysis": "process_affinity_comparison",
                "concurrency": key[0], "tensor_parallel": key[1], "precision": key[2],
                "independent_repetitions": min(len(affinity_results["pinned"]), len(affinity_results["unpinned"])),
                "unpinned_p99_median_ms": round(median(before), 3),
                "pinned_p99_median_ms": round(median(after), 3),
                "unpinned_p99_iqr_ms": round(before_jitter, 3),
                "pinned_p99_iqr_ms": round(after_jitter, 3),
                "jitter_reduction_percent": round(reduction, 1),
                "synthetic": synthetic,
                "label": "SYNTHETIC SCENARIO" if synthetic else "MEASURED HARDWARE COMPARISON",
            }
        )
    return records
