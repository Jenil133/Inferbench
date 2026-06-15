"""Nsight Systems selected OS-runtime wait/blocking-time adapter."""

from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


class NsightParseError(RuntimeError):
    """Nsight output did not contain usable selected OS-runtime rows."""


@dataclass(frozen=True)
class SelectedOsRuntimeWaitSummary:
    selected_os_runtime_wait_ms: float
    selected_calls: int
    api_wait_ms: dict[str, float]
    selected_api_patterns: tuple[str, ...]
    source: str
    scope: str = "complete server-process profile; not trial-window isolated"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


SELECTED_API_PATTERNS = (
    r"(^|::|_)futex($|::|_)",
    r"(^|::|_)(p?poll|select|epoll_wait|sleep|usleep|nanosleep|sched_yield|waitpid)($|::|_)",
)


def _selected(name: str) -> bool:
    normalized = name.lower().replace(" ", "_")
    return any(re.search(pattern, normalized) for pattern in SELECTED_API_PATTERNS)


def _number(value: str) -> float:
    cleaned = re.sub(r"[^0-9.eE+-]", "", value)
    try:
        return float(cleaned)
    except ValueError as exc:
        raise NsightParseError(f"invalid numeric value in Nsight CSV: {value!r}") from exc


def _duration_ms(value: str, heading: str) -> float:
    normalized = heading.lower().replace("μ", "u").replace("µ", "u")
    match = re.search(r"(?:\(|\[)\s*(ns|us|ms|s)\s*(?:\)|\])", normalized)
    if not match:
        raise NsightParseError(f"unsupported or missing duration unit in column {heading!r}")
    return _number(value) * {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1_000.0}[match.group(1)]


def parse_osrt_csv(
    text: str, source: str = "nsys osrt_sum CSV"
) -> SelectedOsRuntimeWaitSummary:
    """Normalize explicitly selected OS-runtime wait/blocking rows to milliseconds."""
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("Processing")]
    header_index = next((i for i, line in enumerate(lines) if "name" in line.lower() and "," in line), -1)
    if header_index < 0:
        raise NsightParseError(f"no OS-runtime CSV header found in {source}")
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    original = [str(field).strip() for field in (reader.fieldnames or [])]
    headings = {field.lower(): field for field in original}
    duration_lower = next((key for key in headings if "total time" in key or "total duration" in key), None)
    count_lower = next((key for key in headings if "instances" in key or "calls" in key), None)
    name_key = headings.get("name")
    if duration_lower is None or count_lower is None or name_key is None:
        raise NsightParseError(f"missing name, duration, or call-count column in {source}")
    duration_key, count_key = headings[duration_lower], headings[count_lower]
    totals: dict[str, float] = {}
    calls = 0
    for row in reader:
        name = str(row.get(name_key, "")).strip()
        if not _selected(name):
            continue
        duration = _duration_ms(str(row.get(duration_key, "")), duration_key)
        totals[name] = totals.get(name, 0.0) + duration
        calls += int(_number(str(row.get(count_key, ""))))
    if not totals:
        raise NsightParseError(f"no selected OS-runtime wait/blocking API rows found in {source}")
    rounded = {name: round(value, 6) for name, value in sorted(totals.items())}
    return SelectedOsRuntimeWaitSummary(
        round(sum(totals.values()), 3), calls, rounded, SELECTED_API_PATTERNS, source
    )


def extract_profile(profile: str | Path) -> SelectedOsRuntimeWaitSummary:
    path = Path(profile)
    if not shutil.which("nsys"):
        raise RuntimeError("nsys not found; cannot extract profile statistics")
    completed = subprocess.run(
        ["nsys", "stats", "--report", "osrt_sum", "--format", "csv", str(path)],
        capture_output=True, text=True, timeout=120, check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[:1000]
        raise RuntimeError(f"nsys stats failed with exit {completed.returncode}: {detail}")
    return parse_osrt_csv(completed.stdout, str(path))
