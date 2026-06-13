"""OpenAI-compatible streaming HTTP load driver with bounded execution."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from subprocess import Popen
from typing import Any

from inferbench.models import RequestMetric, TrialMetric
from inferbench.statistics import percentile


class BackendError(RuntimeError):
    """External serving backend failed."""


@dataclass(frozen=True)
class HttpSettings:
    endpoint: str
    model: str
    prompt: str
    max_tokens: int
    timeout_seconds: float
    trial_timeout_seconds: float
    api_key: str | None = None
    request_format: str = "completions"
    max_event_bytes: int = 1_048_576


class RequestControl:
    """Allow the trial coordinator to interrupt an active HTTP response."""

    def __init__(self, trial_deadline: float) -> None:
        self.trial_deadline = trial_deadline
        self.request_deadline: float | None = None
        self.cancelled = threading.Event()
        self._response: Any = None
        self._lock = threading.Lock()

    def begin(self, request_timeout: float) -> float:
        self.request_deadline = min(time.monotonic() + request_timeout, self.trial_deadline)
        return self.request_deadline

    def set_response(self, response: Any) -> None:
        with self._lock:
            self._response = response
            if self.cancelled.is_set():
                self._interrupt_locked()

    def expired(self) -> bool:
        return self.request_deadline is not None and time.monotonic() >= self.request_deadline

    def cancel(self) -> None:
        self.cancelled.set()
        with self._lock:
            self._interrupt_locked()

    def clear(self) -> None:
        with self._lock:
            self._response = None

    def _interrupt_locked(self) -> None:
        response = self._response
        if response is None:
            return
        try:
            raw = getattr(getattr(response, "fp", None), "raw", None)
            sock = getattr(raw, "_sock", None)
            if sock is not None:
                sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        with suppress(OSError):
            response.close()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def readiness(
    url: str,
    timeout_seconds: float,
    process: Popen[bytes] | None = None,
    api_key: str | None = None,
    accepted_statuses: tuple[int, ...] = (200,),
    log_path: Path | None = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    error = "not attempted"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            suffix = f"; see {log_path}" if log_path else ""
            raise BackendError(f"server exited before readiness (code {process.returncode}){suffix}")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            remaining = max(0.1, min(2.0, deadline - time.monotonic()))
            with urllib.request.urlopen(request, timeout=remaining) as response:  # noqa: S310
                if response.status in accepted_statuses:
                    return
                error = f"unexpected HTTP status {response.status}"
        except urllib.error.HTTPError as exc:
            if exc.code in accepted_statuses:
                return
            error = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = str(exc)
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    raise BackendError(f"server readiness timed out after {timeout_seconds}s: {error}")


def _payload(settings: HttpSettings) -> bytes:
    common: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if settings.request_format == "chat":
        common["messages"] = [{"role": "user", "content": settings.prompt}]
    else:
        common["prompt"] = settings.prompt
    return json.dumps(common).encode()


def _sse_events(
    response: Any, deadline: float, max_event_bytes: int, control: RequestControl
) -> Iterator[str]:
    """Decode SSE event framing, including comments and multi-line data fields."""
    data: list[str] = []
    size = 0
    for raw_line in response:
        if control.cancelled.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("absolute request deadline exceeded while reading stream")
        if len(raw_line) > max_event_bytes:
            raise BackendError("SSE line exceeds configured safety limit")
        line = raw_line.decode("utf-8", errors="strict").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data, size = [], 0
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field != "data":
            continue
        value = value[1:] if separator and value.startswith(" ") else value
        size += len(value.encode("utf-8"))
        if size > max_event_bytes:
            raise BackendError("SSE event exceeds configured safety limit")
        data.append(value)
    if data:
        yield "\n".join(data)


def _bounded_error(value: object, limit: int = 500) -> str:
    return str(value).replace("\n", " ")[:limit]


def _request(settings: HttpSettings, sequence: int, control: RequestControl) -> RequestMetric:
    request_id = str(uuid.uuid4())
    started_at = _utc_now()
    started = time.perf_counter()
    deadline = control.begin(settings.timeout_seconds)
    first_token: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    response_id: str | None = None
    events = 0
    status: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    request = urllib.request.Request(settings.endpoint, _payload(settings), headers, method="POST")
    try:
        network_timeout = max(0.001, deadline - time.monotonic())
        response = urllib.request.urlopen(request, timeout=network_timeout)  # noqa: S310
        control.set_response(response)
        with response:
            status = int(response.status)
            done = False
            for payload in _sse_events(response, deadline, settings.max_event_bytes, control):
                if payload == "[DONE]":
                    done = True
                    continue
                try:
                    event: dict[str, Any] = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise BackendError(f"malformed SSE JSON event: {exc.msg}") from exc
                events += 1
                if event.get("error"):
                    raise BackendError(f"serving API error: {_bounded_error(event['error'])}")
                response_id = str(event.get("id")) if event.get("id") is not None else response_id
                for choice in event.get("choices") or []:
                    text = choice.get("text", "") or (choice.get("delta") or {}).get("content", "")
                    if text and first_token is None:
                        first_token = time.perf_counter()
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
                usage = event.get("usage") or {}
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = usage.get(key)
                    if key in usage and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                        raise BackendError(f"invalid non-negative integer usage.{key}")
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                total_tokens = usage.get("total_tokens", total_tokens)
            if not done:
                raise BackendError("stream ended without an explicit [DONE] event")
            if first_token is None:
                raise BackendError("stream completed without generated text")
            if completion_tokens is None:
                raise BackendError("stream omitted usage.completion_tokens")
    except urllib.error.HTTPError as exc:
        status = exc.code
        error_type = "http_error"
        error_message = f"HTTP {exc.code}: {_bounded_error(exc.read().decode('utf-8', errors='replace'))}"
    except (TimeoutError, urllib.error.URLError) as exc:
        error_type = "timeout" if isinstance(exc, TimeoutError) else "network_error"
        error_message = _bounded_error(exc)
    except (BackendError, OSError, UnicodeDecodeError) as exc:
        error_type = "protocol_error" if isinstance(exc, (BackendError, UnicodeDecodeError)) else "io_error"
        error_message = _bounded_error(exc)
    finally:
        control.clear()
    if control.cancelled.is_set():
        error_type = "timeout"
        error_message = "absolute request or trial deadline exceeded"
    finished = time.perf_counter()
    return RequestMetric(
        request_id=request_id,
        sequence=sequence,
        success=error_type is None,
        started_at=started_at,
        finished_at=_utc_now(),
        ttft_ms=round((first_token - started) * 1000, 3) if first_token is not None else None,
        latency_ms=round((finished - started) * 1000, 3),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        finish_reason=finish_reason,
        error_type=error_type,
        error_message=error_message,
        http_status=status,
        response_id=response_id,
        events=events,
    )


def _deadline_metric(sequence: int, started_at: str) -> RequestMetric:
    return RequestMetric(
        request_id=str(uuid.uuid4()), sequence=sequence, success=False, started_at=started_at,
        finished_at=_utc_now(), ttft_ms=None, latency_ms=0.0, error_type="trial_timeout",
        error_message="trial deadline elapsed before request started",
    )


def run_trial(
    settings: HttpSettings,
    concurrency: int,
    request_count: int,
    trial: int,
    repetition: int,
    process: Popen[bytes] | None = None,
) -> TrialMetric:
    """Run a fixed worker pool; interrupt and join every worker before returning."""
    started_at = _utc_now()
    started = time.perf_counter()
    trial_deadline = time.monotonic() + settings.trial_timeout_seconds
    pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="inferbench-request")
    controls = {sequence: RequestControl(trial_deadline) for sequence in range(1, request_count + 1)}
    futures: dict[Future[RequestMetric], int] = {
        pool.submit(_request, settings, sequence, controls[sequence]): sequence
        for sequence in range(1, request_count + 1)
    }
    metrics: list[RequestMetric] = []
    pending = set(futures)
    try:
        while pending:
            if process is not None and process.poll() is not None:
                for control in controls.values():
                    control.cancel()
                raise BackendError(f"server exited during trial with code {process.returncode}")
            now = time.monotonic()
            for control in controls.values():
                if control.expired():
                    control.cancel()
            if now >= trial_deadline:
                for control in controls.values():
                    control.cancel()
                break
            completed, pending = wait(
                pending, timeout=min(0.05, trial_deadline - now), return_when=FIRST_COMPLETED
            )
            for future in completed:
                metrics.append(future.result())
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        for future in pending:
            sequence = futures[future]
            metrics.append(
                _deadline_metric(sequence, started_at) if future.cancelled() else future.result()
            )
    finally:
        for control in controls.values():
            control.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    metrics.sort(key=lambda item: item.sequence)
    wall_seconds = time.perf_counter() - started
    successful = [metric for metric in metrics if metric.success]
    ttft = [float(metric.ttft_ms) for metric in successful if metric.ttft_ms is not None]
    latency = [metric.latency_ms for metric in successful]
    tokens = sum(metric.completion_tokens or 0 for metric in successful)
    failures = len(metrics) - len(successful)
    return TrialMetric(
        trial=trial, repetition=repetition,
        status="completed" if failures == 0 else ("partial" if successful else "failed"),
        concurrency=concurrency, request_count=request_count,
        successful_requests=len(successful), failed_requests=failures,
        wall_time_s=round(wall_seconds, 6),
        throughput_tokens_s=round(tokens / wall_seconds, 3) if successful and wall_seconds else None,
        ttft_p50_ms=round(percentile(ttft, 0.50), 3) if ttft else None,
        ttft_p99_ms=round(percentile(ttft, 0.99), 3) if ttft else None,
        latency_p50_ms=round(percentile(latency, 0.50), 3) if latency else None,
        latency_p99_ms=round(percentile(latency, 0.99), 3) if latency else None,
        requests=metrics,
    )
