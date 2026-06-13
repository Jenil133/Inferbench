"""Safe process adapters for TensorRT-LLM server and Nsight Systems."""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, BinaryIO

from inferbench.models import MatrixPoint


class ToolError(RuntimeError):
    """An external tool is unavailable or failed."""


def format_command(template: list[str], point: MatrixPoint, model: str) -> list[str]:
    values = {
        "batch_size": point.concurrency,
        "concurrency": point.concurrency,
        "tp": point.tensor_parallel,
        "precision": point.precision,
        "model": model,
    }
    try:
        return [part.format(**values) for part in template]
    except (KeyError, ValueError) as exc:
        raise ToolError(f"invalid server command template: {exc}") from exc


def build_server_command(
    config: dict[str, Any], point: MatrixPoint, profile: Path | None
) -> tuple[list[str], str]:
    server = config["server"]
    model = str(server.get("model_by_precision", {}).get(point.precision, config["model"]))
    command = format_command(server["command"], point, model)
    if point.affinity == "pinned":
        cpus = str(server["affinity"]["cpus"])
        if platform.system() != "Linux" or not shutil.which("taskset"):
            raise ToolError("pinned affinity requires Linux taskset")
        command = ["taskset", "-c", cpus, *command]
    profile_status = "disabled"
    if profile is not None:
        if not shutil.which("nsys"):
            return command, "missing_tool"
        profile.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "nsys", "profile", "--force-overwrite=false", "--trace=cuda,nvtx,osrt",
            "--sample=process-tree", "--output", str(profile), "--", *command,
        ]
        profile_status = "capture_requested"
    return command, profile_status


def start_process(
    command: list[str], log_path: Path, env: dict[str, str] | None = None
) -> tuple[subprocess.Popen[bytes], BinaryIO]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("xb")
    merged_env = os.environ.copy()
    merged_env.update(env or {})
    try:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=merged_env, start_new_session=True
        )
    except OSError as exc:
        log.close()
        raise ToolError(f"cannot start {shlex.join(command)}: {exc}") from exc
    return process, log


def stop_process(
    process: subprocess.Popen[bytes], log: BinaryIO, grace_seconds: float = 20
) -> dict[str, Any]:
    term_sent = False
    kill_sent = False
    started = time.monotonic()
    try:
        if process.poll() is None:
            term_sent = True
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                kill_sent = True
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired as exc:
                    raise ToolError("process group did not exit after SIGKILL") from exc
        return {
            "pid": process.pid,
            "return_code": process.returncode,
            "term_sent": term_sent,
            "kill_sent": kill_sent,
            "shutdown_seconds": round(time.monotonic() - started, 3),
        }
    finally:
        log.close()


def tool_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    commands = (
        ("python", ["python3", "--version"]),
        ("nvidia-smi", ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        ("nsys", ["nsys", "--version"]),
        ("trtllm-serve", ["trtllm-serve", "--version"]),
        ("taskset", ["taskset", "--version"]),
    )
    for tool, args in commands:
        if not shutil.which(args[0]):
            versions[tool] = "not found"
            continue
        try:
            completed = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
            output = (completed.stdout or completed.stderr).strip().splitlines()
            versions[tool] = output[0] if output else f"exit {completed.returncode} (no output)"
        except (OSError, subprocess.TimeoutExpired) as exc:
            versions[tool] = f"probe failed: {exc}"
    return versions
