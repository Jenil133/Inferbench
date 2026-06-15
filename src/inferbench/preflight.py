"""Hardware preflight checks and resolved execution-plan rendering."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from inferbench.config import Config
from inferbench.external import build_server_command


class PreflightError(RuntimeError):
    """Hardware prerequisites are not satisfied."""


def _gpu_count() -> tuple[int | None, str]:
    if not shutil.which("nvidia-smi"):
        return None, "nvidia-smi not found"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"GPU probe failed: {exc}"
    if result.returncode:
        return None, f"GPU probe exit {result.returncode}: {result.stderr.strip()[:300]}"
    return len([line for line in result.stdout.splitlines() if line.strip()]), "ok"


def run_preflight(config: Config, execution_dir: Path | None = None) -> dict[str, Any]:
    if config.mode != "hardware":
        return {"status": "not_applicable", "mode": config.mode, "checks": [], "resolved_commands": []}
    raw, server = config.raw, config.raw["server"]
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []
    cpus = str(server.get("affinity", {}).get("cpus", ""))
    cpu_syntax_ok = not cpus or bool(re.fullmatch(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*", cpus))
    checks.append({"name": "cpu_list_syntax", "ok": cpu_syntax_ok, "value": cpus or None})
    if not cpu_syntax_ok:
        errors.append(f"invalid taskset CPU list syntax: {cpus!r}")
    if any(point.affinity == "pinned" for point in config.points()) and (
        platform.system() != "Linux" or not shutil.which("taskset")
    ):
        errors.append("pinned points require Linux and taskset")
    model_map = server.get("model_by_precision", {})
    for precision in raw["matrix"]["precisions"]:
        artifact = model_map.get(str(precision).lower())
        if artifact:
            exists = Path(artifact).exists()
            checks.append({"name": f"model_artifact_{precision}", "ok": exists, "path": artifact})
            if not exists:
                errors.append(f"model artifact does not exist: {artifact}")
    executable = str(server["command"][0])
    executable_ok = bool(shutil.which(executable) or Path(executable).is_file())
    checks.append({"name": "server_executable", "ok": executable_ok, "value": executable})
    if not executable_ok:
        errors.append(f"server executable not found: {executable}")
    gpu_count, gpu_detail = _gpu_count()
    required_gpus = max(int(value) for value in raw["matrix"]["tensor_parallel"])
    gpu_ok = gpu_count is not None and gpu_count >= required_gpus
    checks.append({"name": "visible_gpu_count", "ok": gpu_ok, "count": gpu_count, "required": required_gpus, "detail": gpu_detail})
    if not gpu_ok:
        errors.append(f"visible GPU count {gpu_count!r} is below required TP {required_gpus}: {gpu_detail}")
    nsight = raw.get("nsight", {})
    if nsight.get("enabled", True) and not shutil.which("nsys"):
        message = "nsys not found; profiling cannot be captured"
        if nsight.get("required", True):
            errors.append(message)
        else:
            warnings.append(message + " (best effort; benchmark will continue)")
    output_base = execution_dir or config.output_dir
    try:
        output_base.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output_base, prefix=".preflight-", delete=True):
            pass
        output_ok = os.access(output_base, os.W_OK)
    except OSError as exc:
        output_ok = False
        errors.append(f"output path is not writable: {output_base}: {exc}")
    checks.append({"name": "output_writable", "ok": output_ok, "path": str(output_base)})
    commands: list[dict[str, Any]] = []
    for point in config.points():
        profile = output_base / "profiles" / point.point_id if nsight.get("enabled", True) else None
        try:
            command, profile_status = build_server_command(raw, point, profile)
            commands.append({"point_id": point.point_id, "argv": command, "profile_status": profile_status})
        except RuntimeError as exc:
            errors.append(f"{point.point_id}: {exc}")
    result = {
        "status": "failed" if errors else "passed",
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "resolved_commands": commands,
    }
    return result
