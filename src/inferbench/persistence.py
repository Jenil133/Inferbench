"""Atomic persistence, provenance, and artifact hashing."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def code_provenance(project_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True,
            text=True, timeout=5, check=False,
        )
        result["git_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_root, capture_output=True,
            text=True, timeout=5, check=False,
        )
        result["git_worktree_dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        result.update({"git_revision": None, "git_worktree_dirty": None})
    result["project_root_name"] = project_root.name
    return result


def hash_artifacts(paths: list[Path], root: Path) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        if path.is_file():
            records.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return sorted(records, key=lambda item: item["path"])


def acquire_execution_lock(execution_dir: Path) -> Path:
    """Acquire an exclusive execution lock, recovering a dead local lock."""
    lock_path = execution_dir / "execution.lock"
    payload = {"pid": os.getpid(), "hostname": socket.gethostname(), "created_at": utc_now()}
    while True:
        try:
            descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            return lock_path
        except FileExistsError:
            try:
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
                pid = int(existing["pid"])
                same_host = existing.get("hostname") == socket.gethostname()
                if same_host:
                    os.kill(pid, 0)
                raise RuntimeError(f"execution is locked by {existing}")
            except ProcessLookupError:
                lock_path.unlink()
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"cannot validate execution lock {lock_path}: {exc}") from exc


def release_execution_lock(lock_path: Path) -> None:
    with suppress(FileNotFoundError):
        lock_path.unlink()


def source_tree_digest(project_root: Path) -> dict[str, Any]:
    """Identify maintained source bytes even when no Git commit exists."""
    roots = [project_root / name for name in ("src", "docs", "examples", "scripts")]
    if project_root.name == "inferbench":
        roots.append(project_root)
    files = [
        project_root / name
        for name in (
            "pyproject.toml", "MANIFEST.in", "README.md", "Makefile", "Dockerfile",
            "docker-compose.yml", ".dockerignore", ".gitignore", "LICENSE",
        )
        if (project_root / name).is_file()
    ]
    for root in roots:
        if root.exists():
            files.extend(
                path for path in root.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts and ".egg-info" not in path.as_posix()
            )
    digest = hashlib.sha256()
    records = []
    for path in sorted(set(files)):
        relative = path.relative_to(project_root).as_posix()
        file_hash = sha256_file(path)
        digest.update(relative.encode() + b"\0" + file_hash.encode() + b"\n")
        records.append({"path": relative, "sha256": file_hash, "size_bytes": path.stat().st_size})
    return {"sha256": digest.hexdigest(), "file_count": len(records), "files": records}


def hash_execution_artifacts(root: Path) -> list[dict[str, Any]]:
    """Hash all immutable execution evidence except the self-referential manifest."""
    paths = [
        path for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "execution.lock"}
        and not path.name.startswith(".")
    ]
    return hash_artifacts(paths, root)
