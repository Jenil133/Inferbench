"""CPU-only end-to-end contracts for InferBench.

Only simulation mode and static Nsight CSV parsing run here. No GPU,
TensorRT-LLM server, taskset, or Nsight process is started.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
BUNDLED_SIMULATION = ROOT / "examples" / "simulation-matrix.json"
NSIGHT_SAMPLE = ROOT / "examples" / "nsight-osrt-sample.csv"


class InferBenchCpuE2E(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="inferbench-e2e-")
        self.work = Path(self.temp.name)
        self.env = os.environ.copy()
        self.env["PYTHONPATH"] = str(ROOT / "src")
        self.env["CUDA_VISIBLE_DEVICES"] = ""
        self.env["NVIDIA_VISIBLE_DEVICES"] = "void"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def cli(self, *args: str, expected: int | None = None) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [sys.executable, "-m", "inferbench", *map(str, args)],
            cwd=ROOT,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if expected is not None:
            self.assertEqual(result.returncode, expected, result.stderr)
        return result

    def reduced_config(self, name: str = "reduced.json") -> Path:
        config = {
            "mode": "simulate",
            "model": "e2e-synthetic-model",
            "seed": 2025,
            "warmup_iterations": 1,
            "warmup_request_count": 2,
            "measured_trials": 2,
            "independent_repetitions": 2,
            "request_count": 4,
            "execution_order": "counterbalanced",
            "output_tokens": 8,
            "matrix": {
                "concurrency": [2],
                "tensor_parallel": [1],
                "precisions": ["fp16"],
                "affinity": ["unpinned", "pinned"],
            },
        }
        path = self.work / name
        path.write_text(json.dumps(config, indent=2))
        return path

    def execution_from_stdout(self, stdout: str) -> Path:
        line = next(item for item in stdout.splitlines() if item.startswith("execution: "))
        return Path(line.split(": ", 1)[1])

    def assert_artifact_hashes(self, execution: Path) -> None:
        manifest = json.loads((execution / "manifest.json").read_text())
        paths = []
        for artifact in manifest["artifacts"]:
            path = execution / artifact["path"]
            self.assertTrue(path.is_file(), artifact["path"])
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"])
            paths.append(artifact["path"])
        self.assertEqual(paths, sorted(paths))

    def run_reduced(self, parent: Path, config: Path | None = None) -> tuple[Path, dict]:
        config = config or self.reduced_config()
        result = self.cli(
            "run", "--config", config, "--output-dir", parent, expected=0
        )
        self.assertIn("SYNTHETIC (NO HARDWARE EXECUTION)", result.stdout)
        execution = self.execution_from_stdout(result.stdout)
        return execution, json.loads((execution / "report.json").read_text())

    def test_version_validate_preflight_and_inspect(self) -> None:
        version = self.cli("--version", expected=0)
        self.assertEqual(version.stdout, "inferbench 0.2.0\n")
        config = self.reduced_config()
        validation = json.loads(self.cli("validate", "--config", config, expected=0).stdout)
        self.assertEqual(validation["request_count"], 4)
        self.assertEqual(
            validation["execution_plan"],
            [
                "c2-tp1-fp16-unpinned-r1",
                "c2-tp1-fp16-pinned-r1",
                "c2-tp1-fp16-pinned-r2",
                "c2-tp1-fp16-unpinned-r2",
            ],
        )
        self.assertRegex(validation["config_fingerprint"], r"^[0-9a-f]{64}$")

        preflight = json.loads(self.cli("preflight", "--config", config, expected=0).stdout)
        self.assertEqual(
            preflight,
            {"checks": [], "mode": "simulate", "resolved_commands": [], "status": "not_applicable"},
        )
        inspection = json.loads(self.cli("inspect", expected=0).stdout)
        self.assertTrue(
            {"python", "nvidia-smi", "nsys", "taskset", "trtllm-serve"}
            <= set(inspection)
        )

    def test_nsight_csv_parsing_and_errors(self) -> None:
        parsed = json.loads(
            self.cli("parse-nsight", "--csv", NSIGHT_SAMPLE, expected=0).stdout
        )
        self.assertEqual(parsed["selected_os_runtime_wait_ms"], 60.0)
        self.assertEqual(parsed["selected_calls"], 20)
        self.assertEqual(parsed["api_wait_ms"], {"poll": 48.0, "sched_yield": 12.0})
        self.assertIn("not trial-window isolated", parsed["scope"])

        empty = self.work / "empty.csv"
        empty.write_text("")
        result = self.cli("parse-nsight", "--csv", empty, expected=2)
        self.assertIn("no OS-runtime CSV header", result.stderr)

    def test_reduced_matrix_reports_csv_and_hashes(self) -> None:
        execution, report = self.run_reduced(self.work / "runs")
        self.assertTrue(report["synthetic"])
        self.assertEqual(report["mode"], "simulate")
        self.assertEqual(len(report["results"]), 4)
        for result in report["results"]:
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["profile_status"], "disabled")
            self.assertEqual(len(result["trials"]), 2)
            summary = result["summary"]
            self.assertEqual(summary["pooled_successful_requests"], 8)
            self.assertEqual(summary["pooled_failed_requests"], 0)
            self.assertEqual(summary["pooled_ttft_samples"], 8)
            self.assertEqual(summary["pooled_latency_samples"], 8)

        with (execution / "trials.csv").open(newline="") as stream:
            trials = list(csv.DictReader(stream))
        with (execution / "requests.csv").open(newline="") as stream:
            requests = list(csv.DictReader(stream))
        self.assertEqual(len(trials), 8)
        self.assertEqual(len(requests), 32)
        self.assertTrue(all(row["success"] == "true" for row in requests))
        self.assertTrue(all(row["synthetic"] == "true" for row in requests))

        manifest = json.loads((execution / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(len(manifest["points"]), 4)
        self.assertEqual(len(manifest["artifacts"]), 9)
        self.assertEqual(len(list((execution / "points").glob("*.attempt-1.json"))), 4)
        self.assert_artifact_hashes(execution)
        self.assertIn("SYNTHETIC DATA", (execution / "report.md").read_text())

        resume = self.cli(
            "run", "--config", self.work / "reduced.json", "--resume", execution, expected=2
        )
        self.assertIn("already complete and immutable", resume.stderr)

    def test_deterministic_results_with_independent_execution_ids(self) -> None:
        config = self.reduced_config()
        first_dir, first = self.run_reduced(self.work / "runs", config)
        second_dir, second = self.run_reduced(self.work / "runs", config)
        self.assertNotEqual(first_dir, second_dir)
        self.assertNotEqual(first["execution_id"], second["execution_id"])
        self.assertEqual(first["config_fingerprint"], second["config_fingerprint"])
        self.assertEqual(first["results"], second["results"])
        self.assertEqual(first["correlations"], second["correlations"])

    def test_failure_persistence_and_resume(self) -> None:
        config = self.reduced_config()
        runs = self.work / "runs"
        sys.path.insert(0, str(ROOT / "src"))
        from inferbench import cli as inferbench_cli
        from inferbench import runner

        real = runner._run_synthetic_point
        calls = 0

        def fail_second(*args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected CPU E2E failure")
            return real(*args, **kwargs)

        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(runner, "_run_synthetic_point", side_effect=fail_second):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = inferbench_cli.main(
                    ["run", "--config", str(config), "--output-dir", str(runs)]
                )
        self.assertEqual(code, 2)
        executions = list(runs.iterdir())
        self.assertEqual(len(executions), 1)
        execution = executions[0]
        failed = json.loads((execution / "manifest.json").read_text())
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(len(failed["points"]), 1)
        self.assertFalse((execution / "report.json").exists())
        self.assertFalse((execution / "execution.lock").exists())

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = inferbench_cli.main(
                ["run", "--config", str(config), "--resume", str(execution)]
            )
        self.assertEqual(code, 0, stderr.getvalue())
        completed = json.loads((execution / "manifest.json").read_text())
        report = json.loads((execution / "report.json").read_text())
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["resume_count"], 1)
        self.assertEqual(len(report["results"]), 4)
        self.assertEqual(len({item["point"]["affinity"] + str(item["point"]["repetition"]) for item in report["results"]}), 4)
        self.assertTrue(list((execution / "points").glob("*.attempt-2.json")))
        self.assert_artifact_hashes(execution)

    def test_bundled_48_point_matrix_counts(self) -> None:
        result = self.cli(
            "run",
            "--config",
            BUNDLED_SIMULATION,
            "--output-dir",
            self.work / "full",
            expected=0,
        )
        execution = self.execution_from_stdout(result.stdout)
        report = json.loads((execution / "report.json").read_text())
        self.assertEqual(len(report["results"]), 48)
        self.assertTrue(all(item["status"] == "completed" for item in report["results"]))
        self.assertTrue(
            all(item["summary"]["pooled_latency_samples"] == 576 for item in report["results"])
        )
        with (execution / "trials.csv").open(newline="") as stream:
            self.assertEqual(sum(1 for _ in csv.DictReader(stream)), 432)
        with (execution / "requests.csv").open(newline="") as stream:
            self.assertEqual(sum(1 for _ in csv.DictReader(stream)), 27_648)
        manifest = json.loads((execution / "manifest.json").read_text())
        self.assertEqual(len(manifest["artifacts"]), 53)
        self.assert_artifact_hashes(execution)

    def test_invalid_configuration_is_rejected(self) -> None:
        invalid = json.loads(self.reduced_config().read_text())
        invalid["request_count"] = 0
        path = self.work / "invalid.json"
        path.write_text(json.dumps(invalid))
        result = self.cli("validate", "--config", path, expected=2)
        self.assertIn("request_count must be a positive integer", result.stderr)


if __name__ == "__main__":
    unittest.main()
