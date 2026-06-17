"""InferBench command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inferbench import __version__
from inferbench.config import ConfigError, load_config, with_output_dir
from inferbench.external import tool_versions
from inferbench.nsight import extract_profile, parse_osrt_csv
from inferbench.preflight import run_preflight
from inferbench.reporting import write_report
from inferbench.runner import run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inferbench", description="Reproducible TensorRT-LLM serving benchmarks")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run a benchmark matrix")
    run_parser.add_argument("--config", required=True, help="JSON matrix configuration")
    run_parser.add_argument("--output-dir", help="override execution parent directory")
    run_parser.add_argument("--resume", help="resume a failed execution directory after compatibility checks")
    validate = commands.add_parser("validate", help="validate and expand a configuration")
    validate.add_argument("--config", required=True)
    preflight = commands.add_parser("preflight", help="validate hardware prerequisites and resolved commands")
    preflight.add_argument("--config", required=True)
    commands.add_parser("inspect", help="show local external-tool availability")
    nsys = commands.add_parser("parse-nsight", help="parse selected OS-runtime wait/blocking statistics")
    source = nsys.add_mutually_exclusive_group(required=True)
    source.add_argument("--profile", help=".nsys-rep file; invokes nsys stats")
    source.add_argument("--csv", help="previously exported osrt_sum CSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            print(json.dumps(tool_versions(), indent=2, sort_keys=True))
            return 0
        if args.command == "parse-nsight":
            summary = extract_profile(args.profile) if args.profile else parse_osrt_csv(
                Path(args.csv).read_text(encoding="utf-8"), str(Path(args.csv))
            )
            print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
            return 0
        config = load_config(args.config)
        if args.command == "validate":
            print(json.dumps({
                "mode": config.mode, "config_fingerprint": config.fingerprint(),
                "request_count": config.request_count,
                "execution_plan": [point.point_id for point in config.points()],
            }, indent=2))
            return 0
        if args.command == "preflight":
            result = run_preflight(config)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] in {"passed", "not_applicable"} else 2
        if args.output_dir:
            config = with_output_dir(config, args.output_dir)
        report = run(config, Path(args.resume) if args.resume else None)
        paths = write_report(report)
        label = "SYNTHETIC (NO HARDWARE EXECUTION)" if report.synthetic else "HARDWARE"
        print(f"InferBench run complete: {label}")
        print(f"execution: {report.execution_dir}")
        for path in paths:
            print(path)
        return 0
    except (ConfigError, OSError, RuntimeError, KeyError, ValueError) as exc:
        print(f"inferbench: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
