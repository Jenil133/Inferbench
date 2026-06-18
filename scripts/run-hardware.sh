#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${ROOT}/examples/hardware-matrix.json}"
OUTPUT="${2:-${ROOT}/reports/hardware}"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m inferbench run --config "${CONFIG}" --output-dir "${OUTPUT}"
