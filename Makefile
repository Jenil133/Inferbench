.PHONY: ruff compile validate smoke build

ruff:
	ruff check src

compile:
	PYTHONPATH=src python3 -m compileall -q src

validate: ruff compile
	PYTHONPATH=src python3 -m inferbench validate --config examples/simulation-matrix.json
	PYTHONPATH=src python3 -m inferbench validate --config examples/hardware-matrix.json
	PYTHONPATH=src python3 -m inferbench parse-nsight --csv examples/nsight-osrt-sample.csv

smoke:
	PYTHONPATH=src python3 -m inferbench run --config examples/simulation-matrix.json --output-dir /tmp/inferbench-smoke

build:
	python3 -m build
