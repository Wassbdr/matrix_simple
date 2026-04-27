.PHONY: help install dev test lint format audit benchmark-smoke benchmark-paper clean

PYTHON ?= python

help:
	@echo "Targets:"
	@echo "  make install          Install runtime dependencies (editable)"
	@echo "  make dev              Install dev dependencies + pre-commit hooks"
	@echo "  make test             Run pytest unit tests"
	@echo "  make lint             Run ruff linter"
	@echo "  make format           Run ruff formatter"
	@echo "  make audit            Run pre-publication audit"
	@echo "  make benchmark-smoke  Quick simulated benchmark (1 seed, N=20)"
	@echo "  make benchmark-paper  Full publication-grade benchmark (~4h)"
	@echo "  make clean            Remove caches and generated artifacts"

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install || echo "pre-commit not installed; skipping hook install"

test:
	$(PYTHON) -m pytest -q rca_pipeline/tests

lint:
	$(PYTHON) -m ruff check rca_pipeline

format:
	$(PYTHON) -m ruff format rca_pipeline
	$(PYTHON) -m ruff check --fix rca_pipeline

audit:
	bash scripts/pre_publish_audit.sh

benchmark-smoke:
	$(PYTHON) -m rca_pipeline.benchmark \
		--sizes 20 --seeds 42 \
		--supervised-epochs 30 --late-contrastive-epochs 15 \
		--sta-warmup-epochs 5 --sta-k-anneal-epochs 10 \
		--baseline-epochs 30 \
		--output-json /tmp/benchmark_smoke.json

benchmark-paper:
	bash scripts/run_paper_benchmark.sh

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f /tmp/benchmark_smoke.json
