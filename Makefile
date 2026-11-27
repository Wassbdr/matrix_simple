.PHONY: help install dev test coverage lint format audit benchmark-smoke benchmark-paper clean report report-clean figures paper verify-manifest manifest demo scaling showcase data bench

PYTHON ?= python3
RCA_DATA ?= $(HOME)/data/rcaeval

help:
	@echo "Targets:"
	@echo "  make install          Install runtime dependencies (editable)"
	@echo "  make dev              Install dev dependencies + pre-commit hooks"
	@echo "  make test             Run pytest unit tests (rca_pipeline + serving)"
	@echo "  make data             Download the pinned RCAEval datasets"
	@echo "  make bench            Compare STA with RCAEval references"
	@echo "  make coverage         Run tests with the coverage gate"
	@echo "  make lint             Run ruff linter"
	@echo "  make format           Run ruff formatter"
	@echo "  make audit            Run pre-publication audit"
	@echo "  make benchmark-smoke  Quick simulated benchmark (1 seed, N=20)"
	@echo "  make benchmark-paper  Full publication-grade benchmark (~4h)"
	@echo "  make figures          Regenerate paper figures from results_transmission/*.json"
	@echo "  make paper            Build the arXiv preprint PDF (docs/arxiv)"
	@echo "  make demo             Run the Sock Shop incident demo (STA bundle vs BARO)"
	@echo "  make scaling          Scaling benchmark STA vs full-graph, N=14..500 (CPU, ~1-2h)"
	@echo "  make showcase         Rebuild showcase/ (poster, GIF, MP4, showcase.json)"
	@echo "  make verify-manifest  Check every result artifact against its SHA-256"
	@echo "  make manifest         Rewrite the results manifest (after regenerating artifacts)"
	@echo "  make clean            Remove caches and generated artifacts"

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install || echo "pre-commit not installed; skipping hook install"

test:
	$(PYTHON) -m pytest -q

data:
	$(PYTHON) scripts/download_rcaeval.py --root "$(RCA_DATA)"

bench:
	$(PYTHON) -m rca_pipeline.bench $(BENCH_ARGS)

coverage:
	$(PYTHON) -m pytest --cov --cov-report=term

lint:
	$(PYTHON) -m ruff check rca_pipeline serving conftest.py

format:
	$(PYTHON) -m ruff format rca_pipeline
	$(PYTHON) -m ruff check --fix rca_pipeline

verify-manifest:
	$(PYTHON) -m rca_pipeline.provenance verify

manifest:
	$(PYTHON) -m rca_pipeline.provenance build

demo:
	$(PYTHON) -m rca_pipeline.demo_incident run

scaling:
	$(PYTHON) -m rca_pipeline.scaling_benchmark --sizes 14,50,100,200 --train-threads 4 --skip-latency --output-json results/scaling_benchmark.json
	$(PYTHON) -m rca_pipeline.scaling_benchmark --sizes 500 --models STA,FullGraphGCN --train-threads 4 --skip-latency --resume --output-json results/scaling_benchmark.json
	$(PYTHON) -m rca_pipeline.scaling_benchmark --latency-only --profile-latency --latency-k 16 --models STA,FullGraphGCN,LateGATv2 --output-json results/scaling_benchmark.json

showcase:
	bash showcase/make_showcase.sh

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

# Full publication suite: simulated + RCAEval + ObservIT cluster + manifest + SHA256
publication:
	bash scripts/run_publication_suite.sh

publication-dry-run:
	bash scripts/run_publication_suite.sh --dry-run

publication-local:
	bash scripts/run_publication_suite.sh --skip-observit

# Supprime uniquement du régénérable. Ne touche ni à results_transmission/,
# ni à datasets/, checkpoints/, rcaeval/ ou .venv/.
clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -not -path './.venv/*' -not -path './rcaeval/*' -prune -exec rm -rf {} +
	rm -f .coverage .coverage.* coverage.xml
	rm -rf htmlcov
	rm -f dataset_windows.pt test_integration.pt
	rm -f benchmark_cascade_sweep2.log port_forward.log
	rm -f /tmp/benchmark_smoke.json

report:
	$(MAKE) -C docs/rapport pdf

report-clean:
	$(MAKE) -C docs/rapport clean

# Regenerate the paper figures (F1-F4) from the canonical benchmark JSON.
# Pass RESULTS_DIR=... or per-file overrides to point at a re-run (e.g. 10-seed).
figures:
	$(PYTHON) -m rca_pipeline.make_figures --out docs/arxiv/figures
	$(PYTHON) -m rca_pipeline.make_figures --out docs/rapport/figures

paper:
	$(MAKE) -C docs/arxiv pdf
