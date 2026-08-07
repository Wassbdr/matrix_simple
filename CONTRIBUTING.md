# Contributing

New to the project? Read [`HANDOVER.md`](HANDOVER.md) first — it explains how to
install, what is reproducible offline, and where the canonical results live.

## Running the test suite

```bash
pip install -e ".[dev]"
make test                  # or: python -m pytest -q
```

`make test` runs both `rca_pipeline/tests` and `serving/tests`. `make coverage`
adds the coverage gate.

## Linting

```bash
make lint                  # ruff check rca_pipeline serving conftest.py
make format                # ruff format + ruff check --fix
```

The project follows `ruff` with `line-length = 100`, `target-version = py39`,
and rules `E, F, I, UP, B`. Fix errors before opening a PR. `pre-commit install`
(via `make dev`) wires the same checks into your commits.

## Code conventions

- Python ≥ 3.9. Torch + `torch_geometric` for the GNN layers.
- **No dependency that is not declared in `pyproject.toml`** — it is the single
  source of truth, there is no `requirements.txt` to keep in sync.
- Default seeds: `42, 123, 7, 1337, 2025`. Default simulated sizes: `20, 50, 100`.
  Changing either invalidates comparison with the published tables.
- Every entry point accepts `--config <yaml>` validated by pydantic
  (`rca_pipeline/config.py`); unknown keys fail loudly. Versioned configs live in
  [`configs/`](configs/).

## Guardrails

These are not style preferences — breaking one of them silently invalidates a
published result.

- **`--sta-contrastive-warmup 0` on every real-world benchmark.** With
  `warmup = 10`, TrainTicket Avg@1 collapses from 73.3 % to 0.0 % (3 seeds,
  20 cases). Evidence: `results_transmission/warmup_real_trainticket_w{0,10}.json`.
  The guardrail is enforced at config-load time — do not weaken it.
- **Never modify `results_transmission/canonical_results_manifest.json*`.** It is
  the frozen defense artifact; its SHA-256 is what makes the published numbers
  verifiable. Regenerating results means writing a *new* file, not editing this one.
- **Never commit `*.pt`, `checkpoints/`, `datasets/`, `rcaeval/data/`, or
  generated `benchmark*.json`.** They are in `.gitignore` for a reason (`rcaeval/`
  alone is ~39 GB).
- **Do not mock the loaders or the DB in integration tests.** An integration test
  that mocks its data source tests nothing.
- Cite `paper_benchmark_*.json` only. The `rca_pipeline/*_prepublish.json` files
  are short-epoch legacy artifacts whose `reference_top1` fields are stale.

## Commit convention

Use the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
feat: add trace feature extractor for Jaeger spans
fix: handle missing edge_index in GraphBuilder
docs: update COMPARABILITY_MATRIX with MM strict results
```

## Reporting bugs

Open an issue on GitHub. Include:
1. The exact command you ran.
2. The full stack trace.
3. Python version (`python --version`) and PyTorch version (`python -c "import torch; print(torch.__version__)"`).

## Branch policy

- `main` is the stable branch; all results in `results_transmission/` are tied to it.
- Open a PR for any change; CI must be green before merge.
