# Contributing

## Running the test suite

```bash
pip install -e ".[dev]"
python -m pytest -q rca_pipeline/tests
```

## Linting

```bash
python -m ruff check rca_pipeline
```

The project follows `ruff` with `line-length = 100`, `target-version = py39`,
and rules `E, F, I, UP, B`. Fix errors before opening a PR.

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
- Do **not** commit `*.pt` checkpoints, generated `benchmark*.json` files,
  or raw dataset files — they are in `.gitignore` for a reason.
