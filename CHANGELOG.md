# Changelog

All notable changes to this project will be documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- MIT `LICENSE`, `CITATION.cff`, `SECURITY.md`.
- `CONTRIBUTING.md` and GitHub issue / PR templates.
- `Makefile` with `install`, `dev`, `test`, `lint`, `format`, `audit`,
  `benchmark-smoke`, `benchmark-paper`, `clean` targets.
- `.pre-commit-config.yaml` (ruff, large-file guard, private-key detection).
- `scripts/pre_publish_audit.sh` — repeatable repo audit before public push.
- `scripts/run_paper_benchmark.sh` — publication-grade benchmark suite
  (5 seeds, deterministic, simulated + RCAEval metrics-only + multimodal strict).
- `requirements-frozen.txt` — exact dependency pins for reproducibility.
- `.env.example` — environment-variable template (Loki, Prometheus, kubeconfig).

### Changed
- `pyproject.toml` declares MIT license, authors, keywords, PyPI classifiers.
- `pyproject.toml` moves `locust` to optional `[k8s]` extras (it is a load
  generator, not required for model code).
- README rewritten in English with sourced result tables, ASCII architecture
  diagram, badges, and BibTeX citation.
- Internal supervisor-facing documents moved to `docs/internal/`
  (`Week1.md` … `Week5.md`, `rapport_stage.md`, `resume_rapport.md`,
  `ACCOMPAGNEMENT_MAITRE_STAGE.txt`, `README_resultats.txt`).

### Removed
- Untracked generated artifacts: `rca_pipeline/graph_heatmap.png`,
  `rca_pipeline/simulated_dataset.csv`. Files remain on disk;
  removed from git index and added to `.gitignore`.

### Security
- Documented sandbox-only `mypassword` placeholder in `k8s/app/{db,backend}.yaml`
  (`SECURITY.md`).
- Confirmed via git history audit that no real secrets, kubeconfig, or
  credential files were ever committed.
