# PAPER BENCHMARKS

This note defines the benchmark perimeter used for report/soutenance claims.

## Scope

- Dataset family: RCAEval real datasets.
- Primary datasets: trainticket, online-boutique, sockshop.
- Primary metrics: Avg@1, Avg@3, Avg@5, MRR, NDCG@3, NDCG@5.

## Presets

- smoke:
  - seeds: 42,123
  - epochs: 20
  - max_cases: 20
- long:
  - seeds: 42,123,7,1337,2025
  - epochs: 100
  - max_cases: all
- paper:
  - seeds: 42,123,7,1337,2025,314,555,888,999,4242
  - epochs: 100
  - max_cases: all

Reference implementation: rca_pipeline/validation_protocol.py.

## Multimodal Perimeter

- metrics_only run is in scope for all three datasets.
- strict multimodal runs are in scope for:
  - trainticket
  - online-boutique
- sockshop strict multimodal is intentionally excluded from claim perimeter in this protocol.

Rationale:

- strict MM requires stable modality signal and comparable execution.
- sockshop MM runs are marked skipped_with_reason in validation stubs to avoid ambiguous global MM claims.

## Claim Rules

- Any global MM claim must include dataset-level status and skip reasons.
- If one dataset is skipped/non-comparable, global MM aggregation is blocked.
- Baseline comparisons must be qualified by the comparability matrix.

## Required Artifacts For Submission

- rca_pipeline/benchmark_real_long_prepublish.json
- rca_pipeline/benchmark_real_long_mm_strict_prepublish.json
- rca_pipeline/validation_runs/*.json
- docs/COMPARABILITY_MATRIX.md
- results_transmission/canonical_results_manifest.json
