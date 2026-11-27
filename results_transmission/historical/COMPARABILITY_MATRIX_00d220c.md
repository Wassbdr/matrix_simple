# COMPARABILITY MATRIX

Generated at: 2026-04-01T12:36:27.667737+00:00

## Protocol Signature (Ours)

| Dataset | Mode | Seeds | Epochs | Max cases | Split mode | Drift | Normalized | MM status | Protocol source |
|---|---|---|---:|---|---|---|---|---|---|
| trainticket | metrics_only | 42,123,7 | 100 | 90 | varying_split_by_run_seed | on | no | not_applicable | fallback_inferred |
| sockshop | metrics_only | 42,123,7 | 100 | 90 | varying_split_by_run_seed | on | no | not_applicable | fallback_inferred |
| online-boutique | metrics_only | 42,123,7 | 100 | 90 | varying_split_by_run_seed | on | no | not_applicable | fallback_inferred |
| trainticket | full_mm_strict | 42,123,7 | 100 | 90 | varying_split_by_run_seed | on | no | comparable | fallback_inferred |
| sockshop | full_mm_strict | 42,123,7 | 100 | 90 | varying_split_by_run_seed | on | no | skipped_with_reason | fallback_inferred |
| online-boutique | full_mm_strict | 42,123,7 | 100 | 90 | varying_split_by_run_seed | on | no | comparable | fallback_inferred |

## Published Baselines Available In Repository

| Method | Avg@1 | Avg@3 | Avg@5 | MRR | NDCG@3 | NDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| BARO | 0.71 | 0.82 | 0.86 | 0.77 | 0.75 | 0.79 |
| CausaRank | 0.52 | 0.70 | 0.79 | 0.63 | 0.59 | 0.66 |
| E-Diagnosis | 0.64 | 0.77 | 0.83 | 0.72 | 0.70 | 0.74 |
| MicroRCA | 0.49 | 0.68 | 0.77 | 0.60 | 0.57 | 0.63 |

## Isoperimeter Verdict

| Claim | Evidence In Repo | Verdict | Missing For Full Isoperimeter |
|---|---|---|---|
| Ours vs published RCAEval baselines (metrics-only) | Baseline values available, but no per-dataset split/preprocessing signatures. | partially_comparable | Published baseline split protocol per dataset; Published preprocessing/normalization details; Case filtering policy alignment (max_cases, exclusions) |
| Ours MM strict vs published baselines | Published baselines in repo are not multimodal and have no MM protocol signature. | not_isoperimeter | Published multimodal baseline references with matching modality scope; Dataset-level MM comparability status for all compared datasets |

## Decision Use

- Use this matrix to qualify every baseline claim in slides/report.
- A claim marked not_isoperimeter must be framed as indicative, not conclusive SOTA evidence.
