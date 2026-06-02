# COMPARABILITY MATRIX

Generated at: 2026-06-02T15:33:31.330826+00:00

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

## Isoperimetric BARO (our protocol)

BARO re-run on the *same* cases, candidate node set, per-seed split and metric functions as our models (see `baseline_baro.py`). Avg@1 (%).

| Dataset | BARO (ours) | BARO (published) | STA (ours) | GATv2 (ours) | Verdict |
|---|---:|---:|---:|---:|---|
| trainticket | 80.0 | 71.0 | 96.0 | 95.0 | ours_better |
| sockshop | 84.0 | 71.0 | 85.0 | 81.0 | within_noise |
| online-boutique | 86.0 | 71.0 | 65.0 | 68.0 | baro_better |

> Under identical conditions BARO is far stronger than its published 0.71. Our models win on trainticket, are within noise on sockshop, and **lose on online-boutique** — we make no uniform SOTA claim over BARO.

## Isoperimeter Verdict

| Claim | Evidence In Repo | Verdict | Missing For Full Isoperimeter |
|---|---|---|---|
| Ours vs BARO (metrics-only), isoperimetric (our protocol) | BARO re-run on identical cases, candidate node set, per-seed split and metric functions (results_transmission/baselines_rcaeval_ourprotocol.json). Under identical conditions BARO is far stronger than its published 0.71: ours wins on trainticket, ties on sockshop, loses on online-boutique — no uniform SOTA claim. | comparable |  |
| Ours vs *published* RCAEval baseline numbers (BARO 0.71 etc.) | Published constants have no per-dataset split/preprocessing signature; superseded by the isoperimetric row above. | partially_comparable | Published baseline split protocol per dataset; Published preprocessing/normalization details |
| Ours MM strict vs published baselines | Published baselines in repo are not multimodal and have no MM protocol signature. | not_isoperimeter | Published multimodal baseline references with matching modality scope; Dataset-level MM comparability status for all compared datasets |

## Decision Use

- Use this matrix to qualify every baseline claim in slides/report.
- A claim marked not_isoperimeter must be framed as indicative, not conclusive SOTA evidence.
