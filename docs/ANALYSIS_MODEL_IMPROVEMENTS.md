# Model Improvement Campaign (post-soutenance, Phase 1A)

Improvements explored on top of the canonical results. The frozen artifacts
in `results_transmission/` are untouched; every number here comes from new
runs (commands given inline). Protocol: same as README §5 (stratified split
80/20 per seed, `--sta-contrastive-warmup 0` unless stated, 3 seeds 42/123/7
for exploration, 10 seeds for final claims).

## 1. Post-hoc STA+GATv2 ensemble — negative result on RCAEval

**Hypothesis** (per-scenario breakdown): STA and GATv2 specialize (STA wins
hard failures, GATv2 resists cascades), so a soft vote might combine their
strengths.

**Command**: `python -m rca_pipeline.benchmark_real --config
configs/benchmark_real.yaml --ensemble --output-json <out>`

**Result (3 seeds, Avg@1)**:

| Dataset | STA | GATv2 | Soft vote 50/50 | Conf-weighted | Oracle (perfect router) | Top-1 disagreement |
|---|---|---|---|---|---|---|
| TrainTicket | **0.967** | 0.950 | 0.933 | 0.933 | 0.983 | 5.0% |
| SockShop | **0.800** | 0.767 | 0.800 | 0.800 | 0.817 | 10.0% |
| Online Boutique | 0.683 | **0.717** | 0.700 | 0.700 | 0.767 | 16.7% |

**Reading:**

- The 50/50 soft vote never beats the better single model, and on
  TrainTicket it is *worse than both* — it sits at the worst point of the
  weight sweep (w=0.25 → 0.950, w=0.5 → 0.933, w=0.75 → 0.950). When the
  models disagree with similar confidence, averaging promotes the wrong
  candidate.
- The oracle (picking per case whichever model ranks the truth first)
  bounds any routing scheme at +1.7 to +5 pp over the best single model —
  with 3-seed noise of the same magnitude (±2.4 to ±10 pp std).
- The ensemble is heavily **underconfident**: ECE 0.39–0.58 vs 0.08–0.20
  for the single models (averaging two distributions caps top-1 confidence
  near 0.5 while accuracy stays ≥ 0.7).

**Decision:** no ensemble in the serving path on small-N RCAEval-like
topologies; pick the better model per deployment. Routing remains worth
testing only where the specialization is documented and large — the
ObservIT cascade scenario (§3).

## 2. Real-data contrastive-warmup collapse — mechanism + fix

**Documented failure** (README §5.4): warmup=10 drops TrainTicket Avg@1
from 73% to 0%; all real benchmarks therefore run warmup=0 (guardrail).

**Mechanism found in `benchmark_real.train_and_eval_sta`:** the warmup pool
contains only anomalous windows (~5 root-cause classes), and the legacy
scheme draws anchors and negatives as random halves of that same pool. Most
negatives then share the anchor's root cause: InfoNCE actively pushes apart
embeddings of windows with the *same* label, destroying the selector before
supervised training starts.

**Fix:** `losses.class_aware_info_nce_loss` masks same-class pairs out of
the negative set (the augmented view remains the positive).
`--sta-warmup-mode class_aware` is the new default; `legacy` is kept to
reproduce the collapse.

**A/B (TrainTicket, 3 seeds, epochs=100, warmup=10, Avg@1):**

| Variant | Avg@1 |
|---|---|
| warmup=0 (canonical baseline) | 0.967 |
| warmup=10, legacy negatives | 0.200 ±0.000 (= chance on N=5: total selector collapse) |
| warmup=10, class-aware negatives | _pending_ |

The legacy reproduction confirms the mechanism: ±0.000 std across seeds
means the selector converges to the same degenerate solution every time —
this is not noise, it is the InfoNCE objective fighting the labels.

## 3. Cascade propagation prior — _pending ObservIT validation_

**Documented failure** (SYNTHESE_FINALE §6.4): on the
`network-delay-orders-to-orders-db` cascade, the root cause is metric-quiet
while its neighbors scream; STA collapses to Avg@1 0.043, GATv2 0.459.

**Approach:** inference-time blend with a propagation prior that scores the
inverse pattern — quiet nodes with loud neighborhoods
(`rca_pipeline/cascade_prior.py`):

```
prior(v) = mean(anomaly(neighbors(v))) * (1 - anomaly(v))
p_final  = (1-γ) softmax(model) + γ softmax(prior/τ)
```

**Command**: `python -m rca_pipeline.benchmark_real --observit-csv
datasets/sockshop_runs_full --observit-topology sock_shop --seeds
42,123,7,1337,2025 --epochs 100 --sta-contrastive-warmup 0
--cascade-prior-gamma 0.3 --output-json <out>` then
`python -m rca_pipeline.analyze_per_scenario` on the output.

**Acceptance:** cascade scenario Avg@1 ≥ 0.15 for STA+cascade AND no
regression on the four hard/gray scenarios (per-root breakdown).

## 4. 10-seed robustness

`results_transmission/paper_benchmark_real_10seed.json` already provides
10-seed runs with reliable bootstrap CIs (`seed_level_bootstrap_reliable:
true`; STA Avg@1: TrainTicket 0.945, SockShop 0.805, Online Boutique 0.670).
Final step of this campaign: re-run the winning variants of §2–§3 at 10
seeds to quantify their gains at the same standard.
