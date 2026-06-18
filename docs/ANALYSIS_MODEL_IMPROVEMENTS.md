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
| warmup=10, class-aware negatives | 0.200 ±0.000 — **also collapses** |

The ±0.000 std (exactly chance level, all seeds) shows the selector
converges to the same degenerate solution every time. The class-aware
negative masking does **not** fix it — same-class InfoNCE conflict was a
plausible but wrong hypothesis. Component ablation (one term zeroed at a
time, 1 seed, TrainTicket):

| Ablation (warmup=10) | Avg@1 | MRR |
|---|---|---|
| baseline warmup=0 | 0.950 | 0.975 |
| full (NCE + GIB + temperature) | **0.200** | 0.457 |
| NCE zeroed → GIB + temperature | **0.200** | 0.457 |
| GIB zeroed → NCE + temperature | 0.950 | 0.975 |
| both losses zeroed → temperature only | 0.950 | 0.975 |

**The culprit is the variational GIB term, not InfoNCE.** On RCAEval
N=5, `get_optimal_k(5) = 5`, so `target_rate = k/N = 1.0`. The intended
clamp `q = clamp(target_rate, eps, 1-eps)` with `eps=1e-8` is a **no-op in
float32** (`1.0 - 1e-8 == 1.0` exactly; float32 epsilon ≈ 1.2e-7), so
`KL(p ‖ Bernoulli(1))` evaluates to `inf` with **NaN gradients**: one
optimizer step NaN-poisons the weights, every subsequent forward returns
NaN logits, and ranking NaN gives a constant order — exact chance level,
±0.000 across seeds.

**Fix (two layers):**
1. `variational_gib_loss` now clamps with a float32-representable margin
   (`1e-6`) — no more inf/NaN for any target rate (regression-tested in
   `tests/test_gib_saturation.py`).
2. `benchmark_real.train_and_eval_sta` skips the GIB term entirely when
   `k >= N`: with everything selected there is no bottleneck to regularize,
   and even a finite KL toward Bernoulli(≈1) still saturates the selector.
   On topologies with k < N (e.g. ObservIT N=14) the term keeps its
   intended role.

The class-aware InfoNCE stays as the default warmup objective
(theoretically sounder on labeled pools), now decoupled from the collapse.

**Validation A/B with the fix (TrainTicket, 3 seeds, epochs=100, warmup=10):**

| Variant | Avg@1 STA |
|---|---|
| warmup=0 (canonical baseline) | 0.967 ±0.024 |
| warmup=10 before fix (either NCE mode) | 0.200 ±0.000 |
| warmup=10 fixed, class-aware NCE | **0.967 ±0.024** (= baseline) |
| warmup=10 fixed, legacy NCE | 0.933 ±0.024 |

Conclusions: (1) the collapse is eliminated; (2) class-aware negatives do
matter — legacy stays 3.4 pp under baseline, confirming the same-class
conflict as a real but secondary effect; (3) warmup is now *safe* but
*neutral* on TrainTicket (no gain over warmup=0). The warmup=0 guardrail
remains the canonical default for real data; the footgun is defused and
warmup-based research (e.g. multimodal pretraining) is unblocked.

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

**Command** (the gamma sweep is post-hoc — one training, several blends):

```bash
python -m rca_pipeline.benchmark_real \
    --observit-csv datasets/sockshop_runs_full \
    --observit-window 12 --observit-topology sock_shop \
    --seeds 42,123,7,1337,2025 --epochs 100 \
    --sta-contrastive-warmup 0 --deterministic \
    --cascade-prior-gamma 0.2,0.3,0.5 \
    --output-json /tmp/observit_cascade_sweep.json
```

then compare the `STA+cascade@γ` / `GATv2+cascade@γ` per-root entries
against the baseline `STA` / `GATv2` ones (the cascade scenario maps to
root cause `orders-db` in `analyze_per_scenario.CHAOS_SCENARIOS`).

**Acceptance:** cascade scenario Avg@1 ≥ 0.15 for STA+cascade AND no
regression on the four hard/gray scenarios (per-root breakdown).

**Mechanism validated on the real sock_shop N=14 topology** (no training
needed — `cascade_prior` is post-hoc; this isolates the prior from model
noise). Injecting the documented cascade signature (root cause `orders-db`
metric-quiet, neighbors `orders`/`front-end` loud) into a local-evidence
model that ranks the loudest node first:

| Node | anomaly | prior |
|---|---|---|
| orders-db (root cause) | 0.026 | **0.974** |
| orders (loud symptom) | 1.000 | 0.000 |
| front-end | 0.748 | 0.053 |
| payment | 0.010 | 0.865 |

| γ (τ=0.1) | top-1 | `orders-db` rank | p(orders-db) |
|---|---|---|---|
| 0.0 (model only) | orders | 3 | 0.000 |
| 0.3 | orders | 2 | 0.221 |
| 0.5 | orders | 2 | 0.369 |
| 0.7 | **orders-db** | **1** | 0.516 |

The prior correctly assigns the highest score to the quiet root cause and
recovers it to top-1 at γ≈0.7. (`payment` also scores high — it is a leaf
neighbor of the loud `orders`; in a real cascade its own metrics would not
be flat, so the model term keeps it down. This is exactly why the blend,
not the prior alone, is the right tool.)

**Status — full benchmark blocked by host RAM, not by the method.** The
N=14 ObservIT benchmark (5 seeds × 100 epochs × {STA, GATv2}) OOM-kills on
this machine: a full sock-shop microservice stack (locust, recommendation/
reviews servers, watcher) already occupies ~7 GB of the 8 GB, leaving
< 500 MB. The validation command above is ready to run once memory frees
up (close the stack, or `wsl --shutdown` then relaunch). Until then the
mechanism is verified analytically; the end-to-end per-scenario numbers
remain pending.

## 4. 10-seed robustness

`results_transmission/paper_benchmark_real_10seed.json` already provides
10-seed runs with reliable bootstrap CIs (`seed_level_bootstrap_reliable:
true`; STA Avg@1: TrainTicket 0.945, SockShop 0.805, Online Boutique 0.670).
Final step of this campaign: re-run the winning variants of §2–§3 at 10
seeds to quantify their gains at the same standard.

Same RAM blocker as §3: deferred until a machine with free memory. Command:

```bash
# warmup fix (§2) and cascade prior (§3), 10 seeds, real datasets
python -m rca_pipeline.benchmark_real --config configs/benchmark_real.yaml \
    --seeds 42,123,7,1337,2025,314,555,888,999,4242 \
    --output-json /tmp/bench_real_10seed_improved.json
python -m rca_pipeline.benchmark_real \
    --observit-csv datasets/sockshop_runs_full --observit-topology sock_shop \
    --seeds 42,123,7,1337,2025,314,555,888,999,4242 --epochs 100 \
    --sta-contrastive-warmup 0 --deterministic \
    --cascade-prior-tau 0.1 --cascade-prior-gamma 0.5,0.7 \
    --output-json /tmp/observit_cascade_10seed.json
```

## Reproducibility note

All §1 numbers are from fresh 3-seed RCAEval runs (data under
`rcaeval/data/`). §2 numbers are direct A/B runs on TrainTicket. §3 is
validated analytically on the real `sock_shop` topology
(`tests/test_cascade_prior_sockshop.py`) because the full N=14 training
benchmark exceeds available host RAM. The frozen
`results_transmission/` artifacts were never modified; every command writes
to `/tmp` or a new file.
