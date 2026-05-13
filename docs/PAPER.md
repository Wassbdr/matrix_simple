# STA: A Select-Then-Analyze Framework for Scalable Root Cause Analysis in Microservice Architectures

**Wassim Badraoui**
*Devoteam, Paris, France · EPITA, Paris, France*
wassim.badraoui@epita.fr

---

## Abstract

Root Cause Analysis (RCA) in microservice architectures is a critical yet challenging task: failures propagate across service dependency graphs, observable signals are noisy, and the search space grows with the number of instrumented services. I present STA (Select-Then-Analyze), an early-bottleneck spatio-temporal graph neural network that reduces RCA complexity from O(N²) to O(N + k²) by selecting a compact candidate subgraph before deep spatial analysis. STA combines a zero-hop temporal encoder with a differentiable Straight-Through Estimator selector and Variational Graph Information Bottleneck regularization. I evaluate four model variants—STA, GATv2, STA-MM, and GATv2-MM—across three experimental regimes: synthetic Barabási-Albert graphs (N ∈ {20, 50, 100}), three real-world RCAEval datasets (N=5), and a private Kubernetes cluster (N=14) with Chaos Mesh fault injection. Three findings, deliberately scoped: (i) on synthetic graphs, STA achieves Avg@1 = 0.553 vs. GATv2's 0.490 at N=100 (+6.3 pp), confirming its asymptotic efficiency advantage in the regime where the bottleneck is meaningful; (ii) at small graph scale (N≤14), where all my real-data evaluations live, STA and GATv2 are within the noise envelope of 5-seed evaluation; (iii) multimodal fusion (metrics + logs + traces) degrades performance on public RCAEval datasets (−5% Avg@1 on trainticket) but shows no regression on the private cluster (STA-MM Avg@1 = 0.772 vs. STA Avg@1 = 0.796, both within ±0.043), pointing to log/trace data quality as a likely confounder. I provide a fully reproducible pipeline with canonical SHA-256 artifacts, an explicit comparability matrix for baseline claims (I do not assert SOTA), and an operational deployment stack including ADWIN-based drift detection and a four-mechanism gray failure detector.

---

## 1. Introduction

Modern cloud-native applications deploy business logic across tens to hundreds of loosely coupled microservices. When an incident occurs, identifying the root cause (RC) service—the node that first entered a faulty state—is a prerequisite for any automated remediation. In production environments with dozens of instrumented services, the mean time to identify a root cause through manual triage routinely reaches tens of minutes, creating direct business impact.

RCA for microservices is hard for three related reasons. First, **failures propagate**: a single faulty service sends error signals upstream to its callers, causing a cascade of symptomatic anomalies that masks the original cause. Second, **signals are heterogeneous**: the observable evidence spans time-series metrics (CPU, memory, latency, error rates), unstructured logs, and distributed traces—each modality carrying partially overlapping but distinct information. Third, **the search space grows**: at N = 100 services, a naive pairwise graph analysis costs O(N²) = 10,000 operations at the spatial convolution level; at N = 500 this becomes prohibitive for real-time diagnosis.

Recent work has addressed these challenges using graph neural networks (GNNs) operating on the service dependency graph, with architectures ranging from full-graph GCNs to attention-based spatio-temporal models. However, several important questions remain under-explored in the literature:

1. **Scalability**: Does the spatial bottleneck in full-graph GNNs limit their practical deployment at large N, and can a principled selection step reduce this cost without sacrificing recall?
2. **Graph vs. temporal**: At the small-N scale of public benchmarks, does incorporating graph structure actually improve over temporal-only encoders, or does it introduce noise?
3. **Multimodal integration**: Do logs and traces complement metrics, or do they introduce gradient conflicts and optimization overhead that net-hurt performance?
4. **Operational robustness**: How do RCA models behave during Kubernetes deployment transitions (concept drift) and under gray failures—anomalies that stay within individual SLA thresholds yet degrade system performance?

This paper makes four contributions to address these questions:

1. **STA: Select-Then-Analyze** (Section 5.3), an early-bottleneck GNN that achieves O(N + k²) complexity via differentiable top-k node selection with Variational GIB regularization. STA allows deep spatio-temporal analysis on a compact candidate subgraph while maintaining end-to-end trainability through a Straight-Through Estimator.

2. **Systematic multi-regime evaluation** (Sections 6–7): I benchmark four model variants across synthetic graphs (N ∈ {20, 50, 100}), three public RCAEval datasets, and a live Kubernetes cluster with five Chaos Mesh fault scenarios. I report Avg@k, MRR, and NDCG@3 with standard deviations across five independent seeds.

3. **A robust negative result on multimodal fusion** (Section 7.3): I show empirically that on RCAEval datasets, fusing logs and traces with metrics degrades or matches metrics-only performance, and I propose three explanatory hypotheses. This result has direct practical implications for deployment decisions.

4. **An operational RCA stack** (Section 5.6): I contribute an ADWIN-based concept drift detector with per-metric cooldown logic, a four-mechanism composite gray failure detector (P99/P50 micro-stutters, inter-metric decorrelation, Mann-Kendall trend, Limplock signal), and three variants of a cascade-aware loss function including a fully learnable propagation matrix.

The rest of this paper is organized as follows: Section 2 surveys related work. Section 3 formalizes the RCA problem. Section 4 describes my datasets and experimental protocol. Section 5 details all model architectures and pipeline components. Section 6 presents experimental setup. Section 7 reports results and ablations. Section 8 discusses findings and limitations. Section 9 concludes.

---

## 2. Related Work

### 2.1 Root Cause Analysis for Microservices

Early microservice RCA systems rely on correlation-based propagation: **MicroRCA** constructs a weighted service graph from Pearson correlations and applies random walks to rank root cause candidates. **CausaRank** improves this with a causal ranking formulation. **E-Diagnosis** introduces disentangled representation learning to separate anomaly types. **BARO** uses bootstrap-based anomaly ranking on service metrics and achieves the strongest published performance among these four baselines.

The **RCAEval** benchmark (https://github.com/phamquiluan/RCAEval) provides a standardized evaluation framework spanning trainticket, sockshop, online-boutique, and other microservice applications, together with reference implementations and reported numbers for the four baselines above. I evaluate against this benchmark and explicitly qualify my comparisons using an isoperimeter comparability matrix (Section 4.3), since the published baseline numbers I cite do not come with a fully specified training protocol (split policy, number of seeds, case filtering).

### 2.2 GNN-based Fault Localization

Spatio-temporal GNNs have shown strong performance on traffic prediction (STGCN [Yu et al., 2018]) and structural health monitoring. For RCA, the key challenge is adapting these models to directed service dependency graphs with highly variable fault propagation patterns.

The Information Bottleneck principle [Tishby and Zaslavsky, 2015] and its graph adaptation, Graph Information Bottleneck (GIB) [Wu et al., 2020], provide a principled framework for learning sparse graph representations by maximizing I(Z; Y) while minimizing I(Z; X) — retaining only the information about the root cause label while discarding irrelevant node features. Prior work applies GIB at the graph level; I adapt it to node-level mask learning for selective subgraph extraction.

GATv2 [Brody et al., 2022] addresses a known expressiveness limitation of the original GAT [Veličković et al., 2018]: in GAT, the attention mechanism degenerates to a linear function of the query node alone in certain configurations, regardless of the key. GATv2 fixes this by applying the linear projection to the concatenation of query and key before the nonlinearity, enabling truly contextual attention.

### 2.3 Multimodal Fault Diagnosis

The intuition behind multimodal fusion for RCA is that different observability signals capture complementary aspects of system health: metrics track resource utilization and application performance, logs capture discrete events and error codes, and traces provide end-to-end request paths. A common assumption in the multimodal fault-diagnosis literature is that combining these modalities improves over any one of them in isolation.

However, the quality and density of log/trace data varies widely across deployments. In sparse or low-signal observability environments, additional modalities can introduce optimization interference—a phenomenon related to gradient conflicts in multi-task learning [Yu et al., 2020]. My empirical results (Section 7.3) provide evidence for this hypothesis on the RCAEval benchmark.

### 2.4 Gray Failures

Gray failures [Huang et al., 2017] are partial failures where individual metrics remain within their alert thresholds while the system is effectively degraded. Industry post-mortems consistently report that such partial failures account for a sizable fraction of production incidents at large-scale providers, while remaining invisible to per-metric SLO alerting. Two related patterns are particularly relevant for microservice RCA: metastable degraded states, where the system stays in a self-sustaining slow regime after a triggering event; and saturation-under-idleness ("limplock"), where queues grow monotonically while CPU stays flat, indicating a saturated-but-idle system. My gray failure detector targets these patterns explicitly (Section 5.6).

### 2.5 Concept Drift in Deployed Systems

Kubernetes deployments, rolling updates, and autoscaling cause metric distributions to shift over time. Training a static RCA model on pre-deployment data and deploying it without adaptation leads to false positives during legitimate system changes. ADWIN (ADaptive WINdowing) [Bifet and Gavaldà, 2007] is a landmark algorithm for streaming drift detection with O(log n) amortized complexity that maintains a compressed bucket structure and detects changes in mean via a statistical test on all possible sub-windows.

---

## 3. Problem Formulation

### 3.1 System Model

I model a microservice system as a directed graph G = (V, E) where V = {v₁, ..., v_N} is the set of N services and E ⊆ V × V encodes runtime call dependencies (vᵢ → vⱼ if service i calls service j). Each service vᵢ is observed through a multivariate time series X_i ∈ ℝ^(T × F) where T is the observation window length and F is the number of monitored metrics per service. The full system observation is X ∈ ℝ^(T × N × F).

I define three fault taxonomy classes inspired by the multivariate time-series anomaly literature (Chandola et al., 2009):
- **Hard failures**: the root cause service exhibits a sharp, isolated anomaly (CPU spike, OOM kill, pod termination). Signal is strong and localized.
- **Cascade failures**: a fault at v_rc propagates through the dependency graph over multiple hops, generating secondary symptoms at downstream services. The causal chain is v_rc → v₁ → ... → v_k (front-end).
- **Gray failures**: all individual metrics at all services remain within SLA thresholds, yet joint behavior degrades. Detectable only through inter-metric decorrelation or subtle trend analysis.

### 3.2 RCA as a Sparse Inverse Problem

Following an inverse problem formulation, I model fault injection as a sparse perturbation U ∈ ℝ^N applied to the system dynamics:

$$
\min_U \; \lVert U \rVert_0 \quad \text{s.t.} \quad Y \approx g_\phi(X), \quad \lVert U \rVert_0 \le 1
\tag{1}
$$

where Y ∈ {0, 1}^N is the root cause indicator vector (exactly one component is non-zero in the single-fault regime), g_φ is the learned scoring function parameterized by φ, and X is the observed metric tensor. The sparsity constraint ||U||₀ ≤ 1 in Eq. (1) encodes the assumption that incidents originate from a single primary failure node (which may subsequently trigger cascades).

### 3.3 Evaluation Criteria

Given a model producing a ranked list of candidate root causes, I evaluate with:
- **Avg@k** (k ∈ {1, 3, 5}): fraction of test cases where the true RC appears in the top-k predictions. Avg@1 is the primary metric—it measures whether immediate automated remediation is possible.
- **MRR** (Mean Reciprocal Rank): mean of 1/rank_RC across test cases. Rewards early correct placement.
- **NDCG@k** (Normalized Discounted Cumulative Gain): position-weighted accuracy measure.

---

## 4. Datasets and Experimental Protocol

### 4.1 Datasets

**Synthetic (Barabási-Albert graphs).** I generate random scale-free graphs using the Barabási-Albert preferential attachment model with N ∈ {20, 50, 100} nodes and connectivity parameter m = 2. For each graph, fault scenarios are sampled by selecting a random root cause node and simulating metric perturbations according to the fault taxonomy. I generate approximately 3,000 windows per configuration with balanced fault types (hard, cascade, gray) and a 20% healthy-window ratio.

**RCAEval benchmark.** I evaluate on three datasets from the RCAEval suite (https://github.com/phamquiluan/RCAEval):
- **trainticket**: N=5 instrumented services, Java-based ticket booking system, 90 fault scenarios per evaluation.
- **sockshop**: N=5 instrumented services, Go-based e-commerce application, 90 fault scenarios.
- **online-boutique**: N=5 instrumented services, Google's polyglot demo application, 90 fault scenarios.

All three datasets are collected from Kubernetes deployments with Chaos Mesh fault injection, covering CPU stress, memory stress, network delay, network loss, and pod kill scenarios.

**ObservIT cluster.** I operate a private Kubernetes cluster (observit-cluster1) running the full sock-shop application with 14 services. Fault scenarios are injected using five Chaos Mesh manifests designed to cover the three fault taxonomy classes:

| Scenario | Root Cause | Type | Mechanism |
|---|---|---|---|
| stress-cpu-catalogue | catalogue | hard | CPU stress workers @ 80% |
| stress-mem-carts | carts | hard | 256 MB memory allocation |
| network-delay-orders-to-orders-db | orders-db | cascade | 400 ms latency + 80 ms jitter |
| network-loss-catalogue-to-catalogue-db | catalogue-db | gray | 15% packet loss |
| pod-kill-user | user | hard | Periodic pod termination |

### 4.2 Protocol

All experiments use 5 independent seeds: {42, 123, 7, 1337, 2025}. For each seed, a distinct stratified train/test split is produced (train_ratio = 0.8, varying_split_by_run_seed mode). This split mode ensures that my variance estimates reflect both model initialization variance and data split variance—a conservative choice that yields wider confidence intervals.

Training configuration: 100 epochs for real-data benchmarks, 150 supervised epochs for synthetic benchmarks (with an additional 80-epoch contrastive pre-training phase for the Late Bottleneck family). Hidden dimension: 32. Optimizer: Adam (lr=1e-3, weight_decay=1e-4). Batch size: 256.

**Statistical disclaimer.** I report mean and standard deviation across 5 seeds. With n=5, bootstrap-based 95% confidence intervals and formal significance tests (Wilcoxon, paired t-test) are not statistically reliable—the sample size is below the threshold where the assumed sampling distribution becomes a faithful approximation. The reported ± values are standard deviations only. Throughout Section 7, I treat a difference between two models as **meaningful** only when it exceeds the larger of the two reported standard deviations; otherwise I explicitly mark it as **within noise**.

For all real-data evaluations, `--sta-contrastive-warmup 0` is enforced: contrastive warmup destabilizes STA on real datasets by overfitting to within-dataset topology biases (see Appendix).

### 4.3 Baseline Comparability

The published baselines (BARO, E-Diagnosis, CausaRank, MicroRCA) are reproduced from the RCAEval reference numbers shipped with the benchmark. I did **not** re-implement these baselines under my own training protocol. Their split strategy, number of seeds, case filtering policy, and preprocessing normalization are not fully specified in the available documentation. I therefore classify my baseline comparisons as **partially comparable** for metrics-only evaluations and **not isoperimeter** for multimodal evaluations.

Practically, this means that when my models report a higher Avg@1 than a reference baseline, I cannot rule out that the gap is partly explained by protocol differences (e.g., a different train/test split, different normalization, a stricter case filter). All comparisons to published baselines in Section 7 should be read as **indicative competitiveness**, not as conclusive SOTA evidence. A rigorous SOTA claim would require re-implementing each baseline under my isoperimeter protocol—deliberately out of scope for this work, which focuses on the relative comparison of my four model variants under a single shared protocol.

---

## 5. Methods

### 5.1 Architecture Overview

My pipeline follows a sequential structure (Fig. 1):

```
                  Figure 1. End-to-end RCA pipeline.

  Prometheus    Loki      Jaeger              [F=6 metrics / N services]
      │           │          │
      └───────────┼──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ CausalGraphBuilder  │  PC → FCI → Granger → BA fallback
       └──────────┬──────────┘
                  │ edge_index, edge_weights
                  ▼
       ┌─────────────────────┐
       │  ADWIN Drift        │  cooldown flag if K8s rollout in progress
       │  Detector           │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │  Gray Failure       │  pre-filter boost
       │  Detector (×4)      │  P99/P50, decorrelation, Mann-Kendall, Limplock
       └──────────┬──────────┘
                  │
                  ▼
       ┌────────────────────────────────────────────────┐
       │  GNN: STA  | GATv2  | STA-MM  | GATv2-MM       │
       │  (Section 5.3–5.5)                             │
       └──────────┬─────────────────────────────────────┘
                  │
                  ▼
          ranked root causes
          Avg@1, Avg@3, MRR, NDCG@3
```

At inference time, the drift detector suppresses RCA output during Kubernetes deployment transitions. The gray failure detector can boost the score of candidate nodes exhibiting subtle correlation anomalies before they enter the main GNN.

The STA model itself is a five-stage pipeline (Fig. 2):

```
       Figure 2. STA: Select-Then-Analyze (five stages).

   X ∈ ℝ^(T × N × F)
        │
        ▼
  [1] LightEncoder        Conv1d × 2 + temporal attention   O(N)
        │                 (no graph message passing)
        ▼
   H_fast ∈ ℝ^(N × H)
        │
        ▼
  [2] STESelector         Gumbel-sigmoid + top-k + STE      O(N)
        │
        ▼
   M_hard ∈ {0,1}^N  (k selected nodes)
        │
        ▼
  [3] Subgraph            pyg_subgraph (vectorized)         O(E)
      extraction
        │
        ▼
   S = G[supp(M_hard)]    (k nodes, k² edges max)
        │
        ▼
  [4] DeepSTGCN           Conv1d × 2 + GCNConv × 2          O(k²)
        │
        ▼
   H_gcn ∈ ℝ^(k × H)
        │
        ▼
  [5] SubgraphScorer      logits = (1-α)·scorer + α·selector   O(k)
        │
        ▼
   ranked logits ∈ ℝ^N    (−∞ on unselected nodes)
```

### 5.2 Shared Temporal Encoder

All four models share a **LightEncoder** as their first stage: a zero-hop temporal encoder that processes each node's metric time series independently (no message passing at this stage).

Given node features X_i ∈ ℝ^(T × F), the LightEncoder produces a node embedding h_i ∈ ℝ^H via:

```
h_i^(1) = ReLU(Conv1d(X_i, kernel=3))    # [T, H]
h_i^(2) = ReLU(Conv1d(h_i^(1), kernel=3)) # [T, H]
α_i     = softmax(W_α · tanh(W_q · h_i^(2)))  # temporal attention [T, 1]
h_i     = Σ_t α_i^t · h_i^(2)_t          # [H]
```

The zero-hop constraint (no GCN before selection) is deliberate: applying graph convolution before the node selector at large N causes over-smoothing [Li et al., 2018], blending node representations and reducing the discriminability of individual fault signatures. The selector must see local node features, not neighborhood-averaged ones.

### 5.3 STA: Select-Then-Analyze

STA implements an early bottleneck architecture in five stages.

**Stage 1 (O(N)): LightEncoder.** Produces H_fast = {h_i}_{i=1}^N ∈ ℝ^(N × H) as described above.

**Stage 2 (O(N)): STESelector.** Computes a differentiable binary mask M ∈ {0, 1}^N via:

$$
\begin{aligned}
s_i &= \mathrm{MLP_{sel}}(h_i) + \mathrm{Gumbel}(0, \beta) \\
M_{\mathrm{soft},i} &= \sigma(s_i / \tau) \\
M_{\mathrm{hard}} &= \mathrm{TopK}(M_{\mathrm{soft}}, k) \\
M_{\mathrm{STE}} &= M_{\mathrm{hard}} + M_{\mathrm{soft}} - \mathrm{sg}(M_{\mathrm{soft}})
\end{aligned}
\tag{2}
$$

The Gumbel noise with annealed temperature τ (Gumbel-Softmax relaxation [Jang et al., 2017]) prevents premature convergence of the selector to a deterministic top-k. The Straight-Through Estimator [Bengio et al., 2013] maintains gradient flow through the discrete binarization: during the forward pass, M_hard is used; during the backward pass, gradients flow through M_soft.

**Stage 3 (O(E)): Subgraph extraction.** A single vectorized `pyg_subgraph` call extracts the induced subgraph S on the selected k nodes: S = G[{i : M_hard_i = 1}].

**Stage 4 (O(k²)): DeepSTGCN.** Two-layer temporal convolution followed by two-layer GCN [Kipf and Welling, 2017], applied exclusively on the subgraph S:

```
H_temp = Conv1d_2layer(X_S)              # temporal encoding on S
H_gcn  = GCNConv(GCNConv(H_temp, A_S), A_S)  # spatial encoding on S
```

where A_S is the adjacency matrix restricted to the k selected nodes. The computational cost is O(k²) rather than O(N²).

**Stage 5 (O(k)): SubgraphScorer with residual blend.**

$$
\ell_{\mathrm{final}} = (1-\alpha) \cdot \mathrm{MLP_{score}}(H_{\mathrm{gcn}}) + \alpha \cdot s_{\mathrm{selected}}
\tag{3}
$$

The learned scalar α ∈ [0,1] in Eq. (3) controls the blend: the deep analyzer can refine the selector's initial estimates, but cannot contradict them (the residual constraint). Unselected nodes receive logit = −∞.

**Variational GIB regularization.** Rather than an L₁ penalty on M_soft, I use an asymmetric Variational GIB loss that encodes the dynamical system assumption directly:

- For healthy windows (U=0): push M_soft → 0 (no root cause active). Loss: L_healthy = E[M_soft²].
- For anomalous windows (||U||₀=1): encourage a concentrated, sparse M_soft. Loss: L_anomalous = −Entropy(M_soft) + ||mean(M_soft) − 1/N||².

This asymmetry enforces that the selector produces near-zero masks for healthy observations (reducing false positive rates) while encouraging one-hot concentration for fault observations (improving Avg@1 over L₁ regularization).

**Total training loss.**

$$
\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \lambda_{\mathrm{nce}} \mathcal{L}_{\mathrm{InfoNCE}} + \lambda_{\mathrm{gib}} \mathcal{L}_{\mathrm{GIB}} + \lambda_{\mathrm{cascade}} \mathcal{L}_{\mathrm{cascade}}
\tag{4}
$$

where L_CE is cross-entropy over the root cause label, L_InfoNCE is a contrastive loss pulling healthy graph embeddings together while pushing anomalous ones apart, and L_cascade is the cascade-aware structural regularizer (Section 5.6).

**Complexity.** For a graph of N nodes and k selected nodes (k ≪ N), the per-forward-pass cost is O(N + k²). At N=100 and k=15 (15% of nodes), the deep spatial stage processes 225 pairs rather than 10,000.

### 5.4 GATv2: Late Bottleneck with Contextual Attention

GATv2 processes the full graph in two GATv2 spatial layers before deriving node importance from attention weights.

**Architecture.**

```
H_temp  = LightEncoder(X)                 # [N, H] — shared stage
H_gat1  = GATv2Conv(H_temp, edge_index, heads=4)   # [N, 4H]
H_gat2  = GATv2Conv(H_gat1, edge_index, heads=1)   # [N, H]
imp_i   = Σ_{j: (j,i)∈E} α_{ji}           # node importance = incoming attention sum
logits  = MLP_score(H_gat2) + β · imp_i   # [N]
```

GATv2 [Brody et al., 2022] computes attention coefficients as:

```
e_{ij} = a^T · LeakyReLU(W · [h_i || h_j])
α_{ij} = softmax_j(e_{ij})
```

where [||] denotes concatenation. Unlike GAT v1 where the attention depends only on h_i after applying the linear transform, GATv2 applies the linear transform to the concatenation of source and target features before the nonlinearity, yielding strictly more expressive attention patterns.

The node importance signal (sum of incoming attention weights) provides a soft ranking without enforcing hard sparsity. This makes GATv2 better suited to cascade scenarios where multiple services along the propagation chain are jointly anomalous: STA's binary top-k selector may exclude the true root cause if its local score is masked by a high-scoring downstream symptom node.

### 5.5 Multimodal Variants

**STA-MM.** Three modality-specific encoders (MetricEncoder: Conv1d; LogEncoder: MLP + LayerNorm; TraceEncoder: MLP + LayerNorm) each produce an independent selection mask:

```
M^(m) = Selector_m(Encoder_m(X_m))   ∀ m ∈ {metrics, logs, traces}
```

A cross-modal attention module fuses the three masks:

```
M_fused = CrossAttn([M^(m); M^(l); M^(t)])
```

The fused mask feeds into the same STESelector + DeepSTGCN + SubgraphScorer pipeline as the unimodal STA. If a modality is absent (e.g., traces unavailable), its encoder produces M^(m) = uniform (1/N per node), contributing zero net signal to the fusion.

**GATv2-MM.** A late fusion approach: encode all three modalities separately, concatenate, project, then apply GATv2:

```
H_fused = FusionMLP([H_metrics || H_logs || H_traces])
H_graph = GATv2(H_fused, edge_index)
logits  = NodeScorer(H_graph)
```

Zero-padding is applied if any modality tensor is missing.

### 5.6 Pipeline Components

**Cascade-Aware Loss.** I implement three variants of a structural regularizer that enforces distance-proportional score gaps between the root cause and downstream cascade nodes.

The *structural* variant (CascadeAwareLoss) applies hinge constraints on graph hop distance:

```
L_cascade = E[ReLU(s_{1-hop} − s_{RC} + margin₁)]
           + E[ReLU(s_{2-hop} − s_{1-hop} + margin₂)]
```

with margin₁ = 0.5 and margin₂ = 0.3. The *causal-weighted* variant (CausalCascadeAwareLoss) scales margins by edge weights from the causal graph builder, encoding the intuition that strongly causal edges should impose larger score gaps. The *learnable* variant (LearnedPropagationCascadeLoss) trains a full N×N propagation matrix B initialized from causal graph weights, jointly with the main model.

**ADWIN Drift Detector.** I adapt the ADWIN algorithm [Bifet and Gavaldà, 2007] to monitor all F=6 Prometheus metrics simultaneously per service. Drift is declared when more than 50% of metrics simultaneously trigger an ADWIN change. Upon drift detection, a cooldown window of 30 timesteps suppresses RCA output to prevent false positives during Kubernetes rolling updates, autoscaling events, and cold-start periods.

**Gray Failure Detector.** I implement four orthogonal detection mechanisms, triggered in priority order:

1. **P99/P50 micro-stutter**: ratio p99/p50 > 3.0, indicating tail latency amplification from CPU steal, L3 cache contention, or GC pauses.
2. **Inter-metric decorrelation**: mean pairwise Pearson correlation drop > 0.4 relative to a rolling baseline, signaling joint behavior degradation invisible to per-metric alerts.
3. **Mann-Kendall trend** [Mann, 1945]: monotonic trend statistic |S/S_max| > 0.6, targeting nascent metastable degraded states and memory leaks.
4. **Limplock signal**: ratio queue_depth_rate / cpu_usage > 5.0, detecting saturation under apparent idleness.

**Causal Graph Builder.** I implement a cascade of causal discovery algorithms with automatic fallback: PC algorithm [Spirtes et al., 2000] (Fisher's z-test, α=0.05) → FCI (for latent confounders) → Granger causality [Granger, 1969] (lags 1–3) → structural Barabási-Albert edges as default. The final graph is the weighted union of discovered causal edges (weight 2.0) and structural edges (weight 1.0).

---

## 6. Experimental Setup

### 6.1 Model Configurations

| Model | Bottleneck | Graph Module | Modalities | Parameters |
|---|---|---|---|---|
| STA | Early (top-k) | GCNConv × 2 | Metrics | 15,494 |
| GATv2 | Late (soft attn) | GATv2Conv × 2 | Metrics | 9,794 |
| Late STGCN+GIB | Late (hard GIB) | GCNConv × 2 | Metrics | 8,579 |
| Full-Graph GCN | None | GCNConv × 2 | Metrics | 7,490 |
| STA-MM | Early (top-k) | GCNConv × 2 | Metrics + Logs + Traces | ~22,000 |
| GATv2-MM | Late (soft attn) | GATv2Conv × 2 | Metrics + Logs + Traces | ~18,000 |

### 6.2 Evaluation Metrics (Formal Definitions)

For a test set of M fault scenarios, let rank_m be the position of the true root cause in the predicted ranking for scenario m:

```
Avg@k = (1/M) Σ_m 1[rank_m ≤ k]
MRR   = (1/M) Σ_m 1/rank_m
NDCG@k = (1/M) Σ_m DCG@k(m) / IDCG@k
```

where DCG@k(m) = Σ_{i=1}^k rel_i / log₂(i+1) with rel_i = 1 if position i is the true RC, else 0.

### 6.3 Reference Baselines

| Method | Avg@1 | Avg@3 | Avg@5 | MRR | Source |
|---|---:|---:|---:|---:|---|
| BARO | 0.71 | 0.82 | 0.86 | 0.77 | RCAEval reference numbers |
| E-Diagnosis | 0.64 | 0.77 | 0.83 | 0.72 | RCAEval reference numbers |
| CausaRank | 0.52 | 0.70 | 0.79 | 0.63 | RCAEval reference numbers |
| MicroRCA | 0.49 | 0.68 | 0.77 | 0.60 | RCAEval reference numbers |

*Note: Baseline figures are from published papers; their training protocol (split strategy, seeds, max_cases) is not fully specified. Comparisons in Section 7.2 are indicative only.*

---

## 7. Results

### 7.1 Synthetic Benchmark: Scalability Analysis

Table 1 reports performance across graph sizes N ∈ {20, 50, 100} on Barabási-Albert graphs (5 seeds).

**Table 1. Synthetic benchmark (BA graphs). Avg@1 ± std, 5 seeds.**

| Model | N=20 | N=50 | N=100 |
|---|---:|---:|---:|
| STA | 0.805 ± 0.122 | **0.589** ± 0.049 | **0.552** ± 0.046 |
| GATv2 | **0.828** ± 0.039 | 0.571 ± 0.023 | 0.490 ± 0.029 |
| STGCN+GIB | 0.823 ± 0.047 | 0.469 ± 0.068 | 0.471 ± 0.097 |
| Full-Graph GCN | 0.830 ± 0.029 | 0.526 ± 0.026 | 0.466 ± 0.042 |
| Random Forest | 0.685 ± 0.027 | 0.312 ± 0.016 | 0.065 ± 0.009 |
| MLP | 0.084 ± 0.024 | 0.029 ± 0.005 | 0.011 ± 0.004 |

**Table 2. Synthetic benchmark. Full metrics at N=100, 5 seeds.**

| Model | Avg@1 | Avg@3 | Avg@5 | MRR | NDCG@3 |
|---|---:|---:|---:|---:|---:|
| STA (k=15) | **0.552** ± 0.046 | **0.606** ± 0.054 | **0.639** ± 0.051 | **0.600** ± 0.048 | **0.584** ± 0.051 |
| GATv2 | 0.490 ± 0.029 | 0.543 ± 0.036 | 0.570 ± 0.039 | 0.538 ± 0.033 | 0.522 ± 0.033 |
| STGCN+GIB | 0.471 ± 0.097 | 0.557 ± 0.105 | 0.587 ± 0.093 | 0.532 ± 0.094 | 0.521 ± 0.101 |
| Full-Graph GCN | 0.466 ± 0.042 | 0.537 ± 0.039 | 0.567 ± 0.037 | 0.521 ± 0.038 | 0.508 ± 0.041 |

**Analysis.** At N=20, GATv2 and Full-Graph GCN marginally outperform STA (0.828–0.830 vs. 0.805). At this scale, k=5 ≈ N/4, and the selected subgraph captures most of the graph's connectivity; the selection bottleneck provides little benefit while introducing variance from the Gumbel selection noise. From N=50, STA's efficiency advantage materializes: STA achieves 0.589 vs. GATv2's 0.571 (+1.8 pp). At N=100, the gap widens to +6.3 pp (0.552 vs. 0.490). Random Forest collapses at N=100 (0.065), confirming that non-structural approaches do not generalize to large heterogeneous graphs.

The high variance of STGCN+GIB at N=100 (±0.097) reflects optimization instability when the hard GIB mask must identify a single root cause among 100 nodes—exactly the regime where STA's softer early selection is more stable.

### 7.2 Real-World Benchmark: RCAEval

**Table 3. RCAEval benchmark, metrics-only. Avg@1 ± std (5 seeds, 100 epochs).**

| Dataset | STA | GATv2 | BARO† | E-Diag.† | CausaRank† | MicroRCA† |
|---|---:|---:|---:|---:|---:|---:|
| trainticket | **0.960** ± 0.037 | 0.950 ± 0.045 | 0.71 | 0.64 | 0.52 | 0.49 |
| sockshop | **0.850** ± 0.089 | 0.810 ± 0.080 | — | — | — | — |
| online-boutique | 0.650 ± 0.095 | **0.680** ± 0.051 | — | — | — | — |

† Published figures; protocol not fully isoperimeter. See Section 4.3.

**Table 4. RCAEval benchmark, full metrics (5 seeds, 100 epochs).**

| Dataset | Model | Avg@1 | Avg@3 | Avg@5 | MRR | NDCG@3 |
|---|---|---:|---:|---:|---:|---:|
| trainticket | STA | **0.960** ± 0.037 | **1.000** | **1.000** | **0.980** ± 0.019 | **0.985** ± 0.014 |
| trainticket | GATv2 | 0.950 ± 0.045 | **1.000** | **1.000** | 0.975 ± 0.022 | 0.982 ± 0.017 |
| sockshop | STA | **0.850** ± 0.089 | **0.960** ± 0.058 | 1.000 | **0.908** ± 0.058 | 0.915 ± 0.065 |
| sockshop | GATv2 | 0.810 ± 0.080 | 0.910 ± 0.086 | 1.000 | 0.876 ± 0.058 | 0.877 ± 0.085 |
| online-boutique | STA | 0.650 ± 0.095 | **0.890** ± 0.020 | 1.000 | 0.779 ± 0.054 | 0.789 ± 0.060 |
| online-boutique | GATv2 | **0.680** ± 0.051 | 0.840 ± 0.080 | 1.000 | **0.788** ± 0.042 | **0.792** ± 0.057 |

**Analysis.** At N=5 (the scale of all three RCAEval datasets), STA and GATv2 produce results that are **within noise** of one another:
- trainticket: STA 0.960 vs. GATv2 0.950 — gap of 1.0 pp, well below either std (±0.037, ±0.045);
- sockshop: STA 0.850 vs. GATv2 0.810 — gap of 4.0 pp, below both stds (±0.089, ±0.080);
- online-boutique: GATv2 0.680 vs. STA 0.650 — gap of 3.0 pp, below STA's std of ±0.095. The variance on online-boutique is large for both models, which limits any strong ordering claim on this dataset.

I therefore do **not** claim a winner among STA and GATv2 at N=5. What the results do show is that both architectures are competitive at this scale: the absolute Avg@1 levels (0.65–0.96) are substantially above what non-graph baselines achieve at small N.

This is consistent with the hypothesis that at N=5, the selection bottleneck offers no complexity benefit (k ≈ N in practice), and the relative utility of spatial structure is dataset-dependent. Online-boutique's service graph has more diffuse fault propagation patterns—GATv2's contextual multi-head attention may be better suited there, but the noise envelope is too wide to assert this conclusively.

Compared to the RCAEval reference numbers (indicative, **not isoperimeter**—see Section 4.3), both STA and GATv2 report higher Avg@1 than BARO's 0.71 on trainticket and sockshop. On online-boutique, the gap to BARO is much smaller and likely lies inside the combined uncertainty of both protocols. I refrain from claiming SOTA: with 5 seeds and a non-aligned baseline protocol, the most I can defend is that my models are **at the level of the published baselines** on these datasets.

### 7.3 Multimodal Evaluation

**Table 5. RCAEval with multimodal fusion (MM-strict). Avg@1 ± std (5 seeds, 100 epochs).**

| Dataset | STA (metrics) | STA-MM | Δ STA | GATv2 (metrics) | GATv2-MM | Δ GATv2 |
|---|---:|---:|---:|---:|---:|---:|
| trainticket | 0.960 | 0.910 ± 0.020 | **−5.0%** | 0.950 | 0.880 ± 0.051 | **−7.0%** |
| online-boutique | 0.650 | 0.650 ± 0.155 | 0% | 0.680 | **0.790** ± 0.086 | **+11.0%** |
| sockshop | 0.850 | — (skipped) | — | 0.810 | — (skipped) | — |

*Sockshop MM excluded: trace coverage insufficient for strict multimodal evaluation on this dataset.*

**Finding 1: MM degrades on trainticket.** Both STA-MM (−5%) and GATv2-MM (−7%) underperform their unimodal counterparts on trainticket, despite complete log and trace coverage (logs_ratio = traces_ratio = 1.0 in both datasets). The degradation is not attributable to missing modalities.

**Finding 2: GATv2-MM gains on online-boutique (+11%).** This is the only positive MM result across RCAEval. The late-fusion architecture of GATv2-MM appears more tolerant to low-signal auxiliary modalities: when logs/traces carry little discriminative information, the FusionMLP can implicitly down-weight these modalities, whereas STA-MM's cross-modal attention can still produce gradient interference.

**Finding 3: STA-MM on online-boutique shows high variance (±0.155 vs. STA ±0.095).** This 1.6× variance inflation is a direct signature of multimodal optimization instability—the additional gradient paths create inconsistent convergence across seeds.

### 7.4 ObservIT Cluster Evaluation

**Table 6. ObservIT live Kubernetes cluster. 5 seeds, 100 epochs.**

| Model | Avg@1 ± std | Avg@3 ± std | Avg@5 ± std | MRR ± std |
|---|---:|---:|---:|---:|
| STA | **0.796** ± 0.043 | 0.921 ± 0.075 | 0.942 ± 0.062 | **0.868** ± 0.033 |
| GATv2 | 0.762 ± 0.034 | **0.978** ± 0.010 | **0.988** ± 0.006 | 0.860 ± 0.031 |
| STA-MM | 0.772 ± 0.021 | **0.982** ± 0.007 | **0.992** ± 0.003 | 0.878 ± 0.010 |
| GATv2-MM | 0.762 ± 0.016 | 0.937 ± 0.083 | 0.950 ± 0.086 | 0.846 ± 0.035 |

**Analysis.** On the cluster, STA achieves the best Avg@1 (0.796 vs. GATv2's 0.762), but GATv2 achieves dramatically better Avg@3 (0.978 vs. STA's 0.921). This pattern suggests that while STA correctly identifies the root cause in 79.6% of immediate top-1 predictions, GATv2 almost never misses the true RC within the top 3—a practically important property for human-assisted triage.

Critically, **STA-MM shows no regression on the cluster** (Avg@1 = 0.772, −0.024 vs. unimodal STA). The variance is also lower (±0.021 vs. ±0.043 for STA). This contrasts sharply with the RCAEval degradation results, suggesting that cluster-collected observability data contains richer log/trace signals than the public datasets. The ObservIT cluster result is the first positive evidence for multimodal fusion in my evaluation suite.

### 7.5 Ablation Studies

#### 7.5.1 Component Ablation (Simulated Data)

**Table 7. Component ablation, Late Bottleneck family, simulated data. Avg@1 ± std (3 seeds).**

Note: ablation runs use a distinct experimental configuration (3 seeds, fixed split, shorter window); values are not directly comparable to Table 1 but appropriate for relative analysis.

| Variant | N=20 | N=100 |
|---|---:|---:|
| A1: Full STGCN+GIB | 0.949 ± 0.017 | 0.939 ± 0.015 |
| A2: STGCN, no GIB (L₁ replaced) | **0.969** ± 0.012 | 0.933 ± 0.017 |
| A3: Spatial-only (no temporal) | 0.584 ± 0.044 | 0.478 ± 0.008 |
| A4: Temporal-only (no GCN) | **0.992** ± 0.000 | **0.956** ± 0.040 |

The dominance of temporal-only (A4) at N=100 in simulated data is striking: the pure temporal encoder, without any graph convolution, achieves 0.956 vs. 0.933–0.939 for the spatial models. At N=100 on BA graphs, the random scale-free topology provides weaker structural signal than on real microservice graphs with known dependency patterns.

Critically, A3 (spatial-only) drops to 0.478 at N=100—confirming that spatial information alone is insufficient and the temporal encoder is the load-bearing component.

#### 7.5.2 Graph vs. Temporal on Real Data (Ablation T5)

**Table 8. STA vs. temporal-only on real RCAEval datasets. Avg@1 ± std (5 seeds).**

| Dataset | STA | Temporal-only | Δ (STA − Temporal) |
|---|---:|---:|---:|
| trainticket | **0.960** ± 0.037 | 0.940 ± 0.049 | +0.020 |
| sockshop | 0.850 ± 0.089 | **0.860** ± 0.089 | −0.010 |
| online-boutique | 0.650 ± 0.095 | **0.690** ± 0.098 | −0.040 |

The graph provides a net positive contribution only on trainticket (+2%). On sockshop and online-boutique, removing the GCN layer and keeping only the temporal encoder matches or improves performance. This result is consistent with the small-N hypothesis: at N=5, k ≈ N and the DeepSTGCN operates on essentially the full graph, providing no computational advantage and adding noise through over-parameterization.

#### 7.5.3 STA k-Sensitivity (N=100, Simulated)

**Table 9. STA k-sensitivity at N=100 (3 seeds, simulated data).**

| k (% of nodes) | k value | Avg@1 ± std |
|---|---|---:|
| 10% | 10 | 0.924 ± 0.036 |
| 15% | 15 | 0.931 ± 0.042 |
| 20% | 20 | 0.940 ± 0.025 |
| 30% | 30 | 0.949 ± 0.013 |

Performance increases monotonically with k, as expected: a larger subgraph is less likely to exclude the true root cause. The trade-off is between recall (larger k = better coverage) and efficiency (smaller k = faster inference). At k=10% (k=10 nodes), Avg@1 = 0.924 — reasonable coverage at the lowest computational cost. In my main benchmark, I use k=15 (15%) which balances performance and efficiency.

#### 7.5.4 Training Protocol Ablation (STA)

**Table 10. STA training protocol variants, N=100 (3 seeds, simulated).**

| Variant | Avg@1 ± std |
|---|---:|
| D1: Full (warmup + k-annealing) | 0.924 ± 0.036 |
| D2: No contrastive warmup | **0.947** ± 0.031 |
| D3: No k-annealing | **0.949** ± 0.003 |

Counter-intuitively, removing the contrastive warmup (D2) and k-annealing (D3) improves simulated performance. This result motivated the `--sta-contrastive-warmup 0` guardrail for all real-data experiments: the warmup phases appear to over-regularize on simulated data and destabilize on real data.

---

## 8. Discussion

### 8.1 When Does STA's Efficiency Advantage Materialize?

The evidence is consistent: STA's O(N + k²) advantage is measurable from N=50 and substantial at N=100 (+6.3 pp Avg@1 over GATv2). Below N=20, the selection bottleneck provides no benefit—k approaches N, the subgraph extraction is effectively a no-op, and the additional Gumbel noise of the STESelector introduces variance.

For practical deployment: systems with fewer than 20 instrumented services should prefer GATv2 for its simplicity and lower hyperparameter sensitivity. Systems with 50+ services should strongly prefer STA—both for performance and for inference latency predictability (STA's per-query cost is bounded by O(k²), independent of N growth).

### 8.2 Why Does the Graph Help at Large N but Marginally at Small N?

At N=5 (RCAEval), the graph has 4–6 edges at most. The dependency structure is shallow (maximum 2–3 hops between any pair of services). In this regime, the temporal signal alone—detecting the service whose metrics deviate earliest and most strongly—is nearly sufficient for root cause identification.

At N=100 (BA graphs), the graph has approximately N × m = 200 edges, and propagation paths can span 5+ hops. Identifying the originating fault node from downstream symptom signals requires multi-hop graph reasoning that temporal encoders cannot perform.

This finding has an important practical implication: teams should not assume graph-based RCA to be universally superior. For small, well-monitored systems, a temporal-only encoder with proper normalization may be the most cost-effective baseline.

### 8.3 The Negative Multimodal Result

On RCAEval, multimodal fusion consistently fails to improve over metrics-only performance. I propose three non-exclusive explanations:

**Hypothesis 1: Low discriminative signal in RCAEval logs/traces.** The public RCAEval datasets were collected in controlled experimental environments with scripted fault injection. Log messages and trace spans in such environments may be stereotyped—the same log entries appear in both healthy and faulty states—leaving metrics as the sole source of discriminative information.

**Hypothesis 2: Gradient conflict in cross-modal fusion.** When one modality (metrics) carries the signal and two others (logs, traces) carry noise, the cross-modal attention mechanism in STA-MM must learn to down-weight two noisy modalities while amplifying one. This is a challenging optimization problem that may not converge reliably with 5-seed, 100-epoch training budgets, manifesting as the variance inflation observed in STA-MM on online-boutique (±0.155 vs. ±0.095 for STA).

**Hypothesis 3: Inductive bias mismatch.** STA-MM forces each modality to independently propose a root cause node and then fuses these proposals. For metrics, the proposal is meaningful (the temporally anomalous node). For logs and traces with low discriminative content, the proposal is near-uniform—contributing noise to the fusion without signal. The per-modality selector design, while theoretically robust to missing modalities, does not handle the case where a modality is present but uninformative.

The positive cluster result (STA-MM no regression, lower variance) points toward the data quality hypothesis: observability data from a live Kubernetes cluster, collected with Loki and Jaeger under production-like load, contains richer temporal and structural log/trace patterns than the scripted RCAEval collection environment.

### 8.4 Gray Failures: The Hard Case

The catalog analysis (network-loss-catalogue-to-catalogue-db, 15% packet loss) represents the hardest test for any RCA system: the individual metric deviation at catalogue-db is sub-threshold. Network receive rate drops by approximately 15%, which may be within the normal variance band on a busy link.

My gray failure detector's inter-metric decorrelation mechanism targets exactly this pattern: while catalogue-db's individual metrics stay near-nominal, the correlation between its network_rx and the catalogue service's network_tx breaks down. This decorrelation signal, boosted before entering the GNN, can elevate catalogue-db into STA's top-k or increase its attention weight in GATv2.

I have not yet produced per-scenario evaluation results on the ObservIT cluster (ongoing data collection). The results in Table 6 aggregate across all fault types. A per-scenario breakdown is the primary subject of future work.

### 8.5 Limitations

I list the limitations explicitly so that the contributions can be judged for what they are, not for what they are not.

**Scale of real-data evaluation — the central limitation.** All publicly available RCAEval datasets operate at N=5. My private ObservIT cluster reaches N=14. Neither tests the performance regime (N≥50) where STA's design choices are most impactful. **The asymptotic claim—STA outperforms full-graph models at large N—is empirically validated only on synthetic Barabási-Albert graphs.** The extrapolation from synthetic BA at N=100 to real microservice systems at the same scale is an inference, not a measurement. I believe the extrapolation is reasonable for two reasons: (i) the BA preferential-attachment model is the standard stylized abstraction of scale-free service graphs and reproduces their hub-and-spoke degree distribution; (ii) the complexity benefit of STA is structural (O(N + k²) vs. O(N²) in graph operations) and does not depend on dataset semantics. But I cannot demonstrate this empirically without access to a real microservice deployment at N≥50—future work.

**Statistical rigor.** Five seeds with varying splits produce standard deviations that conflate model variance with data split variance. With n=5, bootstrap confidence intervals are not statistically reliable, and formal significance testing (Wilcoxon signed-rank, paired t-test, Bonferroni correction) is not appropriate at this sample size. I adopt the conservative reading throughout Section 7: a result is **meaningful** only when the gap exceeds both reported standard deviations, otherwise **within noise**. Under this rule, several of my headline comparisons at small N are within noise (notably STA vs. GATv2 at N=5), and I say so. On online-boutique specifically, the variances are large (STA ±0.095, STA-MM ±0.155) and any ordering claim on this dataset should be treated cautiously.

**Baseline comparison is not isoperimeter.** Sections 4.3 and 7.2 develop this point at length. In short: I report higher Avg@1 than BARO/E-Diagnosis on trainticket and sockshop, but their training protocol is not specified well enough to guarantee that the comparison is on equal footing. A rigorous SOTA claim would require re-implementing each baseline under my protocol. I did not do this; my contribution lies in the relative comparison of my four model variants and in the negative MM result, both of which are protocol-internal.

**Multimodal baseline absence.** No published RCA baseline that I could verify reports multimodal results with a comparable protocol. My MM evaluation is therefore relative (unimodal vs. MM for my own models) but cannot be situated in the broader landscape of multimodal approaches.

**Gray failure evaluation.** The per-scenario breakdown of gray failure detection performance is not yet available for my live cluster experiments. The four-mechanism detector is implemented and unit-tested, but its end-to-end uplift on RCA Avg@1 under specifically gray scenarios remains to be measured.

---

## 9. Conclusion

I presented STA (Select-Then-Analyze), an early-bottleneck GNN for microservice root cause analysis that achieves O(N + k²) complexity by selecting a compact candidate subgraph before deep spatio-temporal analysis. My systematic evaluation across synthetic graphs, three RCAEval datasets, and a live Kubernetes cluster yields three durable findings:

**Finding 1.** STA's asymptotic efficiency advantage is empirically confirmed at N≥50 **on synthetic Barabási-Albert graphs**: +6.3 pp Avg@1 over GATv2 at N=100, with a per-query inference cost bounded by O(k²) independent of N. At N≤14—the scale of all my real-data evaluations, public and private—STA and GATv2 are within noise of one another. I cannot yet demonstrate STA's large-N advantage on a real microservice system; this remains the main open empirical question of this work. Practically, teams considering deployment at scale should treat my synthetic result as a structural argument for STA, complemented by a head-to-head measurement on their own topology.

**Finding 2.** Multimodal fusion (metrics + logs + traces) does not improve—and often degrades—performance on RCAEval datasets. This negative result is robust across models and datasets, and is consistent with the hypothesis that log/trace quality in controlled experimental environments is insufficient to provide discriminative signal beyond metrics. On a live cluster with richer observability, STA-MM shows no regression and reduced variance, suggesting that data quality is the rate-limiting factor.

**Finding 3.** I contribute an operational RCA pipeline that goes beyond model evaluation: ADWIN-based concept drift detection with multi-metric cooldown, a four-mechanism gray failure composite detector targeting micro-stutters, decorrelation, memory leaks, and Limplock patterns, and three variants of a cascade-aware loss function. These components address the gap between academic benchmarks and production deployment needs.

Future work will focus on: (i) per-scenario evaluation of gray failure detection on the ObservIT cluster, (ii) scaling evaluation to N>14 on real microservice topologies, and (iii) investigating modality-specific signal quality metrics to predict when multimodal fusion will help before committing to the additional training cost.

---

## Acknowledgments

This work was conducted during an industrial research internship at **Devoteam**, in partial fulfillment of the engineering degree at **EPITA**. I thank my company supervisor at Devoteam for access to the ObservIT Kubernetes cluster and guidance on the chaos engineering scenarios, and my academic advisor at EPITA for guidance on the experimental protocol and the comparability matrix design.

---

## Data and Code Availability

**Code.** The complete pipeline, including all four model implementations, training scripts, benchmark runners, chaos manifests, and the full evaluation protocol, is openly available at:

> https://github.com/Wassbdr/matrix_simple

The exact commit corresponding to the results reported in this paper is recorded in `results_transmission/canonical_results_manifest.json`.

**Public datasets.** Results in Tables 3–5 use the RCAEval benchmark, publicly available at https://github.com/phamquiluan/RCAEval. I use the trainticket, sockshop, and online-boutique subsets without modification.

**Private dataset.** The ObservIT cluster results (Table 6) were collected on a private Kubernetes cluster operated for this study. The collection scripts, chaos manifests (`k8s/chaos/sockshop/`), Locust workload (`rca_pipeline/locustfile_sockshop.py`), and topology definitions (`rca_pipeline/topologies.py`) are released; the raw cluster traces are not redistributed.

**Result artifacts.** All numerical results in this paper are reproducible from the JSON artifacts in `results_transmission/`, each verified by SHA-256 in `paper_benchmark.sha256`. The manifest format and verification commands are documented in `docs/COMPARABILITY_MATRIX.md`.

---

## Reproducibility

Each results table maps to a single, deterministic command (5 seeds, fixed splits):

| Table | Artifact | Command |
|---|---|---|
| 1, 2 (synthetic) | `paper_benchmark_sim.json` | `python -m rca_pipeline.benchmark --output-json <path>` |
| 3, 4 (RCAEval metrics) | `paper_benchmark_real.json` | `python -m rca_pipeline.benchmark_real --dataset all --seeds 42,123,7,1337,2025 --epochs 100 --sta-contrastive-warmup 0 --output-json <path>` |
| 5 (RCAEval MM) | `paper_benchmark_real_mm.json` | append `--use-mm --mm-strict` to the above |
| 6 (ObservIT) | `paper_benchmark_observit.json`, `paper_benchmark_observit_mm.json` | `python -m rca_pipeline.benchmark_real --dataset observit --observit-topology sock_shop ...` |
| 7–10 (ablations) | `rca_pipeline/ablation_prepublish.json`, `validation_runs/ablation_temporal_real.json` | `python -m rca_pipeline.benchmark_ablation` |

Reproducing the full benchmark suite from scratch takes approximately 4 hours on a single CPU node (no GPU required for the small-N regime).

---

## References

Bengio, Y., Léonard, N., and Courville, A. (2013). *Estimating or propagating gradients through stochastic neurons for conditional computation*. arXiv preprint arXiv:1308.3432.

Bifet, A., and Gavaldà, R. (2007). *Learning from time-changing data with adaptive windowing*. In Proceedings of the 2007 SIAM International Conference on Data Mining (SDM 2007).

Brody, S., Alon, U., and Yahav, E. (2022). *How attentive are graph attention networks?* In International Conference on Learning Representations (ICLR 2022).

Chandola, V., Banerjee, A., and Kumar, V. (2009). *Anomaly detection: A survey*. ACM Computing Surveys, 41(3), 1–58.

Granger, C. W. J. (1969). *Investigating causal relations by econometric models and cross-spectral methods*. Econometrica, 37(3), 424–438.

Huang, P., Guo, C., Lorch, J. R., Zhou, L., and Dang, Y. (2017). *Gray failure: The Achilles' heel of cloud-scale systems*. In Proceedings of the 16th Workshop on Hot Topics in Operating Systems (HotOS 2017).

Jang, E., Gu, S., and Poole, B. (2017). *Categorical reparameterization with Gumbel-Softmax*. In International Conference on Learning Representations (ICLR 2017).

Kipf, T. N., and Welling, M. (2017). *Semi-supervised classification with graph convolutional networks*. In International Conference on Learning Representations (ICLR 2017).

Li, Q., Han, Z., and Wu, X.-M. (2018). *Deeper insights into graph convolutional networks for semi-supervised classification*. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2018).

Mann, H. B. (1945). *Nonparametric tests against trend*. Econometrica, 13(3), 245–259.

Spirtes, P., Glymour, C., and Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press.

Tishby, N., and Zaslavsky, N. (2015). *Deep learning and the information bottleneck principle*. In IEEE Information Theory Workshop (ITW 2015).

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., and Bengio, Y. (2018). *Graph attention networks*. In International Conference on Learning Representations (ICLR 2018).

Wu, T., Ren, H., Li, P., and Leskovec, J. (2020). *Graph information bottleneck*. In Advances in Neural Information Processing Systems (NeurIPS 2020).

Yu, B., Yin, H., and Zhu, Z. (2018). *Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting*. In Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI 2018).

Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., and Finn, C. (2020). *Gradient surgery for multi-task learning*. In Advances in Neural Information Processing Systems (NeurIPS 2020).

Zheng, X., Aragam, B., Ravikumar, P., and Xing, E. P. (2018). *DAGs with NO TEARS: Continuous optimization for structure learning*. In Advances in Neural Information Processing Systems (NeurIPS 2018).

### Software and tools

Chaos Mesh: https://chaos-mesh.org/ — fault injection framework for Kubernetes.

RCAEval benchmark: https://github.com/phamquiluan/RCAEval — RCA evaluation suite.

PyTorch Geometric (Fey & Lenssen, 2019): https://pytorch-geometric.readthedocs.io/ — GNN library.
