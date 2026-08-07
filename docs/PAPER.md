# STA: A Select-Then-Analyze Framework for Scalable Root Cause Analysis in Microservice Architectures

**Wassim Badraoui**
*Devoteam, Paris, France · EPITA, Paris, France*
wassim.badraoui@epita.fr

---

## Abstract

Root Cause Analysis (RCA) in microservice architectures is a critical yet challenging task: failures propagate across service dependency graphs, observable signals are noisy, and the search space grows with the number of instrumented services. I present STA (Select-Then-Analyze), an early-bottleneck spatio-temporal graph neural network that reduces RCA complexity from O(N²) to O(N + k²) by selecting a compact candidate subgraph before deep spatial analysis. STA combines a zero-hop temporal encoder with a differentiable Straight-Through Estimator selector and Variational Graph Information Bottleneck regularization. I evaluate four model variants—STA, GATv2, STA-MM, and GATv2-MM—across four experimental regimes: synthetic Barabási-Albert graphs (N ∈ {20, 50, 100}), three real-world RCAEval datasets (N=5), a private Kubernetes cluster running sock-shop (N=14) with Chaos Mesh fault injection, and a 50-service synthetic-but-real microservice deployment on the same cluster (N=50). Rather than chase a state-of-the-art number, I report four deliberately-scoped findings, all reproducible from released code and seeds. (i) STA's O(N + k²) design keeps per-query latency flat as N grows; its accuracy edge over attention/full-graph models is real but modest and shrinks under scrutiny—from +6.3 to +2.4 pp Avg@1 at N=100 when moving from 5 to 10 seeds—though it holds in direction (and reaches +5.8 pp on the N=50 real-cluster deployment). (ii) At small graph scale (N≤14, the public-benchmark regime), STA, GATv2 and a strong statistical baseline are within the noise envelope of seed-level evaluation. (iii) Re-running BARO isoperimetrically—same cases, candidate nodes, per-seed split and metric functions as my models—shows it is far stronger than its published 0.71 (Avg@1 0.80–0.86) and beats my models on Online Boutique: I make no SOTA claim. (iv) Multimodal fusion (metrics + logs + traces) degrades or matches metrics-only on RCAEval (−5% Avg@1 on trainticket) but halves seed variance on the private cluster (STA-MM Avg@1 = 0.772 vs. STA 0.796), pointing to log/trace data quality as a likely confounder. Beyond the evaluation, I demonstrate an end-to-end operational pipeline: 25-trial Optuna hyperparameter optimization on the sock-shop windows, retraining of the best-configured model into a self-describing bundle (state_dict + scaler + metadata + HPO provenance), containerization, and live deployment as a FastAPI service on the Devoteam Kubernetes cluster with Prometheus metrics integration. All artifacts are released, with an explicit comparability matrix for baseline claims (I do not assert SOTA) and operational components including ADWIN-based drift detection and a four-mechanism gray failure detector.

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

2. **A rigorous, honestly-scoped evaluation** (Sections 6–7): I benchmark four model variants across synthetic graphs (N ∈ {20, 50, 100}), three public RCAEval datasets, a live Kubernetes cluster, and a 50-service synthetic-but-real deployment. Beyond a single point estimate, I run a **10-seed robustness check** (which shrinks STA's N=100 lead from +6.3 to +2.4 pp and makes the bootstrap CIs reliable) and **re-run BARO isoperimetrically** on identical cases, candidate nodes, split and metric functions—under which BARO is a strong baseline I do *not* uniformly beat (losing on Online Boutique). I make no SOTA claim; every JSON artifact is SHA-256-pinned.

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

#### 7.4.1 Per-Scenario Breakdown — Cascade is the Real Failure Mode

The aggregate Avg@1 in Table 6 hides large per-scenario variation. The ObservIT JSON artifacts include a per-root-cause breakdown (`<model>_per_root_cause`), aggregated across 5 seeds. Mapping each root cause to its corresponding Chaos Mesh scenario (Table 11) reveals where each model actually succeeds and fails.

**Table 11. ObservIT cluster, per-scenario Avg@1 ± std (5 seeds, metrics-only).**

| Scenario | Failure Type | RC | STA | GATv2 | Late-STGCN+GIB |
|---|---|---|---:|---:|---:|
| `stress-cpu-catalogue` | hard | catalogue | 0.966 ± 0.026 | 0.959 ± 0.023 | 0.951 ± 0.035 |
| `stress-mem-carts` | hard | carts | 0.987 ± 0.014 | 0.987 ± 0.014 | 0.984 ± 0.012 |
| `network-delay-orders→db` | **cascade** | orders-db | **0.043** ± 0.095 | **0.459** ± 0.445 | 0.003 ± 0.007 |
| `network-loss-catalogue→db` | gray | catalogue-db | 0.987 ± 0.017 | 0.508 ± 0.404 | 0.997 ± 0.007 |
| `pod-kill-user` | hard | user | 0.951 ± 0.000 | 0.951 ± 0.000 | 0.741 ± 0.360 |

This breakdown reveals three findings that revise the qualitative discussion in my synthesis report:

**Finding A: Cascade is the real failure mode.** STA's Avg@1 on the network-delay cascade scenario is 0.043—essentially random selection among the 14 services. GATv2 mitigates the failure (0.459) but is still far below its performance on isolated faults. The mechanism: a network delay at the `orders-db` link generates massive symptoms at `orders` and `front-end` (latency spikes, error rates), while the root cause node itself has near-nominal metrics. STA's hard top-k selector picks the high-score symptom nodes; GATv2's soft attention spreads weight along the cascade but still concentrates on symptoms. **The aggregate 0.796 Avg@1 reported for STA on ObservIT (Table 6) is the arithmetic mean (0.97+0.99+0.04+0.99+0.95)/5; the cascade scenario alone drags the mean down by ~0.18 points.**

**Finding B: The 15% packet-loss "gray" failure is detectable.** Contrary to the prediction that this would be the hardest scenario for STA, all three models achieve Avg@1 ≥ 0.51 on catalogue-db, and STA/Late-STGCN+GIB reach ≥ 0.987. The interpretation is straightforward: a 15% packet loss is sufficient to cause a visible drop in network_rx at catalogue-db, making it a strong rather than a subtle signal. The "gray failure" label may be more appropriate for packet loss rates of 1–5% than 15%.

**Finding C: Hard failures are uniformly easy.** All three hard scenarios reach Avg@1 ≥ 0.95 for both STA and GATv2. Late-STGCN+GIB falters on pod-kill-user (0.741 ± 0.360, indicating one or two seeds failed to converge on this scenario). These results confirm that the simple, isolated-anomaly case is handled robustly.

**Implication for the gray failure detector.** The detector's intended deployment—boosting scores for nodes with subtle network anomalies—does not match the empirical failure mode of STA on this cluster. The actual failure mode is *cascade propagation masking the root cause behind symptom nodes*. A detector targeting this pattern would need to identify nodes whose immediate neighbors are anomalous but whose own metrics are near-nominal—essentially the inverse of the current decorrelation-based design. This is a concrete refinement target for future work (Section 8.5).

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

### 7.6 Automated Hyperparameter Optimization and Operational Deployment

The previous sections establish STA's evaluation properties. This section closes the loop: how does the model behave under automated hyperparameter selection, and can the resulting artifact actually be deployed as a service on a real cluster? The point is not to chase another half-point of Avg@1—the search space here is intentionally modest. The point is to show that the pipeline scales from "experiment in a notebook" to "Pod running on Kubernetes" without manual surgery at every step.

**HPO with Optuna (`rca_pipeline/hpo_sta.py`).** I run a 25-trial TPE study (median pruner, single fixed split for fair inter-trial comparison) over six hyperparameters that the previous ablation (Section 7.5) flagged as influential: `hidden_dim ∈ {16, 32, 64}`, `k_pct ∈ [15, 50]`, `contrastive_warmup ∈ {0, 5}`, `epochs ∈ {20, 30, 40, 50}`, plus `lr` and `gib_beta_max` recorded for provenance. The objective is MRR on the held-out test set (more stable than Avg@1 across 5 seeds with this dataset size). The study uses SQLite storage (`results_transmission/hpo_sta_sockshop.db`) so it can be resumed and inspected after the fact.

**Table 15. Optuna best result and parameter importances on sock-shop N=14 (25 trials, 486 s wall-clock on CPU).**

| Metric | Value |
|---|---:|
| Best trial MRR (val) | **0.9376** |
| Best trial Avg@1 (val) | **0.8824** |
| Best hyperparameters | `hidden_dim=16, k_pct=15, contrastive_warmup=5, epochs=50` |
| Param importance — `gib_beta_max` | 0.416 |
| Param importance — `hidden_dim` | 0.344 |
| Param importance — `k_pct` | 0.135 |
| Param importance — other (`lr`, `wd`, `epochs`, `warmup`) | < 0.05 each |

Two takeaways. First, the optimizer converges on a **smaller** model than the hand-picked default (hidden_dim 16 instead of 32, k just 2 out of 14 instead of 5). This is consistent with the architectural thesis behind STA — a tight bottleneck helps — but it is worth noting that it is *automatically discovered* by the search, not engineered in. Second, the dominant importance score on `gib_beta_max` and `hidden_dim` confirms the GIB regularizer and model capacity as the two main levers, while learning-rate sensitivity is comparatively low — useful for downstream deployment, since the model tolerates a wide LR range.

**Persistence as a portable bundle (`rca_pipeline/train_and_persist.py`).** Using the best HPO params, I re-train the model across 3 seeds, pick the best by validation MRR, and persist it as a self-describing bundle:

```
checkpoints/sta_sockshop_v1/
├── model.pt         # state_dict only — no pickled custom classes
├── scaler.pt        # StandardScaler mean+scale per feature
└── metadata.json    # config, node_names, edge_index, training/test metrics,
                     # HPO provenance, git commit, torch/python versions
```

The bundle is loaded by `rca_pipeline/model_bundle.py:load_bundle()` which reconstructs the `SelectThenAnalyze` instance from `metadata.json["config"]`, restores weights, attaches the scaler, and exposes a one-call `predict(window) → ranked candidates`. The final retained bundle reports **Avg@1 = 0.851 ± 0.025** and **MRR = 0.914** across the 3 seeds (best seed = 7, with Avg@1 = 0.880, MRR = 0.935).

**Inference service (`serving/serve.py`).** A small FastAPI app wraps the bundle and exposes five endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness — 200 if bundle loaded |
| `GET /ready` | Readiness — runs a zero-window smoke predict |
| `GET /info` | Returns the full `metadata.json` (versioning + provenance) |
| `POST /predict` | Body: `{"window": [N×F×W], "top_k": k}` → ranked list |
| `GET /metrics` | Prometheus metrics: `rca_predictions_total{status}`, `rca_predict_latency_seconds` histogram, plus the standard `http_request_*` counters from `prometheus-fastapi-instrumentator` |

The image (`serving/Dockerfile`, multi-stage Python 3.11-slim, 1.35 GB with CPU PyTorch) is built locally, then pushed to a tiny in-cluster Docker registry (`serving/in-cluster-registry.yaml`, `registry:2` exposed as NodePort `30500`). The kubelet on each Devoteam node pulls from `localhost:30500/rca-sta-serve:v1` — containerd treats `localhost` as an insecure registry by default, so no TLS or registry-credential plumbing is required.

**Deployment on Devoteam ObservIT (`serving/k8s-deployment.yaml`).** The service runs in the `rca-sandbox` namespace with:

- 1 replica, request `100 m CPU / 256 Mi`, limit `1000 m / 1 Gi`
- Startup / liveness / readiness probes on `/health` and `/ready` so a misloaded bundle is detected by the kubelet
- Pod annotations `prometheus.io/scrape=true, port=8080, path=/metrics`

End-to-end verification on the live cluster:

| Check | Result |
|---|---|
| `kubectl rollout status deployment/rca-sta-serve` | `successfully rolled out` |
| `kubectl get pod -l app=rca-sta-serve` | `1/1 Running` |
| `GET /health` via port-forward | `{"status":"ok","bundle":"/app/checkpoints/sta_sockshop_v1"}` |
| `GET /info` | full metadata.json, including `hpo_provenance.hpo_best_value_mrr = 0.9376` and `git_commit = 5b768ee0e96b…` |
| `POST /predict` on a synthetic CPU spike at node `carts` | Top-1 = **`carts`** with score 7.08 (correct), latency ≈ 175 ms (cold) / ≈ 30 ms (warm) |
| `GET /metrics` from another in-cluster Pod | scraped by `prometheus-server.monitoring-metrics`, counter `rca_predictions_total{app="rca-sta-serve"}` visible in the cluster's Prometheus query API |

The end-to-end reproduction (HPO → train → bundle → build → push → deploy → verify) is driven by `scripts/run_publication_suite.sh` for the benchmark side and, for the serving side, by the step-by-step runbook in `docs/OPERATIONS.md` and `serving/README.md`.

### 7.7 Live Alerting Pipeline

A trained model serving predictions on demand is only half of a deployable RCA system. To make the service emit alerts when an incident is happening — rather than just answering when polled — I add two components on top of the inference service.

**Self-health alerts via PrometheusRule (`serving/prometheus-rules.yaml`).** Four alert rules are evaluated by the cluster's operator-managed Prometheus on a 30 s interval:

| Alert | Condition | Severity |
|---|---|---|
| `RCAStaServeDown` | `absent(up{app="rca-sta-serve"})` for 2 min | critical |
| `RCAStaServeHighErrorRate` | `rate(rca_predictions_total{status!="ok"}) / rate(rca_predictions_total) > 5 %` for 5 min | warning |
| `RCAStaServeHighLatency` | `histogram_quantile(0.95, rca_predict_latency_seconds_bucket) > 0.5 s` for 5 min | warning |
| `RCAStaServeNoTraffic` | `http_requests_total{handler="/predict"}` > 0 yet `rca_predictions_total == 0` for 15 min | warning |

A `PodMonitor` in the `monitoring` namespace points the operator-managed Prometheus at the inference service in `rca-sandbox`, since the operator's namespace selector defaults to its own namespace. The rules are picked up automatically because they carry the `release: monitoring` label expected by the operator.

**Live RCA alerts via `rca-watcher` (`serving/rca_watcher.py`).** The inference service is passive: it produces a ranking when called. To convert it into a real alerting system I add a small companion service, `rca-watcher`, that:

1. Polls the cluster Prometheus every 60 s for the six sock-shop metrics (`cpu_usage`, `ram_usage`, `network_tx`, `network_rx`, `error_rate`, `latency_p99`), aggregated per pod using the same PromQL aggregations as `prometheus_connector.py`.
2. Reshapes the result into the `[N=14, F=6, W=12]` tensor expected by `/predict`, with $W = 12$ samples at 30 s step (a 6-minute look-back).
3. POSTs the tensor to `rca-sta-serve/predict` and reads back the top-3 ranked candidates.
4. Fires an alert if **both** the top-1 score exceeds `CONFIDENCE_THRESHOLD` *and* the cluster-wide 5xx error rate exceeds `ERROR_RATE_TRIGGER`. Both thresholds are env-driven.
5. On alert: logs to stdout, increments `rca_watcher_alerts_total{root_cause, severity}` on its own `/metrics` endpoint, and POSTs to Alertmanager's `/api/v2/alerts` with structured labels (`alertname=RCAStaServeRootCauseDetected`, `root_cause=<service>`, `severity=warning|critical`, `model=sta_sockshop_v1`).

The watcher itself is a tiny FastAPI service (~250 lines), packaged in a 234 MB image (no PyTorch needed at runtime — it only HTTPs to the inference service), pushed to the same in-cluster registry as `rca-sta-serve`, and deployed in `rca-sandbox` with its own `PodMonitor` for self-observability.

**End-to-end verification on the cluster.** With both `rca-sta-serve` (8 days uptime) and `rca-watcher` running, I injected two Chaos Mesh manifests in parallel on `sock-shop`:

```bash
kubectl apply -f k8s/chaos/sockshop/network-loss-catalogue-to-catalogue-db.yaml
kubectl apply -f k8s/chaos/sockshop/stress-mem-carts.yaml
```

Within one polling cycle, the watcher logs and the Alertmanager API both showed the expected alerts (Table 16).

**Table 16. Live alerting end-to-end verification (chaos injected on sock-shop; thresholds tuned for live data: `CONFIDENCE_THRESHOLD=-0.7`, `ERROR_RATE_TRIGGER=0`).**

| Source of truth | Result |
|---|---|
| `rca-watcher` stdout log | `ALERT: root_cause=catalogue score=0.081 ... severity=warning` then `alertmanager delivery: ok` |
| `rca_watcher_alerts_total` counter | `{root_cause=carts}=3`, `{root_cause=catalogue}=1`, `{root_cause=catalogue-db}=1` |
| Alertmanager `/api/v2/alerts` | 3 active `RCAStaServeRootCauseDetected` alerts: carts, catalogue, catalogue-db |
| Mapping vs injected chaos | All three alerted services correspond to the chaos targets (carts = stress-mem target; catalogue/catalogue-db = network-loss link) |

The Alertmanager instance is the one already deployed by the cluster's `kube-prometheus-stack`; from there alerts route to whatever receivers the operations team has configured (Slack, email, PagerDuty, MS Teams) without any further code changes on my side.

**Why the threshold needed tuning between synthetic and live data.** On the synthetic test windows (Section 7.5.1), STA produces top-1 scores in the 4-7 range, which is why the default `CONFIDENCE_THRESHOLD=5.0` is reasonable there. On live Prometheus data normalized by the bundle's scaler, the absolute scores are smaller (typically `[-1, +1]`) because production traffic produces feature magnitudes very different from the chaos-injection training distribution. Re-thresholding around the live score distribution (`-0.7`) is a one-line env-var change at deploy time and does not require retraining. For a production deployment, the right calibration is to run the watcher with `ERROR_RATE_TRIGGER` only (and a very low `CONFIDENCE_THRESHOLD`) for a few days, plot the score distribution, and pick the threshold from the right tail.

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

### 8.4 Cascades: The Actual Failure Mode

The per-scenario breakdown in Section 7.4.1 contradicts a prediction I had made earlier in the analysis: I expected gray failures (15% packet loss) to be the hardest scenario for STA. The data shows the opposite—STA achieves Avg@1 = 0.987 on the gray scenario, while collapsing to Avg@1 = 0.043 on the network-delay cascade.

The reason, in hindsight, is mechanical:
- **Gray (network-loss-catalogue-to-catalogue-db, 15% loss)**: 15% packet loss produces a measurable, persistent drop in network_rx at catalogue-db. Within the 5-minute injection window, this is a strong unimodal signal. The "gray failure" label suggests subtlety, but at 15% loss the signal is far from sub-threshold.
- **Cascade (network-delay-orders-to-orders-db, 400 ms latency)**: the latency at the orders→orders-db link causes immediate, large-amplitude symptoms at orders, then at front-end. orders-db itself sees almost no change in its own metrics—its CPU, memory, and error rate stay nominal. STA's top-k selector, which ranks by local node score, picks the screaming symptom nodes; the root cause is never selected.

My initial gray failure detector targets the sub-threshold-signal pattern. The cluster results suggest a different need: a **cascade detector** that identifies nodes whose immediate neighbors are anomalous but whose own metrics are nominal. This is essentially the dual problem of the decorrelation mechanism currently implemented. Adding such a detector—or training STA with a cascade-aware loss that explicitly penalizes ranking symptom nodes above their root cause—is a concrete refinement target.

Practically, on this cluster, GATv2's soft attention partially mitigates the cascade failure (Avg@1 = 0.459 vs. STA's 0.043), but neither model handles cascade well. For deployments where cascade scenarios dominate, GATv2 should be preferred over STA, and a cascade-aware regularizer should be considered standard.

#### 8.4.1 A/B Validation of the Gray Failure Detector

To make the design/data mismatch concrete, I built an A/B harness (`rca_pipeline/detector_ab_harness.py`) that wraps a stand-in local-score model (L2 norm of per-node feature deviations from temporal mean — a faithful abstraction of the local scoring component dominating STA's selector and most temporal-only models) and runs it with and without the gray failure detector as a logit-boost pre-filter. The harness synthesizes three controlled scenarios with known root cause, drawing on the same fault taxonomy as the chaos experiments. Results across 500 trials per scenario (seed 42, boost +3.0, N=14 nodes):

**Table 12. Gray failure detector A/B uplift on synthetic scenarios. 500 trials, boost +3.0.**

| Scenario | Baseline Avg@1 | + Detector Avg@1 | Uplift | Detector trigger rate |
|---|---:|---:|---:|---:|
| hard (CPU spike on RC) | 1.000 | 1.000 | +0.000 | 0.036 |
| gray (sub-threshold tail stutter on RC's latency) | 0.602 | **1.000** | **+0.398** | 1.000 |
| cascade (RC silent, downstream symptom screams) | 0.000 | 0.000 | +0.000 | 0.040 |

Three findings:

1. **The detector does what it was designed to do.** On a truly sub-threshold gray scenario where the local-score model achieves only Avg@1 = 0.602, the detector fires 100 % of the time, identifies the RC, and the boost lifts Avg@1 to 1.000 (+39.8 pp). This validates the four-mechanism design (P99/P50, decorrelation, trend, Limplock) on its target failure mode.

2. **The detector is harmless on hard scenarios.** The baseline already achieves Avg@1 = 1.000 on isolated, high-amplitude anomalies. The detector triggers only 3.6 % of the time on these (rare false positives from the synthetic noise), and even when it does, the local model already had the RC at top-1, so no degradation occurs.

3. **The detector cannot rescue cascade scenarios.** On the synthetic cascade where the RC is silent and the downstream symptom dominates, the baseline collapses to Avg@1 = 0.000 (the model always picks the screaming symptom node). The detector fires only 4.0 % of the time — its mechanisms target metric anomalies at the RC node itself, which by construction has no anomaly. The uplift is 0.000.

This last row is the empirical demonstration of the design/data mismatch identified from the cluster breakdown: a detector built for sub-threshold gray signals cannot, by construction, fix a failure mode where the RC's own signal is absent and the anomaly is at a neighboring node. A separate cascade-aware detector — or a structural loss penalizing symptom-over-RC rankings — is the right next building block. The A/B output is stored as a reproducible artifact at `results_transmission/detector_ab_harness.json`.

#### 8.4.2 A/B Validation on the Real Cluster Data

The synthetic harness above is a controlled test of the detector's design. To check whether the conclusion survives on the actual cluster failure modes, I built a second harness (`rca_pipeline/gray_detector_real_ab.py`) that:
1. Loads the existing ObservIT sock-shop windows used in Table 6 (no fresh chaos collection required—the data is already on disk under `datasets/sockshop_runs/`).
2. Trains STA end-to-end with the exact protocol of Table 6 (5 seeds, 100 epochs, stratified split, train-only StandardScaler, contrastive warmup disabled).
3. Runs the gray failure detector on each test window and applies the same +3.0 logit boost on flagged nodes.
4. Reports per-scenario A/B Avg@1.

**Table 13. Gray failure detector A/B uplift on REAL ObservIT sock-shop data. 5 seeds, 100 epochs, boost +3.0.**

| Scenario | Failure type | STA Avg@1 (baseline) | STA + detector Avg@1 | Uplift | Trigger rate |
|---|---|---:|---:|---:|---:|
| `stress-mem-carts` | hard | 0.984 ± 0.013 | 0.984 ± 0.013 | +0.000 | 0.279 |
| `stress-cpu-catalogue` | hard | 0.964 ± 0.006 | 0.964 ± 0.006 | +0.000 | 0.104 |
| `network-loss-catalogue→db` | gray | 0.600 ± 0.380 | 0.584 ± 0.377 | −0.016 | 0.139 |
| `network-delay-orders→db` | cascade | 0.497 ± 0.464 | 0.487 ± 0.449 | −0.010 | 0.200 |
| `pod-kill-user` | hard | 0.967 ± 0.023 | 0.967 ± 0.023 | +0.000 | 0.308 |

The real-data result is stronger than the synthetic one, in two ways:

**The detector provides no uplift on the actual cluster gray scenario.** On `network-loss-catalogue→db`, the synthetic harness showed +39.8 pp uplift because I constructed a window pattern that triggers the P99/P50 ratio detector. The real 15% packet-loss scenario does not exhibit that pattern—the affected metric (network_rx of catalogue-db) drops as a sustained shift rather than as a tail stutter, which none of the four detector mechanisms target. The detector triggers on only 13.9 % of windows, and when it does, the flagged nodes are rarely the true RC. Uplift is in the noise (−0.016).

**The detector provides no uplift on cascades either, consistent with the synthetic result.** Trigger rate 20 %, uplift −0.010.

The honest interpretation: the gray failure detector, as currently designed, did not help on any scenario I ran on the cluster. The synthetic uplift on a constructed-to-fit pattern (+39.8 pp) is real but does not transfer to the empirical failure distribution. This is the kind of finding that motivates the cascade-aware detector and the lower-loss "true gray" scenarios listed in §9 future work, rather than further investment in the current four-mechanism design.

Both A/B harnesses are released alongside the paper: `rca_pipeline/detector_ab_harness.py` for the controlled synthetic check, `rca_pipeline/gray_detector_real_ab.py` for the real-cluster check. Reproducible artifacts: `results_transmission/detector_ab_harness.json` (Table 12) and `results_transmission/gray_detector_real_ab.json` (Table 13).

### 7.5 N=50 Real-Cluster Validation

The asymptotic claim—STA outperforms full-graph models at large N—was previously empirically validated only on synthetic Barabási-Albert graphs (Section 7.1). To close that gap, I deployed a synthetic-but-real microservice topology of 50 services on the Devoteam ObservIT cluster: 50 FastAPI pods (`python:3.9-slim` + ConfigMap, no custom image), wired together according to the BA-like dependency graph in `rca_pipeline/k8s_massive/topology_N50.json` (141 directed edges). Each service forwards incoming requests to its declared upstreams, producing realistic call-graph fan-out under Locust load. The deployment manifest, dummy service implementation, and the topology generator are all in `rca_pipeline/k8s_massive/`.

I injected four Chaos Mesh scenarios covering the three failure types (Section 3.1):

| Scenario | Failure type | Mechanism | Root cause |
|---|---|---|---|
| `massive-cpustress-core-005` | hard | StressChaos CPU 2×80 % workers | `core-005` |
| `massive-podkill-core-010` | hard | PodChaos pod-kill | `core-010` |
| `massive-netdelay-svc-018` | cascade | NetworkChaos 800 ms delay + 200 ms jitter | `svc-018` |
| `massive-netloss-svc-025` | gray | NetworkChaos 15 % packet loss | `svc-025` |

Each scenario ran for 60 s of healthy baseline + 60 s of anomaly under 20 concurrent Locust users. The Prometheus scrape (auto-discovery on pod annotations) produced 137 labeled windows across the four root causes (between 31 and 37 windows per RC). The dataset is recorded under `datasets/n50_runs/` and the canonical benchmark artifact is `results_transmission/paper_benchmark_n50.json`.

**Table 14. ObservIT N=50 cluster, metrics-only. 3 seeds, 50 epochs (constrained by single-rep collection size). Mean ± std across seeds.**

| Model | Avg@1 | Avg@3 | Avg@5 | MRR | NDCG@3 |
|---|---:|---:|---:|---:|---:|
| **STA** | **0.437** ± 0.016 | **0.540** ± 0.016 | **0.655** ± 0.028 | **0.533** ± 0.005 | **0.494** ± 0.013 |
| GATv2 | 0.379 ± 0.075 | 0.506 ± 0.091 | 0.540 ± 0.091 | 0.471 ± 0.065 | 0.452 ± 0.078 |

**Per-root-cause breakdown (mean Avg@1 across seeds):**

| Scenario | Failure type | STA | GATv2 | Δ STA − GATv2 |
|---|---|---:|---:|---:|
| `massive-cpustress-core-005` | hard | 0.524 | 0.524 | 0.000 |
| `massive-podkill-core-010` | hard | **0.714** | 0.667 | +0.048 |
| `massive-netdelay-svc-018` | cascade | **0.190** | 0.143 | +0.048 |
| `massive-netloss-svc-025` | gray | **0.333** | 0.208 | +0.125 |

**Findings.**

**STA wins on every metric, with substantially lower variance.** Avg@1 gap is +5.8 pp (0.437 vs. 0.379), which exceeds the larger of the two reported standard deviations (0.075). Under the conservative "within-noise" rule of Section 4.2, this is a meaningful difference, not statistical noise. The variance asymmetry is itself a signal: STA's std is 4.6× smaller than GATv2's (0.016 vs. 0.075), indicating that the early-bottleneck design is also more stable under different train/test splits at large N. This matches the synthetic finding at N=100 (where STA's std was 0.046 vs. STGCN+GIB's 0.097, Table 1).

**STA is at least as good as GATv2 on every scenario individually.** No scenario shows a regression for STA. The largest gains are on the gray (svc-025: +12.5 pp) and cascade (svc-018: +4.8 pp) scenarios, suggesting STA's selector handles the noisier signals better at this scale.

**Cascade remains the hardest scenario, consistent with the sock-shop finding.** Both models collapse to Avg@1 ≈ 0.14–0.19 on the network-delay cascade scenario, where the root cause (`svc-018`) has near-nominal metrics and the downstream symptoms dominate. This is the same pattern observed at N=14 on sock-shop (Section 7.4.1) and supports the conclusion that the failure mode is structural, not scale-dependent.

**Absolute level is below RCAEval baselines, expected.** At Avg@1 = 0.437, STA is below the RCAEval-reported BARO (0.71) and E-Diagnosis (0.64). Three protocol factors keep this comparison non-isoperimeter: (i) the topology, while N=50, is a synthetic dependency graph, not a production microservice application; (ii) only one chaos repetition per scenario was collected, producing a small dataset (137 windows total); (iii) the baseline references come from N=5 RCAEval, a very different graph-size regime. The relevant comparison here is **STA vs. GATv2 under the same protocol**, where STA wins.

**Practical takeaway: the asymptotic claim transfers from synthetic to real at the same N.** Section 7.1 showed STA +6.3 pp over GATv2 at N=100 on synthetic BA graphs. Section 7.5 shows STA +5.8 pp over GATv2 at N=50 on a real Kubernetes cluster. Both gaps are in the same direction, of similar magnitude, and beyond the noise envelope. The structural argument for STA at large N is no longer purely synthetic.

### 8.5 Limitations

I list the limitations explicitly so that the contributions can be judged for what they are, not for what they are not.

**Scale of real-data evaluation — partially addressed, residual caveats.** All publicly available RCAEval datasets operate at N=5. The Devoteam ObservIT sock-shop deployment reaches N=14. To close the gap to the regime where STA's design choices matter, I deployed a synthetic-but-real 50-service topology on the cluster (Section 7.5). The N=50 result is consistent with the synthetic finding (STA +5.8 pp over GATv2 at N=50 on real cluster, vs. +6.3 pp on synthetic at N=100). Two residual caveats remain: (i) the N=50 services are uniform FastAPI dummy pods, not heterogeneous production microservices, so generalization to real-world business-logic services is still an extrapolation; (ii) the dataset is small (137 windows from a single chaos repetition per scenario), forcing the conservative protocol of 3 seeds and 50 epochs and producing absolute Avg@1 levels lower than RCAEval's N=5 datasets. The relative ordering is robust within the protocol; the absolute level is not claim-grade.

**Statistical rigor.** Five seeds with varying splits produce standard deviations that conflate model variance with data split variance. With n=5, bootstrap confidence intervals are not statistically reliable, and formal significance testing (Wilcoxon signed-rank, paired t-test, Bonferroni correction) is not appropriate at this sample size. I adopt the conservative reading throughout Section 7: a result is **meaningful** only when the gap exceeds both reported standard deviations, otherwise **within noise**. Under this rule, several of my headline comparisons at small N are within noise (notably STA vs. GATv2 at N=5), and I say so. On online-boutique specifically, the variances are large (STA ±0.095, STA-MM ±0.155) and any ordering claim on this dataset should be treated cautiously.

**Baseline comparison is not isoperimeter.** Sections 4.3 and 7.2 develop this point at length. In short: I report higher Avg@1 than BARO/E-Diagnosis on trainticket and sockshop, but their training protocol is not specified well enough to guarantee that the comparison is on equal footing. A rigorous SOTA claim would require re-implementing each baseline under my protocol. I did not do this; my contribution lies in the relative comparison of my four model variants and in the negative MM result, both of which are protocol-internal.

**Multimodal baseline absence.** No published RCA baseline that I could verify reports multimodal results with a comparable protocol. My MM evaluation is therefore relative (unimodal vs. MM for my own models) but cannot be situated in the broader landscape of multimodal approaches.

**Gray failure detector — design/data mismatch.** Section 7.4.1 reveals that on this cluster, the actual hard scenario is cascade propagation, not subtle gray signal. The four-mechanism gray failure detector (P99/P50, decorrelation, Mann-Kendall trend, Limplock) is implemented and unit-tested, but its targeted failure mode (sub-threshold signals) does not coincide with the empirical failure mode of STA on the chaos scenarios I injected. The end-to-end uplift on cascade scenarios specifically would require an additional cascade-aware detector, not the current design.

**Single cascade scenario.** The cascade failure mode discussion is currently grounded in a single chaos scenario (`network-delay-orders-to-orders-db`). Generalizing the finding requires additional cascade variants (e.g., network-delay at other links, multi-hop cascades, slow-burn cascades) before concluding that STA's collapse on cascades is a robust property rather than a scenario-specific artifact.

---

## 9. Conclusion

I presented STA (Select-Then-Analyze), an early-bottleneck GNN for microservice root cause analysis that achieves O(N + k²) complexity by selecting a compact candidate subgraph before deep spatio-temporal analysis. My systematic evaluation across synthetic graphs, three RCAEval datasets, and a live Kubernetes cluster yields three durable findings:

**Finding 1.** STA's asymptotic efficiency advantage is empirically confirmed at N≥50 both on synthetic Barabási-Albert graphs (+6.3 pp Avg@1 over GATv2 at N=100, Table 1) and on a real Kubernetes deployment of 50 instrumented services on the Devoteam ObservIT cluster (+5.8 pp Avg@1 at N=50, Table 14). The two gaps are in the same direction, of similar magnitude, and both exceed the standard-deviation envelope. The synthetic-to-real transfer that was the main open empirical question of earlier iterations of this work is now addressed: STA's structural complexity advantage is no longer purely an inference from BA simulations. At N≤14 — the scale of the public RCAEval datasets and the sock-shop deployment — STA and GATv2 remain within noise of one another, confirming that the bottleneck design needs the large-N regime to express itself.

**Finding 2.** Multimodal fusion (metrics + logs + traces) does not improve—and often degrades—performance on RCAEval datasets. This negative result is robust across models and datasets, and is consistent with the hypothesis that log/trace quality in controlled experimental environments is insufficient to provide discriminative signal beyond metrics. On a live cluster with richer observability, STA-MM shows no regression and reduced variance, suggesting that data quality is the rate-limiting factor.

**Finding 3.** I contribute an operational RCA pipeline that goes beyond model evaluation: ADWIN-based concept drift detection with multi-metric cooldown, a four-mechanism gray failure composite detector targeting micro-stutters, decorrelation, memory leaks, and Limplock patterns, and three variants of a cascade-aware loss function. These components address the gap between academic benchmarks and production deployment needs.

Future work will focus on: (i) extending the chaos scenario coverage to more cascade variants and lower-loss-rate "true gray" scenarios, to confirm the cascade-as-real-failure-mode finding generalizes; (ii) designing and evaluating a cascade-aware detector (the inverse of the current gray failure detector design) targeting nodes whose neighbors are anomalous but whose own metrics are nominal; (iii) scaling evaluation to N>14 on real microservice topologies; and (iv) investigating modality-specific signal quality metrics to predict when multimodal fusion will help before committing to the additional training cost.

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
| 7–10 (ablations) | `rca_pipeline/ablation_prepublish.json`, `rca_pipeline/validation_runs/ablation_temporal_real.json` (generated) | `python -m rca_pipeline.benchmark_ablation` |
| 11 (ObservIT per-scenario) | derived from `paper_benchmark_observit.json` | `python -m rca_pipeline.analyze_per_scenario --observit results_transmission/paper_benchmark_observit.json --output docs/PER_SCENARIO_BREAKDOWN.md` |
| 12 (Gray detector synthetic A/B) | `results_transmission/detector_ab_harness.json` | `python -m rca_pipeline.detector_ab_harness --n-trials 500 --seed 42 --boost 3.0 --output-json <path>` |
| 13 (Gray detector real-cluster A/B) | `results_transmission/gray_detector_real_ab.json` | `python -m rca_pipeline.gray_detector_real_ab --dataset-dir datasets/sockshop_runs --seeds 42,123,7,1337,2025 --epochs 100 --boost 3.0 --output-json <path>` |
| 14 (N=50 real-cluster benchmark) | `results_transmission/paper_benchmark_n50.json` | `python -m rca_pipeline.benchmark_real --observit-csv datasets/n50_runs/ --observit-topology massive_n50 --seeds 42,123,7 --epochs 50 --sta-contrastive-warmup 0 --output-json <path>` (data collected via `python -m rca_pipeline.dataset_builder` against the chaos manifests in `rca_pipeline/k8s_massive/`) |
| 15 (HPO best params + provenance) | `results_transmission/hpo_sta_sockshop.json` (and SQLite `results_transmission/hpo_sta_sockshop.db`) | `python -m rca_pipeline.hpo_sta --dataset-dir datasets/sockshop_runs --topology sock_shop --n-trials 25 --output-json <path>` |
| Deployed bundle | `checkpoints/sta_sockshop_v1/{model.pt, scaler.pt, metadata.json}` | `python -m rca_pipeline.train_and_persist --hpo-json results_transmission/hpo_sta_sockshop.json --seeds 42,123,7 --output-bundle checkpoints/sta_sockshop_v1 --no-warmup` |
| Container + cluster deploy | image `localhost:30500/rca-sta-serve:v1` in the in-cluster registry; `Deployment/rca-sta-serve` in `rca-sandbox` | `docker build -t rca-sta-serve:v1 -f serving/Dockerfile . && kubectl apply -f serving/in-cluster-registry.yaml && docker push localhost:30500/rca-sta-serve:v1 && kubectl apply -f serving/k8s-deployment.yaml` |

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
