# Research proposal: CESTA

## Name

**CESTA**: Communication-Efficient Spatial-Temporal Aggregation.

The method name is intentionally one uppercase word for paper and implementation consistency.

## Motivation

Fault diagnosis in sensor networks benefits from temporal modeling, but dense spatial models can waste energy by communicating with all neighbors even when local temporal evidence is sufficient. Current project results show strong temporal baselines and weak ST-GCN performance, so a convincing Q1-level contribution should not merely beat ST-GCN. The method should exceed temporal-only macro-F1 while reducing communication energy relative to dense spatial communication.

## Research questions

1. Can a lightweight spatial-temporal model exceed temporal-only macro-F1 by selectively requesting neighbor information?
2. Can learned request and compression decisions reduce measured and theoretical communication energy while preserving or improving accuracy?
3. Does a trainable communication controller outperform rule-based uncertainty/change-triggered communication at the same energy budget?
4. Is Gumbel-Softmax or reinforcement learning better for learning the communication policy under a Pareto criterion?

## Hypotheses

1. CESTA improves average macro-F1 over the best temporal-only model across Intel fault ratios.
2. CESTA reduces communication energy compared with dense spatial message passing by requesting only useful neighbor embeddings and compressing transmitted messages.
3. Receiver-side local uncertainty and temporal embedding state are sufficient to decide when neighbor information is useful without inspecting neighbor embeddings before communication.
4. Gumbel-Softmax will be easier to train and more reproducible than RL, but the final main design will be selected by Pareto dominance rather than preference.

## Success criteria

Primary success criterion:

```text
Average Δ macro-F1 >= +0.01 over the best temporal-only model per fault ratio
and Pareto-superior energy/accuracy behavior against dense spatial communication.
```

Preferred Q1-level target:

```text
Average Δ macro-F1 >= +0.03 to +0.04 over the best temporal-only model per fault ratio
and substantial measured communication-energy reduction against dense spatial communication.
```

Secondary criteria:

- improve over a fixed temporal backbone used inside CESTA;
- outperform or match required spatial baselines including HiFiNet, if HiFiNet targets sensor/graph fault diagnosis;
- reduce theoretical TX+RX radio energy compared with dense spatial message passing;
- show lower or comparable edge inference cost than dense spatial models;
- remain lightweight enough to classify as edge-oriented, with ESP32-S3 as a loose lower target.

## Core design

CESTA is a distributed receiver-side request model. Each node first encodes its own local temporal window, estimates local diagnosis uncertainty, and decides which existing graph neighbors to request from and at what compression ratio. The current dense upper-bound line also includes optional local diagnostic refinements: logit correction and a CRF transition decoder. These refinements do not add message payloads and should be treated as dense-accuracy improvements before selective communication is evaluated.

The graph topology is a fixed directed candidate edge set derived from the existing connectivity data, but runtime communication availability is dynamic. CESTA learns request and compression decisions over currently available candidate edges; it does not perform unconstrained graph discovery in the main method.

### Dynamic network construction

Graph preparation should store a directed candidate edge list instead of a dense static adjacency artifact:

```text
edge_index[0, e] = sender node
edge_index[1, e] = receiver node
edge_prob[e] = raw connectivity probability p_sender,receiver
```

Candidate edges are thresholded exactly as directed entries in `connectivity.txt`; self-loops are excluded. A bursty link simulator runs once during graph preparation and stores a per-timestamp sampled link-success mask plus simulation metadata. Node observation availability remains separate from communication availability:

```text
M[t, i] = 1 if node i has an observed reading at timestamp t
L[t, e] = 1 if directed communication edge e succeeds at timestamp t
active_edge[t, e] = L[t, e] & M[t, sender(e)] & M[t, receiver(e)]
```

The burst simulator uses raw `p_ij` for both state transitions and packet success. Each undirected node pair has a shared GOOD/BAD environment chain plus direction-specific GOOD/BAD chains; the effective directed state is BAD if either chain is BAD. Initial states are sampled from each edge's stationary distribution. This keeps the candidate topology fixed while making the communication network realistic and reproducible.

### Local temporal encoder

Each node processes its own window:

```text
x_i ∈ R^{T × F}
h_i,1:T = TemporalEncoder(x_i)
```

Recommended first encoders:

- lightweight GRU;
- lightweight depthwise-separable TCN as a later alternative.

The encoder should be small enough that gains cannot be explained only by a larger temporal backbone.

### Receiver-side communication decision module

For receiver node `i` and candidate neighbor `j`, the gate uses only local information from receiver `i`:

```text
g_i,j = RequestGate(pool(h_i,1:T), uncertainty_i, edge_features_i,j)
r_i,j = CompressionGate(pool(h_i,1:T), uncertainty_i, edge_features_i,j)
```

where:

- `g_i,j` is request/no-request;
- `r_i,j` selects compression ratio;
- `uncertainty_i` can include entropy or margin from the local classifier;
- `edge_features_i,j` may include existing connectivity weight, hop flag, or normalized node degree.

The gate does not inspect neighbor embeddings before requesting. This avoids hidden pre-communication cost and keeps the energy model honest.

### Message compression

If requested, neighbor `j` sends a compressed temporal embedding:

```text
m_j→i = Compress_r(h_j,1:T)
```

Compression ratios should include staged options such as:

```text
r ∈ {0.25, 0.5, 1.0}
```

The aggregation module projects received messages back to the common hidden dimension before fusion.

### Neighbor aggregation

CESTA aggregates only requested messages:

```text
a_i,1:T = Aggregate({Up(m_j→i) | j ∈ N(i), g_i,j = 1})
```

Aggregation uses a GAT-inspired single-head attention mechanism, executed entirely receiver-side from already-received messages:

```text
Q_i   = W_q · h_i                    ← local state projects to query
K_j   = W_k · h_j                    ← each received message projects to key
V_j   = W_v · h_j                    ← each received message projects to value
α_ij  = softmax_j(Q_i^T K_j / √d)   ← attention over received neighbors only
a_i   = Σ_j α_ij · V_j              ← attention-weighted sum
```

When zero neighbors are requested, `a_i = 0` (no attention computed). When a requested neighbor's reply fails to arrive because its dynamic directed link is unavailable or a deployment deadline expires (e.g. 2× one-hop MQTT RTT), attention runs over the partial received set — the softmax denominator shrinks to match what actually arrived, which is identical to the gradient regime the centralized trainer sees when that neighbor is masked out. Deadline thresholds are documented as a system parameter, not tuned per experiment.

Key properties:
- **Zero communication overhead.** The Q/K/V projections and softmax use only the hidden states already received; no additional bits transmitted.
- **Dynamic receptive field.** The attention softmax operates over the exact variable-cardinality set of actually-requested neighbors, not a padded fixed-size tensor. Masked interpretation matches the distributed inference loop exactly: loop over arrived messages, compute exp(Q·K_j) accumulators, normalize.
- **Single-head by default.** Multi-head attention over small neighbor sets (typically 1–3 neighbors in a sparse sensor graph) fragments the already-small embedding dimension without improving expressivity. Single-head is the initial design; multi-head may be explored as an ablation if neighbor sets grow large.
- **Fusion gate retained.** The learned `fusion` MLP combining `[h_i, a_i]` with a residual connection is preserved after attention aggregation.

### Classifier

The final classifier combines local and aggregated spatial context:

```text
z_i,1:T = Fusion(h_i,1:T, a_i,1:T)
y_i,1:T = Classifier(z_i,1:T)
```

The output remains many-to-many per node and timestep to match current CESTA graph-model evaluation:

```text
(batch, window_size, num_nodes, num_classes)
```

## Training strategies

Two controller-training options will be explored:

1. **Gumbel-Softmax / straight-through estimators** for differentiable request and compression decisions.
2. **Reinforcement learning** for non-differentiable energy/accuracy reward optimization.

The main paper design will be whichever is Pareto-superior:

```text
higher macro-F1 at equal/lower measured energy
or lower measured energy at equal/higher macro-F1.
```

If tied, prefer Gumbel-Softmax because it is simpler, more reproducible, and easier to integrate into the existing supervised training loop.

## Energy model

The theoretical communication energy model should count both transmission and receiving energy for every active message.

For a message of `k` bits over distance `d`:

```text
E_tx(k, d) = E_elec · k + E_amp · k · d^n
E_rx(k) = E_elec · k
E_msg(k, d) = E_tx(k, d) + E_rx(k)
```

Use the cited first-order radio model from Mahajan et al. when writing the paper:

- free-space mode for short distances;
- multipath mode for long distances;
- threshold distance `d0 = sqrt(E_fs / E_mp)`;
- include TX and RX energy in all CESTA and baseline communication totals.

The paper should also measure on-device energy on ESP32-S3 or the smallest feasible edge-class target available.

## Efficiency metrics

Primary metrics should be based on energy consumption:

- measured on-device energy per inference/window;
- theoretical communication energy per window using TX+RX;
- energy reduction versus dense spatial communication;
- macro-F1 per Joule or macro-F1 per communication-energy unit;
- average requested edges per node/window;
- transmitted bits per node/window;
- compression-ratio distribution;
- parameter count and latency.

## Required baselines

1. Best temporal-only model per fault ratio.
2. Fixed temporal backbone matching CESTA's encoder without communication.
3. ST-GCN.
4. HiFiNet, if it targets sensor/graph fault diagnosis.
5. Dense learned message passing with all currently available directed candidate edges and full embeddings.
6. Rule-based uncertainty/change-triggered communication with matched communication budget.
7. Static top-k graph communication using strongest connectivity edges.
8. Random communication at matched average budget.

## Scope

In scope:

- Intel graph datasets across all four fault ratios;
- temp-only input for comparability with existing baselines;
- learned receiver-side request and compression over currently available directed candidate edges;
- Gumbel and RL training comparison;
- theoretical TX+RX energy and on-device edge energy evaluation.

Out of scope for the first paper iteration:

- unconstrained latent graph discovery;
- dependence on quantization as the core novelty;
- multi-hop communication protocols beyond existing graph neighbors;
- assuming ESP32-S3 feasibility without measurement.

## Edge deployment position

ESP32-S3 is a loose target, not a strict hard requirement. The model should aim for the smallest feasible configuration and be defensibly classified as edge-oriented. Quantization and pruning are evaluation tools only, not part of the main algorithmic contribution.

## Current evidence

The current decisive development setting is `Intel_fault15` with `features: ["temp"]`. Dense CESTA has now exceeded the same-split GRU temporal reference, but only modestly:

| Model | Test macro-F1 | Δ vs GRU 0.9018 | Main interpretation |
|---|---:|---:|---|
| Same-split GRU | 0.9018 | — | current temporal reference |
| Dense CESTA + logit correction | 0.9145 | +0.0127 | clears the minimum fault15 target |
| Dense CESTA + boundary head/gated correction | 0.9087 | +0.0069 | underperforms; naive boundary supervision is risky |
| Dense CESTA + logit correction + CRF | 0.9160 | +0.0142 | current dense upper-bound candidate |

The CRF variant improves DRIFT/STUCK sequence consistency without changing communication, but it reduces SPIKE F1. This supports the architecture story that CESTA should handle spatial evidence while a lightweight structured decoder can handle temporal label consistency. However, the current gain is far below the preferred +0.03 to +0.04 Q1-level margin, so dense upper-bound tuning remains the priority before communication-efficiency claims.

## Main risks

1. **Temporal baselines are already strong.** A 3–4 macro-F1 point improvement may require better spatial signal than the current Intel connectivity graph provides.
2. **Current dense gains are modest.** The best dense result on `Intel_fault15` is about +0.014 macro-F1 over same-split GRU, enough for the minimum target on this ratio but not yet a strong Q1-level margin.
3. **ST-GCN is too weak to be decisive.** The paper must include stronger dense and rule-based spatial baselines.
4. **Gate collapse.** Learned requests may become all-on or all-off without careful penalties and temperature/reward schedules.
5. **Energy accounting ambiguity.** All communication claims must include TX and RX energy, plus measured edge energy when possible.
6. **Structured decoding tradeoffs.** CRF-style smoothing helps DRIFT/STUCK but can suppress short SPIKE events unless tuned carefully.
7. **Boundary modeling fragility.** A boundary auxiliary head and simple boundary-gated correction hurt macro-F1 in the current experiment.
8. **HiFiNet availability.** If HiFiNet is the closest spatial method, it must be reproduced or clearly bounded as unavailable/inapplicable.

## Implementation milestones

1. Implement CESTA with a GRU encoder, receiver-side request gate, compression selector, lightweight aggregation, and graph-model output shape.
2. Add auxiliary communication-energy/loss support to training.
3. Add evaluation logging for requested edges, transmitted bits, theoretical TX+RX energy, and model compute metrics.
4. Run a sanity experiment on `Intel_fault15`.
5. Run staged ablations for request-only, compression-only, and full request+compression.
6. Compare Gumbel-Softmax and RL controllers under Pareto dominance.
7. Run full average-performance evaluation across all fault ratios.
