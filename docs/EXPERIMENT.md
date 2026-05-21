# Experiment plan: CESTA

## Question

Can CESTA exceed temporal-only macro-F1 while reducing communication energy through receiver-side learned request and compression decisions over the existing sensor graph?

## Primary hypothesis

CESTA will improve average macro-F1 over the best temporal-only model per fault ratio by at least +0.01, preferably +0.03 to +0.04, while reducing communication energy relative to dense spatial message passing.

## Data

Use the current harder Intel injected graph datasets:

```text
data/injected/Intel_fault05
data/injected/Intel_fault10
data/injected/Intel_fault15
data/injected/Intel_fault20
```

Use temp-only input for comparability:

```text
features: ["temp"]
```

Graph preparation should use a directed candidate edge list from `connectivity.txt` and a once-sampled bursty link-success mask. Runtime graph availability is dynamic:

```text
active_edge[t,e] = link_success[t,e] & node_observed[t,sender(e)] & node_observed[t,receiver(e)]
```

Missing node labels are stored as `-1` and excluded by masked loss/metrics. Complete-case timestamp filtering is not viable for the current Intel graph data because no timestamp contains all 55 nodes.

The first decisive development dataset is:

```text
data/injected/Intel_fault15
```

## Current temporal targets

Current HPO retrain macro-F1 targets from existing results:

| Fault ratio | Best temporal baseline | Macro-F1 |
|---|---|---:|
| fault05 | GRU | 0.8574 |
| fault10 | LSTM | 0.8764 |
| fault15 | GRU | 0.8999 |
| fault20 | GRU | 0.9042 |

Minimum paper target by fault ratio is approximately:

| Fault ratio | Minimum target |
|---|---:|
| fault05 | 0.8674 |
| fault10 | 0.8864 |
| fault15 | 0.9099 |
| fault20 | 0.9142 |

Preferred Q1-level target is +0.03 to +0.04 average macro-F1 over those best temporal baselines.

## Current Intel_fault15 diagnosis results

All results below use the hard temp-only setting:

```yaml
features: ["temp"]
```

The same-split GRU reference for the current `Intel_fault15` diagnosis runs is macro-F1 `0.9018`. The best dense CESTA candidate so far is dense CESTA with logit correction plus a CRF transition layer, at macro-F1 `0.9160`. This clears the minimum +0.01 development target on `fault15` by about `+0.0142` over the same-split GRU, but it is still below the preferred +0.03 to +0.04 Q1-level margin.

| Variant | Run | Val macro-F1 | Test macro-F1 | Δ vs GRU 0.9018 | Δ vs dense logit correction | Notes |
|---|---|---:|---:|---:|---:|---|
| Same-split GRU | prior run | — | 0.9018 | — | — | temporal-only reference for current split |
| Previous dense CESTA | prior run | — | 0.9126 | +0.0108 | -0.0019 | first dense spatial upper-bound candidate |
| Dense CESTA + logit correction | `runs/cesta/20260520T100015Z_cesta_seed42_a6931bb` | 0.9141 | 0.9145 | +0.0127 | — | best pre-CRF dense candidate |
| Dense CESTA + logit correction + boundary head | `runs/cesta/20260520T164518Z_cesta_seed42_a6931bb` | 0.9118 | 0.9087 | +0.0069 | -0.0058 | boundary auxiliary head and soft boundary-gated correction hurt DRIFT |
| Dense CESTA + logit correction + CRF | `runs/cesta/20260520T173159Z_cesta_seed42_a6931bb` | 0.9266 | 0.9160 | +0.0142 | +0.0014 | current best dense upper-bound candidate |

Per-class test F1 for the key dense variants:

| Variant | NORMAL | SPIKE | DRIFT | STUCK | Accuracy | Transmitted bits |
|---|---:|---:|---:|---:|---:|---:|
| Dense + logit correction | 0.9831 | 0.9873 | 0.8458 | 0.8419 | 0.9691 | 449560576 |
| Boundary-aware dense | 0.9819 | 0.9856 | 0.8266 | 0.8407 | 0.9670 | 449560576 |
| CRF dense | 0.9828 | 0.9748 | 0.8493 | 0.8570 | 0.9691 | 449560576 |

Confusion matrices show the tradeoff: the CRF improves DRIFT and STUCK sequence consistency but smooths some true SPIKE cases. Relative to dense + logit correction, CRF changes the main minority F1s as follows:

```text
SPIKE: 0.9873 -> 0.9748  (-0.0125)
DRIFT: 0.8458 -> 0.8493  (+0.0035)
STUCK: 0.8419 -> 0.8570  (+0.0151)
```

Current interpretation:

1. Dense CESTA can exceed the same-split temporal baseline under `features: ["temp"]`, but the current margin is modest.
2. Logit correction is useful and remains part of the dense upper-bound path.
3. Boundary supervision alone is not a reliable fix: it reduced test macro-F1 and especially DRIFT F1.
4. A lightweight CRF transition layer is the strongest dense result so far and adds no communication overhead.
5. Sparse/gated communication experiments should wait until dense CESTA is either tuned further or accepted as a modest upper-bound improvement.

## Baselines and controls

### Temporal baselines

1. Best temporal-only model per fault ratio.
2. Fixed CESTA temporal encoder without communication.
3. GRU/LSTM/ModernTCN HPO retrain results as strong temporal references.

### Spatial baselines

1. ST-GCN.
2. HiFiNet, if the supplied paper confirms it targets sensor/graph fault diagnosis.
3. Dense learned message passing: same encoder/aggregator as CESTA, all currently available directed candidate edges active, full embeddings transmitted.
4. Static top-k graph communication using strongest connectivity edges.
5. Random communication at matched average communication budget.

### Rule-based controls

1. Uncertainty-triggered communication using entropy or prediction margin.
2. Change-triggered communication using local change/anomaly magnitude.
3. Combined uncertainty + change trigger.

Rule-based thresholds must be tuned to match CESTA's average communication budget for fair comparison.

## Metrics

### Accuracy metrics

- macro-F1;
- per-class F1;
- accuracy;
- confusion matrix;
- average Δ macro-F1 against the best temporal-only model per fault ratio;
- average Δ macro-F1 against a fixed temporal backbone.

### Communication and energy metrics

Primary energy metrics should be based on energy consumption:

- measured on-device energy per window/inference;
- theoretical TX+RX communication energy per window;
- energy reduction versus dense learned message passing;
- macro-F1 per Joule or macro-F1 per communication-energy unit;
- active request ratio;
- requested edges per node/window;
- transmitted bits per node/window;
- compression-ratio distribution;
- receiver RX energy share;
- sender TX energy share.

### Edge metrics

- parameter count;
- serialized model size;
- inference latency on edge-class target;
- peak memory estimate;
- effect of int8/dynamic quantization as evaluation only.

## Theoretical energy calculation

For every active receiver-side request from sender node `j` to receiver node `i`, count both TX and RX energy only when the directed candidate edge is available at that timestamp/window.

```text
E_tx(k, d) = E_elec · k + E_amp · k · d^n
E_rx(k) = E_elec · k
E_msg(k, d) = E_tx(k, d) + E_rx(k)
```

Use free-space or multipath amplifier constants according to threshold distance:

```text
d0 = sqrt(E_fs / E_mp)
```

For CESTA:

```text
E_CESTA = Σ_windows Σ_t Σ_edges j→i available[t,j→i] · g_i,j,t · E_msg(k_i,j, d_i,j)
```

where `k_i,j` depends on hidden dimension, compression ratio, numeric precision, and protocol overhead if modeled.

For dense learned message passing:

```text
E_dense = Σ_windows Σ_t Σ_edges j→i available[t,j→i] · E_msg(k_full, d_j,i)
```

Report reduction:

```text
reduction = 1 - E_CESTA / E_dense
```

## Staged experiments

### Stage 0: feasibility checks

Goal: verify graph data shape, output shape, and training loop compatibility.

Run on `Intel_fault15` for a very small epoch budget.

Checks:

- graph batch carries `x`, `y`, `node_mask`, directed `edge_index`, and per-window `edge_mask`;
- logits shape is `(batch, window_size, num_nodes, num_classes)`;
- loss computes against per-node labels only where `node_mask` is true;
- communication stats are non-empty;
- requested edge ratio is not NaN;
- model can overfit a tiny batch.

### Stage 1: temporal encoder baseline

Train the CESTA temporal encoder without communication.

Purpose:

- establish the fixed backbone baseline;
- separate temporal encoder strength from spatial communication contribution.

Required outputs:

- macro-F1;
- per-class F1;
- parameter count;
- latency estimate.

### Stage 2: dense learned message passing

Train the same encoder and aggregation module with all currently available directed candidate edges active and full embeddings transmitted.

Purpose:

- establish the upper bound for the CESTA architecture without communication limits;
- provide a stronger spatial baseline than ST-GCN.

Status on `Intel_fault15`:

- dense CESTA + logit correction reached macro-F1 `0.9145`;
- boundary-aware auxiliary supervision and boundary-gated correction reached only `0.9087`, so this path is deprioritized unless redesigned as full segment-level refinement;
- dense CESTA + logit correction + CRF reached macro-F1 `0.9160`, making it the current dense upper-bound candidate;
- all dense variants above have the same transmitted-bit estimate (`449560576`) because boundary and CRF additions are local/decoder-side and do not alter message payloads.

Required outputs:

- macro-F1;
- per-class F1;
- theoretical TX+RX energy;
- measured edge energy if available.

### Stage 3: request-only CESTA

Train receiver-side learned request gates with full embedding transmission when active.

Compare:

- Gumbel-Softmax request gate;
- RL request policy.

Purpose:

- isolate the benefit of deciding whether to communicate.

### Stage 4: compression-only CESTA

Keep all currently available directed candidate edges active but learn/select compression ratio.

Compare:

- fixed compression ratios;
- Gumbel-Softmax compression selector;
- RL compression selector if feasible.

Purpose:

- isolate the benefit of reducing payload size.

### Stage 5: full CESTA

Train receiver-side request gate and compression selector together.

Compare:

- Gumbel request + Gumbel compression;
- RL request + RL compression;
- hybrid Gumbel pretraining followed by RL fine-tuning if neither pure method dominates.

The main design is selected by Pareto dominance:

```text
higher macro-F1 at equal/lower measured energy
or lower measured energy at equal/higher macro-F1.
```

If Pareto-tied, choose the simpler Gumbel design.

### Stage 6: rule-based controls

Evaluate rule-based controllers at matched average communication budgets:

1. entropy threshold;
2. prediction-margin threshold;
3. local change magnitude threshold;
4. combined uncertainty + change threshold.

Purpose:

- prove that learned communication is better than simple triggering.

### Stage 7: full benchmark across all fault ratios

Run the selected CESTA variants across all four fault ratios.

Required comparisons:

- best temporal per fault ratio;
- fixed temporal backbone;
- ST-GCN;
- HiFiNet if applicable;
- dense learned message passing;
- static top-k;
- random budget-matched;
- best rule-based budget-matched controller.

## Ablations

Required ablations:

1. no communication;
2. dense full communication;
3. request-only;
4. compression-only;
5. request + compression;
6. uncertainty removed from gate input;
7. local embedding removed from gate input;
8. fusion gate removed;
9. static top-k neighbors;
10. random neighbors at matched budget;
11. Gumbel-Softmax versus RL;
12. per-window versus per-timestep decision if implementation cost permits;
13. compression ratios `{0.25, 0.5, 1.0}` versus smaller/larger sets;
14. different energy penalty weights;
15. quantized versus non-quantized edge evaluation;
16. GAT single-head attention versus degree-normalized mean aggregation;
17. attention over received set only versus softmax over padded full neighbor set;
18. multi-head attention versus single-head if neighbor sets grow large (>4).

## Hyperparameter sweeps

Minimum sweep axes:

- hidden size: `32`, `64`, `128`;
- gate penalty weight;
- bits/energy penalty weight;
- compression-ratio set;
- dropout;
- Gumbel temperature schedule;
- RL reward weights if RL is used.

The selection metric should not be macro-F1 alone. Use Pareto frontier analysis over macro-F1 and measured/theoretical energy.

## Expected outcomes

Best case:

- CESTA improves average macro-F1 by +0.03 to +0.04 over best temporal baselines;
- communication energy falls substantially versus dense spatial communication;
- learned request/compression dominates rule-based controls;
- gate activation is higher for uncertain, DRIFT, and STUCK windows.

Minimum acceptable outcome:

- CESTA improves average macro-F1 by at least +0.01 over best temporal baselines;
- CESTA is Pareto-superior to dense learned message passing or at least to ST-GCN/HiFiNet if dense message passing is too costly.

Negative outcome:

- CESTA cannot exceed best temporal baselines;
- communication is only useful at all-on or near-all-on budgets;
- rule-based triggers match learned gates.

If negative, reposition the contribution as an energy-aware spatial communication study only if energy savings are strong and accuracy remains close to temporal baselines.

## Failure modes

1. Gate collapse to all-off due to energy penalty overpowering classification loss.
2. Gate collapse to all-on because spatial messages are too useful or penalty is too weak.
3. Compression selector always chooses full embeddings.
4. RL policy instability or high variance.
5. The existing Intel connectivity graph lacks useful spatial signal.
6. Energy model overstates savings relative to measured ESP32-S3 behavior because radio wake/sleep overhead dominates.
7. Dense learned message passing beats CESTA by too much, weakening selective-communication claims.
8. HiFiNet outperforms CESTA without much extra cost.
9. Structured decoding improves STUCK/DRIFT but oversmooths short SPIKE events.
10. Boundary auxiliary objectives overemphasize unstable transition regions and hurt plateau classification.

## Diagnosis findings so far

The main `Intel_fault15` error analysis indicates that long fault segments are not the dominant problem. DRIFT has only a small number of test segments longer than the 60-step window and those long segments were classified correctly; STUCK has no test segments longer than the window. Errors concentrate near fault starts and ends, especially the first 10 timesteps after onset.

Observed boundary-region pattern before the boundary-head experiment:

```text
DRIFT overall accuracy: 0.8354
DRIFT first 10 steps after start: 0.7367
DRIFT away from start: 0.9202
STUCK overall accuracy: 0.7954
STUCK first 10 steps after start: 0.7510
STUCK away from start: 0.8918
```

Tested remedies:

1. **Boundary head + boundary-gated correction**: implemented auxiliary focal BCE supervision on label transitions with dilation and used predicted boundary probability to boost logit correction. Result: macro-F1 dropped from `0.9145` to `0.9087`, mainly from DRIFT F1 dropping `0.8458 -> 0.8266`. Conclusion: naive boundary gating is not sufficient and may destabilize DRIFT decisions.
2. **CRF transition layer**: added a learned linear-chain transition matrix with masked CRF negative log-likelihood and Viterbi decoding. Result: macro-F1 improved to `0.9160`, with STUCK F1 improving `0.8419 -> 0.8570` and DRIFT F1 improving slightly, but SPIKE F1 dropping `0.9873 -> 0.9748`. Conclusion: structured decoding is promising but must be tuned to avoid oversmoothing short faults.

Promising next dense-upper-bound actions:

1. Tune CRF strength (`crf_loss_weight`) and consider transition initialization from train label transition frequencies.
2. Protect SPIKE with class-aware transition regularization or lower CRF loss weight.
3. If revisiting boundaries, use segment-level refinement or boundary-preserving smoothing rather than a simple correction multiplier.
4. Only proceed to sparse/gated communication claims once dense CESTA's upper bound is stable enough to serve as the proper comparator.

## Reproducibility notes

Record for every run:

- dataset path and fault ratio;
- selected features;
- graph threshold, directed edge count, node count, dynamic-link seed, and burst-simulation parameters;
- random seed;
- model config;
- training controller type: Gumbel, RL, or rule-based;
- energy constants and distance assumptions;
- measured-energy hardware setup;
- communication stats;
- run manifest and git state.

## First implementation checkpoint

Implement a minimal CESTA variant first:

```text
GRU temporal encoder
receiver-side local gate
Gumbel request only
full embedding when active
GAT-inspired single-head attention aggregation
communication stats logging
```

Run this on `Intel_fault15` before adding compression or RL.
