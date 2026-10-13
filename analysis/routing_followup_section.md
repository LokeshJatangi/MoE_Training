## 7. Completed hard-routing repair and per-expert learning audit

This follow-up tests whether the hard-routing failure can be repaired without changing the trained parent, model capacity, top-k or token budget. All **five policies completed 750 T4 updates** from the same pinned parent: 384,000 additional targets per branch, with 12,030,336 total / 3,182,976 selected-convention parameters. Seed 17, identical sampler order, CE-only objective, reset AdamW, LR 0.0003, betas (0.9,0.95), weight decay 0.01 and clip 1.0 are unchanged.

[Complete step records](../MoE_Routing_Followup-20261013-041132-668911/results/hard-routing-20261013-041132-668911/metrics.jsonl) · [Final summary](../MoE_Routing_Followup-20261013-041132-668911/results/hard-routing-20261013-041132-668911/summary.json) · [Config](../MoE_Routing_Followup-20261013-041132-668911/results/hard-routing-20261013-041132-668911/config.json) · [Checkpoint/source audit](routing_followup_audit.json)

### Policies and controlled changes

| Policy | Exact change | Routing after initialization |
|---|---|---|
| `hard` | Unmodified copied router rows; no balancing | Hard top-2 throughout |
| `noise` | Independent Gaussian noise in copied router weights, std 1% of the parent-router RMS per layer | Hard top-2 throughout; no token-level sampling |
| `bias` | Permanent detached selection-bias balancing, rate 0.001 | Hard top-2 throughout |
| `family_bias` | Same bias balancing, select two distinct parent families and one child per family | Hard family/child selection throughout |
| `warmup` | Gumbel top-k without replacement for 125 updates | Hard top-2 for the remaining 625 updates |

Router perturbation is applied **once**, independently across rows/layers; expert weights remain exact copies. Recorded layer noise standard deviations were 0.000208853, 0.000208608, 0.000216426 and 0.000210778. This breaks equal clone scores without introducing stochastic routing during continuation.

For selection-bias balancing, selection uses `router_logits + routing_bias`, but normalized mixture weights use **unbiased** selected logits. After each training forward, update `bias += 0.001*sign(mean(layer_counts)-expert_counts)`, then subtract the layer's mean bias. Bias is a non-gradient buffer, persists throughout training, and is frozen during evaluation. It prioritizes underloaded experts without an auxiliary loss. Family selection ranks mean child scores, then chooses one child from each selected family.

Family-aware growth preserved the parent function on the four-sequence growth probe: maximum logit drift **0**, start CE **4.123746**. Flat hard growth began at **4.147820**, with probe drift **3.612950**. The original clone study's 5.675947 drift used a larger calibration probe; those maxima are not computed over the same token set. Noise began at 4.147373. Final quality comparisons share the same validation/test sets, but family-aware initialization's lower start must be acknowledged when interpreting loss reduction.

### Final loss and exact dead-expert counts

| Policy | Start validation CE | Final validation CE | Test CE | Change versus hard | Lifetime / window / static-probe unused experts |
| --- | --- | --- | --- | --- | --- |
| hard | 4.147820 | 3.856335 | 3.880374 | +0.000000 | 16 / 16 / 16 |
| noise | 4.147373 | 3.835305 | 3.859548 | -0.021030 | 0 / 0 / 0 |
| bias | 4.147820 | 3.845077 | 3.862233 | -0.011258 | 0 / 0 / 0 |
| family_bias | 4.123746 | 3.850813 | 3.866495 | -0.005522 | 0 / 0 / 0 |
| warmup | 4.147820 | 3.839930 | 3.866121 | -0.016405 | 0 / 0 / 0 |

Counts are across **48 expert modules**, four layers × twelve. Every hard layer had `[2,5,8,11]` dead by lifetime exposure: **four per layer, sixteen total**. All other policies ended at **[0,0,0,0]** for both the final training window and static hard probe, with no lifetime-unexposed expert.

Router perturbation had the lowest final validation and test CE: **3.835305 / 3.859548**, versus hard **3.856335 / 3.880374**. The validation improvement is **0.021030 CE**. Bias balancing was slightly worse than perturbation on loss but considerably more balanced. Family-aware selection retained the parent mixture at conversion, yet did not deliver the best final quality. These are observed single-seed differences, not a universal ranking.

![Completed policy comparison: task loss and unused experts](routing_followup_comparison.png)

![Exact zero-traffic cells in the completed policies](routing_followup_zero_cells.png)

**Read the allocation heatmap:** black cells have exactly zero traffic in the final 50-update training window. Only `hard` has black cells—one third of each layer. The other four rows have no zero-traffic cells. Color intensity shows allocation fraction; nonzero does not imply uniform traffic. White separators divide layers, and indices start at zero.

### Did experts receive task gradients and actually update?

| Policy | Experts with CE gradients / Adam moments | Unchanged experts | Minimum lifetime routes | Minimum CE-gradient updates | Minimum gradient updates in final 50 | Effective-expert range per layer |
| --- | --- | --- | --- | --- | --- | --- |
| hard | 32/48 | 16 | 0 | 0 | 0 | 7.419–7.477 |
| noise | 48/48 | 0 | 1,897 | 506 | 19 | 7.184–8.561 |
| bias | 48/48 | 0 | 48,202 | 747 | 50 | 11.938–11.966 |
| family_bias | 48/48 | 0 | 58,306 | 748 | 50 | 11.893–11.963 |
| warmup | 48/48 | 0 | 14,403 | 620 | 16 | 7.681–9.356 |

The source snapshot optimizes CE without an auxiliary term, so gradient norms and Adam moments here are task-backed. Checkpoint review verified final token counters, positive-gradient counters, every state-bearing matrix's shape and Adam step, and nonzero second moments. Hard had **32/48** gradient-backed experts; each repaired policy had **48/48**. All repaired-policy experts changed weights, while sixteen hard experts remained exactly unchanged.

Persistent bias policies had every expert receiving positive CE gradients in **all 50 final updates**, and effective expert counts close to the maximum of twelve. Router noise eliminated strict dead experts but retained marked imbalance: effective counts were **7.184–8.561**. Its least-exposed expert received only **1,897 lifetime routes** versus a uniform expectation of 64,000. Bias and family-bias raised the minimum to **48,202 and 58,306**, respectively. Thus zero-dead status is weaker than robust balanced learning.

Metrics distinguish:

- **Exposure:** lifetime counts, last-token update and rolling allocations.
- **CE gradient:** norm before clipping, positive-gradient update counts, and recent-window counts. The conditional mean norm excludes updates with zero gradient.
- **Movement beyond decay:** every 25 updates, subtract the AdamW decay-only prediction from the actual post-update weights. For `grad=None`, the prediction is unchanged weights because AdamW skips decay too. Remaining movement includes current/past gradient moments, not merely the current gradient.
- **Function change:** relative weight drift and sibling output distances on fixed normalized parent-hidden probes shared across policies.
- **Held-out effect:** reset one expert to its original parent weights while retaining all other weights/router/bias, then re-evaluate a fixed validation probe.

The last sampled update contained **five zero expert-update norms under noise and two under warmup**. They are **not dead experts**: each had lifetime exposure, Adam state, positive gradients and nonzero allocations within the final window. A sparse expert need not be selected on every batch. Bias/family-bias had zero such last-update omissions. This distinction explains zero cells in an update heatmap without contradicting zero sustained dead experts.

### Does the learned expert help held-out loss?

Each expert was reset separately on four fixed validation sequences, **512 targets**, at the final checkpoint. Positive `CE_reset - CE_current` means learned expert weights help that probe/current model context; negative means resetting helps. This is not a full validation score or a statistical significance test.

| Policy | Reset hurts (>1e-6) | Reset helps (<-1e-6) | Negligible change | Zero decay-removed updates on last sampled update |
| --- | --- | --- | --- | --- |
| hard | 22 | 10 | 16 | 16 |
| noise | 25 | 19 | 4 | 5 |
| bias | 25 | 23 | 0 | 0 |
| family_bias | 28 | 20 | 0 | 0 |
| warmup | 34 | 10 | 4 | 2 |

All sixteen hard dead experts had zero ablation effect. Healthy training did **not** imply positive benefit for every expert: noise had 25 positive and 19 negative reset effects; bias had 25/23; family-bias 28/20; warmup 34/10, with remaining effects negligible. The warmup probe-local positive count was higher, although noise had better full validation/test CE. Small-probe effects are not additive and must not be summed into a model-quality estimate. Resetting early experts can change downstream hidden states/routes, which is part of the intervention.

| Detailed per-expert plot | What to inspect |
|---|---|
| [Hard expert heatmaps](../MoE_Routing_Followup-20261013-041132-668911/results/hard-routing-20261013-041132-668911/experts-hard.png) | Repeated zero exposure/gradient/update rows and zero reset effects |
| [Router-noise heatmaps](../MoE_Routing_Followup-20261013-041132-668911/results/hard-routing-20261013-041132-668911/experts-noise.png) | All experts learned, uneven traffic and occasional zero current-batch updates |
| [Bias-balancing heatmaps](../MoE_Routing_Followup-20261013-041132-668911/results/hard-routing-20261013-041132-668911/experts-bias.png) | Uniform exposure, task gradients and ongoing decay-removed updates |
| [Family-bias heatmaps](../MoE_Routing_Followup-20261013-041132-668911/results/hard-routing-20261013-041132-668911/experts-family_bias.png) | Function-preserving family growth and balanced expert use |
| [Warmup heatmaps](../MoE_Routing_Followup-20261013-041132-668911/results/hard-routing-20261013-041132-668911/experts-warmup.png) | Sustained hard-training survival with heterogeneous expert use |

### Cost of fixing utilization

| Policy | Seconds through final probes/telemetry | Peak allocated MiB |
|---|---|---|
| hard | 55.920 | 247.576 |
| noise | 74.208 | 296.032 |
| bias | 73.906 | 296.032 |
| family_bias | 75.011 | 296.032 |
| warmup | 73.935 | 296.032 |

The final timer includes regular validation/calibration and learning telemetry; final test/reset ablations and checkpoint saving occur afterward. Peak allocation includes the final diagnostic work. Repairs were approximately 1.32–1.34× the hard arm's timed continuation. The hard arm's lower cost partly reflects fewer expert dispatches/state-bearing parameters; it is not a useful-capacity efficiency advantage. All policies have the same selected FFN width and total stored capacity.

### Verified conclusion

The original failure reproduced exactly in loss and dead-expert pattern. **One-time router perturbation repaired hard routing without a sampled training phase.** Persistent bias balancing also repaired it and produced much more even learning opportunities. Family constraints prevented initial clone crowding but were not necessary for zero final dead counts. Sampled warmup remained an effective control. In this completed run, all four alternatives removed the observed lifetime/window/static-probe failure; none is claimed to guarantee future starvation-free routing at arbitrary scale.
