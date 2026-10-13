# Continuation strategies and the clone-collapse lab

11 October 2026. The user requested our own experiments on a Colab T4/16 GB, inspired by a colleague's observations. We are not reproducing their exact model, dataset sample, drop/partition operators or parameter counts. Colab has not been accessed or run by this agent.

## Four independent choices

| Axis | Strategy | What it tests / tradeoff |
|---|---|---|
| Expert initialization | Exact copy | Reuses learned behavior. Dense-to-MoE conversion can preserve the function with normalized gates; later flat expert-to-more-experts growth is not automatically function preserving |
| Expert initialization | Copy plus small perturbation | Breaks symmetry quickly; introduces conversion drift. Test expert noise and router noise separately |
| Expert initialization | Partial reinitialization or dropout-style masking | Adds diversity at the cost of learned information. Define whether “drop” means zero, mask, prune, or reinitialize before comparing results |
| Expert initialization | Partition trained hidden units | Splits capacity across experts. For SwiGLU, coordinate gate/up output rows with down input columns. Active/all-expert sums and scaling determine preservation |
| Expert initialization | Shared trained block plus routed additions | Preserves a common path, but shared + copied outputs can double the initial function unless contribution scaling is specified |
| Routing | Immediate hard top-k | Strong baseline for exposing clone starvation and tie effects |
| Routing | Fixed sampled-routing warmup | Gives clones opportunities to update, but the duration does not establish hard-routing survival |
| Routing | Gradual exploration decay | Avoids an abrupt switch; may still starve experts as exploration approaches zero |
| Routing | Probe-gated switch | Candidate switch based on hard-routing behavior and minimum exposure, followed by sustained hard training. No universal threshold |
| Routing | Permanent exploration | May retain coverage; changes training behavior and can leave a mismatch with hard inference |
| Routing | Auxiliary balancing | Encourages use through gradients; coefficient can interfere with the task |
| Routing | Detached routing-bias balancing | Changes selection using measured load while leaving mixture weights unbiased; requires update-scale tuning |
| Routing | Family-aware selection | Select distinct parent families first, then a child. Controls sibling competition; adds a hierarchy that can limit freedom |
| Optimization | Reset AdamW | Simple and controlled; removes old momentum and second moments |
| Optimization | Copy optimizer state | Preserves history but copied moments may be poorly matched to new token distributions. Specify inherited/new parameter treatment |
| Optimization | Lower/re-warm LR, separate router/new-expert LR | Controls transition scale; compare LR schedules separately from routing |
| Trainable weights | Joint training | Backbone, router and experts can co-adapt; may hide which component repaired a bad conversion |
| Trainable weights | Freeze backbone initially | Isolates routing/new experts; reduces adaptation freedom |
| Trainable weights | Router-only or alternating phases | Diagnoses allocation versus expert capacity. Identical experts may supply little router task signal initially |

The first lab fixes exact copying, top-k=2, dropless dispatch, full joint training and optimizer reset. Vary routing before comparing copy/drop/partition. The existing GPT uses nonlinear SwiGLU experts; it is distinct from the colleague's described one-linear-map FFN.

## Corrections to the explanation supplied by Gemini

1. **Identical expert weights do not imply identical routing logits.** Scores come from router parameters and token hidden states. Scores tie within a family when router rows/biases are cloned identically. Randomly initialized router rows are a different case.
2. **Sampled sparse routing is not the same as dense soft routing.** This lab uses Gumbel-top-k sampling without replacement to choose k experts, then normalizes their unbiased gate logits. The discrete choice is not differentiated; this is not a dense all-expert mixture or a straight-through Gumbel-Softmax estimator.
3. **Stochastic selection does not guarantee every expert receives tokens or a nonzero task gradient.** Highly concentrated scores, finite batches and symmetry can still cause starvation. We record routings rather than assuming coverage.
4. **A hard-top-k loser is not necessarily permanently dead.** Auxiliary gradients, router updates from other tokens, changing hidden states or explicit biases can later change the ranking. Persistent starvation must be measured over time.
5. **Weight distance is not a sufficient switch rule.** Different matrices can implement similar functions, and a router can ignore distinct experts. Output differences on common probes are more informative, but still do not prove useful routing or specialization.
6. **Flat expert growth can change the function immediately.** If a top-2 parent route selected A and B, cloned flat routing can instead select A1 and A2. The sum then approximates A rather than the former mixture of A and B. The lab records growth logit drift; family-aware routing is a preservation control.

Primary references: [Sparse Upcycling](https://arxiv.org/abs/2212.05055), [Loss-Free Balancing](https://arxiv.org/abs/2408.15664), [Router Upcycling](https://arxiv.org/abs/2509.00679). These support studying initialization, routing and load balance; they do not supply a universal warmup step count for our proposed model. An [expert-expansion implementation](https://github.com/amazon-science/expert-upcycling) also documents router perturbation and ongoing balancing; it is a reference, not code copied into this lab.

## Our T4 experiment

Dataset: frozen small TinyStories sample, byte-level BPE trained on training stories only, target vocabulary 4,096. Preparation records the dataset revision, tokenizer hash, token counts and exact document cross-split deduplication. Training stories are sampled with a bounded shuffle buffer; this is not an unbiased sample of the complete corpus. Near duplicates are not removed. Documents are concatenated with EOS and attention may cross EOS.

Starting GPT: four layers, width 192, six attention heads, context 128, SwiGLU intermediate width 384, tied embedding/output weights. Approximately 2.3M dense parameters for a 4,096-token vocabulary. Train dense, convert to four experts per layer and train that parent MoE. Grow independent branches from the SAME trained four-expert checkpoint:

`4 → {12, 36, 108, 324, 972}`

These are alternative growth branches, not a sequential chain. Hold top-k=2, microbatch=4, FP32 and data order fixed. Counts do not imply corresponding parameter multipliers because attention/embeddings are unchanged. Every arm logs exact total/active counts. Larger stages may be too slow or too large; that is an outcome to record.

| CLI arm | Training policy |
|---|---|
| `hard` | Hard top-k from step zero, no balancing |
| `warmup` | Fully sampled routing through step 124, then hard routing |
| `anneal` | Per-token probability of sampled routing decays linearly from 1 to 0 across 500 steps |
| `adaptive` | Sample until recent hard probes cover every expert and each expert has ≥1,000 cumulative routings; earliest switch 125 steps. This criterion is an experimental heuristic, not a proven rule |
| `aux` | Hard routing with .01 auxiliary balancing |
| `bias` | Hard routing with selection-only bias feedback, update rate .001 |
| `family` | Select two distinct families, sample children during the 125-step warmup, then hard child selection |

Initial notebook run: only `hard`, `warmup`, `anneal` at 12 experts, capped at 3 arm runs. An explicit later notebook cell enables the larger ladder and controls. Large jobs are not launched automatically by the agent.

Each arm has 750 continuation steps and a ten-minute wall-time budget including evaluation; all budgets are starting choices. Optimizer states are reset in every growth arm. Current training is a diagnostic implementation with a Python expert loop, not a fast grouped-GEMM MoE stack. Poor throughput can become the first scaling limit.

Before training an arm, estimate conventional full-state parameter/Adam storage as 16 bytes × total parameters and reserve 25% of currently free GPU memory for other allocations. The estimate cannot guarantee a fit. Estimated memory skips are labeled separately from actual CUDA OOM. CUDA OOM, runtime errors and exhausted time budgets are recorded without rewriting earlier metrics. The parent checkpoint remains on CPU, so several large GPU branches are not resident simultaneously.

## How to decide whether a warmup worked

Track per layer and clone family:

- Zero routings over the last **50 training steps**, and low-utilization counts/coverage derived from stored fractions.
- Hard-route coverage on fixed **training calibration** sequences, even while training is stochastic; validation is not used to choose the switching point.
- Cumulative routings and sampled within-family output distance on the same hidden probes (up to eight evenly spaced siblings per family).
- Hard-routing validation CE, growth logit drift, total/active parameters, wall time and peak allocated GPU memory.
- Step of the switch and duration of actual hard training afterward.

A candidate switch requires more than random-training coverage. Follow it with at least several hundred hard-training steps. If starvation returns, record the failure rather than restarting the same run until it looks successful. Longer random warmup might help, but persistent balancing or a changed routing topology may be required. The lab currently does not revert a failed switch automatically, which keeps the failure visible.

For N token vectors and k selected experts among E, independent uniform routing would give about `Nk/E` routings per expert, and expected zero-routing experts about `E(1-k/E)^N`. Real LM tokens are correlated, so this is a rough sampling sanity check, not a statistical test. With microbatch 4 × context 128 and E=324, a last-step zero count can be unsurprising; a long rolling-window zero count is more informative.

A result can be: learned but underutilized; balanced but no quality gain; switch never became ready; switch appeared ready but later collapsed; memory limited; throughput limited; or stable within the tested horizon. “No dead experts” alone is not a success criterion for an LLM.

## Execution status

Code and notebook preparation occur locally. Only targeted tests and a tiny CPU toy-token execution smoke are authorized as local checks for this lab. No TinyStories language-training result or Colab result is claimed until the notebook runs on the user's GPU. Three-seed follow-up and exact optimizer/recovery equivalence for the new stochastic routing runner are still required before research conclusions.
