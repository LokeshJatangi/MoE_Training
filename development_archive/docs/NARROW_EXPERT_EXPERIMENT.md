# Narrower experts with a larger expert count

Current task, agreed 12 October 2026: partition a trained dense GPT FFN into smaller experts, increase the expert count, and study conversion and continued training. Sparse activation and expert granularity are separate choices. Our earlier full-copy experiment remains a control.

## Frozen first comparison

Run `notebooks/Narrow_Experts_T4.ipynb` on Colab with GPU enabled. All three branches start from the same dense checkpoint. Each uses the same additional sample order, reset AdamW, token budget and optimizer settings. The dataset is a bounded TinyStories subset with train-only BPE (4,096 vocabulary target), exact cross-split document deduplication and recorded dataset revision. No TinyStories or Colab execution has happened locally.

| Branch | Individual intermediate width | Experts | Top-k | Selected intermediate units | Stored FFN weights relative to dense |
|---|---:|---:|---:|---:|---:|
| Dense continuation | 384 | 1 | 1 | 384 | 1× |
| Full-copy MoE | 384 | 4 | 1 | 384 | 4× |
| Narrow partition MoE | 96 | 16 | 4 | 384 | 4× |

Every expert still accepts and emits the same residual width, 192. Only its intermediate width shrinks. The stored FFN capacity is equal between the two MoEs; the narrower-expert router is larger. FFN matrix multiply-accumulates per token per layer are `3*d*k*h_expert`. This excludes attention, activations, selection, dispatch, routing and optimizer work. Matching that quantity is not a runtime or full-training-compute guarantee. Resident expert parameters still require optimizer memory even when inactive for a token.

## Copy and scale

For our bias-free SwiGLU, `f(x) = down(silu(gate(x))*up(x))`. Split the dense intermediate units into four contiguous groups. Copy corresponding rows of gate/up and columns of down into each expert. Four replicas of each group give 16 narrow experts. These are copies of parameter subsets, not compressed approximations of the complete dense function.

The sum of one expert from every partition reconstructs the original FFN. Our router uses softmax weights over selected experts, so the narrow mixture is multiplied by the number of partitions. Uniform weights over one of each partition would reconstruct the sum. Flat top-4 does not enforce distinct partitions, and learned unequal mixture weights also change the sum. Conversion therefore need not preserve logits. The scale is fixed in advance; alternative scales can be separate ablations, not silently tuned using held-out results.

## Evidence and acceptance

Measure dense pretraining loss, conversion logit drift and CE jump, post-conversion CE reduction, recovery below the dense checkpoint's CE, expert routing fractions, expert weight changes, finite gradients, checkpoint recovery, memory and elapsed time. Report failures without overwriting them. Single-probe zero allocations do not prove persistent death. Use the clone-routing lab for rolling starvation and exploration policies after establishing this granularity baseline.

All branches are jointly trained; there is no freeze stage or sampling warmup here. Auxiliary balancing has coefficient 0.01. Full-copy top-1 is a deliberate compute control; its task gradient through the selected softmax weight is zero, while the balancing objective trains its router. This differs from our earlier top-2 baseline.

## Local CPU pilot: measured, one seed

Tiny Shakespeare, two layers, width 64, dense FFN 128; narrow experts width 32, 16 experts, top-4. Seed 17, 150 dense steps plus 150 per continuation. FP32 CPU execution, 1,024 next-character targets/step.

- Dense validation CE: **4.190773 → 2.624921**.
- Dense continuation: **2.505817**.
- Full-copy continuation: **2.624921 → 2.511586**.
- Narrow-expert continuation: **2.655241 → 2.498899**, including a **0.030320** conversion CE increase.

The narrow branch continued learning and recovered the dense checkpoint's loss. A small one-seed difference does not establish superior quality or efficiency. See `results/granular-cpu-pilot/REPORT.md` and `summary.json` for complete measurements. Fourteen repository tests passed; the notebook embeds seven GPT/partition tests. Partition reconstruction, coordinated copying, sparse dispatch, gradients and parameter accounting are tested. Every training phase's saved model/Adam/sampler reproduced the next update.

```bash
python3 -m src.granular_experiment --config configs/granular_cpu.json --data data/tinyshakespeare.txt --device cpu --output results/new-granular-cpu
# On Colab after preparing TinyStories:
python3 -m src.granular_experiment --config configs/granular_t4.json --data data/tinystories_lab.npz --device cuda --output results/new-granular-t4
```

The notebook is self-contained, embeds source, refuses result overwrites and downloads an evidence archive. Its optional finer comparison uses eight partitions, width 48, 32 experts and top-8: selected width and stored FFN weights remain constant while router/dispatch costs grow. Later vary replicas to increase total capacity independently. Confirm across seeds before claiming differences. The Colab notebook was syntax-checked, not GPU-executed.
