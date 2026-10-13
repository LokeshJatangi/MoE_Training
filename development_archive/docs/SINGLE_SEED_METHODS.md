# Single-seed continuation methods

The user authorized continuing locally with one seed and different methods on 12 October 2026. This experiment varies narrow-expert initialization, keeping hard top-4 routing and training settings fixed. It complements the separate routing-policy laboratory.

## Configuration frozen before execution

Start from `checkpoints/granular-cpu-pilot/17/dense_pretrain.pt`, the trained dense GPT after 150 steps. Its dataset checksum is verified against the original evidence. Two layers, residual width 64, intermediate width 128, context 64, Tiny Shakespeare character vocabulary 65, FP32 CPU, seed 17.

Each narrow arm has 16 width-32 experts, top-4 and output scale 4. All parameters train jointly. Reset AdamW in every branch: learning rate 0.0005, betas (0.9, 0.95), weight decay 0.01, gradient clipping 1.0, balance coefficient 0.01. Continue for 300 steps of 16 sequences × 64 targets, using the same sampler seed and batches. Fixed train/validation probes use 32 sequences. Final test loss is calculated only after all arms finish; it is not used to select methods.

| Method | What changes in the expert bank |
|---|---|
| Contiguous partition | Disjoint adjacent unit groups, four replicas of each |
| Shuffled partition | Fixed permutation before dividing into disjoint groups, four replicas of each |
| Independent subset/drop | Retain a separately sampled quarter of trained units per expert; overlap across experts permitted |
| Perturbed partition | Contiguous partition with Gaussian noise at 1% of each copied matrix's RMS |
| Random experts | Initialize the narrow experts afresh; retain the trained GPT attention, embeddings and norms |

Dense continuation and full-copy top-1 are controls. All five narrow variants have equal parameter counts and identical initial router matrices. Initialization can change hidden states and thus initial token routings. Our drop definition is explicit and independent of the colleague's unavailable code.

Partition copying always selects corresponding gate/up rows and down columns. Shuffled disjoint partitions still reconstruct the full FFN when summed once each. Independent subsets may overlap or leave units uncovered. Flat routing and unequal gate weights can change predictions for every partition/subset method. Random experts provide a control for whether transferred FFN weights help recovery.

Noise tests initial clone symmetry, but does not guarantee routing balance or specialization. Fixed-probe loads describe that probe only; this run does not measure rolling starvation. The five initialization methods use the same routing policy; family constraints and probabilistic warmups should be subsequent, separately controlled comparisons.

```bash
python3 -m unittest discover -s tests -v
python3 -m src.compare_narrow_methods --output results/new-single-seed-comparison
python3 scripts/report_narrow_methods.py --results results/new-single-seed-comparison
```

The runner refuses to overwrite results and saves per-arm checkpoints, source/checkpoint hashes, unit indices, loss curves, parameter changes and next-update recovery checks. The original dense checkpoint must be present; use the earlier granular CPU command to regenerate it in a new directory and point the config at that new checkpoint/evidence if starting from a fresh checkout. Checkpoints and corpus remain ignored by Git.

Measured results: `results/narrow-methods-cpu-seed17/REPORT.md`. One seed and a small corpus support a mechanism demonstration, not a reliable ranking of methods or a claim about larger language models. TinyStories and Colab remain unexecuted.

## What this run taught us

All seven branches reduced validation CE and recovered the trained dense checkpoint's 2.624921 CE. Final CE: dense 2.460981, full-copy 2.466103, contiguous 2.446182, shuffled 2.451317, drop 2.456125, perturbed 2.450853, random 2.448603.

Transferred expert weights protect the starting function much better: conversion CE jumps were about 0.03 for copied narrow variants versus 1.10 for random experts. At step 50, contiguous copying had CE 2.575785 versus random 2.697639. Random experts caught up substantially by step 300. The trained non-FFN components stayed intact in the random arm, and the corpus/model are small; this is not evidence that expert transfer is unnecessary at scale.

Small perturbations did not improve final CE over contiguous copying in this seed. Shuffling and independent subsets also did not improve it. These are observations from the frozen run, not a robust ordering. All experts had nonzero final fixed-probe loads and all expert matrices changed. Loads were still unequal, and no persistent-starvation conclusion follows from a single probe.

The CPU narrow branches took about 13–14 seconds each, including evaluation, versus about 5 seconds dense and 7 seconds full-copy. More Python expert dispatches and a larger router add overhead. These timings are not optimized GPU benchmarks. Active FFN width is an architectural compute proxy, not a speed guarantee.

Next controlled axis: keep the contiguous weight transfer fixed and compare routing exploration/family constraints while growing replicas. Preserve this experiment and its budget; use a new config/result directory for that study. No new larger-data or GPU run has been started.
