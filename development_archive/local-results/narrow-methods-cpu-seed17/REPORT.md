# Single-seed narrow-expert initialization comparison

Measured **CPU** execution, Tiny Shakespeare, seed 17. Same trained dense checkpoint (validation CE **2.624921**), same sample order and 300 continuation updates per branch. Each branch resets AdamW and sees 307,200 additional next-character targets. Original dense pretraining saw 153,600 targets.

| Method | CE immediately after conversion | Final validation CE | Conversion CE jump | Recovers dense checkpoint | Final fixed-probe zero experts by layer |
|---|---:|---:|---:|---|---|
| dense_control | 2.624921 | 2.460981 | +0.000000 | True | [] |
| full_copy | 2.624921 | 2.466103 | +0.000000 | True | [0, 0] |
| contiguous | 2.655241 | 2.446182 | +0.030320 | True | [0, 0] |
| shuffled | 2.660617 | 2.451317 | +0.035696 | True | [0, 0] |
| drop | 2.656356 | 2.456125 | +0.031435 | True | [0, 0] |
| perturbed | 2.655223 | 2.450853 | +0.030302 | True | [0, 0] |
| random | 3.729246 | 2.448603 | +1.104324 | True | [0, 0] |

![Measured loss and load curves](curves.png)

## Definitions

- `contiguous`: disjoint adjacent groups of trained intermediate units; four replicas of each group.
- `shuffled`: disjoint groups after a fixed random permutation; four replicas of each group.
- `drop`: each expert independently retains 32 of the 128 trained units without replacement. Different experts may overlap; no complete-coverage guarantee.
- `perturbed`: contiguous copies plus independent Gaussian noise at 1% of each copied matrix’s RMS.
- `random`: newly initialized narrow expert weights; trained attention, embeddings and norms are retained. This is not an entirely random GPT.
- `full_copy`: four complete width-128 experts/top-1. `dense_control` continues the original GPT.

## Controls and interpretation

All five narrow variants have width-32 experts, 16 experts and top-4, normalized selected gates, output scale 4 and balancing coefficient 0.01. Their routers start with identical weights across methods. This isolates expert initialization. Different post-conversion hidden states can still produce different initial routing decisions. Drop is our explicitly defined subset-retention method, not a claimed reproduction of a colleague’s implementation.

Every branch selects 128 intermediate units per token (24,576 FFN matrix MACs per token/layer). Dense has 90,816 total parameters; full-copy has 238,784 and narrow variants have 240,320. Routing, dispatch and optimizer overhead differ, so no equal-runtime claim is made. Full-copy top-1 receives router task gradient zero through its selected softmax weight; its auxiliary objective still trains the router.

| Method | Active parameter convention | Wall seconds including evaluation | All expert matrices updated | Checkpoint next-update parameter error |
|---|---:|---:|---|---:|
| dense_control | 90816 | 5.07 | None | 0 |
| full_copy | 91328 | 7.39 | True | 0 |
| contiguous | 92864 | 13.33 | True | 0 |
| shuffled | 92864 | 12.85 | True | 0 |
| drop | 92864 | 13.71 | True | 0 |
| perturbed | 92864 | 14.39 | True | 0 |
| random | 92864 | 13.29 | True | 0 |

The active convention counts the full tied embedding table and nonexpert weights plus selected expert weights; it is not a FLOP count. Per-matrix changes, train CE, gradient norms, router entropy, allocation fractions, unit mappings and final test CE are in summary.json/metrics.jsonl. Source and checkpoint hashes are in environment.json. Test was evaluated after all predeclared arms completed and was not used to choose settings.

Fixed-probe zero counts are not rolling starvation measurements. A parameter change alone does not prove useful specialization; weight decay also changes parameters. Finite task/gradient checks and reproducible next optimizer updates passed during training. The comparison is a one-seed small-data mechanism study; do not infer a robust method ranking or frontier-scale behavior from small differences. Colab/TinyStories runs remain unexecuted.
