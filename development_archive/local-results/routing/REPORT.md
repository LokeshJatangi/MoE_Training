# Preliminary literal-linear results

Scope correction (11 October 2026): the user clarified that the intended routing baseline is a small dense GPT. This preserved result tests literal-linear upcycling; see `results/gpt-cpu-pilot/` for the current GPT experiment.

Measured locally on CPU; Colab execution remains pending. No hyperparameter search was performed for this result.

| Seed | Initial val CE | Trained linear | Continued dense | Continued MoE | MoE test CE |
|---|---:|---:|---:|---:|---:|
| 17 | 4.176091 | 2.342965 | 2.239687 | 2.157658 | 2.177699 |
| 42 | 4.176660 | 2.337062 | 2.239597 | 2.145183 | 2.181228 |
| 123 | 4.190873 | 2.335082 | 2.237464 | 2.143636 | 2.169030 |

Mean ± sample standard deviation across seeds (not a confidence interval):

- Initial validation: 4.181208 ± 0.008375
- Trained linear validation: 2.338370 ± 0.004101
- Continued dense validation: 2.238916 ± 0.001258
- Continued MoE validation: 2.148826 ± 0.007688
- Trained linear train: 2.261702 ± 0.004332
- Continued MoE train: 2.058425 ± 0.003367

Linear parameters: **67,665**. MoE total: **274,824**. Parameters in selected experts plus router per target: **139,494**.

## What passed

- Dense training and validation loss decreased in every seed.
- MoE training and validation loss decreased after conversion in every seed.
- Maximum conversion logit error: 9.5367431640625e-07.
- All experts changed weights; final routing fractions are plotted above.
- Saved model, optimizer and sampler states reproduced the next update (all parameter errors < 1e-6).

## Interpretation and limits

This completes the literal linear-to-MoE routing at small scale. The model predicts one next character from 16 preceding characters. Its logits are an affine map of a fixed one-hot encoding; it is not a Transformer. The routed mixture is nonlinear because the router depends on the input.

The MoE outperformed the continued linear control at equal target tokens in this experiment. This is not a compute-matched or total-parameter-matched superiority claim: two experts execute per target, the router adds work, and sparse dispatch adds overhead. The baseline was still learning at conversion; the task asks for continued learning, not convergence before conversion.

Each branch sees 204,800 additional next-character targets after 204,800 pretraining targets. Context reads are not counted as additional supervised targets. Evaluation uses fixed random samples of 4,096 windows per split, with replacement. Split raw text before making windows; vocabulary comes only from training text. The held-out test sample is evaluated only at final branch checkpoints. Tiny Shakespeare is a mechanism test, not evidence of general language ability or scaling performance.

Both continuation branches reset AdamW at the branch point, use learning rate 0.002, and consume identical sampled training windows. There is no loss smoothing or best-checkpoint selection. Curves show means and min/max seed bands. Task CE is reported separately from the balancing term (coefficient 0.01).

## Reproduce

```bash
python -m unittest discover -s tests -v
python -m src.train --device cpu --output results/reproduction
python scripts/report.py --results results/reproduction
```

Evidence: `config.json`, `data_manifest.json`, `environment.json`, `metrics.jsonl`, `metrics.csv`, `summary.json`. Local checkpoints are in `checkpoints/routing/`; these and the downloadable corpus are ignored by Git. Source SHA-256 values identify the executed implementation even when the initial Git commit is not yet present.

![Loss curves](loss_curves.png)
