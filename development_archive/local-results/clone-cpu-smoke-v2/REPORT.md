# Clone routing lab outcomes

All validation and calibration probes use HARD routing, even when training is stochastic. CPU smoke results on toy tokens only verify execution; they do not answer the language-training question.

| Arm | Status | Total params | Switch | Post-switch steps | Final hard-probe zero count |
|---|---|---:|---:|---:|---|
| E12-hard | completed | 20,080 | 0 | 10 | [0] |
| E12-warmup | completed | 20,080 | 4 | 6 | [0] |
| E12-anneal | completed | 20,080 | 8 | 2 | [1] |
| E12-adaptive | completed | 20,080 | 4 | 6 | [0] |
| E12-family | completed | 20,080 | 4 | 6 | [1] |
| E12-bias | completed | 20,080 | 0 | 10 | [0] |
| E12-aux | completed | 20,080 | 0 | 10 | [0] |

A zero expert count on one batch is not permanent death. The training-window metric aggregates the last configured number of steps; the hard-probe metric aggregates recent fixed calibration probes. Neither guarantees future survival. Output distances are sampled within clone families and are diagnostics, not a validated switching threshold.

Estimated memory skips, CUDA OOM, runtime failures and time limits are different outcomes. No GPU result should be inferred from the local smoke run.

![Clone curves](clone_curves.png)
