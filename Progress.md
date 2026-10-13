# Progress — completed Colab study

Updated: 12 October 2026 (Asia/Kolkata).

## Completed evidence

- Character dense GPT → four full-size experts/top-2, seed 17: dense validation 4.216622→1.889442; continued dense 1.691272; continued MoE 1.657208. Final tests 1.899665 dense / 1.876497 MoE.
- TinyStories narrow-expert comparison: dense checkpoint 4.278877; full-copy top-1 final 4.081297; narrow conversion 4.350333→4.065797; continued dense 4.051691.
- TinyStories four→twelve expert growth: hard/warmup/anneal completed 750 updates. Final hard-validation CE 3.856335/3.839930/3.848049. Hard routing left four experts/layer with zero lifetime tokens, unchanged weights and no Adam state; warmup/anneal gave every expert exposure and state.

All uploaded experiments ran on Tesla T4, PyTorch 2.11.0+cu130, Python 3.13.15. Uploaded source snapshots remain unchanged. Seventeen environment-recorded source hashes match; both TinyStories tokenizer/manifests match. Optimizer states and cumulative exposure were inspected on CPU using map_location='cpu'. Derived audits and regenerated plots are in analysis/.

## Submission documentation

README.md now starts with a terse summary and result/size tables, followed by completed experiment details: datasets, shapes, transfer mathematics, optimizer/reset/replay states, route schedules, compute and memory, tests, reproduction and resolved errors. Static imported report device labels are corrected by the authoritative environment-backed README, without editing raw evidence.

Earlier development records and removed exploratory notebook branches were preserved outside this project rather than deleted. Runnable notebooks/configs are restricted to the documented completed studies. Corpus provenance now uses data_specs/tinyshakespeare.json independently of earlier development files.

## Engineering resolutions

- Persistent streamed child stdout/stderr replaced opaque subprocess wrapper errors.
- Full-copy conversion retained exact weight equality and added independent FP64 CPU validation with bounded FP32 error checks; completed T4 FP64 error was 3.463896e-14 with zero CE change.
- Checkpoint discovery avoids stale hard-coded paths and can restore dense pretraining before branching into reset optimizers.
- CPU checkpoint review explicitly maps CUDA tensor storages to CPU.
- Evidence/source/checkpoint audits distinguish utilization failure, conversion recovery and numerical roundoff.

## Validation

- All 26 mathematical/logging/recovery tests passed after final packaging changes.
- Completed source hashes and tokenizer checksums verified by scripts/audit_colab_evidence.py.
- Plots regenerated from raw metrics and visually inspected.
- README local links, tensor/parameter arithmetic, summary-derived result tables and forbidden legacy references checked.
- No new training was launched while preparing this report.


## Completed follow-up consolidated — 13 October 2026

Uploaded package MoE_Routing_Followup-20261013-041132-668911 contains five completed 750-update T4 arms. Audited all four recorded source hashes, parent checkpoint digest, packed-data digest and 155 metric records. Loaded all five checkpoints on CPU and verified exposure/gradient counters, Adam matrix shapes/state/counters and nonzero second moments. All repaired policies have 48/48 task-backed experts; hard has 32/48, with sixteen zero lifetime/current-window/static-probe experts.

Noise/bias/family_bias/warmup final CE: 3.835305/3.845077/3.850813/3.839930; hard 3.856335. Bias policies effective expert counts near twelve, whereas noise wins loss with less uniform traffic. Reset-ablation signs remain mixed; no universal per-expert usefulness claim.

Consolidated the main README's headline tables, detailed completed methods/metrics, optimizer interpretation, runtime, expert plots and conclusions. Added source/checkpoint audit and comparison/zero-cell figures; visually inspected both. Updated the sibling follow-up README/status from prepared to completed. Original uploaded evidence and prior completed records remain unchanged. No training rerun needed.
