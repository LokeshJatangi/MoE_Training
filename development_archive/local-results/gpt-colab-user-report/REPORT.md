# User-reported Colab dense phase and conversion failure

Source: pasted Colab subprocess output supplied by the user on 12 October 2026. This was not run or independently reproduced by the local agent. GPU type and the full runtime archive were not supplied; the training command used CUDA.

Seed 17 dense phase completed 1,000 steps / 2,048,000 next-character targets. Train CE 4.209708→1.781826; validation CE 4.216622→1.889444. Reported elapsed time including evaluation was 9.373663 seconds; no cross-hardware speed claim is made from this pasted output.

The job stopped at conversion, before dense-control or MoE continuation: `AssertionError: 2.2411346435546875e-05` against the old absolute-only limit `1e-5`. The output does not contain the reference logits or per-logit error distribution, so passing the new relative/absolute check cannot be inferred from this maximum alone. Expert copying is mathematically function-preserving; floating-point dispatch differences are a plausible explanation requiring the new diagnostic probe.

`reported_log.txt` preserves the pasted output, `metrics.reported.jsonl` contains its deduplicated dense records, and `provenance.json` identifies the source and failure. No successful MoE GPU result is claimed.
