# Hard-routing follow-up — complete

Finalized 13 October 2026.

All five policies completed 750 continuation updates on Tesla T4 using the same pinned parent, tokenizer and sampled batches. Source hashes, parent/data checksums and all optimizer/exposure checkpoints were audited.

Hard routing reproduced the failure: 16/48 experts never received tokens/CE gradients/Adam state and stayed unchanged. Noise, bias, family_bias and warmup each finished with zero lifetime/window/static-probe dead experts and task-backed Adam second moments in all 48 experts. Noise had best validation/test CE (3.835305/3.859548); bias and family_bias had the most even traffic.

Main consolidated README and zero-cell/comparison figures are in the sibling MoE_Training project. This README now reports actual completed T4 results. Synthetic smoke records remain separate historical execution checks. Four local invariant tests preceded the run; no new training was needed to audit/finalize results.
