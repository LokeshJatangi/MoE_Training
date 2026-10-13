# User-reported T4 FP32 conversion diagnostic failure

Seed 17 recovered its completed dense checkpoint on Tesla T4, PyTorch 2.11.0+cu130. The scale-aware per-logit check still rejected six logits. Max absolute error 3.3378601e-5; RMS 1.2258761e-6; maximum probability error 3.5166740e-6. Dense/MoE probe CE both 1.8894422054290771, with zero reported change. MoE continuation was not reached.

These measurements come from user-pasted output and are not independently reproduced by the agent. This output does not contain an FP64 reference check. The subsequent implementation adds that independent verification and bounded FP32 criteria; no successful GPU result is claimed yet. Original rejected diagnostics are preserved unchanged in conversion.reported.json.
