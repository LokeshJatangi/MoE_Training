# What we trained, why it learned, and what remains uncertain

Updated 11 October 2026 after the user clarified that the routing's starting model means a **small dense GPT**. The earlier literal affine model is now a preliminary routing experiment. The current relevant routing result is `results/gpt-cpu-pilot/`.

## Two experiments, with different questions

| | Preliminary literal-linear test | Current GPT pilot |
|---|---|---|
| Input | 16 preceding characters, fixed one-hot features | A sequence of 64 character IDs |
| Dense model | One affine predictor | 2 causal Transformer blocks, width 64, 4 attention heads, SwiGLU FFNs |
| Objective | Predict one next character per example | Predict the next character at every position |
| Dense parameters | 67,665 | 90,816 |
| Conversion | Clone whole affine predictor | Clone each block's complete trained FFN; preserve attention/embeddings/norms/head |
| MoE | Four experts, top-2 | Four experts per FFN, top-2 at every token |
| MoE total / active convention | 274,824 / 139,494 | 238,784 / 140,480 |
| Seeds measured | 17, 42, 123 | 17, CPU pilot only |
| Validation CE | 4.181 → 2.338 → 2.149, seed means | 4.191 → 2.484 → 2.367 |

Both use character tokens. Despite using the same corpus, the losses are not directly comparable: the GPT is trained and evaluated across sequence positions with varying available context; the affine model always receives a full fixed window for a single target. The linear experiment's multi-seed improvement cannot be transferred into a GPT quality claim.

## What a dense GPT means here

Dense means all of a block's ordinary FFN weights participate for every token. A dense GPT already contains embeddings, causal attention, residual paths, normalization and nonlinear FFNs. It is not a mathematical linear model. “FFN” is a component inside the GPT; MoE conversion replaces that component rather than replacing the complete language model.

The GPT pilot uses learned positional embeddings, pre-normalized residual blocks, causal scaled-dot-product attention, SwiGLU feed-forward blocks and tied embedding/output weights. No dropout or mixed precision is used in the first baseline, which simplifies conversion and recovery checks.

For width `d=64`, intermediate width `h=128`, one expert contains three matrices:

- Gate: `[128, 64]`
- Up: `[128, 64]`
- Down: `[64, 128]`

It computes `down(silu(gate(x)) * up(x))`. Each expert therefore has `3dh = 24,576` parameters. Copying all three matrices matters. A router maps a token's hidden vector `[64]` into four scores. It selects two experts and normalizes their scores into mixing weights. Each selected expert receives only its assigned token vectors; the outputs are weighted and combined back into their original sequence positions.

If all cloned experts initially implement `F(x)`, then `a1*F(x) + a2*F(x) = F(x)` because `a1+a2=1`. Attention, norms, embeddings and output head are copied too. Full-GPT logits agreed within **2.62e-6** after conversion, under the asserted 1e-5 tolerance.

## The dataset and the meaning of loss

Tiny Shakespeare contains **1,115,394 characters**, approximately 1.1 MB. The training vocabulary is 65 characters, including whitespace, punctuation and letters. We split raw text into 80% train, 10% validation, 10% test before constructing sequences; windows cannot cross these boundaries.

In the GPT, `x = text[i:i+64]` and `y = text[i+1:i+65]`. Causal attention prevents prediction at position t from reading positions later than t. Cross entropy measures the probability assigned to the correct next character. Uniform guesses over 65 characters give roughly `ln(65)=4.174`; actual initialized logits give a nearby value. Lower CE means the model assigns more probability to true next characters.

The CPU pilot used 16 sequences × 64 targets = **1,024 supervised character targets per step**. It trained dense for 300 steps, then each branch for 300 additional steps: **614,400 target exposures per branch path** in total. These are sampled exposures, not guaranteed unique characters. Evaluation used 32 fixed sequences per split, so the pilot estimate is small and windows within a split can overlap.

This corpus is sufficient to find implementation failures and show continued learning. It is too narrow to assess broad language knowledge, code/math ability, robust expert specialization or scaling laws. Character-level tokenization is convenient for inspection; the next larger language experiment should use a fixed subword tokenizer and report its own token-based metrics.

## Why MoE can get stuck, and what our evidence says

| Issue | Mechanism | What we did / what remains untested |
|---|---|---|
| Few experts get all the traffic | Router preferences reinforce themselves; inactive experts get little task gradient | Four experts and top-2 routing, plus coefficient .01 balance loss. Final GPT fractions: layer 0 ≈18%,22%,24%,36%; layer 1 ≈22%,23%,26%,28%. All expert matrices updated. Balance is imperfect, and specialization is not proven |
| Router task gradients vanish at conversion | Identical expert outputs make changing their mixture initially useless | Random router scores create unequal token routings; experts then receive different gradients. The balancing objective also supplies router gradients. Exact cloned symmetry is preserved initially rather than hidden with noise |
| Top-1 renormalization kills router gradients | A single selected normalized weight is identically 1 | We used top-2. The tests demonstrate gradient flow for distinct expert outputs. A later top-1 experiment must explicitly choose its gate/scaling rule |
| Conversion causes a loss jump | Random experts, missing matrices or unnormalized gates change the function | Clone trained complete FFNs, normalize selected weights and verify full-model logits. No observed material jump |
| Token dropping loses training signal | Capacity limits reject tokens assigned to overloaded experts | Dropless dispatch; every token reaches two experts. This avoids capacity handling, so we have not learned that failure mode yet |
| Balancing improves while language loss worsens | Auxiliary gradients can compete with the task | Task CE is logged independently from balance and total objective. We used a small coefficient; its optimality is not established |
| Loss spikes or unstable updates | LR, attention logits, numerical precision and uneven gradients can interact | Lower continuation LR, FP32 and gradient clipping; finite-gradient checks. We have not stress-tested FP8 or large-scale instability |
| Bad comparison masquerades as an MoE gain | More active work or optimizer/data differences favor one branch | Dense and MoE reset AdamW and consume identical additional batches. Tokens match; FLOPs, wall time and total parameters do not match |
| Sparse model is slower in practice | Dispatch and small matrix multiplies dominate; cross-device communication adds cost | In this pilot, measured MoE continuation took ~9.03s versus dense ~4.80s including evaluation. This is CPU evidence, not a GPU prediction |
| Expert states exhaust memory | Every expert still has weights, gradients and optimizer state | Tiny models fit locally; total and active counts are reported separately. We have not tested distributed sharding/offload |
| Multi-GPU stragglers/communication | Load skew creates unequal work; dispatch needs exchanges between devices | No expert-parallel communication in this experiment; those problems were absent, not solved |
| Memorization or narrow data hides problems | A small homogeneous corpus can give misleading quality gains | Held-out splits and test measurements reduce leakage risk. Larger and multi-domain data plus intervention tests are needed for broader conclusions |

The GPT's final validation losses were **2.37348 dense continuation** and **2.36706 MoE continuation**. This difference is only about **0.00643** in one seed. The supported conclusion is continued learning after conversion. A meaningful quality advantage requires repeated seeds and fair resource comparisons.

## When to grow data and model size

Increasing both at once makes the cause of a change hard to identify. Use this progression:

1. **Tiny Shakespeare:** establish masks, gradients, conversion and recovery; reproduce the notebook across three seeds.
2. **TinyStories subset:** propose 10–50M subword training tokens and a ~5–10M dense GPT. Keep tokenizer, official validation-derived holdouts and token budgets fixed; compare dense continuation versus upcycling. Start with a bounded pilot on the actual Colab GPU.
3. **Broader corpus such as FineWeb-Edu:** use a small frozen shard and distinct held-out documents when testing broad text and routing across domains. A larger corpus alone does not create a good domain-specialization test; label and evaluate the domains deliberately.
4. Increase model size only after measured underfitting, learning curves and hardware capacity justify it. Compare iso-token and eventually iso-FLOP runs separately.

If training CE keeps falling while validation worsens, reduce overfitting and investigate data/splits. If both remain high and improve slowly, investigate optimization, model capacity and token budget. If both flatten on a narrow corpus, more diverse data may help, but a plateau alone does not identify the cause. The present short pilot does not establish a data bottleneck.

Dataset references: [TinyStories official card](https://huggingface.co/datasets/roneneldan/TinyStories), [FineWeb-Edu official card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). Next-stage token/model sizes above are proposed experiments, not author-published requirements or completed runs.

## What to run next

Open `notebooks/GPT_to_MoE.ipynb` in Colab, select a GPU runtime and Run all. It embeds the current source, verifies the corpus, runs the checks, trains the 4-layer dense/MoE comparison in three seeds, makes plots and downloads an evidence archive. This notebook is syntax-checked but has not been executed on Colab. The larger-dataset experiment follows after that result is inspected.
