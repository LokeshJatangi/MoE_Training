# Progress

Updated: 12 October 2026 (Asia/Kolkata).

## State

**ACTIVE — resumed on 11 October 2026 after switching to Medium.** User clarification: the routing means a **small dense GPT**, not a literal affine model. Previous claims of routing completion applied to the old interpretation. Preserve that experiment as a preliminary mechanism test.

## Current GPT implementation and CPU pilot

- Implemented `src/gpt.py`: causal Transformer, SwiGLU FFNs, tied embeddings/head, normalized top-2 FFN expert routing, actual dropless sparse dispatch.
- Implemented `src/train_gpt.py`: causal next-character targets at every position, matched-order dense control, per-layer expert loads/entropy, separate task/auxiliary losses, gradients, checkpoints and exact next-update checks.
- Configs: `configs/gpt_cpu.json` (2 layers, width 64, context 64, seed 17) and `configs/gpt_colab.json` (4 layers, width 128, context 128, 3 seeds).
- Eight tests passed, including causality, full-GPT conversion preservation and sparse/reference equivalence.
- CPU pilot completed: 300 pretraining + 300 continuation steps per branch, batch 16, 1,024 next-character targets per step.
- Validation CE: **4.190773 → 2.483576 → 2.367056**. Dense control: **2.373484**. MoE improvement relative to dense control is small and unconfirmed across seeds.
- Dense parameters **90,816**; MoE **238,784 total / 140,480 active under the documented convention**. Conversion max logit error **2.6226e-6**. All expert matrices updated and all acceptance checks passed.
- Evidence: `results/gpt-cpu-pilot/`. Four phase checkpoints in `checkpoints/gpt-cpu-pilot/17/`.
- Added `scripts/report_gpt.py`, `docs/EXPERIMENT_EXPLAINED.md` and self-contained `notebooks/GPT_to_MoE.ipynb`. Notebook cells syntax-checked; Colab execution pending.
- The dataset has not yet shown a demonstrated bottleneck; larger-data stage proposed: frozen TinyStories subset plus fixed subword tokenizer, after the Colab reproduction.

Commands:

```bash
python3 -m unittest discover -s tests -v
python3 -m src.train_gpt --config configs/gpt_cpu.json --output results/gpt-cpu-pilot
python3 scripts/report_gpt.py
```

## Decisions

- Dataset: small text corpus, Tiny Shakespeare, next-character prediction.
- Future experiment environment: Google Colab GPU; GPU model/VRAM not yet known.
- First routing: literal linear predictor → four cloned experts, top-2 sparse routing.
- Eventual goal: pretrain a 70B sparse model from scratch, not achieve that large-model result here. 70B total is an explicit provisional assumption; active count remains open.
- All project files moved together, including local Git metadata, corpus and checkpoints, from `outputs/moe-lab/` to `outputs/MoE_Training/`.

## Completed and measured

| Item | Evidence |
|---|---|
| Local repository initialization | `.git/`, branch `main`; no remote |
| CPU hardware/environment inspection | Ryzen 5 5500U; ~14 GiB RAM; PyTorch 2.9.1; CUDA unavailable |
| Corpus acquisition / reproducible split | `results/routing/data_manifest.json` |
| Linear training: three seeds | `results/routing/metrics.jsonl` |
| Function-preserving sparse MoE conversion | Max logit difference 9.5367431640625e-7 |
| Continued MoE and matched-token dense control | `results/routing/summary.json` |
| All expert weights updated | Per-seed expert weight changes in summary |
| Model + optimizer + sampler recovery | Next update matched within 1e-6 in every phase/seed |
| Four mathematical/gradient tests | `python -m unittest discover -s tests -v` passed |
| Report and loss/expert-load plots | `results/routing/REPORT.md`, PNG and SVG; PNG visually inspected |
| Primary-source research | `docs/SOURCES.md` |
| Project status overview | `readme.md` |

Mean validation CE: initial **4.181208** → trained linear **2.338370** → continued MoE **2.148826**. Continued dense control: **2.238916**. Mean training evaluation CE: trained linear **2.261702** → continued MoE **2.058425**. All three seeds passed the routing acceptance checks.

## Executed commands

From the repository directory:

```bash
python3 -m src.train --device cpu --output results/routing
python3 -m unittest discover -s tests -v
python3 scripts/report.py
```

Seeds: 17, 42, 123. Steps: 800 dense pretraining + 800 per continuation branch. Batch size: 256. Task CE is separate from the routing auxiliary term. The first attempt to download data was blocked by shell networking; the user granted network permission and download succeeded. One initial help command ran from the parent directory and failed to find `src`; corrected by using the repository directory. Neither issue affected the recorded training run.

## Not done

- Colab GPU execution (notebook packaging is complete).
- Three-seed GPT confirmation and later architecture ablations. Tiny GPT/FFN upcycling has a passing one-seed CPU pilot.
- Faithful implementations of current frontier components, distributed training or large-model benchmarking.
- Remote GitHub repository publication.
- Any 70B model training or compute purchase.

Roadmaps describe planned work; E00 has earlier linear evidence and the clarified GPT baseline has a new one-seed CPU pilot. No claim of equal-compute superiority is made. The initial training happened before the first Git commit, so the environment file has a null commit but includes exact executed source hashes.

## Resume procedure

Read `readme.md`, this file and `docs/EXPERIMENT_EXPLAINED.md`. Run `notebooks/GPT_to_MoE.ipynb` on the actual Colab GPU; preserve the original local evidence. Inspect the three-seed result before increasing data/model size. Roadmap dates/parked notes are historical and superseded by the latest user instruction.

## Agent work

Astra subagent: completed final review/creation of the three roadmap documents only. Required core experiments are separated from optional probes, and dates begin on resumption. Parent: project relocation, status/readme, source catalog and repository preservation. The user has resumed training. Astra remains limited to roadmap work; current implementation uses the parent agent.

## Clone-continuation lab preparation — 11 October 2026

User requested our own clone-growth experiment, inspired by a colleague; clarified 20M→460M meant parameters and chose T4/16 GB, but asked us to use our own sizes. Added `src/routing_lab.py`, `src/clone_sweep.py`, TinyStories/BPE preparation, T4 config, diagnostics/reporting and `notebooks/Clone_Collapse_T4.ipynb`. Strategies implemented: immediate hard, fixed sampled warmup, annealing, probe-gated switch, auxiliary balancing, detached bias balancing and family-aware routing. Exact copying/joint training/Adam reset are held fixed.

Eleven unit tests passed. All seven arms executed in a tiny **CPU toy-token smoke check** at 12 experts; this is execution evidence only, not a language-training conclusion. Updated output-distance diagnostic samples at most eight siblings per family; final smoke saved separately in `results/clone-cpu-smoke-v2/`. Notebook syntax checked. No TinyStories download/training or Colab execution occurred locally.

Primary distinction: identical expert weights do not necessarily imply identical router rows. Flat growth can change the original parent mixture. Single-batch dead counts differ from rolling starvation. Adaptive readiness is an experimental heuristic; sustained hard-training survival and useful task loss are the actual questions. Stochastic checkpoint restart equivalence and three-seed confirmation are still pending. See `docs/CONTINUATION_STRATEGIES.md`.

## Narrow-expert task — 12 October 2026

User made smaller experts/more experts the current task and requested a Colab notebook. Implemented `src/granular.py`: coordinated gate/up row and down column partition copying, replicated narrow experts, normalized sparse routing with explicit partition-count output scaling. Implemented `src/granular_experiment.py`: same dense checkpoint and data order for dense continuation, full-copy top-1 and narrow top-4, with fixed FFN matrix MACs across branches. Record conversion CE jump/drift, recovery, loads, parameter changes, memory and checkpoint checks. Whole-runtime equivalence is not claimed.

**Measured CPU pilot:** `results/granular-cpu-pilot/`, Tiny Shakespeare, seed 17, 150+150 steps. Dense CE 4.190773→2.624921; continued dense 2.505817; full-copy 2.511586; narrow conversion 2.655241→2.498899. All phase recovery checks passed. Fourteen unit tests passed. This is a one-seed mechanism demonstration, not TinyStories/Colab evidence.

**Prepared, unrun on GPU:** self-contained `notebooks/Narrow_Experts_T4.ipynb`, TinyStories/BPE, three branches, plots, evidence download and opt-in finer partition comparison. Code cells compiled and embedded source checked. See `docs/NARROW_EXPERT_EXPERIMENT.md`. This notebook is the next action; larger clone growth remains a separate lab.

Command executed:

```bash
python3 -m src.granular_experiment --config configs/granular_cpu.json --data data/tinyshakespeare.txt --device cpu --output results/granular-cpu-pilot
```

## Single-seed initialization comparison — 12 October 2026

User authorized continuing locally with one seed and different methods. Added `src/narrow_methods.py`, `src/compare_narrow_methods.py`, `configs/narrow_methods_cpu.json`, `scripts/report_narrow_methods.py` and `docs/SINGLE_SEED_METHODS.md`. Reused the original seed-17 dense checkpoint (150 updates) rather than retraining. Dataset/checkpoint/source hashes captured. Five narrow initialization arms: contiguous partition, shuffled disjoint partition, independent quarter-unit retention/drop, contiguous + 1% matrix-RMS noise, fresh random experts. Controls: dense continuation and full-copy top-1. All narrow variants have 16 experts of width 32, top-4, scale 4, identical initial routers, identical batches and reset AdamW; 300 continuation steps each.

**Measured local CPU/Tiny Shakespeare**, evidence `results/narrow-methods-cpu-seed17/`: dense checkpoint CE 2.624921. Final CE: dense 2.460981; full-copy 2.466103; contiguous 2.446182; shuffled 2.451317; drop 2.456125; perturbed 2.450853; random 2.448603. All branches improved and recovered the dense checkpoint. Random expert conversion started at 3.729246 versus copied narrow variants around 2.655–2.661. At step 50 contiguous had 2.575785 vs random 2.697639: transfer helped early recovery. Final differences are too small/underconfirmed to establish a method ranking.

Seventeen repository tests passed. All phase checkpoint next-update checks passed; all expert matrices changed; every expert received tokens on the final fixed validation probe. This does not establish persistent starvation absence or useful specialization. Source hashes verified after execution. PNG plot visually inspected. Narrow branches took 13–14 seconds including evaluation vs dense 5 seconds: constant active FFN width did not imply constant runtime. Colab/TinyStories remain unrun. Original notebook/source snapshots and prior measured evidence remain unchanged.

Commands:

```bash
python3 -m unittest discover -s tests -v
python3 -m src.compare_narrow_methods --output results/narrow-methods-cpu-seed17
python3 scripts/report_narrow_methods.py --results results/narrow-methods-cpu-seed17
```

Next study: fix initialization and vary routing/exploration or replica count. Keep one-seed observations separate from validated quality/efficiency claims.


## Detailed canonical README — 12 October 2026

Created `README.md` with all measured language experiments, both clone smoke histories, exact summary-derived results, parameter/compute accounting, conversion mathematics, datasets, optimizer/token budgets, limitations, reproduction commands, tests/evidence and prepared Colab notebooks. Replaced lowercase `readme.md` with a compatibility pointer and updated AGENTS.md to read the canonical document. Roadmap parked wording is explicitly historical; no roadmap implementation claim was added.

Documentation checked against saved configs/summaries and local link targets. No training or test suite rerun was needed for this documentation-only change. All measured training remains CPU; TinyStories/Colab execution remains pending.


## User-reported Colab GPT failure — 12 October 2026

User attempted GPT_to_MoE on Colab and reported subprocess exit status 1. No child stderr/traceback was available. Root cause remains undiagnosed and no successful GPU result is verified. Added `scripts/run_logged.py`, preserving streamed stdout/stderr and final failure details without log overwrites. Updated GPT notebook embedded source and training cell; microsecond output names, persistent logs and failure/archive guidance. Added self-contained `scripts/colab_gpt_training_cell.py` to patch the user's existing runtime without a restart. No model settings, optimizer behavior, conversion/recovery tolerances or old result files changed.

Two meaningful failure/success logging tests passed; the full 19-test suite passed. Notebook code cells compile, embedded source matches local snapshot. GPU training was not run by this agent. Next action: user executes diagnostic cell and shares the underlying failure/log so we can fix its actual cause.


## Conversion assertion identified and recovery prepared — 12 October 2026

User supplied the underlying Colab traceback. Saved pasted evidence separately in `results/gpt-colab-user-report/`, explicitly user-reported CUDA output, not local measurements. Dense seed 17 completed 1,000 steps/2,048,000 targets: validation 4.216622→1.889444, train 4.209708→1.781826. Conversion then failed the old absolute-only 1e-5 cutoff with max logit difference 2.2411346e-5. MoE continuation was not reached; GPU type remains unknown.

Added exact expert-weight-copy validation, scale-aware per-entry logit criterion atol=rtol=1e-5, finite/CE-change checks and persistent conversion diagnostics. This changes the acceptance definition explicitly; the old user maximum alone cannot establish whether the new test passes. Added `--resume-dense-run` to load completed dense checkpoints and inherited records with config/data validation and checkpoint hash. Continued Adam still resets by design. Updated only the GPT notebook snapshot; other historic notebook snapshots remain unchanged. Created self-contained Colab recovery cell targeting the user's failed diagnostic run and seed 17.

Full 21-test suite passed. CPU integration smoke (`work/resume-check`, two continuation steps) restored the checkpoint, passed conversion and both phase next-update checks; final MoE validation-loss-drop acceptance was false at that short budget and the run exited with acceptance failure. Preserved it; no extra training to turn this smoke positive. GPU remains unverified. Next action: user runs recovery cell and inspects conversion diagnostics and continuation metrics.


## Restart-safe Colab recovery — 12 October 2026

User restarted the runtime; the prior hard-coded result path was absent. Rebuilt the recovery cell as self-contained: bootstrap required source and checksum-verifying data preparation, auto-discover completed GPT dense evidence/checkpoint, resume if present, otherwise start a fresh seed-17 run. No timestamp edits or old initialized runtime are required; GPU must be enabled. New runs/logs preserve surviving evidence, and success downloads an archive. This fixes path handling, not a newly observed training failure. Original failed user evidence remains unchanged.

Two discovery tests passed; the full 23-test suite passed. Embedded source and code syntax checked. No GPU run executed by the agent. User-supplied untracked evidence directories `MoE_Clone_Lab-evidence/` and `MoE_Granular_Lab-evidence/` were left untouched.


## Independent FP64 conversion check — 12 October 2026

User-reported T4 (torch 2.11.0+cu130) recovery restored the dense checkpoint, then failed six FP32 per-logit entries: max 3.3378601e-5, RMS 1.2258761e-6, probability max 3.5166740e-6, CE difference 0. Saved these reported diagnostics/provenance separately in `results/gpt-colab-user-report-fp32/`; no GPU continuation occurred.

Added CPU FP64 model reference probe, including FP64 router arithmetic (ordinary FP32 behavior unchanged). Tight FP64 per-logit and CE limits remain 1e-10. Retained old per-logit FP32 diagnostic; a fallback requires independent FP64 equivalence plus finite FP32 max/RMS/probability/CE bounds (1e-4/1e-5/1e-5/1e-5). Exact expert parameter equality still required. This is an explicit acceptance-method change, not a silently relaxed single assertion. Saved stats name the acceptance path.

Updated GPT notebook and restart-safe self-contained recovery snapshot. All 26 tests passed, including corruption/fallback guard tests. Existing trained local GPT reference check passed with max FP64 logit error 4.4408921e-15 and CE difference 0 (`work/fp64-trained-model-check.json`). No new training run was needed. Actual T4/reference recovery remains to be executed by user. Existing measured source hashes identify earlier implementations; prior result files and other historic notebook snapshots were left unchanged.
