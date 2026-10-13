"""Render measured narrow-expert initialization comparison."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--results', required=True)
a=p.parse_args()
root=Path(a.results)
s=json.loads((root/'summary.json').read_text())
cfg=json.loads((root/'config.json').read_text())
rows=[json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
fig,axes=plt.subplots(1,2,figsize=(13,4.8))
for method,b in s['branches'].items():
    records=[r for r in rows if r['phase']==method]
    axes[0].plot([r['step'] for r in records],[r['validation_ce'] for r in records],label=method,marker='.')
    if 'expert_load_cv' in b['final']:
        axes[1].plot([r['step'] for r in records], [max(r['expert_load_cv']) for r in records],label=method)
axes[0].axhline(s['dense_checkpoint_validation_ce'],color='black',linestyle='--',label='dense checkpoint')
axes[0].set(xlabel='Continuation steps (1,024 target tokens/step)',ylabel='Validation task CE')
axes[1].set(xlabel='Continuation steps',ylabel='Maximum layer load CV on fixed validation probe')
for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(root/'curves.png',dpi=160)
fig.savefig(root/'curves.svg')
lines=['# Single-seed narrow-expert initialization comparison','',
       f"Measured **CPU** execution, Tiny Shakespeare, seed {s['seed']}. Same trained dense checkpoint (validation CE **{s['dense_checkpoint_validation_ce']:.6f}**), same sample order and 300 continuation updates per branch. Each branch resets AdamW and sees 307,200 additional next-character targets. Original dense pretraining saw 153,600 targets.",'',
       '| Method | CE immediately after conversion | Final validation CE | Conversion CE jump | Recovers dense checkpoint | Final fixed-probe zero experts by layer |',
       '|---|---:|---:|---:|---|---|']
for method,b in s['branches'].items():
    lines.append(f"| {method} | {b['start']['validation_ce']:.6f} | {b['final']['validation_ce']:.6f} | {b['conversion_ce_jump']:+.6f} | {b['recovers_dense']} | {b['final_probe_zero_experts_per_layer']} |")
lines+=['','![Measured loss and load curves](curves.png)','','## Definitions','',
        '- `contiguous`: disjoint adjacent groups of trained intermediate units; four replicas of each group.',
        '- `shuffled`: disjoint groups after a fixed random permutation; four replicas of each group.',
        '- `drop`: each expert independently retains 32 of the 128 trained units without replacement. Different experts may overlap; no complete-coverage guarantee.',
        '- `perturbed`: contiguous copies plus independent Gaussian noise at 1% of each copied matrix’s RMS.',
        '- `random`: newly initialized narrow expert weights; trained attention, embeddings and norms are retained. This is not an entirely random GPT.',
        '- `full_copy`: four complete width-128 experts/top-1. `dense_control` continues the original GPT.',
        '', '## Controls and interpretation','',
        'All five narrow variants have width-32 experts, 16 experts and top-4, normalized selected gates, output scale 4 and balancing coefficient 0.01. Their routers start with identical weights across methods. This isolates expert initialization. Different post-conversion hidden states can still produce different initial routing decisions. Drop is our explicitly defined subset-retention method, not a claimed reproduction of a colleague’s implementation.',
        '', 'Every branch selects 128 intermediate units per token (24,576 FFN matrix MACs per token/layer). Dense has 90,816 total parameters; full-copy has 238,784 and narrow variants have 240,320. Routing, dispatch and optimizer overhead differ, so no equal-runtime claim is made. Full-copy top-1 receives router task gradient zero through its selected softmax weight; its auxiliary objective still trains the router.',
        '', '| Method | Active parameter convention | Wall seconds including evaluation | All expert matrices updated | Checkpoint next-update parameter error |',
        '|---|---:|---:|---|---:|']
for method,b in s['branches'].items():
    lines.append(f"| {method} | {b['parameters']['active_parameter_convention']} | {b['wall_seconds_including_eval']:.2f} | {b['all_expert_matrices_updated']} | {b['resume_next_update_parameter_error']:.2g} |")
lines+=['', 'The active convention counts the full tied embedding table and nonexpert weights plus selected expert weights; it is not a FLOP count. Per-matrix changes, train CE, gradient norms, router entropy, allocation fractions, unit mappings and final test CE are in summary.json/metrics.jsonl. Source and checkpoint hashes are in environment.json. Test was evaluated after all predeclared arms completed and was not used to choose settings.',
        '', 'Fixed-probe zero counts are not rolling starvation measurements. A parameter change alone does not prove useful specialization; weight decay also changes parameters. Finite task/gradient checks and reproducible next optimizer updates passed during training. The comparison is a one-seed small-data mechanism study; do not infer a robust method ranking or frontier-scale behavior from small differences. Colab/TinyStories runs remain unexecuted.']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:14]))
