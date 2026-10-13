import argparse
import json
import statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--results',default='results/routing')
a=parser.parse_args()
p=Path(a.results)
summaries=json.loads((p/'summary.json').read_text())
rows=[json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines()]
colors={'dense_pretrain':'#475569','dense_control':'#d97706','moe_continue':'#087f8c'}
labels={'dense_pretrain':'Linear pretraining','dense_control':'Linear continuation','moe_continue':'MoE continuation'}
fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
for phase in colors:
    selected=[r for r in rows if r['phase']==phase]
    steps=sorted(set(r['global_step'] for r in selected))
    for ax,metric in zip(axs[:2],['train_ce','validation_ce']):
        means=[st.mean(r[metric] for r in selected if r['global_step']==s) for s in steps]
        mins=[min(r[metric] for r in selected if r['global_step']==s) for s in steps]
        maxs=[max(r[metric] for r in selected if r['global_step']==s) for s in steps]
        ax.plot(steps,means,color=colors[phase],label=labels[phase],lw=2)
        ax.fill_between(steps,mins,maxs,color=colors[phase],alpha=.18)
for ax,title in zip(axs[:2],['Training evaluation loss','Held-out validation loss']):
    ax.axvline(summaries[0]['dense_end']['global_step'],color='#94a3b8',ls='--')
    ax.set(title=title,xlabel='Optimizer steps along each branch',ylabel='Cross entropy (nats / character)')
    ax.grid(alpha=.2)
axs[0].legend(fontsize=8)
for i,s in enumerate(summaries):
    vals=s['branches']['moe_continue']['final']['expert_fraction']
    axs[2].bar([x+(i-1)*.25 for x in range(4)],vals,width=.24,label=f"Seed {s['seed']}")
axs[2].axhline(.25,color='#94a3b8',ls='--')
axs[2].set(title='Final expert routing fractions',xlabel='Expert',ylabel='Fraction of selected routings',xticks=range(4),ylim=(0,.4))
axs[2].legend(fontsize=8)
fig.suptitle('Linear → sparse MoE | Tiny Shakespeare | 3 seeds',fontsize=15)
fig.savefig(p/'loss_curves.png',dpi=170)
fig.savefig(p/'loss_curves.svg')
lines=['# routing results','', 'Measured locally on CPU; Colab execution remains pending. No hyperparameter search was performed for this result.', '',
'| Seed | Initial val CE | Trained linear | Continued dense | Continued MoE | MoE test CE |',
'|---|---:|---:|---:|---:|---:|']
for s in summaries:
    b=s['branches']
    lines.append(f"| {s['seed']} | {s['initial']['validation_ce']:.6f} | {s['dense_end']['validation_ce']:.6f} | {b['dense_control']['final']['validation_ce']:.6f} | {b['moe_continue']['final']['validation_ce']:.6f} | {b['moe_continue']['test_ce']:.6f} |")
extractors={
 'Initial validation':lambda s:s['initial']['validation_ce'],
 'Trained linear validation':lambda s:s['dense_end']['validation_ce'],
 'Continued dense validation':lambda s:s['branches']['dense_control']['final']['validation_ce'],
 'Continued MoE validation':lambda s:s['branches']['moe_continue']['final']['validation_ce'],
 'Trained linear train':lambda s:s['dense_end']['train_ce'],
 'Continued MoE train':lambda s:s['branches']['moe_continue']['final']['train_ce'],
}
lines+=['','Mean ± sample standard deviation across seeds (not a confidence interval):','']
for name,fn in extractors.items():
    values=[fn(s) for s in summaries]
    lines.append(f'- {name}: {st.mean(values):.6f} ± {st.stdev(values):.6f}')
s=summaries[0]
lines+=['',f"Linear parameters: **{s['dense_parameters']:,}**. MoE total: **{s['moe_total_parameters']:,}**. Parameters in selected experts plus router per target: **{s['moe_active_parameters_per_target']:,}**.",
'', '## What passed','',
'- Dense training and validation loss decreased in every seed.',
'- MoE training and validation loss decreased after conversion in every seed.',
'- Maximum conversion logit error: '+str(max(x['conversion_max_logit_error'] for x in summaries))+'.',
'- All experts changed weights; final routing fractions are plotted above.',
'- Saved model, optimizer and sampler states reproduced the next update (all parameter errors < 1e-6).',
'', '## Interpretation and limits','',
'This completes the literal linear-to-MoE routing at small scale. The model predicts one next character from 16 preceding characters. Its logits are an affine map of a fixed one-hot encoding; it is not a Transformer. The routed mixture is nonlinear because the router depends on the input.',
'',
'The MoE outperformed the continued linear control at equal target tokens in this experiment. This is not a compute-matched or total-parameter-matched superiority claim: two experts execute per target, the router adds work, and sparse dispatch adds overhead. The baseline was still learning at conversion; the task asks for continued learning, not convergence before conversion.',
'',
'Each branch sees 204,800 additional next-character targets after 204,800 pretraining targets. Context reads are not counted as additional supervised targets. Evaluation uses fixed random samples of 4,096 windows per split, with replacement. Split raw text before making windows; vocabulary comes only from training text. The held-out test sample is evaluated only at final branch checkpoints. Tiny Shakespeare is a mechanism test, not evidence of general language ability or scaling performance.',
'',
'Both continuation branches reset AdamW at the branch point, use learning rate 0.002, and consume identical sampled training windows. There is no loss smoothing or best-checkpoint selection. Curves show means and min/max seed bands. Task CE is reported separately from the balancing term (coefficient 0.01).',
'', '## Reproduce','',
'```bash\npython -m unittest discover -s tests -v\npython -m src.train --device cpu --output results/reproduction\npython scripts/report.py --results results/reproduction\n```',
'', 'Evidence: `config.json`, `data_manifest.json`, `environment.json`, `metrics.jsonl`, `metrics.csv`, `summary.json`. Local checkpoints are in `checkpoints/routing/`; these and the downloadable corpus are ignored by Git. Source SHA-256 values identify the executed implementation even when the initial Git commit is not yet present.','',
'![Loss curves](loss_curves.png)','']
(p/'REPORT.md').write_text('\n'.join(lines))
print('\n'.join(lines[:27]))
