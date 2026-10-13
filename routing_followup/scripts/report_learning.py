"""Expert-learning and routing report from measured records only."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

p=argparse.ArgumentParser();p.add_argument('--results',required=True);a=p.parse_args()
root=Path(a.results);summary=json.loads((root/'summary.json').read_text())
rows=[json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
provenance=json.loads((root/'provenance.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
for arm in summary:
    name=arm['arm'];records=[r for r in rows if r['arm']==name];trained=[r for r in records if r['step']>0]
    axes[0].plot([r['step'] for r in records],[r['validation_ce'] for r in records],label=name)
    axes[1].plot([r['step'] for r in trained],[max(r['window_zero_experts']) for r in trained],label=name)
    axes[2].plot([r['step'] for r in trained],[max(r['hard_probe_zero_experts']) for r in trained],label=name)
for ax,title in zip(axes,['Hard-evaluation validation CE','Max layer zero traffic (training window)','Max layer zero traffic (static hard probe)']):
    ax.set(title=title,xlabel='Continuation update');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(root/'routing_curves.png',dpi=160);plt.close(fig)
lines=['# Hard-routing repair and expert-learning measurements','',
       '**CPU synthetic execution smoke; not language-training evidence.**' if provenance.get('toy') else 'Measured continuation from the pinned trained parent; all arms use matched data order and reset AdamW.',
       '', '| Arm | Start / final validation CE | Lifetime zero experts | Final window zero experts | Static probe zero experts | Zero CE-gradient experts | No Adam state | Positive reset deltas (>1e-6) |',
       '|---|---|---|---|---|---|---|---|']
for arm in summary:
    f=arm['final'];name=arm['arm']
    lifetime=np.asarray(f['cumulative_tokens']);grads=np.asarray(f['positive_task_gradient_steps'])
    no_state=sum(s['state_matrices']==0 for row in f['expert_optimizer_state'] for s in row)
    deltas=np.asarray(arm['reset_ablation']['expert_reset_ce_delta']) if arm['reset_ablation'] else np.array([])
    positives=int((deltas>1e-6).sum()) if deltas.size else 'not measured'
    lines.append(f"| {name} | {arm['initial']['validation_ce']:.6f} / {f['validation_ce']:.6f} | {int((lifetime==0).sum())} | {sum(f['window_zero_experts'])} | {sum(f['hard_probe_zero_experts'])} | {int((grads==0).sum())} | {no_state} | {positives} |")
    arrays=[np.log10(lifetime.flatten()+1),np.log10(np.asarray(f['mean_task_gradient_norm_when_nonzero']).flatten()+1e-12),
            np.log10(np.asarray(f['last_decay_removed_update_norm']).flatten()+1e-12),deltas.flatten() if deltas.size else np.zeros(lifetime.size)]
    labels=['Lifetime token routes + 1\n(log10)','CE gradient norm when nonzero\n(log10)',
            f"Decay-removed update, step {f['last_update_sample_step']}\n(log10)",'Reset-to-parent CE change']
    fig,axs=plt.subplots(1,4,figsize=(14,max(4.5,lifetime.size*.13)),layout='constrained')
    for idx,(ax,data,label) in enumerate(zip(axs,arrays,labels)):
        kwargs={}
        if idx==3:
            limit=max(abs(data).max(),1e-8);kwargs={'norm':TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),'cmap':'RdBu_r'}
        im=ax.imshow(data[:,None],aspect='auto',**kwargs);ax.set(xticks=[]);ax.set_title(label,fontsize=10)
        ax.set_yticks(range(lifetime.size),[f'L{l+1}/E{e}' for l in range(lifetime.shape[0]) for e in range(lifetime.shape[1])],fontsize=6)
        fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle(name);fig.savefig(root/f'experts-{name}.png',dpi=150);plt.close(fig)
lines+=['','![Routing/loss comparison](routing_curves.png)','','## Interpretation',
        '', 'Token traffic is exposure, not proof of learning. CE gradients are measured before clipping and contain no auxiliary-loss term in this comparison. AdamW parameter drift includes decay; sampled update norms subtract the decay-only prediction (including the grad=None skip rule). Those updates include current and retained Adam moments, not solely this step’s gradient.',
        '', 'Output diversity compares sibling functions on the same fixed parent-hidden inputs across arms. Diversity alone does not prove useful specialization. Reset ablation measures held-out CE after replacing one expert’s current weights with its original parent weights while retaining router weights/bias. Positive deltas suggest those learned weights help this specific probe/current context; negative deltas favor resetting. Subsequent hidden states and routes may change as part of that causal effect.',
        '', 'The positive-ablation threshold 1e-6 is a numerical display convention, not statistical significance. The small fixed ablation probe cannot establish generalization per expert. State/coverage and task loss must be read together. Training-window coverage under continually updated selection biases can coexist with starvation on a static hard probe; the two columns deliberately remain separate.',
        '', 'Full per-layer/per-expert telemetry and optimizer counters are in summary.json and metrics.jsonl. Checkpoints retain weights, moments, biases and RNG/counter state. No exact stochastic-resume equivalence claim is made.']
for arm in summary:
    lines+=['',f"## {arm['arm']} expert telemetry",'',f"![Expert metrics](experts-{arm['arm']}.png)"]
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:13]))
