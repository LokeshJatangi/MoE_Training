"""Consolidate completed T4 expert-learning evidence without retraining."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.gpt import GPTConfig,TinyGPT
from src.routing_lab import grow

ROOT=Path(__file__).resolve().parents[1]
package=ROOT/'MoE_Routing_Followup-20261013-041132-668911'
run=package/'results/hard-routing-20261013-041132-668911'
out=ROOT/'analysis';out.mkdir(exist_ok=True)
previous_audit=json.loads((ROOT/'provenance/original_checkpoint_review.json').read_text())
previous_arms={a['arm']:a for a in previous_audit['arms']}
cfg=json.loads((run/'config.json').read_text());env=json.loads((run/'environment.json').read_text())
summary=json.loads((run/'summary.json').read_text())
records=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
for name,digest in env['source_sha256'].items():
    assert hashlib.sha256((package/name).read_bytes()).hexdigest()==digest,name
provenance=json.loads((run/'provenance.json').read_text())
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint-root',type=Path,help='External original MoE_Training directory containing excluded checkpoint files')
args=parser.parse_args()
checkpoint_package=args.checkpoint_root/package.name if args.checkpoint_root else package
parent_path=checkpoint_package/'artifacts/parent.pt'
if args.checkpoint_root:
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest()==provenance['parent_sha256']
original_parent=(args.checkpoint_root or ROOT)/'MoE_Clone_Lab-evidence/checkpoints/clone-first-20261012-133612/parent.pt'
original_env=json.loads((ROOT/'MoE_Clone_Lab-evidence/results/clone-first-20261012-133612/environment.json').read_text())
if args.checkpoint_root:
    assert hashlib.sha256(original_parent.read_bytes()).hexdigest()==provenance['parent_sha256']
assert original_env['data_sha256']==provenance['npz_sha256']
assert len(summary)==5
result={'package':str(package.relative_to(ROOT)),'run':run.name,'device':env,
        'source_hashes_verified':len(env['source_sha256']),
        'parent_sha256_recorded':provenance['parent_sha256'],'parent_checkpoint_verified':bool(args.checkpoint_root),
        'packed_data_sha256':provenance['npz_sha256'],
        'parent_matches_original_clone':bool(args.checkpoint_root),'packed_data_matches_original_clone':True,'arms':[]}
for arm in summary:
    name=arm['arm'];final=arm['final'];assert final['step']==cfg['steps']==750
    assert len([r for r in records if r['arm']==name])==31
    counts=np.asarray(final['cumulative_tokens']);positive=np.asarray(final['positive_task_gradient_steps'])
    if args.checkpoint_root:
        ck_path=checkpoint_package/'checkpoints'/run.name/f'{name}.pt'
        ck=torch.load(ck_path,map_location='cpu',weights_only=True)
        model=grow(TinyGPT(GPTConfig(**cfg['model'])).upcycle(),cfg['copies'])
        parameters=list(model.named_parameters());ids=ck['optimizer']['param_groups'][0]['params']
        assert len(parameters)==len(ids)==176
        states={n:ck['optimizer']['state'].get(i,{}) for (n,_),i in zip(parameters,ids)}
        counts=np.asarray(final['cumulative_tokens']);positive=np.asarray(final['positive_task_gradient_steps'])
        assert np.array_equal(counts,ck['cumulative_tokens'].numpy())
        assert np.array_equal(positive,ck['positive_task_gradient_steps'].numpy())
        assert (counts.sum(-1)==750*4*128*2).all()
        nonzero_moments=[];no_states=[];steps=[]
        for layer in range(4):
            moments=[];absent=[];step_row=[]
            for e in range(12):
                moment=0.;state_count=0;step=[]
                for matrix in ['gate','up','down']:
                    n=f'blocks.{layer}.ffn.experts.{e}.{matrix}.weight';s=states[n]
                    if s:
                        assert s['exp_avg_sq'].shape==ck['model'][n].shape
                        moment+=s['exp_avg_sq'].sum().item();state_count+=1;step.append(int(s['step']))
                    else:step.append(0)
                moments.append(moment>0);absent.append(state_count==0);step_row.append(step)
                assert state_count==final['expert_optimizer_state'][layer][e]['state_matrices']
                assert step==final['expert_optimizer_state'][layer][e]['matrix_steps']
            nonzero_moments.append(moments);no_states.append(absent);steps.append(step_row)
    else:
        no_states=[[e['state_matrices']==0 for e in layer] for layer in final['expert_optimizer_state']]
    deltas=np.asarray(arm['reset_ablation']['expert_reset_ce_delta'])
    row={'arm':name,'initial_validation_ce':arm['initial']['validation_ce'],'final_validation_ce':final['validation_ce'],
         'test_ce':arm['test_ce'],'lifetime_zero_experts':int((counts==0).sum()),
         'window_zero_experts_by_layer':final['window_zero_experts'],'static_probe_zero_experts_by_layer':final['hard_probe_zero_experts'],
         'no_task_gradient_experts':int((positive==0).sum()),'no_adam_state_experts':int(np.asarray(no_states).sum()),
         'nonzero_task_second_moment_experts':int(np.asarray(nonzero_moments).sum()) if args.checkpoint_root else previous_arms[name]['nonzero_task_second_moment_experts'],
         'unchanged_experts':sum(d['l2']==0 for layer in final['weight_drift'] for d in layer),
         'last_update_zero_experts':int((np.asarray(final['last_decay_removed_update_norm'])==0).sum()),
         'ablation_positive':int((deltas>1e-6).sum()),'ablation_negative':int((deltas< -1e-6).sum()),
         'ablation_negligible':int((abs(deltas)<=1e-6).sum()),'minimum_lifetime_routes':float(counts.min()),
         'minimum_gradient_steps':float(positive.min()),'minimum_recent_gradient_steps':min(v for row in final['window_positive_task_gradient_steps'] for v in row),
         'effective_experts_by_layer':final['effective_experts'],
         'seconds_through_final_probe':final['elapsed_seconds_including_diagnostics'],
         'peak_allocated_bytes':arm['peak_cuda_allocated_bytes'],
         'checkpoint_sha256':hashlib.sha256(ck_path.read_bytes()).hexdigest() if args.checkpoint_root else None,'checkpoint_counts_and_state_verified':bool(args.checkpoint_root),'second_moment_evidence_scope':'external checkpoint recheck' if args.checkpoint_root else 'previous original checkpoint audit'}
    result['arms'].append(row)
fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
for arm in summary:
    r=[item for item in records if item['arm']==arm['arm']];trained=[item for item in r if item['step']>0]
    axs[0].plot([x['step'] for x in r],[x['validation_ce'] for x in r],label=arm['arm'])
    axs[1].plot([x['step'] for x in trained],[sum(x['window_zero_experts']) for x in trained],label=arm['arm'])
    axs[2].plot([x['step'] for x in trained],[sum(x['hard_probe_zero_experts']) for x in trained],label=arm['arm'])
for ax,title in zip(axs,['Validation task CE','Unused experts: last 50 training updates','Unused experts: static hard probe']):
    ax.set(title=title,xlabel='Continuation update');ax.grid(alpha=.2);ax.legend(fontsize=8)
axs[1].set_ylabel('Count across 48 experts');axs[2].set_ylabel('Count across 48 experts')
fig.savefig(out/'routing_followup_comparison.png',dpi=170);plt.close(fig)
# Allocation heatmap explicitly masks exact zeros as black.
fractions=np.array([np.array(a['final']['window_route_fractions']).flatten() for a in summary])
fig,ax=plt.subplots(figsize=(15,3.8),layout='constrained')
cmap=plt.get_cmap('viridis').copy();cmap.set_bad('black')
im=ax.imshow(np.ma.masked_where(fractions==0,fractions),aspect='auto',cmap=cmap,vmin=0,vmax=fractions.max())
ax.set_yticks(range(5),[a['arm'] for a in summary])
ax.set_xticks(range(0,48,3),[f'L{i//12+1}/E{i%12}' for i in range(0,48,3)],fontsize=8)
for boundary in [11.5,23.5,35.5]:ax.axvline(boundary,color='white',lw=2)
ax.set(title='Final training allocation fractions — black cells are exact zero traffic',xlabel='Four layers × twelve experts (indices start at zero)')
fig.colorbar(im,ax=ax,label='Fraction of routed tokens within layer')
fig.savefig(out/'routing_followup_zero_cells.png',dpi=180);plt.close(fig)
(out/'routing_followup_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('Verified submission source hashes and all 155 records. Checkpoint validation: '+('external files verified' if args.checkpoint_root else 'not rerun; excluded from submission.'))
for arm in result['arms']:print(arm['arm'],arm['lifetime_zero_experts'],arm['nonzero_task_second_moment_experts'],arm['final_validation_ce'])
