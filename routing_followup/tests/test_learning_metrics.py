import unittest
import torch
from src.gpt import GPTConfig,TinyGPT
from src.learning_metrics import (build_arm,gradient_norms,snapshot_experts,decay_removed_updates,
    fixed_hidden_probes,function_diversity,ablate_to_initial)
from src.routing_lab import set_policy

class LearningMetricTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17);torch.set_num_threads(2)
        self.parent=TinyGPT(GPTConfig(vocab_size=13,context=8,layers=1,width=16,heads=4,intermediate=32)).upcycle()
        self.cfg={'seed':17,'copies':3,'noise_fraction':.01}
        self.x=torch.randint(13,(3,8));self.y=torch.randint(13,(3,8))

    def test_family_selection_preserves_parent_and_noise_breaks_router_rows(self):
        with torch.no_grad():
            self.parent.blocks[0].ffn.experts[0].down.weight.add_(.02)
            self.parent.blocks[0].ffn.experts[1].gate.weight.mul_(.8)
        m=build_arm(self.parent,'family_bias',self.cfg).eval()
        torch.testing.assert_close(m(self.x)[0],self.parent.eval()(self.x)[0],atol=1e-6,rtol=1e-5)
        n=build_arm(self.parent,'noise',self.cfg)
        self.assertFalse(torch.equal(n.blocks[0].ffn.router.weight[0],n.blocks[0].ffn.router.weight[1]))
        torch.testing.assert_close(n.blocks[0].ffn.experts[0].gate.weight,n.blocks[0].ffn.experts[1].gate.weight)

    def test_decay_only_is_not_counted_as_learned_update(self):
        m=build_arm(self.parent,'hard',self.cfg)
        opt=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.1)
        for p in m.parameters():p.grad=torch.zeros_like(p)
        before=snapshot_experts(m);opt.step()
        self.assertEqual(float(decay_removed_updates(m,before,.01,.1).max()),0.)
        opt.zero_grad(set_to_none=True)
        self.assertEqual(float(gradient_norms(m).max()),0.)

    def test_reset_ablation_zero_for_initial_experts_and_diversity_zero(self):
        m=build_arm(self.parent,'hard',self.cfg).eval()
        probes=fixed_hidden_probes(self.parent,self.x,'cpu')
        self.assertTrue(all(abs(v)<1e-8 for row in function_diversity(m,probes) for v in row))
        states={k:v.clone() for k,v in m.state_dict().items()}
        result=ablate_to_initial(m,self.parent,(self.x,self.y),'cpu')
        self.assertTrue(all(abs(v)<1e-6 for row in result['expert_reset_ce_delta'] for v in row))
        for k,v in m.state_dict().items():torch.testing.assert_close(v,states[k])

    def test_bias_routes_tied_clones_without_random_sampling(self):
        m=build_arm(self.parent,'bias',self.cfg).train()
        with torch.no_grad():m.blocks[0].ffn.router.weight.zero_()
        seen=torch.zeros(12)
        set_policy(m,0.,bias_rate=.01)
        for _ in range(12):
            _,aux=m(self.x);seen+=aux[0]['counts'].detach()
        self.assertTrue(bool((seen>0).all()))
