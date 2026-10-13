import unittest
import torch
from src.gpt import TinyGPT,GPTConfig
from src.routing_lab import grow,set_policy

class RoutingLabTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.parent=TinyGPT(GPTConfig(vocab_size=13,context=8,layers=1,width=16,heads=4,intermediate=32,experts=8,top_k=2)).upcycle()
        self.tokens=torch.randint(13,(4,8))

    def test_cloned_router_rows_and_family_preservation(self):
        # Parent experts distinct: family-aware routing must preserve the parent function.
        with torch.no_grad():
            for i,e in enumerate(self.parent.blocks[0].ffn.experts):
                e.down.weight.add_(i*.01)
        child=grow(self.parent,4,family_routing=True).eval()
        expected=self.parent.eval()(self.tokens)[0]
        torch.testing.assert_close(child(self.tokens)[0],expected,atol=1e-6,rtol=1e-5)
        self.assertTrue(torch.equal(child.blocks[0].ffn.router.weight[0],child.blocks[0].ffn.router.weight[1]))

    def test_sampling_uses_unique_experts_and_hard_probe_is_rng_neutral(self):
        child=grow(self.parent,4)
        ffn=child.blocks[0].ffn
        ffn.route_generator=torch.Generator().manual_seed(10)
        child.train();set_policy(child,1.)
        ids=ffn.select(torch.zeros(2048,32))
        self.assertTrue(torch.all(ids[:,0]!=ids[:,1]))
        self.assertEqual(torch.bincount(ids.flatten(),minlength=32).gt(0).sum().item(),32)
        state=ffn.route_generator.get_state().clone()
        child.eval();child(self.tokens)
        self.assertTrue(torch.equal(state,ffn.route_generator.get_state()))

    def test_positive_bias_favors_starved_expert(self):
        child=grow(self.parent,4)
        ffn=child.blocks[0].ffn
        with torch.no_grad():
            ffn.routing_bias[3]=10.
        ids=ffn.select(torch.zeros(10,32)+ffn.routing_bias)
        self.assertTrue(torch.all((ids==3).any(-1)))

if __name__=='__main__': unittest.main()
