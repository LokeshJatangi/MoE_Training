import unittest
import torch
from src.models import LinearLM, SparseMoE

class ConversionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(2)
        self.dense = LinearLM(4, 7)
        self.x = torch.randint(7, (31, 4))

    def test_conversion_for_different_k(self):
        for k in (1, 2, 4):
            moe = SparseMoE(self.dense, 4, k)
            torch.testing.assert_close(moe(self.x)[0], self.dense(self.x)[0])

    def test_sparse_dispatch_matches_dense_reference(self):
        moe = SparseMoE(self.dense, 4, 2)
        with torch.no_grad():
            for expert in moe.experts:
                expert.weight.add_(torch.randn_like(expert.weight)*.1)
        x = self.dense.features(self.x)
        values, ids = moe.router(x).topk(2, dim=-1)
        weights = values.softmax(-1)
        all_outputs = torch.stack([e(x) for e in moe.experts], dim=1)
        expected = (all_outputs[torch.arange(len(x))[:,None], ids] * weights[:,:,None]).sum(1)
        actual, aux = moe(self.x)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(int(aux['counts'].sum()), 2*len(x))
        actual.square().mean().backward()
        self.assertGreater(moe.router.weight.grad.norm().item(), 0)

    def test_identical_experts_initial_router_task_gradient(self):
        moe = SparseMoE(self.dense, 4, 2)
        moe(self.x)[0].square().mean().backward()
        self.assertLess(moe.router.weight.grad.abs().max().item(), 1e-6)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            SparseMoE(self.dense, 4, 5)

if __name__ == '__main__':
    unittest.main()
