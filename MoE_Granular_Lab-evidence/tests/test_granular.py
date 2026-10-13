import unittest
import torch
from src.gpt import SwiGLU, TinyGPT, GPTConfig
from src.granular import GranularFFN, partition_upcycle


class GranularTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(2)
        self.dense = SwiGLU(16, 32)
        self.x = torch.randn(3, 7, 16, requires_grad=True)

    def test_all_partitions_reconstruct_and_copy_coordinates(self):
        moe = GranularFFN(self.dense, 4, 2, 8)
        with torch.no_grad():
            moe.router.weight.zero_()
        torch.testing.assert_close(moe(self.x)[0], self.dense(self.x))
        for e, idx in zip(moe.experts, moe.unit_indices):
            torch.testing.assert_close(e.gate.weight, self.dense.gate.weight[idx])
            torch.testing.assert_close(e.up.weight, self.dense.up.weight[idx])
            torch.testing.assert_close(e.down.weight, self.dense.down.weight[:, idx])

    def test_sparse_reference_dispatch_and_gradients(self):
        moe = GranularFFN(self.dense, 4, 3, 4)
        flat = self.x.reshape(-1, 16)
        values, ids = moe.router(flat).topk(4, -1)
        ref = torch.stack([e(flat) for e in moe.experts], 1)
        ref = (ref[torch.arange(len(flat))[:, None], ids] * values.softmax(-1)[:, :, None]).sum(1) * 4
        # Hooks verify each expert receives precisely its selected rows.
        observed = [0] * len(moe.experts)
        handles = [e.register_forward_pre_hook(lambda m, args, i=i: observed.__setitem__(i, len(args[0])))
                   for i, e in enumerate(moe.experts)]
        actual, aux = moe(self.x)
        for h in handles:
            h.remove()
        torch.testing.assert_close(actual.reshape(-1, 16), ref)
        self.assertEqual(observed, aux['counts'].int().tolist())
        self.assertEqual(sum(observed), len(flat) * 4)
        actual.square().mean().backward()
        self.assertGreater(moe.router.weight.grad.norm().item(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in moe.parameters() if p.grad is not None))

    def test_model_dimensions_accounting_and_invalid_shapes(self):
        model = TinyGPT(GPTConfig(vocab_size=13, context=8, layers=1, width=16, heads=4, intermediate=32))
        moe = partition_upcycle(model, 4, 3, 4)
        each = sum(p.numel() for p in moe.blocks[0].ffn.experts[0].parameters())
        original = sum(p.numel() for p in model.blocks[0].ffn.parameters())
        self.assertEqual(each * 4, original)
        self.assertEqual(each * 12, original * 3)
        self.assertEqual(moe.cfg.intermediate, 8)
        self.assertEqual(model.cfg.intermediate, 32)
        self.assertEqual(moe(torch.randint(13, (2, 8)))[0].shape, (2, 8, 13))
        with self.assertRaises(ValueError):
            GranularFFN(self.dense, 3, 2, 2)
