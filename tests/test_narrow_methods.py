import unittest
import torch
from src.gpt import GPTConfig, TinyGPT
from src.narrow_methods import narrow_upcycle, METHODS


class NarrowMethodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)
        self.model = TinyGPT(GPTConfig(vocab_size=13, context=8, layers=1, width=16, heads=4, intermediate=32))

    def test_equal_sizes_and_common_routers(self):
        converted = []
        for method in METHODS:
            torch.manual_seed(30017)
            converted.append(narrow_upcycle(self.model, method))
        for m in converted:
            self.assertEqual(m.parameter_counts(), converted[0].parameter_counts())
            torch.testing.assert_close(m.blocks[0].ffn.router.weight, converted[0].blocks[0].ffn.router.weight)

    def test_shuffled_partition_reconstructs_and_drop_copies(self):
        x = torch.randn(4, 16)
        for method in ('shuffled', 'drop'):
            moe = narrow_upcycle(self.model, method).blocks[0].ffn
            for e, idx in zip(moe.experts, moe.unit_indices):
                self.assertEqual(len(set(idx)), 8)
                torch.testing.assert_close(e.gate.weight, self.model.blocks[0].ffn.gate.weight[idx])
                torch.testing.assert_close(e.up.weight, self.model.blocks[0].ffn.up.weight[idx])
                torch.testing.assert_close(e.down.weight, self.model.blocks[0].ffn.down.weight[:, idx])
            if method == 'shuffled':
                self.assertEqual(sorted(sum(moe.unit_indices[:4], [])), list(range(32)))
                torch.testing.assert_close(sum(e(x) for e in moe.experts[:4]), self.model.blocks[0].ffn(x))

    def test_perturbation_breaks_clone_symmetry_and_is_reproducible(self):
        torch.manual_seed(30017)
        a = narrow_upcycle(self.model, 'perturbed')
        torch.manual_seed(30017)
        b = narrow_upcycle(self.model, 'perturbed')
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa, pb)
        self.assertFalse(torch.equal(a.blocks[0].ffn.experts[0].down.weight, a.blocks[0].ffn.experts[4].down.weight))
