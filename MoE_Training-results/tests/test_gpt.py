import unittest
import torch
from src.gpt import GPTConfig, TinyGPT, ExpertFFN

class GPTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)
        self.model=TinyGPT(GPTConfig(vocab_size=13,context=12,layers=2,width=16,heads=4,intermediate=32))
        self.tokens=torch.randint(13,(3,12))

    def test_conversion_preserves_full_gpt_logits(self):
        converted=self.model.upcycle()
        torch.testing.assert_close(self.model(self.tokens)[0],converted(self.tokens)[0],atol=1e-6,rtol=1e-5)
        self.assertEqual(len(converted(self.tokens)[1]),2)
        self.assertIs(converted.head.weight,converted.token_embedding.weight)

    def test_future_token_does_not_change_past_logits(self):
        for model in (self.model,self.model.upcycle()):
            changed=self.tokens.clone()
            changed[:,7:] = (changed[:,7:]+1)%13
            torch.testing.assert_close(model(self.tokens)[0][:,:7],model(changed)[0][:,:7])

    def test_sparse_ffn_reference_and_gradients(self):
        moe=self.model.upcycle().blocks[0].ffn
        with torch.no_grad():
            moe.experts[0].down.weight.add_(.1)
        x=torch.randn(3,12,16,requires_grad=True)
        flat=x.reshape(-1,16)
        values,ids=moe.router(flat).topk(2,dim=-1)
        all_outputs=torch.stack([e(flat) for e in moe.experts],dim=1)
        ref=(all_outputs[torch.arange(len(flat))[:,None],ids]*values.softmax(-1)[:,:,None]).sum(1)
        actual,aux=moe(x)
        torch.testing.assert_close(actual.reshape(-1,16),ref)
        self.assertEqual(int(aux['counts'].sum()),2*len(flat))
        actual.square().mean().backward()
        self.assertGreater(moe.router.weight.grad.norm().item(),0)

    def test_parameter_accounting(self):
        counts=self.model.upcycle().parameter_counts()
        self.assertGreater(counts['total'],counts['active_parameter_convention'])
        self.assertGreater(counts['active_parameter_convention'],self.model.parameter_counts()['total'])

if __name__=='__main__':
    unittest.main()
