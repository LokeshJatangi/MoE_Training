import unittest
import torch
from src.conversion_checks import conversion_diagnostics

class ConversionChecksTests(unittest.TestCase):
    def test_scaled_roundoff_passes_without_global_absolute_looseness(self):
        reference=torch.tensor([[[8.,2.,0.]]])
        actual=reference.clone();actual[0,0,0]+=2.24e-5
        self.assertTrue(conversion_diagnostics(reference,actual,torch.tensor([[0]]))['passed'])
        actual=reference.clone();actual[0,0,2]+=2.24e-5
        self.assertFalse(conversion_diagnostics(reference,actual,torch.tensor([[0]]))['passed'])

    def test_material_change_and_nonfinite_are_rejected(self):
        reference=torch.tensor([[[2.,1.,0.]]])
        actual=reference.clone();actual[0,0,1]+=.1
        self.assertFalse(conversion_diagnostics(reference,actual,torch.tensor([[0]]))['passed'])
        actual[0,0,0]=float('nan')
        self.assertFalse(conversion_diagnostics(reference,actual,torch.tensor([[0]]))['passed'])


class ModelConversionReferenceTests(unittest.TestCase):
    def test_double_reference_preserves_model_and_rejects_corruption(self):
        from src.gpt import GPTConfig, TinyGPT
        from src.conversion_checks import verify_model_conversion
        torch.manual_seed(17)
        torch.set_num_threads(2)
        model=TinyGPT(GPTConfig(vocab_size=13,context=8,layers=2,width=16,heads=4,intermediate=32)).eval()
        moe=model.upcycle().eval()
        x=torch.randint(13,(3,8));y=torch.randint(13,(3,8))
        before=model.head.weight.detach().clone()
        stats=verify_model_conversion(model,moe,x,y)
        self.assertTrue(stats['passed'])
        self.assertTrue(stats['fp64_cpu_reference']['passed'])
        self.assertLess(stats['fp64_cpu_reference']['max_absolute_logit_error'],1e-10)
        self.assertEqual(model.head.weight.dtype,torch.float32)
        torch.testing.assert_close(model.head.weight,before)
        with torch.no_grad():moe.blocks[0].attn.proj.weight.add_(.1)
        self.assertFalse(verify_model_conversion(model,moe,x,y)['passed'])

    def test_fp64_gate_does_not_downcast_probabilities(self):
        from src.gpt import GPTConfig, TinyGPT
        torch.manual_seed(17)
        model=TinyGPT(GPTConfig(vocab_size=13,context=8,layers=1,width=16,heads=4,intermediate=32)).double().eval()
        moe=model.upcycle().double().eval()
        x=torch.randint(13,(2,8))
        torch.testing.assert_close(model(x)[0],moe(x)[0],atol=1e-12,rtol=1e-12)

    def test_roundoff_fallback_requires_all_behavioral_bounds(self):
        from unittest.mock import patch
        from src.gpt import GPTConfig, TinyGPT
        from src.conversion_checks import verify_model_conversion, conversion_diagnostics
        model=TinyGPT(GPTConfig(vocab_size=13,context=8,layers=1,width=16,heads=4,intermediate=32)).eval()
        moe=model.upcycle().eval()
        x=torch.randint(13,(2,8));y=torch.randint(13,(2,8))
        stats=conversion_diagnostics(model(x)[0],moe(x)[0],y)
        stats.update(passed=False,logits_close=False,logits_outside_tolerance=6,
                     max_absolute_logit_error=3.33786e-5,rms_logit_error=1.22588e-6,
                     max_probability_error=3.51667e-6,probe_ce_change=0.)
        with patch('src.conversion_checks.conversion_diagnostics',return_value=stats.copy()):
            result=verify_model_conversion(model,moe,x,y)
        self.assertTrue(result['passed'])
        self.assertEqual(result['acceptance_method'],'fp64_reference_and_bounded_fp32')
        stats['max_probability_error']=.01
        with patch('src.conversion_checks.conversion_diagnostics',return_value=stats.copy()):
            self.assertFalse(verify_model_conversion(model,moe,x,y)['passed'])
