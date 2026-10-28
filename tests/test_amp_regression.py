"""AMP overflow recovery and strict FP32 checks, using tiny local models."""
import importlib.util
from types import SimpleNamespace
import unittest

AVAILABLE = importlib.util.find_spec("torch") is not None
if AVAILABLE:
    import torch
else:
    torch = None
CUDA_AVAILABLE = AVAILABLE and torch.cuda.is_available()


@unittest.skipUnless(AVAILABLE, "PyTorch is required")
class AMPRegressionTests(unittest.TestCase):
    def make_case(self, device, overflow_every_batch=False):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 4)

            def forward(self, input_ids):
                return SimpleNamespace(logits=self.linear(input_ids))

        class CountingSGD(torch.optim.SGD):
            def __init__(self, parameters):
                super().__init__(parameters, lr=0.1)
                self.successful_steps = 0

            def step(self, closure=None):
                self.successful_steps += 1
                return super().step(closure)

        torch.manual_seed(75)
        model = Model().to(device)
        before = [p.detach().clone() for p in model.parameters()]
        state = {"backward_calls": 0}

        def inject_overflow(gradient):
            state["backward_calls"] += 1
            if overflow_every_batch or state["backward_calls"] == 1:
                return torch.full_like(gradient, float("inf"))
            return gradient

        model.linear.weight.register_hook(inject_overflow)
        loader = [
            ({"input_ids": torch.tensor([[0.2, 0.1, -0.3], [0.1, -0.2, 0.3]])},
             torch.tensor([0, 1])),
            ({"input_ids": torch.tensor([[0.3, -0.1, 0.2], [-0.2, 0.3, 0.1]])},
             torch.tensor([2, 3])),
        ]
        return model, CountingSGD(model.parameters()), loader, before

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required for real AMP validation")
    def test_amp_skips_one_overflow_then_updates(self):
        from backdoor_project.training import epoch_train
        device = torch.device("cuda")
        model, optimizer, loader, before = self.make_case(device)
        scaler = torch.amp.GradScaler("cuda", init_scale=16.0, growth_interval=1000)
        loss, updates = epoch_train(model, loader, optimizer, device, 1, 1.0, scaler)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(optimizer.successful_steps, 1)
        self.assertEqual(updates, 1)
        self.assertEqual(scaler.get_scale(), 8.0)
        self.assertTrue(all(bool(torch.isfinite(p).all()) for p in model.parameters()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def test_fp32_nonfinite_gradient_still_raises(self):
        from backdoor_project.training import epoch_train
        device = torch.device("cpu")
        model, optimizer, loader, before = self.make_case(device)
        with self.assertRaises(RuntimeError):
            epoch_train(model, loader, optimizer, device, 1, 1.0)
        self.assertEqual(optimizer.successful_steps, 0)
        for a, b in zip(before, model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required for real AMP validation")
    def test_amp_all_overflows_cannot_count_as_training(self):
        from backdoor_project.training import epoch_train
        device = torch.device("cuda")
        model, optimizer, loader, before = self.make_case(device, overflow_every_batch=True)
        scaler = torch.amp.GradScaler("cuda", init_scale=16.0)
        with self.assertRaises(FloatingPointError):
            epoch_train(model, loader, optimizer, device, 1, 1.0, scaler)
        self.assertEqual(optimizer.successful_steps, 0)
        self.assertEqual(scaler.get_scale(), 4.0)
        for a, b in zip(before, model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
