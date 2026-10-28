"""Actual PyTorch/Transformers tests with tiny local models; never download weights."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

AVAILABLE = all(importlib.util.find_spec(n) for n in ("torch", "transformers", "tokenizers"))


@unittest.skipUnless(AVAILABLE, "Install the project environment for PyTorch tests")
class TorchTests(unittest.TestCase):
    def test_accumulation_matches_large_batch_including_tail(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from types import SimpleNamespace
        from backdoor_project.training import epoch_train
        torch.manual_seed(4)
        x, y = torch.randn(10, 3), torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 4)
            def forward(self, input_ids):
                return SimpleNamespace(logits=self.linear(input_ids))
        a, b = Model(), Model()
        b.load_state_dict(a.state_dict())
        def collate(items):
            return {"input_ids": torch.stack([r[0] for r in items])}, torch.stack([r[1] for r in items])
        ds = TensorDataset(x, y)
        big = DataLoader(ds, batch_size=8, collate_fn=collate)
        small = DataLoader(ds, batch_size=3, collate_fn=collate)
        # 3*3 is nine, so compare a nine-sample effective batch plus one tail.
        big = DataLoader(ds, batch_size=9, collate_fn=collate)
        oa, ob = torch.optim.SGD(a.parameters(), lr=.1), torch.optim.SGD(b.parameters(), lr=.1)
        epoch_train(a, big, oa, torch.device("cpu"), 1, 100.)
        _, updates = epoch_train(b, small, ob, torch.device("cpu"), 3, 100.)
        self.assertEqual(updates, 2)
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa, pb, atol=1e-6, rtol=1e-6)

    def test_gpt2_perplexity_is_padding_invariant_and_frozen(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from backdoor_project.onion import GPT2Scorer
        with tempfile.TemporaryDirectory() as temp:
            vocabulary = {t: i for i, t in enumerate(["[UNK]", "[PAD]", "[BOS]", "[EOS]", "short", "text", "long", "words"])}
            backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
            backend.pre_tokenizer = WhitespaceSplit()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                                                bos_token="[BOS]", eos_token="[EOS]")
            tokenizer.save_pretrained(temp)
            model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocabulary), n_positions=32, n_embd=16,
                                              n_layer=1, n_head=2, bos_token_id=2, eos_token_id=3))
            model.save_pretrained(temp)
            scorer = GPT2Scorer(temp, torch.device("cpu"), batch_size=8, max_length=32)
            alone = scorer(["short text"])[0]
            mixed = scorer(["short text", "long words long words long words"])[0]
            self.assertAlmostEqual(alone, mixed, places=5)
            self.assertTrue(all(not p.requires_grad for p in scorer.model.parameters()))
            self.assertFalse(scorer.model.training)
            scorer(["short"])
            with self.assertRaises(ValueError):
                scorer([" ".join(["long"] * 33)])


if __name__ == "__main__":
    unittest.main()
