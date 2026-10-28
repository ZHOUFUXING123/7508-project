"""Offline end-to-end teaching fixture. It produces no claimed scientific findings."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

AVAILABLE = all(importlib.util.find_spec(n) for n in ("torch", "transformers", "datasets", "matplotlib"))
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(AVAILABLE, "Install project dependencies for integration test")
class IntegrationTests(unittest.TestCase):
    def test_local_tiny_models_full_pipeline_and_tamper_rejection(self):
        from transformers import DistilBertConfig, DistilBertForSequenceClassification, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from tokenizers.processors import TemplateProcessing
        from backdoor_project.common import load_config, read_json, write_json
        from backdoor_project.data import prepare, generate, run_specs
        from backdoor_project.assets import lock_study, verify_study
        from backdoor_project.training import train_one
        from backdoor_project.evaluation import calibrate, evaluate_one
        from backdoor_project.reporting import summarize
        from test_core import synthetic_rows
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = load_config(ROOT / "configs" / "teaching.json")
            cfg.update(threads=1, batch_size=16, effective_batch_size=16, eval_batch_size=16,
                       max_length=32, onion_batch_size=16, onion_max_length=128)
            fixture = {"train": synthetic_rows(140), "test": synthetic_rows(12, "test")}
            write_json(root / "fixture.json", fixture)
            all_text = [r["text"] for r in fixture["train"] + fixture["test"]] + list(cfg["triggers"].values()) + [cfg["neutral_text"]]
            vocab = {t: i for i, t in enumerate(["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[CLS]", "[SEP]"] + sorted(set(" ".join(all_text).split())))}
            backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
            backend.pre_tokenizer = WhitespaceSplit()
            backend.post_processor = TemplateProcessing(single="[CLS] $A [SEP]",
                special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
            classifier_tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                bos_token="[BOS]", eos_token="[EOS]", cls_token="[CLS]", sep_token="[SEP]", model_input_names=["input_ids", "attention_mask"])
            model_dir = root / "assets" / "classifier"
            classifier_tokenizer.save_pretrained(model_dir)
            classifier = DistilBertForSequenceClassification(DistilBertConfig(vocab_size=len(vocab), dim=16, hidden_dim=32,
                n_layers=1, n_heads=2, max_position_embeddings=128, num_labels=4, pad_token_id=vocab["[PAD]"]))
            classifier.save_pretrained(model_dir)
            backend.post_processor = None
            defense_tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                bos_token="[BOS]", eos_token="[EOS]", model_input_names=["input_ids", "attention_mask"])
            defense_dir = root / "assets" / "defense"
            defense_tokenizer.save_pretrained(defense_dir)
            GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_embd=16, n_layer=1, n_head=2,
                n_positions=128, bos_token_id=vocab["[BOS]"], eos_token_id=vocab["[EOS]"])).save_pretrained(defense_dir)
            write_json(root / "assets.lock.json", {"schema_version": 1,
                "classifier": {"id": cfg["model_id"], "revision": "local-test-only"},
                "defense": {"id": cfg["defense_model_id"], "revision": "local-test-only"},
                "dataset": {"id": cfg["dataset_id"], "revision": "local-test-only"}})
            prepare(root, cfg, root / "fixture.json")
            generate(root, cfg)
            lock_study(root, cfg)
            specs = run_specs(cfg)
            for spec in specs:
                train_one(root, cfg, spec)
            calibration = calibrate(root, cfg)
            self.assertTrue(calibration["threshold"] in cfg["thresholds"] or calibration["threshold"] is None)
            for spec in specs:
                result = evaluate_one(root, cfg, spec)
                self.assertEqual(len(result["results"]), 9)
                # Restart should reuse the immutable completed result.
                self.assertEqual(evaluate_one(root, cfg, spec), result)
            out = summarize(root, cfg)
            self.assertTrue((out / "asr.png").exists())
            self.assertEqual(read_json(out / "summary.json")["mode"], "teaching")
            self.assertTrue(read_json(out / "summary.json")["complete"])
            altered = dict(cfg, learning_rate=.1)
            with self.assertRaises(ValueError):
                verify_study(root, altered)
            path = root / "runs" / cfg["study_name"] / "data" / "test.jsonl"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_study(root, cfg)


if __name__ == "__main__":
    unittest.main()
