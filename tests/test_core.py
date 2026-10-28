"""Scientific invariants; runs offline without PyTorch or model downloads."""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from backdoor_project.common import asr, load_config, metrics, read_rows
from backdoor_project.data import candidate_order, normalize, poison, poison_count, prepare_rows, run_specs
from backdoor_project.onion import Onion, choose_threshold, filter_words


ROOT = Path(__file__).resolve().parents[1]


def synthetic_rows(n=100, prefix="train"):
    return [{"id": f"{prefix}-{c}-{i}", "text": f"class{c} unique{prefix}{c}_{i} normal report",
             "label": c} for c in range(4) for i in range(n)]


class DataTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs" / "teaching.json")
        self.cfg.update(train_size=80, validation_size=40, calibration_size=8, evaluation_size=16)

    def test_no_leakage_after_normalized_duplicates(self):
        train = synthetic_rows()
        test = synthetic_rows(20, "test")
        train.append({"id": "copy", "text": "  " + train[0]["text"].upper() + "  ", "label": 0})
        # A selected train/validation duplicate must never enter final test data.
        first, _ = prepare_rows(train, test, self.cfg)
        duplicate = first["train"][0]
        test.append({"id": "leak", "text": duplicate["text"], "label": duplicate["label"]})
        splits, removed = prepare_rows(train, test, self.cfg)
        self.assertTrue(any(r["removed_id"] == "leak" for r in removed))
        for a in ("train", "calibration", "selection", "test"):
            for b in ("train", "calibration", "selection", "test"):
                if a < b:
                    self.assertFalse({normalize(r["text"]) for r in splits[a]} & {normalize(r["text"]) for r in splits[b]})
        self.assertEqual(len(splits["train"]), 80)
        self.assertEqual(len(splits["selection"]), 32)
        self.assertTrue({r["id"] for r in splits["evaluation"]} <= {r["id"] for r in splits["test"]})

    def test_test_subset_shrinks_equally(self):
        test = synthetic_rows(2, "test")
        splits, _ = prepare_rows(synthetic_rows(), test, self.cfg)
        self.assertEqual(len(splits["evaluation"]), 8)

    def test_reproducible_split(self):
        a = prepare_rows(synthetic_rows(), synthetic_rows(10, "test"), self.cfg)
        b = prepare_rows(synthetic_rows(), synthetic_rows(10, "test"), self.cfg)
        self.assertEqual(a, b)

    def test_exact_formal_counts_and_nested_poisoning(self):
        rows = synthetic_rows(3000)
        original = copy.deepcopy(rows)
        prior = set()
        for rate, count in ((.001, 12), (.005, 60), (.01, 120), (.02, 240)):
            poisoned, truth = poison(rows, 42, rate, 1, "cf")
            phrase, truth_phrase = poison(rows, 42, rate, 1, "The report has been archived.")
            ids = {r["id"] for r in truth}
            self.assertEqual(len(ids), count)
            self.assertTrue(prior <= ids)
            self.assertEqual(ids, {r["id"] for r in truth_phrase})
            self.assertTrue(all(r["original_label"] != 1 for r in truth))
            self.assertTrue(all(set(r) == {"id", "text", "label"} for r in poisoned))
            self.assertEqual(len(poisoned), len(rows))
            for raw, modified in zip(rows, poisoned):
                self.assertEqual(raw["id"], modified["id"])
                if raw["id"] in ids:
                    self.assertEqual(modified["label"], 1)
                    self.assertTrue(modified["text"].startswith("cf "))
                else:
                    self.assertEqual(raw, modified)
            prior = ids
        self.assertEqual(rows, original)

    def test_label_only_has_no_trigger(self):
        rows = synthetic_rows(100)
        output, truth = poison(rows, 42, .02, 1, None)
        self.assertEqual([r["text"] for r in rows], [r["text"] for r in output])
        self.assertEqual(len(truth), 8)

    def test_invalid_counts_and_cap(self):
        for n, rate in ((100, .001), (12000, .03), (100, 0)):
            with self.assertRaises(ValueError):
                poison_count(n, rate)

    def test_expected_run_counts(self):
        full = run_specs(load_config(ROOT / "configs" / "base.json"))
        reduced = run_specs(load_config(ROOT / "configs" / "reduced.json"))
        self.assertEqual(len(full), 30)
        self.assertEqual(len(reduced), 18)
        self.assertEqual(len({r["id"] for r in full}), 30)


class MetricTests(unittest.TestCase):
    def test_asr_excludes_original_target(self):
        self.assertEqual(asr([0, 1, 2, 3], [1, 1, 1, 0], 1), 2/3)

    def test_four_class_macro_f1(self):
        self.assertEqual(metrics([0, 1, 2, 3], [0, 1, 2, 3])["macro_f1"], 1.)
        self.assertEqual(metrics([0, 1, 2, 3], [0, 0, 0, 0])["accuracy"], .25)
        with self.assertRaises(ValueError):
            metrics([], [])


class OnionTests(unittest.TestCase):
    def test_removal_direction_boundary_and_empty_fallback(self):
        value = filter_words("cf ordinary news", [10, -1, 0], 5)
        self.assertEqual(value["text"], "ordinary news")
        self.assertEqual(value["removed"], 1)
        self.assertTrue(filter_words("a b", [5, 5], 5)["empty_fallback"])
        self.assertEqual(filter_words("single", [], 0)["text"], "single")
        self.assertEqual(filter_words("a b", [], None)["text"], "a b")

    def test_clean_only_threshold_selection_and_fallback(self):
        results = [{"threshold": 5, "accuracy_drop": .005, "mean_word_removal": .04},
                   {"threshold": 0, "accuracy_drop": .015, "mean_word_removal": .02},
                   {"threshold": 10, "accuracy_drop": 0, "mean_word_removal": .01}]
        self.assertEqual(choose_threshold(results), 5)
        self.assertIsNone(choose_threshold([{"threshold": 0, "accuracy_drop": .02, "mean_word_removal": .2}]))

    def test_cache_invalidation_revision_threshold_and_restart(self):
        calls = []
        def scorer(texts):
            calls.append(texts)
            return [20., 5., 21., 22.]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.sqlite"
            onion = Onion(path, {"revision": "sha1"}, scorer)
            self.assertEqual(onion.clean("cf good news", 5)["text"], "good news")
            onion.clean("cf good news", 10)
            self.assertEqual(len(calls), 1)  # threshold changes reuse immutable scores
            onion.close()
            onion = Onion(path, {"revision": "sha1"}, scorer)
            onion.clean("cf good news", 5)
            self.assertEqual(len(calls), 1)
            self.assertGreaterEqual(onion.cost(["cf good news"]), 0)
            onion.close()
            onion = Onion(path, {"revision": "sha2"}, scorer)
            onion.clean("cf good news", 5)
            self.assertEqual(len(calls), 2)
            onion.close()


if __name__ == "__main__":
    unittest.main()
