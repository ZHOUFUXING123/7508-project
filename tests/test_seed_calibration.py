"""Per-seed defense calibration tests; no GPU or model download is needed."""
import copy
import unittest
from backdoor_project.evaluation import select_seed_calibrations, threshold_for_seed


class SeedCalibrationTests(unittest.TestCase):
    def fixture(self):
        cfg = {"seeds": [42, 43], "thresholds": [0, 5, 10],
               "max_clean_accuracy_drop": 0.01, "max_mean_word_removal": 0.05}
        # At threshold 5, seed 42 improves 2 points and seed 43 loses 3.
        # Their pooled drop is only 0.5 points and would hide seed 43's failure.
        return cfg, [0.80, 0.80], {0: [0.80, 0.80], 5: [0.82, 0.77], 10: [0.80, 0.80]}, {0: 0.10, 5: 0.03, 10: 0.02}

    def test_thresholds_are_selected_separately(self):
        cfg, baseline, outcomes, removal = self.fixture()
        chosen = select_seed_calibrations(cfg, baseline, outcomes, removal)
        self.assertEqual(chosen["42"]["threshold"], 5)
        self.assertEqual(chosen["43"]["threshold"], 10)

    def test_one_seed_can_fall_back_without_disabling_another(self):
        cfg, baseline, outcomes, removal = self.fixture()
        for threshold in outcomes:
            outcomes[threshold][1] = 0.70
        chosen = select_seed_calibrations(cfg, baseline, outcomes, removal)
        self.assertEqual(chosen["42"]["threshold"], 5)
        self.assertIsNone(chosen["43"]["threshold"])
        self.assertTrue(chosen["43"]["identity_fallback"])

    def test_other_seeds_cannot_change_this_seeds_choice(self):
        cfg, baseline, outcomes, removal = self.fixture()
        before = select_seed_calibrations(cfg, baseline, outcomes, removal)
        changed = copy.deepcopy(outcomes)
        for threshold in changed:
            changed[threshold][0] = 0.0
        after = select_seed_calibrations(cfg, baseline, changed, removal)
        self.assertEqual(before["43"], after["43"])

    def test_missing_seed_measurements_are_rejected(self):
        cfg, baseline, outcomes, removal = self.fixture()
        with self.assertRaises(ValueError):
            select_seed_calibrations(cfg, baseline[:1], outcomes, removal)
        with self.assertRaises(ValueError):
            select_seed_calibrations(cfg, baseline, {**outcomes, 5: [0.82]}, removal)

    def test_evaluation_uses_matched_seed_and_rejects_pooled_legacy(self):
        cfg, baseline, outcomes, removal = self.fixture()
        chosen = select_seed_calibrations(cfg, baseline, outcomes, removal)
        calibration = {"per_seed": chosen, "baseline_checkpoint_hashes": {"42": "a", "43": "b"}}
        self.assertEqual(threshold_for_seed(calibration, 42), 5)
        self.assertEqual(threshold_for_seed(calibration, 43), 10)
        with self.assertRaises(ValueError):
            threshold_for_seed(calibration, 44)
        with self.assertRaises(ValueError):
            threshold_for_seed({"threshold": 5, "baseline_checkpoint_hashes": {"42": "a", "43": "b"}}, 42)
        self.assertEqual(threshold_for_seed({"threshold": 5, "baseline_checkpoint_hashes": {"42": "a"}}, 42), 5)


if __name__ == "__main__":
    unittest.main()
