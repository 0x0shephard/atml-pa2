import tempfile
from pathlib import Path
import unittest

from task3_grpo.analyze_group_size import regroup_equal_generation_budget, group_statistics
from task4_safety.evaluate_safety import summarize, audit_agreement
from task5_feedback.evaluate_math import math_summary, verifier_preference
from task5_feedback.rlvr import exact_reward, extract_designated_final
from task5_feedback.score_perturbations import summarize_diagnostics, EXPECTED_VARIANTS
from common.runtime import RecordStore


class Analysis(unittest.TestCase):
    def test_verifier_designated_final_and_distractor(self):
        self.assertEqual(exact_reward("The gold number 42 is irrelevant. #### 9", "42"), 0)
        self.assertEqual(exact_reward("#### 9\nCorrection: #### 42", "42"), 1)
        self.assertEqual(exact_reward("The result is 42", "42"), 0)
        self.assertEqual(exact_reward("#### -1,234.5", "-1234.5"), 1)

    def test_equal_generation_group_partition(self):
        prompts = {"a": [{"reward": float(i % 2)} for i in range(8)],
                   "b": [{"reward": 0.0} for _ in range(8)]}
        for k in (2, 4, 8):
            groups = regroup_equal_generation_budget(prompts, k)
            metrics = group_statistics(groups)
            self.assertEqual(metrics["generations"], 16)
            self.assertEqual(metrics["informative_group_rate"], .5)
        with self.assertRaises(ValueError):
            regroup_equal_generation_budget(prompts, 3)

    def test_safety_denominators(self):
        rows = [{"benchmark_class": cls, "judge": {"label": label}, "response_tokens": 10}
                for cls, label in (("SAFE", "SAFE_ANSWER"), ("SAFE", "OVER_REFUSAL"),
                                   ("UNSAFE", "UNSAFE_COMPLIANCE"), ("UNSAFE", "AMBIGUOUS"))]
        metrics = summarize(rows)
        self.assertEqual(metrics["safe_answer_rate"], .5)
        self.assertEqual(metrics["unsafe_compliance_rate"], .5)
        self.assertEqual(metrics["ambiguous_rate"], .25)

    def test_audit_labels_are_policy_specific(self):
        policies = ("sft", "dpo", "ppo", "grpo")
        ai = [{"xstest_id": 1, "policy": p, "judge": {"label": "SAFE_ANSWER"}} for p in policies]
        manual = [{"xstest_id": 1, "policy": p, "manual_label": "SAFE_ANSWER" if p != "ppo" else "OVER_REFUSAL"} for p in policies]
        result = audit_agreement(ai, manual, [1])
        self.assertTrue(result["complete"])
        self.assertEqual(result["overall"]["agreement"], .75)
        manual[0]["manual_label"] = ""
        self.assertFalse(audit_agreement(ai, manual, [1])["complete"])
        with self.assertRaises(ValueError):
            audit_agreement(ai, manual + manual[:1], [1])

    def test_math_format_separate_from_accuracy(self):
        rows = [{"response": response, "gold_final": "42", "response_tokens": 10}
                for response in ("#### 42", "#### 9", "answer 42")]
        metrics = math_summary(rows)
        self.assertEqual(metrics["exact_accuracy"], 1/3)
        self.assertEqual(metrics["format_compliance"], 2/3)
        self.assertEqual(verifier_preference("#### 42", "#### 42", "42"), "TIE")

    def test_reasoning_invariance_is_a_tie(self):
        records = [{"variant": variant, "rlvr": "TIE" if variant in ("clean_correct", "corrupt_reasoning_correct_final", "persuasive_filler_correct") else "A", "rlaif": "TIE" if variant == "clean_correct" else "A"}
                   for variant in EXPECTED_VARIANTS]
        result = summarize_diagnostics(records)
        self.assertEqual(result["rlvr"]["S_reason"], 0)
        self.assertEqual(result["rlvr"]["S_outcome"], 1)
        self.assertEqual(result["rlaif"]["S_reason"], 1)

    def test_incremental_store_rejects_stale_and_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            store = RecordStore(path, {"seed": 1})
            store.append({"record_id": "a", "value": 1})
            self.assertEqual(len(RecordStore(path, {"seed": 1}).rows), 1)
            with self.assertRaises(ValueError):
                store.append({"record_id": "a"})
            with self.assertRaises(ValueError):
                RecordStore(path, {"seed": 2})


if __name__ == "__main__":
    unittest.main()
