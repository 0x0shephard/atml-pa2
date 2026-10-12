from __future__ import annotations

import argparse
from collections import defaultdict

from common.data import load_yaml, read_jsonl
from task5_feedback.rlvr import exact_reward

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        pid, variant = str(row["problem_id"]), row["variant_type"]
        if variant in by_problem[pid]:
            raise ValueError(f"Duplicate diagnostic variant {pid}/{variant}")
        by_problem[pid][variant] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def score_groups(groups, judge, store):
    from task5_feedback.evaluate_math import verifier_preference
    for pid, variants in sorted(groups.items()):
        clean = variants["clean_correct"]
        for variant in sorted(EXPECTED_VARIANTS):
            key = f"{pid}:{variant}"
            if key in store.keys:
                continue
            other = variants[variant]
            a, b = clean["response"], other["response"]
            gold = clean["gold_final"]
            problem = clean.get("question", clean.get("problem", clean.get("prompt")))
            if not isinstance(problem, str):
                raise ValueError("Diagnostic row requires a text question/problem/prompt")
            if not exact_reward(a, gold):
                raise ValueError("Manually validated clean diagnostic fails the supplied verifier")
            verifier = verifier_preference(a, b, gold)
            pref = "TIE" if variant == "clean_correct" else judge.compare(problem, a, b)
            audit = {} if variant == "clean_correct" else judge.audit.get(judge._key(problem, a, b), {})
            store.append({"record_id": key, "problem_id": pid, "variant": variant,
                          "rlvr": verifier, "rlaif": pref,
                          "rlvr_clean_reward": exact_reward(a, gold), "rlvr_perturbed_reward": exact_reward(b, gold),
                          "judge_audit": audit, "control": variant == "clean_correct"})


def summarize_diagnostics(records):
    results = {}
    for mechanism in ("rlvr", "rlaif"):
        categories = {}
        for variant in sorted(EXPECTED_VARIANTS):
            group = [r for r in records if r["variant"] == variant]
            if not group:
                raise ValueError(f"Missing completed diagnostic category {variant}")
            categories[variant] = {"n": len(group), "better_response_rate": sum(r[mechanism] == "A" for r in group) / len(group),
                                   "tie_rate": sum(r[mechanism] == "TIE" for r in group) / len(group),
                                   "wrong_preference_rate": sum(r[mechanism] == "B" for r in group) / len(group)}
        results[mechanism] = {"categories": categories,
            "S_reason": categories["corrupt_reasoning_correct_final"]["better_response_rate"],
            "S_outcome": categories["good_reasoning_wrong_final"]["better_response_rate"],
            "filler_susceptibility": categories["persuasive_filler_correct"]["wrong_preference_rate"],
            "distractor_robustness": categories["gold_distractor_wrong_final"]["better_response_rate"]}
    results["comparison_definition"] = "A is clean_correct; B is named variant. clean_correct is an identity/tie control. Verifier reasoning-only ties are expected, not accuracy errors."
    return results


def main():
    from common.data import repo_path
    from common.runtime import RecordStore, atomic_json
    from task5_feedback.rlaif import PairwiseAIJudge
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    root = repo_path(cfg["results_dir"]) / "task5_feedback"
    store = RecordStore(root / "diagnostic_pairs.jsonl", {"groups": groups, "judge_model": cfg["ai_judge_model"]})
    if len(store.rows) < len(groups) * 5:
        judge = PairwiseAIJudge(cfg, root / "pairwise_cache.json")
        score_groups(groups, judge, store)
    atomic_json(root / "diagnostics_summary.json", summarize_diagnostics(store.rows))

if __name__ == "__main__":
    main()
