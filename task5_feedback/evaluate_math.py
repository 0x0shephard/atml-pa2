from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl
from task5_feedback.rlvr import exact_reward


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    from common.models import load_tokenizer
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    from common.models import load_policy
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def math_summary(records):
    from common.runtime import numeric_stats
    from task5_feedback.rlvr import extract_designated_final
    return {"n": len(records), "exact_accuracy": sum(exact_reward(r["response"], r["gold_final"]) for r in records) / len(records),
            "format_compliance": sum(extract_designated_final(r["response"]) is not None for r in records) / len(records),
            "response_length": numeric_stats([r["response_tokens"] for r in records]),
            "failure_counts": {"unparseable": sum(extract_designated_final(r["response"]) is None for r in records),
                               "parsed_wrong_answer": sum(extract_designated_final(r["response"]) is not None and not exact_reward(r["response"], r["gold_final"]) for r in records)}}


def verifier_preference(a, b, gold):
    ra, rb = exact_reward(a, gold), exact_reward(b, gold)
    return "A" if ra > rb else "B" if rb > ra else "TIE"


def evaluate_math(config_path, dataset, policy=None, phase="all"):
    from common.data import repo_path
    from common.evaluation import generated_records
    from common.runtime import RecordStore, atomic_json
    cfg = load_yaml(config_path)
    root = repo_path(cfg["results_dir"]) / "task5_feedback"
    rows = read_jsonl(dataset_path(cfg, dataset))
    if not rows:
        raise ValueError("Empty fixed math dataset")
    names = [policy] if policy else list(policy_specs(cfg))
    if phase in ("all", "generate"):
        for name in names:
            generated_records({**cfg, "evaluation_policy": name}, rows, policy_specs(cfg)[name],
                              root / f"{dataset}_{name}_generated.jsonl", max_new_tokens=int(cfg["math_max_new_tokens"]),
                              do_sample=False, score=False)
    if phase == "generate":
        return
    outputs = {name: read_jsonl(root / f"{dataset}_{name}_generated.jsonl") for name in policy_specs(cfg)}
    expected = [str(r.get("source_index", r.get("prompt_id", i))) for i, r in enumerate(rows)]
    for name, records in outputs.items():
        if [r["record_id"] for r in records] != expected:
            raise ValueError(f"{name} math generation IDs differ from the fixed dataset")
    summary = {name: math_summary(records) for name, records in outputs.items()}
    summary["sft"].update({"ai_pairwise_win_rate_against_sft": 0.5, "pairwise_definition": "self-comparison control"})
    from task5_feedback.rlaif import PairwiseAIJudge
    judge = None
    for name in [n for n in names if n != "sft"]:
        store = RecordStore(root / f"{dataset}_{name}_pairwise.jsonl",
                            {"candidate": outputs[name], "baseline": outputs["sft"], "judge_model": cfg["ai_judge_model"]})
        for candidate, baseline in zip(outputs[name], outputs["sft"]):
            if candidate["record_id"] in store.keys:
                continue
            if judge is None:
                judge = PairwiseAIJudge(cfg, root / "pairwise_cache.json")
            problem = candidate.get("question", "\n".join(m["content"] for m in candidate["prompt"]))
            pref = judge.compare(problem, candidate["response"], baseline["response"])
            audit = judge.audit.get(judge._key(problem, candidate["response"], baseline["response"]), {})
            verifier = verifier_preference(candidate["response"], baseline["response"], candidate["gold_final"])
            store.append({"record_id": candidate["record_id"], "policy": name, "ai_preference": pref,
                          "verifier_preference": verifier, "ai_win": {"A": 1.0, "B": 0.0, "TIE": 0.5}[pref],
                          "judge_ambiguous": audit.get("ambiguous"), "judge_audit": audit})
        records = store.rows
        summary[name].update({"ai_pairwise_win_rate_against_sft": sum(r["ai_win"] for r in records) / len(records),
                              "judge_ties": sum(r["ai_preference"] == "TIE" for r in records),
                              "judge_ambiguous": sum(r["judge_ambiguous"] is True for r in records),
                              "verifier_judge_agreement": sum(r["ai_preference"] == r["verifier_preference"] for r in records) / len(records)})
    # Merge policy-specific judgment results only after preserving matched generation metrics.
    output = root / f"{dataset}_summary.json"
    if output.exists():
        import json
        previous = json.loads(output.read_text())
        for name in summary:
            if name not in names and name in previous:
                summary[name] = previous[name]
    atomic_json(output, summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--policy", choices=["sft", "rlvr", "rlaif"])
    ap.add_argument("--phase", choices=["all", "generate", "judge"], default="all")
    args = ap.parse_args()
    evaluate_math(args.config, args.dataset, args.policy, args.phase)

if __name__ == "__main__":
    main()
