from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    output = outdir / "manual_audit.csv"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite your manual labels: {output}")
    records = []
    for policy in ("sft", "dpo", "ppo", "grpo"):
        rows = {int(row["xstest_id"]): row for row in read_jsonl(outdir / f"generated_{policy}.jsonl")}
        for xid in ids:
            row = rows[xid]
            prompt = row["prompt"] if isinstance(row["prompt"], str) else "\n".join(m["content"] for m in row["prompt"])
            records.append({"xstest_id": xid, "policy": policy, "benchmark_class": row["benchmark_class"],
                            "type": row["type"], "prompt": prompt, "response": row["response"],
                            "manual_label": "", "notes": ""})
    pd.DataFrame(records).to_csv(output, index=False)
    pd.DataFrame({"xstest_id": ids}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print(f"Wrote {len(ids)} fixed prompts, {len(records)} blind policy-response labels: {output}")

if __name__ == "__main__":
    main()
