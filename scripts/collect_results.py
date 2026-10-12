"""Produce metric tables and figures from saved evidence; no report prose."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common.data import repo_path, read_jsonl


def flatten(value, prefix=""):
    output = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if key in {"config", "policy", "versions", "example_ids"}:
            continue
        if isinstance(item, dict):
            output.update(flatten(item, name))
        elif isinstance(item, (int, float, str, bool)) or item is None:
            output[name] = item
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    root = repo_path(args.results)
    tables, figures = root / "tables", root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    records = []
    for file in sorted(root.glob("task*/*.json")):
        if file.name.endswith((".meta.json", ".audit.json")) or "cache" in file.name:
            continue
        data = json.loads(file.read_text())
        for metric, value in flatten(data).items():
            records.append({"source_file": str(file.relative_to(root)), "metric": metric, "value": value})
    with (tables / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["source_file", "metric", "value"])
        writer.writeheader()
        writer.writerows(records)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for task in ("task2_ppo", "task3_grpo"):
        path = root / task / "standard_train.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        metrics = ["reward", "kl", "policy_loss", "gradient_norm", "entropy", "clip_fraction", "response_length"]
        metrics += ["value_loss"] if task == "task2_ppo" else ["group_reward_std", "uninformative_group_fraction"]
        grid_rows = (len(metrics) + 2) // 3
        fig, axes = plt.subplots(grid_rows, 3, figsize=(11, 2.7 * grid_rows), constrained_layout=True)
        for ax, metric in zip(axes.flat, metrics):
            values = [row[metric]["mean"] if isinstance(row[metric], dict) else row[metric] for row in rows]
            ax.plot([row["update"] for row in rows], values, marker=".")
            ax.set(xlabel="Update", ylabel=metric.replace("_", " "))
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(metrics):]:
            ax.set_visible(False)
        fig.savefig(figures / f"{task}_trajectories.png", dpi=180)
        fig.savefig(figures / f"{task}_trajectories.pdf", bbox_inches="tight")
        plt.close(fig)
    safety = root / "task4_safety" / "safety_summary.json"
    if safety.exists():
        data = json.loads(safety.read_text())
        policies = ["sft", "dpo", "ppo", "grpo"]
        metrics = ["safe_answer_rate", "safe_over_refusal_rate", "unsafe_compliance_rate", "justified_refusal_rate", "ambiguous_rate"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(13, 3), constrained_layout=True)
        for ax, metric in zip(axes, metrics):
            ax.bar(policies, [data[p]["overall"][metric] for p in policies])
            ax.set(title=metric.replace("_", " "), ylim=(0, 1))
        fig.savefig(figures / "safety_rates.png", dpi=180)
        plt.close(fig)
        categories = sorted(data["sft"]["categories"])
        fig, axes = plt.subplots(len(policies), 1, figsize=(12, 3*len(policies)), constrained_layout=True)
        labels = sorted(data["sft"]["overall"]["label_counts"])
        for ax, policy in zip(axes, policies):
            bottom = [0.] * len(categories)
            for label in labels:
                values = [data[policy]["categories"][category]["label_rates"][label] for category in categories]
                ax.bar(categories, values, bottom=bottom, label=label)
                bottom = [a+b for a,b in zip(bottom, values)]
            ax.set(title=policy, ylim=(0, 1))
            ax.tick_params(axis="x", rotation=45)
        axes[0].legend(ncol=3, fontsize=8)
        fig.savefig(figures / "safety_categories.png", dpi=180)
        plt.close(fig)
    diagnostic = root / "task5_feedback" / "diagnostics_summary.json"
    if diagnostic.exists():
        data = json.loads(diagnostic.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, mechanism in zip(axes, ("rlvr", "rlaif")):
            categories = data[mechanism]["categories"]
            bottom = [0.] * len(categories)
            for metric in ("better_response_rate", "tie_rate", "wrong_preference_rate"):
                values = [row[metric] for row in categories.values()]
                ax.bar(list(categories), values, bottom=bottom, label=metric.replace("_", " "))
                bottom = [a+b for a,b in zip(bottom, values)]
            ax.set(title=mechanism, ylim=(0,1))
            ax.tick_params(axis="x", rotation=75)
            ax.legend(fontsize=8)
        fig.savefig(figures / "feedback_diagnostics.png", dpi=180)
        plt.close(fig)
    print(f"Saved {len(records)} metric entries to {tables}; available figures to {figures}")


if __name__ == "__main__":
    main()
