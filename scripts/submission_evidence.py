"""Export numerical evidence and figures without drafting report analysis."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.kaggle_runner import ROOT

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"


def read(path):
    return json.loads((RESULTS / path).read_text())


def table(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (TABLES / f"{name}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def comparisons(task, names, title):
    rows = []
    for name in names:
        data = read(f"{task}/{name}_metrics.json")
        row = {"condition": name, "source": f"{task}/{name}_metrics.json", "n": data["n"]}
        for key in ("preference_accuracy", "heldout_dpo_loss", "kl", "entropy", "sample_entropy", "truncated_fraction"):
            if key in data:
                row[key] = data[key]
        row.update(reward=data["reward"]["mean"], response_length=data["response_length"]["mean"],
                   response_length_std=data["response_length"]["std"])
        rows.append(row)
    table(title, rows)
    keys = [k for k in ("reward", "kl", "entropy", "response_length", "preference_accuracy", "heldout_dpo_loss") if k in rows[0]]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
    for ax, key in zip(axes.flat, keys):
        ax.bar([r["condition"] for r in rows], [r[key] for r in rows], color="#315b80")
        ax.set_ylabel(key.replace("_", " "))
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    for ax in list(axes.flat)[len(keys):]:
        ax.set_visible(False)
    save(fig, title)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    comparisons("task1_dpo", ["beta_0.03", "beta_0.10", "beta_0.30"], "dpo_beta")
    comparisons("task2_ppo", ["clip_0.05", "clip_0.20", "clip_0.50"], "ppo_clipping")
    comparisons("task2_ppo", ["kl_0.00", "kl_0.10", "kl_0.20"], "ppo_kl")
    # Main continuations have distinct starting states; export separately.
    for task in ("task1_dpo", "task2_ppo", "task3_grpo"):
        comparisons(task, ["standard"], task + "_standard")
    length = read("task1_dpo/length_summary.json")
    table("dpo_length", [{"policy": policy, "stratum": stratum, **stats,
                           "word_limit_compliance": data["word_limit_compliance"]["mean"]}
                          for policy, data in length.items() for stratum, stats in data["strata_preference_accuracy"].items()])
    strata = list(length["standard"]["strata_preference_accuracy"])
    fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    for index, (policy, data) in enumerate(length.items()):
        ax.bar([i + index*.35 for i in range(len(strata))],
               [data["strata_preference_accuracy"][s]["mean"] for s in strata], width=.35, label=policy)
    ax.set(xticks=[i+.175 for i in range(len(strata))], xticklabels=[s.replace("_", " ") for s in strata],
           ylabel="Preference accuracy", ylim=(0, 1))
    ax.legend()
    save(fig, "dpo_length")
    cache = read("task2_ppo/cached_clipping.json")
    table("ppo_cached_clipping", [{"epsilon": eps, **data} for eps, data in cache["summary"].items()])
    groups = read("task3_grpo/group_size_summary.json")["conditions"]
    table("grpo_groups", [{"K": k, "difficulty": difficulty, **data}
                          for k, condition in groups.items()
                          for difficulty, data in {"overall": condition["overall"], **condition["difficulty_bins"]}.items()])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
    for ax, metric in zip(axes, ("informative_group_rate", "mean_group_reward_std")):
        for difficulty in ("lower_mean_reward", "higher_mean_reward"):
            ax.plot([int(k) for k in groups], [d["difficulty_bins"][difficulty][metric] for d in groups.values()],
                    marker="o", label=difficulty.replace("_", " "))
        ax.set(xlabel="K", ylabel=metric.replace("_", " "), xticks=[2,4,8])
    axes[0].legend(fontsize=7)
    save(fig, "grpo_groups")
    normalization = {}
    for condition in ("grpo", "dr_grpo"):
        normalization[condition] = read(f"task3_grpo/normalization_summary_{condition}.json")[condition]
    norm_rows = []
    for condition, data in normalization.items():
        heldout = data["heldout"]
        row = {"condition": condition, "generated_tokens": data["generated_tokens"],
               "reward": heldout["reward"]["mean"], "kl": heldout["kl"], "entropy": heldout["entropy"],
               "response_length": heldout["response_length"]["mean"]}
        norm_rows.append(row)
    table("grpo_normalization", norm_rows)
    gradient_rows = [{"condition": condition, "update": row["update"], "length_bin": length_bin, **stats}
                     for condition, data in normalization.items() for row in data["length_conditioned_gradient"]
                     for length_bin, stats in row.items() if length_bin != "update"]
    table("grpo_length_gradients", gradient_rows)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), layout="constrained")
    for ax, length_bin in zip(axes, ("short", "long")):
        for condition in normalization:
            selected = [r for r in gradient_rows if r["condition"] == condition and r["length_bin"] == length_bin]
            ax.plot([r["update"] for r in selected],
                    [r["mean_abs_logp_gradient_sum"] for r in selected], marker="o", label=condition)
        ax.set(xlabel="Update", ylabel="Mean sum |d loss / d token logp|", title=length_bin + " responses")
        ax.legend()
    save(fig, "grpo_length_gradients")
    fig, axes = plt.subplots(1, 4, figsize=(10, 3), layout="constrained")
    for ax, metric in zip(axes, ("reward", "kl", "entropy", "response_length")):
        ax.bar([r["condition"] for r in norm_rows], [r[metric] for r in norm_rows], color="#315b80")
        ax.set_ylabel(metric.replace("_", " "))
    save(fig, "grpo_normalization")
    for dataset in ("gsm", "transfer"):
        data = read(f"task5_feedback/{dataset}_summary.json")
        table("math_" + dataset, [{"policy": p, **{k:v for k,v in d.items() if not isinstance(v,dict)},
                                  "response_length": d["response_length"]["mean"],
                                  "response_length_std": d["response_length"]["std"]} for p,d in data.items()])
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
        for ax, metric in zip(axes, ("exact_accuracy", "format_compliance", "ai_pairwise_win_rate_against_sft")):
            ax.bar(list(data), [d[metric] for d in data.values()], color="#315b80")
            ax.set(ylabel=metric.replace("_", " "), ylim=(0,1))
        save(fig, "math_" + dataset)
    stability = []
    for eps in ("0.05", "0.20", "0.50"):
        data = read(f"task2_ppo/clipping_summary_{eps}.json")["clip_" + eps]
        stability.append({"epsilon": eps, **data["stability_gradient_norm"], "maximum": data["max_gradient_norm"]})
    table("ppo_clipping_stability", stability)
    resources = []
    for task in ("task1_dpo", "task2_ppo", "task3_grpo"):
        run = read(f"{task}/standard_run.json")
        peak = run.get("peak_vram_bytes", 0)
        peak = max(peak.values(), default=0) if isinstance(peak, dict) else peak
        resources.append({"task": task, "wall_clock_seconds": run["wall_clock_seconds"],
                          "peak_per_device_vram_gib": peak / 2**30,
                          "generated_tokens": run.get("generated_tokens"),
                          "updates": run.get("updates"), "epochs": run.get("epochs")})
    table("standard_resources", resources)
    print("Exported comparison tables and PNG/PDF figures")


if __name__ == "__main__":
    main()
