"""Format saved measurements as LaTeX; generates no report interpretation."""
import json
from pathlib import Path
from scripts.kaggle_runner import ROOT


def number(value):
    return "--" if value is None else f"{value:.4g}"


def tabular(path, headings, rows):
    def escape(v):
        return str(v).replace("_", r"\_").replace("%", r"\%")
    lines = [r"\begin{center}\small", r"\begin{tabular}{" + "l" + "r"*(len(headings)-1) + "}",
             r"\toprule", " & ".join(escape(h) for h in headings) + r" \\", r"\midrule"]
    lines += [" & ".join(escape(v) for v in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    path.write_text("\n".join(lines) + "\n")


def read(name):
    return json.loads((ROOT / "results" / name).read_text())


def main():
    out = ROOT / "report/tables"
    out.mkdir(parents=True,exist_ok=True)
    rows = []
    for task in ("task1_dpo", "task2_ppo", "task3_grpo"):
        d = read(f"{task}/standard_run.json")
        peak = d["peak_vram_bytes"]
        peak = max(peak.values()) if isinstance(peak,dict) else peak
        rows.append([task, number(d["wall_clock_seconds"]), number(peak/2**30), d.get("epochs",d.get("updates"))])
    tabular(out/"resources.tex", ["Task", "Train seconds", "Peak GiB/device", "Epochs/updates"], rows)
    rows=[]
    for name in ("standard","beta_0.03","beta_0.10","beta_0.30"):
        d=read(f"task1_dpo/{name}_metrics.json")
        rows.append([name,number(d["heldout_dpo_loss"]),number(d["preference_accuracy"]),number(d["reward"]["mean"]),number(d["kl"]),number(d["response_length"]["mean"])])
    tabular(out/"dpo.tex",["Condition","Loss","Accuracy","Reward","KL","Tokens"],rows)
    d=read("task2_ppo/cached_clipping.json")["summary"]
    tabular(out/"ppo_cached.tex",["Epsilon","Surrogate loss","Affected fraction"],
            [[k,number(v["loss"]),number(v["affected_token_fraction"])] for k,v in d.items()])
    rows=[]
    for dataset in ("gsm","transfer"):
        for policy,d in read(f"task5_feedback/{dataset}_summary.json").items():
            rows.append([dataset,policy,number(d["exact_accuracy"]),number(d["format_compliance"]),number(d["ai_pairwise_win_rate_against_sft"]),number(d["response_length"]["mean"])])
    tabular(out/"math.tex",["Dataset","Policy","Accuracy","Format","Win rate","Tokens"],rows)


if __name__=="__main__":
    main()
