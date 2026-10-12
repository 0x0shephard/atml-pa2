# Submission workflow

All 32 experiment jobs and three benchmarks completed. The ZIP contains Python source, release configurations, tests, result records, job receipts, numerical tables, and figures. It excludes credentials, raw downloaded datasets, and model weights. This is an evidence bundle, not a completed LMS submission.

The user subsequently requested an ungraded sample report. `report/main.pdf` is that completed AI-assisted sample (eight main pages plus references), based on actual results and explicitly noting the unperformed human audit. The student-work steps below still apply if adapting the materials to the graded assignment.

## Student work

1. Open `manual_audit.html` in a browser. Label all 240 responses yourself without viewing automated labels. Export `manual_audit.csv` and replace `results/task4_safety/manual_audit.csv` with that exported file. Preserve the IDs and response text.
2. Run `python -m task4_safety.evaluate_safety`, then `python -m scripts.validate_results`. Audit agreement and confusion matrices must be regenerated from your labels.
3. Regenerate evidence with `python -m scripts.collect_results`, `python -m scripts.submission_evidence`, and `python -m scripts.package_submission`.
4. Write your own eight-page main-content report in the course NeurIPS format. References are excluded from the page count. Follow the manual's required evidence and research questions. Add the accessible repository URL at the end of the abstract.
5. Freeze the final code/results with a meaningful Git commit in your own repository, then upload the PDF and repository link to LMS.

## Evidence locations

- `results/tables/`: machine-readable metrics and condition comparisons.
- `results/figures/`: numerical plots; comparison plots include vector PDF exports.
- `results/task1_dpo/` through `results/task5_feedback/`: source records, per-example judgments, summaries, and logs.
- `results/jobs/`: successful Kaggle job receipts and source-snapshot hashes.
- `file_manifest.json`: SHA-256 hashes of packaged source and evidence.
- `status.json`: experiment completion and pending submission items.

## Measurement details to retain

- DPO pairs with prompts too long for the released 768-token limit were filtered without truncating prompts. Training provenance and evaluation sidecars record retained/excluded coverage. State the effective sample counts.
- DPO standard and short beta forks use different training budgets.
- Online forks use the same prompts, update counts, group sizes, and maximum token allocations. Logs record realized generated-token counts separately; do not describe different realized counts as identical.
- Sampled reference KL can be slightly negative because it is an empirical mean of log-probability differences, not an exact nonnegative distribution KL.
- PPO/GRPO start from supplied continuation checkpoints; main-run comparisons do not share identical starting states.
- GRPO computes its complete-group loss while recomputing individual response forwards to fit the T4 memory limit.
- Task 5 pairwise memoization caches remain in their original downloads because independent jobs use the same cache filename. Saved experiment records and summaries are imported separately.

The manual explicitly prohibits generative AI from writing report language, interpretation, or analysis. The blind audit and report must be completed by the student.
