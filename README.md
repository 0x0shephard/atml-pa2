# ATML PA2: LLM post-training

Implementation of the course DPO, PPO, GRPO, safety-calibration, and feedback-source experiments. Python modules contain the pipeline; generated Kaggle notebooks only install the environment and launch those modules. **All 32 experiment jobs and three GPU benchmarks have completed on Kaggle.** Results, training logs, evaluation records, and per-job receipts are saved under `results/`. The student's blind manual safety audit and report remain pending; `python -m scripts.validate_results` must pass before submission.

## Setup

Run all commands from this repository root, using Python 3.11 or 3.12:

```bash
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
python -m scripts.check_environment
python -m unittest discover -s tests -v
```

Course assets are downloaded from [AbDu11aHHH/ATML-PA2-assets](https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets), revision `0b350481fb03f5525a35bcdec4131bd4fe487f98`. The transfer installer creates the fixed first 100 official SVAMP examples only if that course file is absent. Do not edit supplied data, caches, or checkpoints. Public models download at runtime.

The released configuration is preserved: seed 6304, Qwen2.5-1.5B-Instruct, policy LoRA rank 8 on q/v projections, and the released reward/judge/critic models. On T4 ×2 the policy and critic default to GPU 0, with the frozen reward model on GPU 1. Frozen reward/judge quantization follows the release. Trainable adapters and critic head use float32 parameters for stable FP16 gradient scaling. Device placement can be overridden with `policy_device`, `value_device`, and `reward_device` in a config; it does not change experiment budgets.

## Required experiment commands

### Task 1: DPO

```bash
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard --resume
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml --resume
python -m task1_dpo.analyze_length --config configs/dpo.yaml --resume
```

The standard and length-balanced models train for one epoch. The separate β conditions start fresh and each use the first 600 fixed training examples. Use `--beta 0.03`, `--beta 0.10`, or `--beta 0.30` to run one fork. Standard and short-run budgets must be distinguished in the report. Held-out DPO loss uses each condition's β; accuracy uses the unscaled reference-adjusted margin. Length strata come from course metadata rather than being regenerated.

### Task 2: PPO

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard --resume
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml --resume
python -m task2_ppo.ablate_kl --config configs/ppo.yaml --resume
```

Standard continuation is 20 updates; each fork is 8 updates. Every condition restores the supplied midpoint policy and critic. The critic uses the supplied merged state with a fresh LoRA and trainable saved head. Clipping supports `--cached-only`, or `--forks-only --epsilon 0.05` (also 0.20/0.50). KL forks support `--beta 0.00` (also 0.10/0.20). Clipping stability is reported as the mean, standard deviation, and maximum pre-clipping policy-gradient norm. Cached diagnostics fail explicitly if exact token/logprob alignment cannot be reconstructed.

Training generations are capped at 512 tokens; frozen evaluation uses the released 768-token cap. Learned reward scores are logged separately from missing-EOS penalties and KL-shaped rewards. Dropout is disabled for online rollout/update likelihood comparisons so unchanged policies have unit importance ratios.

### Task 3: GRPO

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard --resume
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml --resume
```

Standard continuation is 20 updates, K=4. Normalization forks are 8 updates each, starting from the same supplied midpoint. Run either separately using `--loss-type grpo` or `--loss-type dr_grpo`. The supplied Dr. GRPO condition changes the sequence denominator only; reward-standard-deviation normalization is retained as in the release helper.

The K study partitions each prompt's eight cached completions, ordered by generation index, into contiguous groups of size 2, 4, or 8. Every condition uses all the same generations. Difficulty bins are fixed across K: rank prompts by the mean of all eight rewards, break ties by source index, and split into lower/higher halves. This is a reward-based difficulty proxy. Informative means population reward standard deviation greater than 1e-6.

Normalization diagnostics directly measure the absolute gradient of loss with respect to sampled-token log probabilities, summed per response and divided into short/long groups at each rollout's median response length. These are score-space gradient statistics, not parameter-gradient norms. Both are saved. Forks match updates, prompts, K, and maximum token allocation; realized token counts are reported separately because EOS behavior can change after policies diverge.

### Task 4: safety

After all three standard adapters exist:

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

Generation and judging support `--policy sft|dpo|ppo|grpo`. They use the released deterministic decoding and fixed categorical judge. The audit sheet contains the release helper's fixed 60 prompt IDs, with 30 safe and 30 unsafe prompts, and **240 policy-specific response rows**. Fill `manual_label` in `results/task4_safety/manual_audit.csv` using the five course labels without viewing automated labels. Then rerun `evaluate_safety`. The exporter refuses to overwrite an existing audit sheet. Pending/partial labels remain explicitly incomplete; they are never filled automatically.

Safety rates use the relevant safe/unsafe denominator. Category distributions include all five labels. Agreement includes ambiguous judgments and a manual-row/AI-column confusion matrix.

### Task 5: feedback source

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

The supplied RLVR/RLAIF adapters are frozen evaluation policies. Separate generation from judgment with `--phase generate --policy sft|rlvr|rlaif`, then `--phase judge` after all three generation files exist. Exact accuracy uses the supplied final-answer parser. Pairwise comparisons use the supplied judge, rubric, hash-based A/B orientation, and tie-as-0.5 convention. Raw judge outputs and parse-failure flags are retained alongside cached decisions.

Verifier–judge agreement compares their three-way A/B/TIE decisions on the same candidate-versus-SFT response pairs. The SFT win-rate entry is an identity comparison, 0.5, not an independently judged performance estimate. Diagnostic comparisons use the clean response as A and each named perturbation as B; the clean category is an identity/tie control. Reasoning-only verifier ties are expected. Individual diagnostic preferences trace to problem IDs and variant types in the unchanged course diagnostic file; raw course datasets are not copied into result metadata.

### Tables, figures, and evidence coverage

```bash
python -m scripts.collect_results
python -m scripts.validate_results
```

Tables and figures are generated only from saved records. Missing results are not fabricated. `validate_results --allow-incomplete` reports progress without returning a failure exit code. Standard continuation logs include true token-distribution entropy and the release sampled-surprisal helper separately. Reported reference KL consistently averages logprob differences over valid generated response tokens; the GRPO training penalty is recorded separately. Response length includes generated EOS when present, excludes padding, and includes mean and standard deviation.

## Kaggle CLI execution

Install the Kaggle CLI separately from the model environment and authorize each account locally. Credentials stay outside Git. The runner takes `--profile /absolute/path/to/account-config` for separate credential profiles and never embeds their values in notebooks. Private datasets and private notebooks are the defaults.

```bash
python -m scripts.kaggle_runner list
python -m scripts.kaggle_runner build benchmark_dpo --username ACCOUNT_A
python -m scripts.kaggle_runner push kaggle_jobs/ACCOUNT_A/benchmark_dpo --profile /path/to/account-a-config
python -m scripts.kaggle_runner status ACCOUNT_A/pa2-benchmark-dpo --profile /path/to/account-a-config
python -m scripts.kaggle_runner output ACCOUNT_A/pa2-benchmark-dpo --directory kaggle_downloads/benchmark_dpo --profile /path/to/account-a-config
python -m scripts.kaggle_runner import benchmark_dpo kaggle_downloads/benchmark_dpo
```

Benchmark PPO and GRPO similarly with `benchmark_ppo` and `benchmark_grpo` on account B. Benchmarks are separate from report results. Use measured throughput and memory to decide the execution schedule before launching all jobs. Each submitted job requests T4 ×2 and an eight-hour timeout. A source allowlist and embedded snapshot hash prevent accidental credential, weight, or dataset uploads. Notebook outputs retain adapters, training state, and small evidence files; downloaded course assets and model caches are kept out of saved outputs.

`experiments/runs.json` declares 35 jobs, commands, dependencies, expected artifacts, and initial account assignment. Run IDs can be built individually. Do not launch an aggregate job until every dependency is complete and imported. Independent work can run on the two accounts; policy evaluations wait for the correct standard adapter.

For a dependent stage, mount accessible prior kernel output using `--kernel owner/kernel-slug`. For cross-account transfer, download/import the upstream outputs and prepare a private dependency bundle under the destination account:

```bash
python -m scripts.kaggle_runner dependencies safety_ppo --username ACCOUNT_A
python -m scripts.kaggle_runner upload-dependencies kaggle_jobs/ACCOUNT_A/safety_ppo_inputs --profile /path/to/account-a-config
# Read the dataset ID from the generated dataset-metadata.json.
python -m scripts.kaggle_runner build safety_ppo --username ACCOUNT_A --dataset ACCOUNT_A/DATASET_SLUG
python -m scripts.kaggle_runner push kaggle_jobs/ACCOUNT_A/safety_ppo --profile /path/to/account-a-config
```

Imports reject failed jobs, missing expected artifacts, and conflicting existing files. Dataset bundles include only the stage's declared dependencies, preserving paths and evaluation provenance sidecars.

Training `--resume` restores trainable parameters, optimizer/scaler state, RNG state, and completed-update progress. It refuses changed configurations, input rows, source code, or GPU counts. To continue an interrupted remote job, rebuild with `--resume-kernel owner/prior-kernel --attempt 2` (increment the attempt for later retries). This copies the previous job's `outputs/` and `results/` before launching; the same source snapshot must be used. The runner rejects a mismatched snapshot. Do not rerun the audit exporter over an existing labeled audit sheet. Restarting a notebook alone does not recover a checkpoint from the previous session. Evaluation records are saved incrementally and resumed only when provenance matches.

## Attribution and submission

Starter infrastructure, fixed objective interfaces, metric helpers, model/data loading, judge rubrics/parsers, verifier, and asset tooling are materially reused from [the official starter](https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining), initial commit `1d64ac65acd5e45d1e4e1f415edc80455e21274e`. Kaggle metadata/CLI behavior follows the [official Kaggle API documentation](https://github.com/Kaggle/kaggle-api/tree/main/docs). The student implementation corrects the three objective defects and completes training, evaluation, ablation, persistence, and orchestration. LLM coding assistance was used; the student must understand and review every submitted line.

Keep code, configs, fixed IDs, small JSON/CSV records, and figures in the public submission repository. Downloaded assets, checkpoints, credentials, and local environments are ignored. Training trajectories and run provenance are copied into `results/` at successful completion so they can be committed independently of model weights. Freeze the reviewed final code state before submission.

The assignment prohibits generative AI from writing report language, interpretation, or analysis. The student must complete the manual audit and write the eight-page NeurIPS-style report, including the repository link at the end of the abstract. This repository produces experiment evidence and figures, not report text.
