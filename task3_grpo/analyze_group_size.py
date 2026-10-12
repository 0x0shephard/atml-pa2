from __future__ import annotations
import argparse
from collections import defaultdict
import numpy as np
from common.data import load_yaml, read_jsonl, repo_path
from common.runtime import atomic_json


def load_k8_cache(path):
    by_prompt = defaultdict(list)
    for row in read_jsonl(path):
        by_prompt[str(row['source_index'])].append(row)
    for pid, rows in by_prompt.items():
        if len(rows) != 8:
            raise ValueError(f'Expected exactly 8 completions per prompt; {pid} has {len(rows)}')
        rows.sort(key=lambda row: int(row['generation_index']))
        if len({row['generation_index'] for row in rows}) != 8:
            raise ValueError(f'Duplicate generation indices for {pid}')
    if not by_prompt:
        raise ValueError('Empty group-size cache')
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k):
    if k not in (2, 4, 8):
        raise ValueError('K must divide the eight cached completions')
    return [rows[start:start+k] for _, rows in sorted(by_prompt.items()) for start in range(0, 8, k)]


def group_statistics(groups):
    stds, signals = [], []
    for rows in groups:
        rewards = np.asarray([row['reward'] for row in rows], dtype=float)
        std = rewards.std()
        stds.append(float(std))
        signals.extend(((rewards - rewards.mean()) / (std + 1e-6)).tolist())
    return {'groups': len(groups), 'generations': sum(len(x) for x in groups),
            'informative_group_rate': float(np.mean(np.asarray(stds) > 1e-6)),
            'uninformative_group_fraction': float(np.mean(np.asarray(stds) <= 1e-6)),
            'mean_group_reward_std': float(np.mean(stds)), 'relative_signal_variance': float(np.var(signals))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/grpo.yaml')
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg['group_cache'])
    ranked = sorted(by_prompt, key=lambda pid: (np.mean([r['reward'] for r in by_prompt[pid]]), pid))
    if len(ranked) < 2:
        raise ValueError('Need at least two prompts for difficulty bins')
    bins = {'lower_mean_reward': set(ranked[:len(ranked)//2]), 'higher_mean_reward': set(ranked[len(ranked)//2:])}
    output = {'difficulty_rule': 'Rank prompts by mean reward across their same eight cached completions, break ties by source_index, split into lower/higher halves. Lower reward is a difficulty proxy.',
              'bin_prompt_ids': {key: sorted(ids) for key, ids in bins.items()}, 'conditions': {}}
    for k in cfg['group_sizes']:
        output['conditions'][str(k)] = {'overall': group_statistics(regroup_equal_generation_budget(by_prompt, k)),
            'difficulty_bins': {label: group_statistics(regroup_equal_generation_budget({pid: by_prompt[pid] for pid in ids}, k)) for label, ids in bins.items()}}
    atomic_json(repo_path(cfg['results_dir']) / 'group_size_summary.json', output)

if __name__ == '__main__':
    main()
