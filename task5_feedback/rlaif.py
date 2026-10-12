from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common.data import repo_path
from common.models import resolve_dtype

PAIRWISE_RUBRIC = """You are comparing two candidate solutions to the same math problem.

Evaluate mathematical correctness, internal consistency, relevance, and whether the stated final answer follows from the reasoning.
Do not prefer a response merely because it is longer, more confident, or more polished.
If neither response is meaningfully better, return TIE.

Return exactly one token: A, B, or TIE.

Problem:
{problem}

Candidate A:
{a}

Candidate B:
{b}

Preference:"""


class PairwiseAIJudge:
    """Fixed course judge with deterministic A/B orientation balancing and persistent caching."""

    def __init__(self, cfg: dict, cache_path: str | Path):
        from common.runtime import atomic_json
        self.cfg = cfg
        self.cache_path = repo_path(cache_path)
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        else:
            self.cache = {}

        self.audit_path = self.cache_path.with_suffix(".audit.json")
        self.audit = json.loads(self.audit_path.read_text()) if self.audit_path.exists() else {}
        from common.runtime import fingerprint
        self.meta_path = self.cache_path.with_suffix(".meta.json")
        provenance = {"model": cfg["ai_judge_model"], "rubric": PAIRWISE_RUBRIC,
                      "quantized": cfg.get("quantize_frozen_models", True), "max_new_tokens": 4}
        if self.meta_path.exists() and json.loads(self.meta_path.read_text())["fingerprint"] != fingerprint(provenance):
            raise ValueError("Pairwise judge cache configuration changed")
        atomic_json(self.meta_path, {"fingerprint": fingerprint(provenance), "provenance": provenance})

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["ai_judge_model"], padding_side="left", use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {"low_cpu_mem_usage": True}
        if torch.cuda.is_available() and bool(cfg.get("quantize_frozen_models", True)):
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=resolve_dtype(cfg.get("dtype", "float16")),
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["dtype"] = resolve_dtype(cfg.get("dtype", "float16"))

        self.model = AutoModelForCausalLM.from_pretrained(cfg["ai_judge_model"], **kwargs)
        self.model.eval()

    def _key(self, problem, a, b):
        payload = json.dumps(
            {"model": self.cfg["ai_judge_model"], "problem": problem, "a": a, "b": b},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @torch.no_grad()
    def compare(self, problem: str, a: str, b: str):
        key = self._key(problem, a, b)
        if key in self.cache:
            return self.cache[key]

        swap = int(key[:8], 16) % 2 == 1
        aa, bb = (b, a) if swap else (a, b)
        text = PAIRWISE_RUBRIC.format(problem=problem, a=aa, b=bb)
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(next(self.model.parameters()).device)
        out = self.model.generate(
            ids,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        decoded = self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().upper()
        m = re.search(r"\b(A|B|TIE)\b", decoded)
        result = m.group(1) if m else "TIE"
        if swap:
            result = {"A": "B", "B": "A", "TIE": "TIE"}[result]

        self.cache[key] = result
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        from common.runtime import atomic_json
        self.audit[key] = {"raw_output": decoded, "ambiguous": m is None, "swapped": swap, "result": result}
        atomic_json(self.cache_path, self.cache)
        atomic_json(self.audit_path, self.audit)
        return result

    def group_rewards(self, problem: str, responses: list[str]):
        k = len(responses)
        wins = np.zeros(k, dtype=float)
        ties = np.zeros(k, dtype=float)
        for i, j in itertools.combinations(range(k), 2):
            pref = self.compare(problem, responses[i], responses[j])
            if pref == "A":
                wins[i] += 1
            elif pref == "B":
                wins[j] += 1
            else:
                ties[i] += 1
                ties[j] += 1
        return ((wins + 0.5 * ties) / max(k - 1, 1)).tolist()
