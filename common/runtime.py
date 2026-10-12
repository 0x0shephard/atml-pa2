"""Run provenance, atomic checkpoints, and numerical safeguards shared by tasks."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from common.data import repo_path


def identifier(row, index):
    return str(row.get("source_index", row.get("prompt_id", row.get("id", index))))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def source_fingerprint():
    source = hashlib.sha256()
    root = repo_path(".")
    for file in sorted(root.rglob("*.py")):
        relative = file.relative_to(root)
        if any(part.startswith(".") or part in {"outputs", "hf_cache", "kaggle_downloads", "kaggle_jobs"} for part in relative.parts):
            continue
        source.update(str(relative).encode())
        source.update(file.read_bytes())
    return source.hexdigest()


def compact_provenance(provenance):
    """Record input identities/hashes without copying raw course datasets into Git logs."""
    result = dict(provenance)
    for key in ("rows", "candidate", "baseline"):
        if key in result:
            rows = result[key]
            result[key] = {"count": len(rows), "sha256": fingerprint(rows),
                           "example_ids": [identifier(row, index) for index, row in enumerate(rows)]}
    if "groups" in result:
        result["groups"] = {"count": len(result["groups"]), "sha256": fingerprint(result["groups"]),
                            "problem_ids": sorted(result["groups"])}
    return result


def atomic_json(path, value):
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def completed(output):
    path = repo_path(output) / "run.json"
    return path.exists() and json.loads(path.read_text()).get("status") == "complete"


def trainable_state(model):
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters()
            if param.requires_grad}


def restore_parameters(model, state):
    params = dict(model.named_parameters())
    expected = {name for name, param in params.items() if param.requires_grad}
    if expected != set(state):
        raise ValueError("Checkpoint trainable parameter names differ from the current model")
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value.to(params[name].device))


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact training resume requires the same GPU count")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


class Run:
    def __init__(self, output, cfg, rows, name, resume=False):
        self.output = repo_path(output)
        self.path = self.output / "training_state.pt"
        self.cfg = cfg
        self.run_id = fingerprint({"config": cfg, "rows": rows, "name": name})
        self.start = time.perf_counter()
        self.elapsed = 0.0
        self.previous_peaks = {}
        self.state = None
        if self.path.exists():
            if not resume:
                raise FileExistsError(f"Existing run at {self.output}; use --resume")
            self.state = torch.load(self.path, map_location="cpu", weights_only=False)
            if self.state["run_id"] != self.run_id:
                raise ValueError("Resume config or input data differs from saved run")
            self.elapsed = self.state.get("elapsed_seconds", 0.0)
            self.previous_peaks = self.state.get("peak_vram_bytes", {})
        elif self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f"Output is nonempty without a resumable checkpoint: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path("."),
                                             stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        versions = {}
        for package in ("torch", "transformers", "peft", "trl", "tokenizers", "bitsandbytes"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        self.metadata = {"name": name, "run_id": self.run_id, "status": "running",
                         "config": cfg, "source_commit": commit, "source_sha256": source_fingerprint(),
                         "versions": versions, "example_ids": [identifier(r, i) for i, r in enumerate(rows)],
                         "asset_revision": "0b350481fb03f5525a35bcdec4131bd4fe487f98"}
        if self.state and self.state.get("source_sha256") != self.metadata["source_sha256"]:
            raise ValueError("Source changed since checkpoint; start a new run instead of mixing code versions")
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)
        atomic_json(self.output / "run.json", self.metadata)

    def seconds(self):
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
        return self.elapsed + time.perf_counter() - self.start

    def peaks(self):
        return {str(i): max(self.previous_peaks.get(str(i), 0), torch.cuda.max_memory_allocated(i))
                for i in range(torch.cuda.device_count())}

    def restore(self, models, optimizers, scaler):
        if not self.state:
            return {"update": 0, "cursor": 0, "epoch": 0}
        for name, model in models.items():
            restore_parameters(model, self.state["models"][name])
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(self.state["optimizers"][name])
        scaler.load_state_dict(self.state["scaler"])
        restore_rng(self.state["rng"])
        # A crash can leave log rows beyond the last atomic checkpoint.
        log = self.output / "train.jsonl"
        if log.exists():
            rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
            rows = [x for x in rows if x["update"] <= self.state["progress"]["update"]]
            log.write_text("".join(json.dumps(x) + "\n" for x in rows))
        return self.state["progress"]

    def save(self, models, optimizers, scaler, progress, record):
        seconds = self.seconds()
        record = {**record, "update": progress["update"], "elapsed_seconds": seconds}
        with (self.output / "train.jsonl").open("a") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")
        state = {"run_id": self.run_id, "source_sha256": self.metadata["source_sha256"],
                 "models": {k: trainable_state(v) for k, v in models.items()},
                 "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
                 "scaler": scaler.state_dict(), "rng": rng_state(), "progress": progress,
                 "elapsed_seconds": seconds, "peak_vram_bytes": self.peaks()}
        temporary = self.path.with_suffix(".tmp")
        torch.save(state, temporary)
        os.replace(temporary, self.path)
        print(json.dumps(record), flush=True)

    def finish(self, model, tokenizer, **extra):
        model.save_pretrained(self.output)
        tokenizer.save_pretrained(self.output)
        final = {**self.metadata, **extra, "status": "complete",
                 "wall_clock_seconds": self.seconds(), "peak_vram_bytes": self.peaks()}
        atomic_json(self.output / "run.json", final)
        results = repo_path(self.cfg.get("results_dir", "results"))
        if self.metadata["name"] == "benchmark":
            results = repo_path("results/benchmarks") / self.cfg.get("algorithm", "dpo")
        results.mkdir(parents=True, exist_ok=True)
        atomic_json(results / f"{self.metadata['name']}_run.json", final)
        if (self.output / "train.jsonl").exists():
            shutil.copyfile(self.output / "train.jsonl", results / f"{self.metadata['name']}_train.jsonl")


def scaler_for(model):
    enabled = next(model.parameters()).is_cuda and next(model.parameters()).dtype == torch.float16
    return torch.amp.GradScaler("cuda", enabled=enabled, init_scale=1024.0)


def optimizer_step(model, optimizer, scaler, max_norm):
    scaler.unscale_(optimizer)
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    if not torch.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Nonfinite gradient; run stopped without reporting a successful update")
    scaler.step(optimizer)
    return float(norm)


def numeric_stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"mean": None, "std": None, "count": 0}
    return {"mean": float(values.mean()), "std": float(values.std()), "count": len(values)}


class RecordStore:
    """Incremental evaluation output, refusing stale provenance and duplicate IDs."""
    def __init__(self, path, provenance):
        self.path = repo_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.meta = self.path.with_suffix(".meta.json")
        provenance = {**provenance, "source_sha256": source_fingerprint()}
        self.digest = fingerprint(provenance)
        if self.path.exists() and not self.meta.exists():
            raise ValueError(f"Missing provenance for existing output {self.path}")
        if self.meta.exists() and json.loads(self.meta.read_text())["fingerprint"] != self.digest:
            raise ValueError(f"Stale evaluation output {self.path}; use a different run name")
        atomic_json(self.meta, {"fingerprint": self.digest, "provenance": compact_provenance(provenance)})
        self.rows = []
        if self.path.exists():
            with self.path.open() as file:
                for line in file:
                    if line.strip():
                        self.rows.append(json.loads(line))
        self.keys = {str(x["record_id"]) for x in self.rows}
        if len(self.keys) != len(self.rows):
            raise ValueError(f"Duplicate record IDs in {self.path}")

    def append(self, row):
        key = str(row["record_id"])
        if key in self.keys:
            raise ValueError(f"Duplicate record ID {key}")
        with self.path.open("a") as file:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            file.flush()
        self.rows.append(row)
        self.keys.add(key)


def policy_provenance(adapter):
    if adapter is None:
        return {"adapter": None}
    path = repo_path(adapter)
    if not path.exists():
        raise FileNotFoundError(path)
    files = {}
    for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "run.json"):
        file = path / name
        if file.exists():
            files[name] = hashlib.sha256(file.read_bytes()).hexdigest()
    return {"adapter": str(adapter), "files": files}
