import json, time, os
from pathlib import Path


class StatsLogger:
    def __init__(self, path="logs/training.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None

    def _write(self, entry):
        entry["timestamp"] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def step(self, phase, step, total, loss, lr=None, grad_norm=None, tok_per_s=None,
             seq_len=None, mtp=None, domain=None, acc=None, per_domain_acc=None):
        self._write({k: v for k, v in {
            "type": "step", "phase": phase, "step": step, "total": total,
            "loss": loss, "lr": lr, "grad_norm": grad_norm,
            "tok_per_s": tok_per_s, "seq_len": seq_len, "mtp": mtp,
            "domain": domain, "acc": acc, "per_domain_acc": per_domain_acc,
        }.items() if v is not None})

    def phase(self, label, status, **extra):
        e = {"type": "phase", "label": label, "status": status}
        e.update(extra)
        self._write(e)

    def summary(self, phase, **metrics):
        e = {"type": "summary", "phase": phase}
        e.update(metrics)
        self._write(e)

    def hw(self, label, cpu_pct, ram_pct, ram_used_gb, ram_total_gb, swap):
        self._write({
            "type": "hw", "label": label,
            "cpu_pct": cpu_pct, "ram_pct": ram_pct,
            "ram_used_gb": ram_used_gb, "ram_total_gb": ram_total_gb,
            "swap": swap,
        })

    def dataset(self, name, domain, split, config, samples, elapsed, yield_rate=None,
                file_size_mb=None):
        self._write({k: v for k, v in {
            "type": "dataset", "name": name, "domain": domain, "split": split,
            "config": config, "samples": samples, "elapsed": elapsed,
            "yield_rate": yield_rate, "file_size_mb": file_size_mb,
        }.items() if v is not None})


GLOBAL = StatsLogger()
