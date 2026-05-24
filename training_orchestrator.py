"""
training_orchestrator.py — Round management loop for Qwen2.5-based AgentMind.

Usage:
  python training_orchestrator.py --rounds 1-5 --save-dir ./checkpoints
  python training_orchestrator.py --rounds 1-5 --no-distill --save-dir ./checkpoints
  python training_orchestrator.py --rounds 1-5 --distill-only
  python training_orchestrator.py --train-router
"""

import json, os, argparse, time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load as load_model
from mlx.utils import tree_flatten

from config import APPRENTICE_ROUNDS
from data.pipeline import AgentDataset
from lora import apply_lora, save_adapter, reset_adapter, load_adapter
from train import train_specialist, distill_backbone
from router import TaskRouter


def parse_rounds(rounds_spec: str) -> list[int]:
    result = []
    for part in rounds_spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            for i in range(int(a.strip()), int(b.strip()) + 1):
                result.append(i - 1)
        else:
            result.append(int(part) - 1)
    return sorted(set(result))


def load_domain_data(data_path, tokenizer):
    ds = AgentDataset.from_raw(data_path, tokenizer=tokenizer)
    print(f"  Loaded {len(ds.samples)} samples from {data_path}")
    return ds


def gather_combined_data(completed_rounds):
    combined = []
    for r in completed_rounds:
        path = Path(r["file"])
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                sample = json.loads(line.strip())
                sample["domain"] = r["domain"]
                combined.append(sample)
    print(f"  Gathered {len(combined)} combined samples from {len(completed_rounds)} domains")
    return combined


def load_adapter_weights(adapter_path):
    loaded = mx.load(adapter_path)
    if 'metadata' in loaded:
        del loaded['metadata']
    return loaded


def train_router(backbone, tokenizer, save_dir):
    domain_names = [r["domain"] for r in APPRENTICE_ROUNDS]
    router = TaskRouter(
        d_model=512,
        hidden=64,
        n_domains=len(domain_names),
        domain_names=domain_names,
    )
    router_data_path = "data/router_training.jsonl"
    if not os.path.exists(router_data_path):
        print(f"  Warning: {router_data_path} not found — skipping router training")
        return
    router_data = []
    with open(router_data_path) as f:
        for line in f:
            router_data.append(json.loads(line.strip()))
    print(f"  Loaded {len(router_data)} router training samples")
    router.train(router_data, backbone, tokenizer=tokenizer, steps=200, lr=1e-3)
    router_save_path = str(save_dir / "router")
    router.save(router_save_path)


def export_backbone(backbone, save_dir):
    print(f"\n{'='*60}")
    print("  Saving final artifacts")
    print(f"{'='*60}")
    backbone_path = save_dir / "backbone.safetensors"
    backbone_params = {k: v for k, v in tree_flatten(backbone.parameters())}
    mx.save_safetensors(str(backbone_path), dict(backbone_params))
    print(f"  Backbone saved -> {backbone_path}")


def run_round(backbone, tokenizer, domain, data_path,
              specialist_steps, save_dir, seq_len=256,
              seq_len_schedule=None, do_distill=False,
              distill_steps=50, existing_adapters=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    reset_adapter(backbone)
    dataset = load_domain_data(data_path, tokenizer)

    print(f"\n{'='*60}")
    print(f"  Training specialist: {domain}")
    print(f"  Steps: {specialist_steps} | Seq len: {seq_len}")
    print(f"{'='*60}")

    adapter_weights = train_specialist(
        backbone, tokenizer, dataset, domain,
        steps=specialist_steps, seq_len=seq_len,
        seq_len_schedule=seq_len_schedule,
    )

    save_adapter(backbone, domain, str(save_dir))
    adapter_path = str(save_dir / f"{domain}.safetensors")

    result = {
        "domain": domain,
        "adapter_path": adapter_path,
    }

    if do_distill and existing_adapters:
        specialists = dict(existing_adapters)
        specialists[domain] = adapter_weights

        existing_domains = list(existing_adapters.keys()) if existing_adapters else []
        completed_domains = existing_domains + [domain]
        rounds_for_data = [r for r in APPRENTICE_ROUNDS if r["domain"] in completed_domains]
        combined = gather_combined_data(rounds_for_data)

        reset_adapter(backbone)
        print(f"\n  Distilling backbone ({distill_steps} steps, seq_len={min(seq_len, 512)})")
        distill_backbone(
            backbone, specialists, combined, tokenizer,
            steps=distill_steps, seq_len=min(seq_len, 512),
        )

    return result


def main():
    parser = argparse.ArgumentParser(description="AgentMind Apprenticeship Training Orchestrator")
    parser.add_argument("--rounds", type=str, default="1-5",
                        help="Rounds to run: '1-5', '1,3,5', '2' (1-indexed)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint dir to resume from")
    parser.add_argument("--save-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--no-distill", action="store_true", default=False,
                        help="Skip distillation after each round")
    parser.add_argument("--distill-only", action="store_true", default=False,
                        help="Only run distillation on existing adapters, skip specialist training")
    parser.add_argument("--train-router", action="store_true", default=False,
                        help="Train the task router on cached hidden states")
    args = parser.parse_args()

    t_start = time.time()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    adapters_dir = save_dir / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    # Load backbone
    print(f"Loading backbone (Qwen/Qwen2.5-0.5B)...")
    model, tokenizer = load_model("Qwen/Qwen2.5-0.5B")

    # Add special tokens to underlying HF tokenizer
    special_tokens = [
        "<|tool_call|>", "<|plan|>", "<|memory|>", "<|scratch|>", "<|observe|>",
        "<|think_start|>", "<|think_end|>", "<|system|>", "<|user|>", "<|assistant|>",
    ]
    tokenizer._tokenizer.add_tokens(special_tokens)
    # Embedding (151936 slots) already has room for these tokens

    # Apply LoRA
    model = apply_lora(model)
    print(f"Model initialized. Trainable params: "
          f"{sum(p.size for _,p in tree_flatten(model.trainable_parameters())):,}")

    if args.train_router:
        train_router(model, tokenizer, save_dir)
        export_backbone(model, save_dir)
        return

    round_indices = parse_rounds(args.rounds)
    print(f"Rounds to run: {[i+1 for i in round_indices]}")

    all_adapter_weights = {}
    completed_round_entries = []

    for idx in round_indices:
        round_cfg = APPRENTICE_ROUNDS[idx]
        domain = round_cfg["domain"]

        # Check for existing adapter
        adapter_path = adapters_dir / f"{domain}.safetensors"
        if adapter_path.exists() and args.resume:
            print(f"  Skipping {domain} — found existing adapter")
            all_adapter_weights[domain] = load_adapter_weights(str(adapter_path))
            completed_round_entries.append(round_cfg)
            continue

        if args.distill_only:
            print(f"  Skipping {domain} — distill-only mode")
            continue

        result = run_round(
            backbone=model,
            tokenizer=tokenizer,
            domain=domain,
            data_path=round_cfg["file"],
            specialist_steps=round_cfg["specialist_steps"],
            save_dir=str(adapters_dir),
            seq_len=round_cfg["seq_len"],
            seq_len_schedule=round_cfg.get("seq_len_schedule"),
            do_distill=not args.no_distill,
            distill_steps=round_cfg.get("distill_steps", 50),
            existing_adapters=all_adapter_weights if all_adapter_weights else None,
        )

        adapter_path = adapters_dir / f"{domain}.safetensors"
        if adapter_path.exists():
            all_adapter_weights[domain] = load_adapter_weights(str(adapter_path))
        completed_round_entries.append(round_cfg)

    # Distillation-only mode
    if args.distill_only and all_adapter_weights:
        combined = gather_combined_data(completed_round_entries)
        reset_adapter(model)
        print(f"\n  Running distillation-only on {len(all_adapter_weights)} specialists")
        for idx in round_indices:
            round_cfg = APPRENTICE_ROUNDS[idx]
            if round_cfg["domain"] in all_adapter_weights:
                distill_backbone(
                    model, all_adapter_weights, combined, tokenizer,
                    steps=round_cfg.get("distill_steps", 50),
                    seq_len=min(round_cfg["seq_len"], 512),
                )
                break

    # Router training (if enough specialists)
    n_completed = len(all_adapter_weights)
    if n_completed >= 3:
        print(f"\n  Training router ({n_completed} specialists)...")
        train_router(model, tokenizer, save_dir)
    else:
        print(f"\n  Skipping router — only {n_completed} specialists (need >=3)")

    export_backbone(model, save_dir)

    total_elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Training complete. All artifacts in {save_dir}")
    print(f"  Total wall time: {total_elapsed//60:.0f}m {total_elapsed%60:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
