"""
training_orchestrator.py — Round management loop for the apprenticeship protocol.

Usage:
  python training_orchestrator.py --rounds 1-5 --save-dir ./checkpoints
  python training_orchestrator.py --rounds 1-3 --resume ./checkpoints --save-dir ./checkpoints
"""

import json, os, argparse, time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from config import AgentMindConfig, APPRENTICE_ROUNDS
from model.agent_lm import AgentMind
from data.pipeline import AgentDataset
from lora import apply_lora, save_adapter, reset_adapter
from init import init_agentmind
from train import train_specialist, distill_backbone
from router import TaskRouter
from training_utils import _nested_weights
from monitor import print_hw
from stats_logger import GLOBAL as log


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


def load_domain_data(data_path: str, backbone, latent_stage: int = 1):
    from tokenizer_setup import load_tokenizer
    tok = load_tokenizer("agentmind_tok.model")
    ds = AgentDataset.from_raw([data_path], tokenizer=tok, cfg=backbone.cfg)
    ds.latent_stage = latent_stage
    print(f"  Loaded {len(ds.samples)} samples from {data_path} (latent_stage={latent_stage})")
    return ds


def gather_combined_data(completed_rounds: list[dict]) -> list[dict]:
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


def load_adapter_weights(adapter_path: str) -> dict:
    loaded = mx.load(adapter_path)
    if 'metadata' in loaded:
        del loaded['metadata']
    return loaded


def train_router(backbone, cfg, tok, save_dir):
    domain_names = [r["domain"] for r in APPRENTICE_ROUNDS]
    router = TaskRouter(
        d_model=cfg.d_model,
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
    router.train(router_data, backbone, tokenizer=tok, steps=200, lr=1e-3)
    router_save_path = str(save_dir / "router")
    router.save(router_save_path)


def export_backbone(backbone, save_dir):
    print(f"\n{'='*60}")
    print("  Saving final artifacts")
    print(f"{'='*60}")
    backbone_path = save_dir / "backbone.safetensors"
    backbone_params = {k: v for k, v in tree_flatten(backbone.parameters())
                       if not k.startswith("last_")}
    mx.save_safetensors(str(backbone_path), dict(backbone_params))
    print(f"  Backbone saved -> {backbone_path}")


def run_round(backbone, domain: str, data_path: str,
              specialist_steps: int, distill_steps: int, save_dir: str,
              seq_len: int = 256, latent_stage: int = 1,
              seq_len_schedule: dict = None,
              existing_adapters: dict = None) -> dict:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    reset_adapter(backbone)
    dataset = load_domain_data(data_path, backbone, latent_stage=latent_stage)

    print(f"\n{'='*60}")
    print(f"  Training specialist: {domain}")
    print(f"  Steps: {specialist_steps} | Seq len: {seq_len} | Latent stage: {latent_stage}")
    print(f"{'='*60}")

    adapter_weights = train_specialist(
        backbone, dataset, domain,
        steps=specialist_steps, seq_len=seq_len, latent_stage=latent_stage,
        seq_len_schedule=seq_len_schedule, syntax_aux_weight=0.2
    )

    save_adapter(backbone, domain, str(save_dir))
    adapter_path = str(save_dir / f"{domain}.safetensors")

    specialists = dict(existing_adapters or {})
    specialists[domain] = adapter_weights
    print(f"  Specialists for distillation: {list(specialists.keys())}")

    existing_domains = list(existing_adapters.keys()) if existing_adapters else []
    completed_domains = existing_domains + [domain]
    rounds_for_data = [r for r in APPRENTICE_ROUNDS if r["domain"] in completed_domains]
    combined = gather_combined_data(rounds_for_data)

    reset_adapter(backbone)
    distill_seq_len = min(seq_len, 512)
    print(f"\n  Distilling backbone ({distill_steps} steps, seq_len={distill_seq_len}, latent_stage={latent_stage})")
    final_loss = distill_backbone(
        backbone, specialists, combined,
        steps=distill_steps, seq_len=distill_seq_len, latent_stage=latent_stage,
        mtp_weight=0.2
    )

    return {
        "domain": domain,
        "adapter_path": adapter_path,
        "latent_stage": latent_stage,
        "distill_loss": final_loss if final_loss is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description="AgentMind Apprenticeship Training Orchestrator")
    parser.add_argument("--rounds", type=str, default="1-5",
                        help="Rounds to run: '1-5', '1,3,5', '2' (1-indexed)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint dir to resume from")
    parser.add_argument("--save-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    args = parser.parse_args()

    round_indices = parse_rounds(args.rounds)
    print(f"Rounds to run: {[i+1 for i in round_indices]}")

    t_start = time.time()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    adapters_dir = save_dir / "adapters"

    cfg = AgentMindConfig()
    from tokenizer_setup import load_tokenizer, get_token_ids, hydrate_config
    tok = load_tokenizer("agentmind_tok.model")
    ids = get_token_ids(tok)
    hydrate_config(cfg, tok)
    print("Config token IDs hydrated from tokenizer.")

    backbone = AgentMind(cfg)
    backbone = init_agentmind(backbone, cfg)

    default_backbone = Path(args.save_dir) / "backbone.npz"
    if default_backbone.exists():
        weights = mx.load(str(default_backbone))
        backbone.update(_nested_weights(weights))
        print(f"Loaded pretrained backbone from {default_backbone}")

    completed_domains = set()
    if args.resume:
        resume_dir = Path(args.resume)
        if resume_dir.exists():
            backbone_path = resume_dir / "backbone.npz"
            if backbone_path.exists():
                weights = mx.load(str(backbone_path))
                backbone.update(_nested_weights(weights))
                print(f"Resumed backbone from {backbone_path}")
            adapters_dir = resume_dir / "adapters"
            if adapters_dir.exists():
                for f in adapters_dir.glob("*.safetensors"):
                    completed_domains.add(f.stem)
                    print(f"  Found completed adapter: {f.stem}")
        else:
            print(f"Resume dir {resume_dir} not found — starting fresh")

    backbone = apply_lora(backbone, rank=32, alpha=32.0)
    print(f"Backbone initialized. Trainable params: "
          f"{sum(p.size for _,p in tree_flatten(backbone.trainable_parameters())):,}")

    log.phase("orchestrator", "start", rounds=args.rounds)
    print_hw("start")

    all_adapter_weights = {}
    completed_round_entries = []

    for idx in round_indices:
        round_cfg = APPRENTICE_ROUNDS[idx]

        if round_cfg["domain"] in completed_domains:
            print(f"  Skipping {round_cfg['domain']} — already completed")
            adapter_path = adapters_dir / f"{round_cfg['domain']}.safetensors"
            if not adapter_path.exists() and args.resume:
                adapter_path = Path(args.resume) / "adapters" / f"{round_cfg['domain']}.safetensors"
            if adapter_path.exists():
                all_adapter_weights[round_cfg['domain']] = load_adapter_weights(str(adapter_path))
            completed_round_entries.append(round_cfg)
            continue

        log.phase("round", "start", domain=round_cfg["domain"],
                  spec_steps=round_cfg["specialist_steps"],
                  seq_len=round_cfg["seq_len"],
                  latent_stage=round_cfg["latent_stage"])
        print_hw(f"round {idx+1} start")
        result = run_round(
            backbone=backbone,
            domain=round_cfg["domain"],
            data_path=round_cfg["file"],
            specialist_steps=round_cfg["specialist_steps"],
            distill_steps=round_cfg["distill_steps"],
            save_dir=str(adapters_dir),
            seq_len=round_cfg["seq_len"],
            latent_stage=round_cfg["latent_stage"],
            seq_len_schedule=round_cfg.get("seq_len_schedule"),
            existing_adapters=all_adapter_weights if all_adapter_weights else None,
        )

        log.phase("round", "complete", domain=round_cfg["domain"],
                  distill_loss=result.get("distill_loss", "N/A"))
        print_hw(f"round {idx+1} end")
        distill_loss = result.get("distill_loss", "N/A")
        print(f"\n  Round {idx+1} ({round_cfg['domain']}) complete")
        print(f"    Adapter: {result['adapter_path']}")
        print(f"    Distill loss: {distill_loss if distill_loss is not None else 'N/A'}")

        adapter_path = adapters_dir / f"{round_cfg['domain']}.safetensors"
        if adapter_path.exists():
            all_adapter_weights[round_cfg['domain']] = load_adapter_weights(str(adapter_path))
        completed_round_entries.append(round_cfg)

    n_completed = len(all_adapter_weights)
    if n_completed < 3:
        print(f"\n  Skipping router training — only {n_completed} specialists exist (need >=3)")
        log.phase("router", "skipped", n_completed=n_completed)
    else:
        log.phase("router", "start")
        print_hw("router")
        print(f"\n{'='*60}")
        print(f"  Training Router ({n_completed} specialists)")
        print(f"{'='*60}")
        train_router(backbone, cfg, tok, save_dir)
        log.phase("router", "complete")

    export_backbone(backbone, save_dir)

    total_elapsed = time.time() - t_start
    log.summary("orchestrator", total_elapsed=total_elapsed,
                rounds=args.rounds, completed=[r["domain"] for r in completed_round_entries])
    print_hw("end")
    print(f"\n{'='*60}")
    print(f"  Training complete. All artifacts in {save_dir}")
    print(f"  Total wall time: {total_elapsed//60:.0f}m {total_elapsed%60:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
