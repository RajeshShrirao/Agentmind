"""
training_orchestrator.py — Round management loop for the apprenticeship protocol.

Runs the full apprenticeship protocol:
  For each domain (tool_caller, planner, recovery, code, research):
    1. Train LoRA specialist on domain data with correct latent stage
    2. Save specialist adapter
    3. Load all completed specialists
    4. Distill specialist knowledge into backbone
  Then:
    5. Train router classifier on backbone hidden states
    6. Final export: backbone + all adapters + router

Usage:
  python training_orchestrator.py --rounds 1-5 --save-dir ./checkpoints
  python training_orchestrator.py --rounds 1-3 --resume ./checkpoints --save-dir ./checkpoints
"""

import json, os, argparse, copy, time
from pathlib import Path
from monitor import print_hw
from stats_logger import GLOBAL as log

import mlx.core as mx
from mlx.utils import tree_flatten

from config import AgentMindConfig
from model.agent_lm import AgentMind
from data.pipeline import AgentDataset
from lora import apply_lora, save_adapter, reset_adapter
from init import init_agentmind
from train import train_specialist, distill_backbone
from router import TaskRouter

# Per-round latent stage mapping — explicit, from the design doc table:
#   Round 1 (tool_caller):         latent stages 1->2 (basic tool calling, then wrap scratch in boundaries)
#   Round 2 (planner):             latent stages 2->3 (planned trajectories, 50% CoT -> latent replacement)
#   Round 3+ (recovery/code/research): latent stage 4 (full latent — CoT removed, only think boundaries)
# This mapping MUST be explicit — NOT derived from global step count.
# Pass latent_stage directly to train_specialist() for each round.

ROUNDS = [
    {
        "domain": "tool_caller",
        "file": "data/apprentice_tool_caller.jsonl",
        "specialist_steps": 2000,
        "seq_len": 256,
        "seq_len_schedule": {0: 128, 200: 256},
        "distill_steps": 200,
        "adversarial": 0.3,
        "latent_stage": 1,
    },
    {
        "domain": "planner",
        "file": "data/apprentice_planner.jsonl",
        "specialist_steps": 300,
        "seq_len": 512,
        "seq_len_schedule": None,
        "distill_steps": 150,
        "adversarial": 0.3,
        "latent_stage": 2,
    },
    {
        "domain": "recovery",
        "file": "data/apprentice_recovery.jsonl",
        "specialist_steps": 300,
        "seq_len": 256,
        "seq_len_schedule": {0: 128, 150: 256},
        "distill_steps": 150,
        "adversarial": 0.4,
        "latent_stage": 2,
    },
    {
        "domain": "code",
        "file": "data/apprentice_code.jsonl",
        "specialist_steps": 300,
        "seq_len": 512,
        "seq_len_schedule": None,
        "distill_steps": 150,
        "adversarial": 0.3,
        "latent_stage": 4,
    },
    {
        "domain": "research",
        "file": "data/apprentice_research.jsonl",
        "specialist_steps": 300,
        "seq_len": 1024,
        "seq_len_schedule": None,
        "distill_steps": 150,
        "adversarial": 0.3,
        "latent_stage": 4,
    },
]


def parse_rounds(rounds_spec: str) -> list[int]:
    """Parse --rounds spec like '1-5' or '1,3,5' or '2' into 0-indexed round indices."""
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
    """Load a JSONL domain dataset and wrap as AgentDataset."""
    from tokenizer_setup import load_tokenizer

    tok = load_tokenizer("agentmind_tok.model")
    ds = AgentDataset.__new__(AgentDataset)
    ds.samples = []
    ds.cfg = backbone.cfg
    ds.tok = tok
    ds.latent_stage = latent_stage
    ds.ids_array = None
    ds.labels_array = None
    ds._cache = {}
    ds.weights = {"instruction": 0.3, "tool_single": 0.3,
                  "agent_multi": 0.25, "recovery": 0.15}

    with open(data_path) as f:
        for line in f:
            ds.samples.append(json.loads(line.strip()))

    print(f"  Loaded {len(ds.samples)} samples from {data_path} (latent_stage={latent_stage})")
    return ds


def gather_combined_data(completed_rounds: list[dict]) -> list[dict]:
    """Gather samples from all completed domains, tagging each with its domain."""
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
    """Load adapter weights from .safetensors, stripping metadata."""
    loaded = mx.load(adapter_path)
    if 'metadata' in loaded:
        del loaded['metadata']
    return loaded


def run_round(backbone, domain: str, data_path: str,
              specialist_steps: int, distill_steps: int, save_dir: str,
              seq_len: int = 256, latent_stage: int = 1,
              seq_len_schedule: dict = None,
              existing_adapters: dict = None) -> dict:
    """
    Run a single apprenticeship round.

    Steps:
      1. Reset LoRA adapter to random init (specialist starts fresh)
      2. Load domain dataset with correct latent_stage
      3. train_specialist() -> adapter_weights
      4. save_adapter()
      5. Build specialists dict (existing + new)
      6. Gather combined data from all completed domains
      7. Reset LoRA again (distillation starts from clean slate)
      8. distill_backbone()

    Returns dict with round results.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reset LoRA to random init
    reset_adapter(backbone)

    # 2. Load domain dataset
    dataset = load_domain_data(data_path, backbone, latent_stage=latent_stage)

    # 3. Train specialist
    print(f"\n{'='*60}")
    print(f"  Training specialist: {domain}")
    print(f"  Steps: {specialist_steps} | Seq len: {seq_len} | Latent stage: {latent_stage}")
    print(f"{'='*60}")

    adapter_weights = train_specialist(
        backbone, dataset, domain,
        steps=specialist_steps, seq_len=seq_len, latent_stage=latent_stage,
        seq_len_schedule=seq_len_schedule, syntax_aux_weight=0.2
    )

    # 4. Save adapter
    save_adapter(backbone, domain, str(save_dir))
    adapter_path = str(save_dir / f"{domain}.safetensors")

    # 5. Build specialists dict (existing + new)
    specialists = dict(existing_adapters or {})
    specialists[domain] = adapter_weights
    print(f"  Specialists for distillation: {list(specialists.keys())}")

    # 6. Gather combined data from all completed domains
    existing_domains = list(existing_adapters.keys()) if existing_adapters else []
    completed_domains = existing_domains + [domain]
    rounds_for_data = [r for r in ROUNDS if r["domain"] in completed_domains]
    combined = gather_combined_data(rounds_for_data)

    # 7. Reset LoRA so distillation doesn't start from specialist weights
    reset_adapter(backbone)

    # 8. Distill backbone with domain seq_len (half for RAM safety on large domains)
    distill_seq_len = min(seq_len, 512)
    print(f"\n  Distilling backbone ({distill_steps} steps, seq_len={distill_seq_len}, latent_stage={latent_stage})")
    final_loss = distill_backbone(
        backbone, specialists, combined,
        steps=distill_steps, seq_len=distill_seq_len, latent_stage=latent_stage,
        mtp_weight=0.0
    )

    return {
        "domain": domain,
        "adapter_path": adapter_path,
        "latent_stage": latent_stage,
        "distill_loss": final_loss if final_loss is not None else None,
    }


# ── CLI ────────────────────────────────────────────────────

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

    # Config
    cfg = AgentMindConfig()

    # Tokenizer
    from tokenizer_setup import load_tokenizer, get_token_ids, hydrate_config
    tok = load_tokenizer("agentmind_tok.model")
    ids = get_token_ids(tok)
    hydrate_config(cfg, tok)
    print("Config token IDs hydrated from tokenizer.")

    # Model
    backbone = AgentMind(cfg)
    backbone = init_agentmind(backbone, cfg)
    backbone = apply_lora(backbone, rank=16, alpha=32.0)
    print(f"Backbone initialized. Trainable params: "
          f"{sum(p.size for _,p in tree_flatten(backbone.trainable_parameters())):,}")

    # Resume support
    completed_domains = set()
    if args.resume:
        resume_dir = Path(args.resume)
        if resume_dir.exists():
            backbone_path = resume_dir / "backbone.safetensors"
            if backbone_path.exists():
                weights = mx.load(str(backbone_path))
                backbone.update(weights)
                print(f"Resumed backbone from {backbone_path}")

            for f in (resume_dir / "adapters").glob("*.safetensors"):
                completed_domains.add(f.stem)
                print(f"  Found completed adapter: {f.stem}")
        else:
            print(f"Resume dir {resume_dir} not found — starting fresh")

    log.phase("orchestrator", "start", rounds=args.rounds)
    print_hw("start")
    # Round loop
    all_adapter_weights = {}
    completed_round_entries = []

    for idx in round_indices:
        round_cfg = ROUNDS[idx]

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

        # Reload adapter weights for subsequent rounds
        adapter_path = adapters_dir / f"{round_cfg['domain']}.safetensors"
        if adapter_path.exists():
            all_adapter_weights[round_cfg['domain']] = load_adapter_weights(str(adapter_path))

        completed_round_entries.append(round_cfg)

    # Router: requires ≥3 specialists before hidden-state manifold is separable
    log.phase("router", "start")
    print_hw("router")
    n_completed = len(all_adapter_weights)
    if n_completed < 3:
        print(f"\n  Skipping router training — only {n_completed} specialists exist (need ≥3)")
        log.phase("router", "skipped", n_completed=n_completed)
    else:
        print(f"\n{'='*60}")
        print(f"  Training Router ({n_completed} specialsts)")
        print(f"{'='*60}")

        domain_names = [r["domain"] for r in ROUNDS]
        router = TaskRouter(
            d_model=cfg.d_model,
            hidden=64,
            n_domains=len(domain_names),
            domain_names=domain_names,
        )

        router_data_path = "data/router_training.jsonl"
        if os.path.exists(router_data_path):
            router_data = []
            with open(router_data_path) as f:
                for line in f:
                    router_data.append(json.loads(line.strip()))
            print(f"  Loaded {len(router_data)} router training samples")

            router.train(router_data, backbone, tokenizer=tok, steps=200, lr=1e-3)

            router_save_path = str(save_dir / "router")
            router.save(router_save_path)
        else:
            print(f"  Warning: {router_data_path} not found — skipping router training")

    log.phase("router", "complete")

    # Final export
    print(f"\n{'='*60}")
    print("  Saving final artifacts")
    print(f"{'='*60}")

    backbone_path = save_dir / "backbone.safetensors"
    backbone_params = {k: v for k, v in tree_flatten(backbone.parameters())
                       if not k.startswith("last_")}
    mx.save_safetensors(str(backbone_path), dict(backbone_params))
    print(f"  Backbone saved -> {backbone_path}")

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
