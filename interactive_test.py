#!/usr/bin/env python3
"""Interactive test harness for trained AgentMind model.

Usage:
  python3 interactive_test.py
  python3 interactive_test.py --prompt "Search arxiv for Mamba papers" --max-tokens 100
"""
import argparse, time, sys
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten

from config import AgentMindConfig
from model.agent_lm import AgentMind
from init import init_agentmind
from lora import apply_lora, load_adapter, reset_adapter
from training_utils import _nested_weights


def load(checkpoints_dir: str, adapter_name: str = None, bare: bool = False):
    checkpoints_dir = Path(checkpoints_dir)
    cfg = AgentMindConfig()

    from tokenizer_setup import load_tokenizer, get_token_ids, hydrate_config
    tok = load_tokenizer("agentmind_tok.model")
    ids = get_token_ids(tok)
    hydrate_config(cfg, tok)

    backbone = AgentMind(cfg)
    backbone = init_agentmind(backbone, cfg)

    # Try safetensors first, then npz
    backbone_path = checkpoints_dir / "backbone.safetensors"
    if not backbone_path.exists():
        backbone_path = checkpoints_dir / "backbone.npz"
    if not backbone_path.exists():
        backbone_path = checkpoints_dir / "backbone.safetensors" / "weights.safetensors"

    if backbone_path.exists():
        weights = mx.load(str(backbone_path))
        # Filter out MTP heads and LoRA A/B
        clean = {k: v for k, v in weights.items()
                 if not k.startswith("mtp.")
                 and not k.endswith(".A") and not k.endswith(".B")
                 and k != "last_hidden" and k != "last_mtp_logits"}

        if bare:
            backbone.update(_nested_weights(clean))
            print(f"Loaded bare backbone from {backbone_path} ({len(clean)} keys)")
        else:
            apply_lora(backbone, rank=16, alpha=32.0)
            backbone.update(_nested_weights(clean))
            print(f"Loaded backbone from {backbone_path} ({len(clean)} keys, "
                  f"skipped {len(weights)-len(clean)} MTP/A/B)")

            if adapter_name:
                adapter_path = checkpoints_dir / "adapters" / f"{adapter_name}.safetensors"
                if adapter_path.exists():
                    backbone = load_adapter(backbone, str(adapter_path))
                else:
                    print(f"Adapter {adapter_name} not found at {adapter_path}")
    else:
        print(f"WARNING: no backbone weights at {backbone_path} — random init")

    backbone.eval()
    return backbone, tok, cfg


def generate(backbone, tok, cfg, prompt: str, max_new_tokens: int = 120):
    prompt_ids = mx.array([tok.encode(prompt)])
    h_states = {}
    out_ids = []

    for _ in range(max_new_tokens):
        logits_t, h_states = backbone.forward_with_state(
            mx.array([[out_ids[-1]]]) if out_ids else prompt_ids,
            h_states if out_ids else h_states,
        )
        ntok = mx.argmax(logits_t[0, -1]).item()
        out_ids.append(ntok)
        if ntok == cfg.eos_id:
            break

    return tok.decode(out_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", default="./checkpoints")
    parser.add_argument("--adapter", default="tool_caller")
    parser.add_argument("--bare", action="store_true", help="Load backbone without LoRA")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=120)
    args = parser.parse_args()

    backbone, tok, cfg = load(args.checkpoints, args.adapter, bare=args.bare)

    if args.prompt:
        full = f"<|user|>{args.prompt}<|assistant|>"
        t0 = time.time()
        out = generate(backbone, tok, cfg, full, args.max_tokens)
        print(f"[{time.time()-t0:.1f}s] {out}")
    else:
        prompts = [
            "Search arxiv for Mamba SSM papers",
            "Get the weather in Tokyo",
            "Find the stock price of Apple",
            "Run a SQL query to find all users",
        ]
        for p in prompts:
            full = f"<|user|>{p}<|assistant|>"
            t0 = time.time()
            out = generate(backbone, tok, cfg, full, args.max_tokens)
            print(f"\n{p}")
            print(f"  [{time.time()-t0:.1f}s] {out}")
