import json, shutil
import mlx.core as mx
from mlx.utils import tree_flatten
from pathlib import Path

HF_CONFIG_TEMPLATE = {
    "architectures": ["AgentMindForCausalLM"],
    "model_type": "agentmind",
    "vocab_size": 32000,
    "d_model": 1024,
    "n_layers": 16,
    "d_state": 64,
    "d_conv": 4,
    "expand": 2,
    "dt_rank": 64,
    "n_heads": 8,
    "attn_window": 256,
    "attn_every": 4,
    "ffn_mult": 8 / 3,
    "max_seq_len": 8192,
    "tie_embeddings": True,
}

def save_hf_format(model, cfg, tok, save_dir: str):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    config = dict(HF_CONFIG_TEMPLATE)
    config["pad_token_id"] = cfg.pad_id
    config["bos_token_id"] = cfg.bos_id
    config["eos_token_id"] = cfg.eos_id
    config["tool_call_token_id"] = cfg.tool_call_id
    config["assistant_token_id"] = cfg.assistant_id
    config["user_token_id"] = cfg.user_id
    config["system_token_id"] = cfg.system_id
    config["think_start_token_id"] = cfg.think_start_id
    config["think_end_token_id"] = cfg.think_end_id
    config["vocab_size"] = cfg.vocab_size

    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    weights = dict(tree_flatten(model.parameters()))
    mx.savez(str(save_path / "weights.npz"), **weights)

    shutil.copy("agentmind_tok.model", save_path / "tokenizer.model")
    with open(save_path / "tokenizer_config.json", "w") as f:
        json.dump({
            "pad_token": "<pad>",
            "bos_token": "<s>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "tokenizer_class": "SentencePieceTokenizer",
        }, f, indent=2)

    num_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Model exported to {save_path}")
    print(f"  config.json  — {len(config)} fields (token IDs from tokenizer)")
    print(f"  weights.npz  — {num_params:,} params")
    print(f"  tokenizer.model + tokenizer_config.json")
