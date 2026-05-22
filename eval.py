import mlx.core as mx
import mlx.nn as nn
import json, re
from data.pipeline import make_dataloader

def compute_loss(model, dataset, tok, cfg, max_batches: int = 50, max_len: int = 512) -> float:
    """Average cross-entropy loss on validation set."""
    total_loss = 0.0
    total_tokens = 0
    loader = make_dataloader(dataset, batch_size=1, shuffle=False, max_len=max_len)

    model.eval()
    for i, (input_ids, targets) in enumerate(loader):
        if i >= max_batches:
            break
        logits, _ = model(input_ids)
        B, L, V = logits.shape
        flat_logits  = logits.reshape(-1, V)
        flat_targets = targets.reshape(-1)

        mask = (flat_targets != -100).astype(mx.float32)
        safe_targets = mx.where(flat_targets == -100, 0, flat_targets)
        loss = nn.losses.cross_entropy(flat_logits, safe_targets, reduction='none')
        total_loss   += (loss * mask).sum().item()
        total_tokens += mask.sum().item()

    model.train()
    if total_tokens == 0:
        return 100.0
    return total_loss / total_tokens

def tool_call_accuracy(model, prompts: list[str], tok, cfg) -> float:
    """
    Check if model reliably produces valid JSON after <|tool_call|>.
    A structurally valid JSON tool call = pass.
    """
    passed = 0
    TOOL_CALL_TOKEN = "<|tool_call|>"

    for prompt in prompts:
        ids = mx.array([tok.encode(prompt, add_bos=True)])
        output_ids = []

        # Generate up to 200 tokens
        for _ in range(200):
            logits, _ = model(ids)
            next_tok = mx.argmax(logits[0, -1]).item()
            output_ids.append(next_tok)
            ids = mx.array([[next_tok]])
            if next_tok == cfg.eos_id:
                break

        decoded = tok.decode(output_ids)

        # Check for valid JSON tool call
        if TOOL_CALL_TOKEN in decoded:
            after = decoded.split(TOOL_CALL_TOKEN)[-1]
            try:
                obj = json.loads(after.split("<|observe|>")[0].strip())
                if "name" in obj and "args" in obj:
                    passed += 1
            except json.JSONDecodeError:
                pass

    return passed / max(len(prompts), 1)

def format_adherence(model, prompts: list[str], tok, cfg) -> dict:
    """
    Check structural output quality:
    - Does it use <|plan|> for multi-step queries?
    - Does it use <|scratch|> for intermediate reasoning?
    - Does it terminate with EOS?
    """
    results = {"plan": 0, "scratch": 0, "eos": 0, "total": len(prompts)}

    for prompt in prompts:
        ids = mx.array([tok.encode(prompt, add_bos=True)])
        output_ids = []

        for _ in range(300):
            logits, _ = model(ids)
            next_tok = mx.argmax(logits[0, -1]).item()
            output_ids.append(next_tok)
            ids = mx.array([[next_tok]])
            if next_tok == cfg.eos_id:
                results["eos"] += 1
                break

        decoded = tok.decode(output_ids)
        if "<|plan|>" in decoded:
            results["plan"] += 1
        if "<|scratch|>" in decoded:
            results["scratch"] += 1

    return results

def evaluate(model, val_dataset, tok, cfg, max_len: int = 512):
    """Combined eval — returns (val_loss, tool_acc)."""
    try:
        val_loss = compute_loss(model, val_dataset, tok, cfg, max_batches=10, max_len=max_len)
    except Exception:
        val_loss = 100.0

    test_prompts = [
        "<|user|>Search arxiv for Mamba SSM papers<|assistant|>",
        "<|user|>Get the weather in Tokyo and Pune<|assistant|>",
        "<|user|>Run the test suite and fix any failures<|assistant|>",
    ]
    try:
        tool_acc = tool_call_accuracy(model, test_prompts, tok, cfg)
    except Exception:
        tool_acc = 0.0

    return val_loss, tool_acc
