import mlx.core as mx
import mlx.nn as nn
import json
from data.pipeline import make_dataloader
from decode import generate_tool_call, validate_tool_call, tool_eval_report, print_tool_report, extract_tool_calls, TOOL_REGISTRY


def compute_loss(model, dataset, tok, cfg, max_batches: int = 50, max_len: int = 512) -> float:
    """Average cross-entropy loss on validation set."""
    total_loss = 0.0
    total_tokens = 0
    loader = make_dataloader(dataset, batch_size=1, shuffle=False, max_len=max_len)

    model.eval()
    for i, (input_ids, targets) in enumerate(loader):
        if i >= max_batches:
            break

        from model.latent import latent_loss_mask
        stage = getattr(dataset, "latent_stage", 1)
        if stage >= 3:
            masked_targets = latent_loss_mask(input_ids, targets, cfg.think_start_id, cfg.think_end_id)
        else:
            masked_targets = targets

        logits, _ = model(input_ids)

        logits = logits[:, :-1, :]
        masked_targets = masked_targets[:, 1:]

        B, L, V = logits.shape
        flat_logits  = logits.reshape(-1, V)
        flat_targets = masked_targets.reshape(-1)

        mask = (flat_targets != -100).astype(mx.float32)
        safe_targets = mx.where(flat_targets == -100, 0, flat_targets)
        loss = nn.losses.cross_entropy(flat_logits, safe_targets, reduction='none')
        total_loss   += (loss * mask).sum().item()
        total_tokens += mask.sum().item()

    model.train()
    if total_tokens == 0:
        return 100.0
    return total_loss / total_tokens


def evaluate_tool_calls(model, prompts: list[str], tok, cfg) -> list[dict]:
    """
    Evaluate tool call generation using structured decoding.
    Each prompt is fed to the model via greedy decode.
    Returns per-call result dicts from decode.py validation.
    """
    results = []
    for prompt in prompts:
        ids = mx.array([tok.encode(prompt, add_bos=True)])
        result = generate_tool_call(model, ids, {}, cfg, tok)
        results.append(result)
    return results


def evaluate_tool_calls_from_text(model, prompts: list[str], tok, cfg) -> list[dict]:
    """
    Legacy-style: generate freeform text, then extract and validate tool calls.
    Useful for comparison. Uses argmax (no temperature).
    """
    results = []
    for prompt in prompts:
        ids = mx.array([tok.encode(prompt, add_bos=True)])
        output_ids = []
        for _ in range(200):
            logits, _ = model(ids)
            next_tok = mx.argmax(logits[0, -1]).item()
            output_ids.append(next_tok)
            ids = mx.concatenate([ids, mx.array([[next_tok]])], axis=1)
            if next_tok == cfg.eos_id:
                break
        decoded = tok.decode(output_ids)
        call_results = extract_tool_calls(decoded)
        if call_results:
            results.extend(call_results)
        else:
            results.append({"valid": False, "name": None, "args": None,
                            "error": "no tool call found", "failure_mode": "parse_error"})
    return results


def format_adherence(model, prompts: list[str], tok, cfg) -> dict:
    """Check structural output quality with greedy decoding."""
    results = {"plan": 0, "scratch": 0, "eos": 0, "total": len(prompts)}

    for prompt in prompts:
        ids = mx.array([tok.encode(prompt, add_bos=True)])
        output_ids = []

        for _ in range(300):
            logits, _ = model(ids)
            next_tok = mx.argmax(logits[0, -1]).item()
            output_ids.append(next_tok)
            ids = mx.concatenate([ids, mx.array([[next_tok]])], axis=1)
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
    """Combined eval — returns (val_loss, tool_report)."""
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
        call_results = evaluate_tool_calls(model, test_prompts, tok, cfg)
        tool_report = tool_eval_report(call_results)
        if tool_report["total"] > 0:
            print_tool_report(tool_report)
    except Exception:
        tool_report = {"total": 0, "valid": 0, "valid_pct": 0.0,
                       "breakdown": {"eval_error": 1}, "tool_counts": {}}

    return val_loss, tool_report


def tool_call_accuracy(model, prompts: list[str], tok, cfg) -> float:
    """Legacy: kept for backward compat. Returns fraction of prompts with valid JSON."""
    results = evaluate_tool_calls(model, prompts, tok, cfg)
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("valid")) / max(len(results), 1)
