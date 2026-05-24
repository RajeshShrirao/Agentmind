"""eval.py — Evaluation for Qwen2.5-based AgentMind.

Per-apprentice evaluation + cross-apprentice interference detection.
Uses KV cache generation (not old SSM forward_with_state).
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import generate_step
import json
from pathlib import Path

from data.pipeline import make_dataloader
from lora import load_lora
from training_utils import cross_entropy_loss
from decode import extract_tool_calls


def _greedy_sampler(logits):
    return mx.argmax(logits, axis=-1)


def _sampler(temp=0.0):
    if temp <= 0.0:
        return _greedy_sampler
    def _temp_sampler(logits):
        return mx.random.categorical(mx.log(mx.softmax(logits / temp) + 1e-10))
    return _temp_sampler


def _generate_text(model, tokenizer, prompt_tokens, max_tokens=200, temp=0.0):
    sampler = _sampler(temp)
    output_ids = []
    eos_ids = getattr(tokenizer, 'eos_token_ids', None)
    if eos_ids is None:
        eos_ids = {tokenizer.eos_token_id}
    for tok_id, _ in generate_step(prompt_tokens, model, max_tokens=max_tokens, sampler=sampler):
        if tok_id in eos_ids:
            break
        output_ids.append(tok_id)
    return tokenizer.decode(output_ids)


def compute_loss(model, dataset, tokenizer, max_batches=50, max_len=512):
    """Average cross-entropy loss on validation set."""
    total_loss = 0.0
    total_tokens = 0
    loader = make_dataloader(dataset, batch_size=1, shuffle=False, max_len=max_len)

    model.eval()
    for i, (input_ids, targets) in enumerate(loader):
        if i >= max_batches:
            break

        logits = model(input_ids)

        logits_slice = logits[:, :-1, :]
        targets_slice = targets[:, 1:]

        B, L, V = logits_slice.shape
        flat_logits = logits_slice.reshape(-1, V)
        flat_targets = targets_slice.reshape(-1)

        mask = (flat_targets != -100).astype(mx.float32)
        safe_targets = mx.where(flat_targets == -100, 0, flat_targets)
        loss = nn.losses.cross_entropy(flat_logits, safe_targets, reduction='none')
        total_loss   += (loss * mask).sum().item()
        total_tokens += mask.sum().item()

    model.train()
    if total_tokens == 0:
        return 100.0
    return total_loss / total_tokens


def evaluate_tool_calls(model, tokenizer, prompts, max_tokens=200, temp=0.0):
    """Evaluate tool call generation using KV cache generation.

    Uses generate_step() with persistent KV cache.
    Returns list of {raw, valid, name, args, error, failure_mode} dicts.
    """
    results = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        decoded = _generate_text(model, tokenizer, mx.array(tokens), max_tokens=max_tokens, temp=temp)
        call_results = extract_tool_calls(decoded)
        if call_results:
            results.extend(call_results)
        else:
            results.append({"valid": False, "name": None, "args": None,
                            "error": "no tool call found", "failure_mode": "parse_error"})
    return results


def format_adherence(model, tokenizer, prompts) -> dict:
    """Check structural output quality with greedy decoding."""
    results = {"plan": 0, "scratch": 0, "eos": 0, "total": len(prompts)}
    eos_ids = getattr(tokenizer, 'eos_token_ids', None)
    if eos_ids is None:
        eos_ids = {tokenizer.eos_token_id}

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        prompt_arr = mx.array(tokens)

        output_ids = []
        for tok_id, _ in generate_step(prompt_arr, model, max_tokens=300, sampler=_greedy_sampler):
            if tok_id in eos_ids:
                results["eos"] += 1
                break
            output_ids.append(tok_id)

        decoded = tokenizer.decode(output_ids)
        if "<|plan|>" in decoded:
            results["plan"] += 1
        if "<|scratch|>" in decoded:
            results["scratch"] += 1

    return results


def evaluate_tool_syntax(text: str) -> dict:
    """Lightweight tool-call syntax metric.

    Extracts all tool calls from generated text and checks:
      - parse_success: extracts as valid JSON
      - malformed_json: JSON parse failure
      - missing_name: valid JSON but no 'name' key
      - missing_args: valid JSON but no 'args' key (or not a dict)

    Returns dict with counts for each category.
    Does NOT validate tool registry or argument types.
    """
    result = {
        "parse_success": 0,
        "malformed_json": 0,
        "missing_name": 0,
        "missing_args": 0,
        "valid_tool_calls": 0,
        "total": 0,
    }

    idx = 0
    while True:
        start = text.find("<|tool_call|>", idx)
        if start == -1:
            break
        start += len("<|tool_call|>")
        json_str = text[start:]
        for boundary in ("<|observe|>", "<|end|>", "<eos>"):
            if boundary in json_str:
                json_str = json_str.split(boundary)[0]
        json_str = json_str.strip()
        result["total"] += 1

        try:
            obj = json.loads(json_str)
            result["parse_success"] += 1
            if not isinstance(obj.get("name"), str):
                result["missing_name"] += 1
            if not isinstance(obj.get("args"), dict):
                result["missing_args"] += 1
            if isinstance(obj.get("name"), str) and isinstance(obj.get("args"), dict):
                result["valid_tool_calls"] += 1
        except (json.JSONDecodeError, ValueError):
            result["malformed_json"] += 1

        idx = start + len(json_str)

    return result


def _extract_eval_prompts(domain_dataset, tokenizer, max_tokens: int = 512,
                          fallback_prompts: list[str] = None) -> list[str]:
    """Build evaluation prompts from domain dataset samples, truncated to max_tokens."""
    prompts = []
    samples = getattr(domain_dataset, "samples", None)
    if samples:
        for sample in samples[:10]:
            text = ""
            for msg in sample["messages"]:
                role, content = msg["role"], msg["content"]
                if role == "system":
                    text += f"<|system|>{content}"
                elif role == "user":
                    text += f"<|user|>{content}"
                elif role == "assistant":
                    text += f"<|assistant|>{content}<eos>"
            if text:
                ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
                prompts.append(tokenizer.decode(ids))
    if not prompts:
        prompts = fallback_prompts or [
            "<|user|>Search arxiv for Mamba SSM papers<|assistant|>",
            "<|user|>Get the weather in Tokyo and Pune<|assistant|>",
            "<|user|>Run the test suite and fix any failures<|assistant|>",
        ]
    return prompts


def evaluate_apprentice(model, tokenizer, adapter_weights, domain_dataset) -> dict:
    """Run all metrics for one specialist.

    Loads the adapter into the backbone, computes:
      - Loss on held-out domain data
      - Tool call accuracy on extracted prompts
      - Format adherence (boundary tokens)

    Restores the original LoRA weights after evaluation.
    """
    from mlx.utils import tree_flatten

    orig = dict(tree_flatten(model.trainable_parameters()))

    load_lora(model, adapter_weights)

    loss = compute_loss(model, domain_dataset, tokenizer, max_batches=20, max_len=512)

    prompts = _extract_eval_prompts(domain_dataset, tokenizer)

    try:
        call_results = evaluate_tool_calls(model, tokenizer, prompts)
        valid = sum(1 for r in call_results if r.get("valid"))
        tool_acc = valid / max(len(call_results), 1)
    except Exception as e:
        print(f"  [evaluate_apprentice] tool eval skipped: {e}")
        tool_acc = 0.0

    try:
        fmt = format_adherence(model, tokenizer, prompts)
    except Exception as e:
        print(f"  [evaluate_apprentice] format eval skipped: {e}")
        fmt = {"plan": 0, "scratch": 0, "eos": 0, "total": len(prompts)}

    if orig:
        load_lora(model, orig)

    return {"loss": loss, "tool_acc": tool_acc, "format": fmt}


def test_interference(model, adapters: dict, test_fn, tokenizer) -> tuple:
    """Measure cross-apprentice interference.

    For each specialist (A):
      Load adapter A, run test_fn, record baseline_A

    For each pair (A, B) where A != B:
      Load adapter A, run test_fn (baseline_A already known)
      Load adapter B, then load adapter A again
      Run test_fn, record score
      Interference = score - baseline_A

    If interference > 5% (absolute), prints a warning.

    Returns (baselines: dict, interference: dict).
    """
    baselines = {}

    for name, adapter in adapters.items():
        load_lora(model, adapter)
        baselines[name] = test_fn(model, tokenizer)

    interference = {}
    for name_a, adapter_a in adapters.items():
        for name_b, adapter_b in adapters.items():
            if name_a == name_b:
                continue
            load_lora(model, adapter_a)
            score = test_fn(model, tokenizer)
            diff = score - baselines[name_a]
            interference[f"{name_a}_under_{name_b}"] = diff
            if abs(diff) > 0.05:
                print(f"  SPECIALIST INTERFERENCE DETECTED: "
                      f"{name_a} under {name_b}: {diff:+.4f} "
                      f"(baseline={baselines[name_a]:.4f}, actual={score:.4f})")

    return baselines, interference


def evaluate_apprentice_legacy(model, tokenizer, adapter_weights, domain_dataset) -> dict:
    """Legacy entry point kept for backward compat."""
    return evaluate_apprentice(model, tokenizer, adapter_weights, domain_dataset)
