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
    Maintains SSM state via forward_with_state for O(L) not O(L²).
    Uses argmax (no temperature).
    """
    results = []
    for prompt in prompts:
        ids = mx.array([tok.encode(prompt, add_bos=True)])
        h_states = {}
        output_ids = []
        for _ in range(200):
            logits, h_states = model.forward_with_state(ids, h_states)
            next_tok = mx.argmax(logits[0, -1]).item()
            output_ids.append(next_tok)
            ids = mx.array([[next_tok]])
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
    """Check structural output quality with greedy decoding. Maintains SSM state."""
    results = {"plan": 0, "scratch": 0, "eos": 0, "total": len(prompts)}

    for prompt in prompts:
        ids = mx.array([tok.encode(prompt, add_bos=True)])
        h_states = {}
        output_ids = []

        for _ in range(300):
            logits, h_states = model.forward_with_state(ids, h_states)
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


def evaluate_tool_syntax(text: str) -> dict:
    """
    Lightweight tool-call syntax metric.

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


def tool_call_accuracy(model, prompts: list[str], tok, cfg) -> float:
    """Legacy: kept for backward compat. Returns fraction of prompts with valid JSON."""
    results = evaluate_tool_calls(model, prompts, tok, cfg)
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("valid")) / max(len(results), 1)


def _extract_eval_prompts(domain_dataset, tok, max_tokens: int = 512,
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
                ids = tok.encode(text, add_bos=True)[:max_tokens]
                prompts.append(tok.decode(ids))
    if not prompts:
        prompts = fallback_prompts or [
            "<|user|>Search arxiv for Mamba SSM papers<|assistant|>",
            "<|user|>Get the weather in Tokyo and Pune<|assistant|>",
            "<|user|>Run the test suite and fix any failures<|assistant|>",
        ]
    return prompts


def evaluate_apprentice(model, adapter_weights: dict, domain_dataset, tok, cfg) -> dict:
    """
    Run all metrics for one specialist.

    Loads the adapter into the backbone, computes:
      - Loss on held-out domain data
      - Tool call accuracy on extracted prompts
      - Format adherence (boundary tokens)

    Restores the original LoRA weights after evaluation.
    """
    from mlx.utils import tree_flatten

    orig = dict(tree_flatten(model.trainable_parameters()))

    model.load_lora(adapter_weights)

    loss = compute_loss(model, domain_dataset, tok, cfg, max_batches=20, max_len=512)

    prompts = _extract_eval_prompts(domain_dataset, tok)

    try:
        call_results = evaluate_tool_calls(model, prompts, tok, cfg)
        valid = sum(1 for r in call_results if r.get("valid"))
        tool_acc = valid / max(len(call_results), 1)
    except Exception as e:
        print(f"  [evaluate_apprentice] tool eval skipped: {e}")
        tool_acc = 0.0

    try:
        fmt = format_adherence(model, prompts, tok, cfg)
    except Exception as e:
        print(f"  [evaluate_apprentice] format eval skipped: {e}")
        fmt = {"plan": 0, "scratch": 0, "eos": 0, "total": len(prompts)}

    if orig:
        model.load_lora(orig)

    return {"loss": loss, "tool_acc": tool_acc, "format": fmt}


def test_interference(model, adapters: dict, test_fn, tok, cfg) -> tuple:
    """
    Measure cross-apprentice interference.

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
        model.load_lora(adapter)
        baselines[name] = test_fn(model, tok, cfg)

    interference = {}
    for name_a, adapter_a in adapters.items():
        for name_b, adapter_b in adapters.items():
            if name_a == name_b:
                continue
            model.load_lora(adapter_a)
            score = test_fn(model, tok, cfg)
            diff = score - baselines[name_a]
            interference[f"{name_a}_under_{name_b}"] = diff
            if abs(diff) > 0.05:
                print(f"  ⚠️  SPECIALIST INTERFERENCE DETECTED: "
                      f"{name_a} under {name_b}: {diff:+.4f} "
                      f"(baseline={baselines[name_a]:.4f}, actual={score:.4f})")

    return baselines, interference
