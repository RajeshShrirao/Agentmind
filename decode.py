"""
Tool call structured decoding and validation.

Separate from the freeform generation path. Always uses greedy
(argmax) decoding for tool calls. Validates against registered
schemas. Logs specific failure modes — not just pass/fail.
"""

import json
import mlx.core as mx
from typing import Any, Optional

# ── Tool Registry with Type Schemas ──────────────────────────

TOOL_REGISTRY = {
    "web_search": {
        "description": "Search the web for current information",
        "params": {
            "query": {"type": "string", "required": True},
        },
    },
    "read_file": {
        "description": "Read a file from disk",
        "params": {
            "path": {"type": "string", "required": True},
        },
    },
    "write_file": {
        "description": "Write content to a file",
        "params": {
            "path": {"type": "string", "required": True},
            "content": {"type": "string", "required": True},
        },
    },
    "run_python": {
        "description": "Execute Python code",
        "params": {
            "code": {"type": "string", "required": True},
        },
    },
    "get_weather": {
        "description": "Get current weather for a city",
        "params": {
            "city": {"type": "string", "required": True},
        },
    },
    "search_arxiv": {
        "description": "Search arxiv for papers",
        "params": {
            "query": {"type": "string", "required": True},
            "days": {"type": "integer", "required": False},
        },
    },
    "fetch_abstract": {
        "description": "Fetch paper abstract by ID",
        "params": {
            "id": {"type": "string", "required": True},
        },
    },
    "execute_sql": {
        "description": "Execute SQL query",
        "params": {
            "query": {"type": "string", "required": True},
        },
    },
    "send_email": {
        "description": "Send an email",
        "params": {
            "to": {"type": "string", "required": True},
            "subject": {"type": "string", "required": True},
            "body": {"type": "string", "required": True},
        },
    },
    "git_commit": {
        "description": "Commit changes to git",
        "params": {
            "message": {"type": "string", "required": True},
        },
    },
    "list_directory": {
        "description": "List files in directory",
        "params": {
            "path": {"type": "string", "required": True},
        },
    },
    "get_stock_price": {
        "description": "Get current stock price",
        "params": {
            "ticker": {"type": "string", "required": True},
        },
    },
    "translate": {
        "description": "Translate text to another language",
        "params": {
            "text": {"type": "string", "required": True},
            "target_lang": {"type": "string", "required": True},
        },
    },
    "summarize": {
        "description": "Summarize long text",
        "params": {
            "text": {"type": "string", "required": True},
        },
    },
}

TYPE_MAP = {"string": str, "integer": int, "number": float, "boolean": bool}


def validate_tool_call(raw: str, registry: Optional[dict] = None) -> dict:
    """
    Validate a raw string as a tool call JSON.
    
    Returns a result dict:
      {"valid": bool, "name": str|None, "args": dict|None, "error": str|None,
       "failure_mode": str|None}

    failure_mode is one of:
      "parse_error"       — JSON does not parse
      "missing_name"      — JSON parsed but no "name" key
      "missing_args"      — JSON parsed but no "args" key
      "unknown_tool"      — name is not in registry
      "missing_param"     — a required arg is absent
      "type_mismatch"     — an arg value has the wrong type
      None                — valid
    """
    if registry is None:
        registry = TOOL_REGISTRY

    raw = raw.strip()

    # Remove trailing structural tokens that may have been captured
    for tok in ["<|observe|>", "<|tool_call|>", "<|end|>", "<eos>"]:
        if tok in raw:
            raw = raw.split(tok)[0]

    raw = raw.strip()

    if not raw:
        return {"valid": False, "name": None, "args": None,
                "error": "empty input", "failure_mode": "parse_error"}

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"valid": False, "name": None, "args": None,
                "error": f"JSON parse error: {e}", "failure_mode": "parse_error"}

    if not isinstance(obj, dict):
        return {"valid": False, "name": None, "args": None,
                "error": "root value is not a JSON object", "failure_mode": "parse_error"}

    if "name" not in obj:
        return {"valid": False, "name": None, "args": None,
                "error": "missing 'name' field", "failure_mode": "missing_name"}

    name = obj["name"]
    if not isinstance(name, str):
        return {"valid": False, "name": str(name), "args": None,
                "error": f"'name' must be a string, got {type(name).__name__}",
                "failure_mode": "missing_name"}

    if name not in registry:
        return {"valid": False, "name": name, "args": None,
                "error": f"unknown tool: '{name}'",
                "failure_mode": "unknown_tool"}

    if "args" not in obj:
        return {"valid": False, "name": name, "args": None,
                "error": "missing 'args' field", "failure_mode": "missing_args"}

    args = obj["args"]
    if not isinstance(args, dict):
        return {"valid": False, "name": name, "args": None,
                "error": f"'args' must be a JSON object, got {type(args).__name__}",
                "failure_mode": "missing_args"}

    schema = registry[name]
    for param_name, param_schema in schema["params"].items():
        if param_schema["required"] and param_name not in args:
            return {"valid": False, "name": name, "args": args,
                    "error": f"missing required arg '{param_name}' for {name}",
                    "failure_mode": "missing_param"}

    for param_name, value in args.items():
        if param_name not in schema["params"]:
            continue
        expected_type = TYPE_MAP.get(schema["params"][param_name]["type"])
        if expected_type and not isinstance(value, expected_type):
            return {"valid": False, "name": name, "args": args,
                    "error": f"arg '{param_name}' expected {schema['params'][param_name]['type']}, got {type(value).__name__}",
                    "failure_mode": "type_mismatch"}

    return {"valid": True, "name": name, "args": args,
            "error": None, "failure_mode": None}


def extract_tool_calls(text: str, registry: Optional[dict] = None) -> list[dict]:
    """
    Extract and validate all tool calls embedded in generated text.

    Parses segments between <|tool_call|> and next boundary token.
    Returns a list of validate_tool_call result dicts.
    """
    results = []
    segments = text.split("<|tool_call|>")
    for seg in segments[1:]:
        raw = seg.strip()
        result = validate_tool_call(raw, registry)
        results.append(result)
    return results


def generate_tool_call(model, ctx: mx.array, h_states: dict, cfg, tok,
                       max_tokens: int = 80) -> dict:
    """
    Greedy-decode a tool call from context. Separate from freeform sampling.
    
    Args:
        model: AgentMind model instance
        ctx: [1, L] token IDs including prompt up to <|assistant|>
        h_states: SSM state dict from prior forward pass
        cfg: AgentMindConfig
        tok: tokenizer
        max_tokens: max tokens to generate for the tool call

    Returns:
        {"raw": str, "valid": bool, "name": str|None, "args": dict|None,
         "error": str|None, "failure_mode": str|None, "tokens": list[int]}
    """
    input_ids = ctx
    past_h = dict(h_states) if h_states else {}
    output_ids = []
    tool_call_token_str = "<|tool_call|>"
    tool_call_id = cfg.tool_call_id
    eos_id = cfg.eos_id
    observe_id = cfg.observe_id

    tool_call_emitted = False

    for _ in range(max_tokens):
        logits, past_h = model.forward_with_state(input_ids, past_h)
        next_tok = mx.argmax(logits[0, -1]).item()
        output_ids.append(next_tok)

        if next_tok == tool_call_id:
            tool_call_emitted = True
        elif tool_call_emitted and next_tok in (eos_id, observe_id):
            break
        elif not tool_call_emitted and next_tok == eos_id:
            break

        input_ids = mx.array([[next_tok]])

    raw = tok.decode(output_ids)

    if not tool_call_emitted:
        return {"raw": raw, "valid": False, "name": None, "args": None,
                "error": "no <|tool_call|> emitted", "failure_mode": "parse_error",
                "tokens": output_ids}

    # Extract JSON after <|tool_call|>
    idx = raw.find(tool_call_token_str)
    if idx == -1:
        return {"raw": raw, "valid": False, "name": None, "args": None,
                "error": "malformed output", "failure_mode": "parse_error",
                "tokens": output_ids}

    json_str = raw[idx + len(tool_call_token_str):]
    result = validate_tool_call(json_str)
    result["raw"] = raw
    result["tokens"] = output_ids
    return result


def tool_eval_report(results: list[dict]) -> dict:
    """
    Aggregate per-call evaluation results into summary metrics.

    Returns:
      {"total": int, "valid": int, "valid_pct": float,
       "breakdown": {failure_mode: count},
       "tool_counts": {tool_name: count}}
    """
    total = len(results)
    valid = sum(1 for r in results if r.get("valid"))
    breakdown: dict[str, int] = {}
    tool_counts: dict[str, int] = {}

    for r in results:
        fm = r.get("failure_mode")
        if fm:
            breakdown[fm] = breakdown.get(fm, 0) + 1
        name = r.get("name")
        if name:
            tool_counts[name] = tool_counts.get(name, 0) + 1

    return {
        "total": total,
        "valid": valid,
        "valid_pct": valid / max(total, 1) * 100,
        "breakdown": dict(sorted(breakdown.items())),
        "tool_counts": dict(sorted(tool_counts.items())),
    }


def print_tool_report(report: dict, label: str = "Tool Eval") -> None:
    """Print a human-readable tool evaluation report."""
    print(f"\n  ── {label} ──")
    print(f"  Total calls: {report['total']}")
    print(f"  Valid:       {report['valid']}/{report['total']} ({report['valid_pct']:.1f}%)")
    if report["breakdown"]:
        print(f"  Failures:")
        for mode, count in report["breakdown"].items():
            pct = count / max(report["total"], 1) * 100
            print(f"    {mode}: {count} ({pct:.1f}%)")
    if report["tool_counts"]:
        print(f"  Tools used:")
        for name, count in report["tool_counts"].items():
            print(f"    {name}: {count}")
