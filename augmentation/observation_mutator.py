"""
Layer 3 — Observation Mutation.

Generate realistic environment outputs: stale cache, timeouts, partial,
malformed, conflicting, permission denied, rate limits, truncated, empty.
"""

import random
import json
from .core import parse_tool_calls, apply_positional, _try_json

OBSERVATION_MODES = {
    "clean": lambda tool, args: _gen_clean(tool, args),
    "stale_cache": lambda tool, args: {"warning": "stale cache", "data": _gen_clean(tool, args), "cached_at": "2024-01-01T00:00:00Z"},
    "timeout": lambda tool, args: {"error": "timeout", "retry": True, "message": "Service did not respond within 30s"},
    "partial": lambda tool, args: {"status": "partial", "data": _gen_partial(tool, args), "message": "Only partial results available"},
    "malformed": lambda tool, args: "unexpected response from <" + tool + "> unable to parse",
    "conflicting": lambda tool, args: {"warning": "Conflicting data from sources", "retry": True, "message": "Source A and Source B disagree"},
    "empty": lambda tool, args: {} if random.random() > 0.5 else {"results": [], "message": "No results found"},
    "permission_denied": lambda tool, args: {"error": "permission_denied", "resource": args.get("path", args.get("ticker", "unknown")), "message": "Access denied"},
    "rate_limit": lambda tool, args: {"error": "rate_limit", "retry_after": random.randint(5, 60), "message": "Too many requests"},
    "truncated": lambda tool, args: {"warning": "response truncated", "data": _gen_clean(tool, args), "truncated": True, "estimated_total": random.randint(100, 9999)},
    "corrupt": lambda tool, args: {"error": "corrupt_response", "message": "Response failed integrity check"},
    "empty_results": lambda tool, args: {"results": [], "total": 0, "message": "Query returned no results"},
}


def _gen_clean(tool, args):
    """Generate a clean observation for a given tool."""
    if tool == "web_search":
        return {"results": [{"title": f"Result about {args.get('query', 'topic')}", "url": "https://example.com"}]}
    elif tool == "read_file":
        return {"content": "# File content\nprint('hello world')"}
    elif tool == "write_file":
        return {"success": True}
    elif tool == "run_python":
        return {"stdout": "42\n", "stderr": ""}
    elif tool == "get_weather":
        return {"temp": random.randint(10, 35), "condition": random.choice(["sunny", "cloudy", "rainy"])}
    elif tool == "search_arxiv":
        return {"results": [{"id": "2405.12345", "title": f"Paper on {args.get('query', 'ML')}"}]}
    elif tool == "fetch_abstract":
        return {"abstract": "We present a novel approach to sequence modeling using structured state space architectures that achieve linear complexity while maintaining quality."}
    elif tool == "execute_sql":
        return {"rows": [{"count": random.randint(10, 50000)}]}
    elif tool == "send_email":
        return {"success": True, "message_id": f"msg_{random.randint(100000, 999999)}"}
    elif tool == "git_commit":
        return {"success": True, "commit_hash": hex(random.randint(0, 16**7))[2:]}
    elif tool == "list_directory":
        return {"files": random.sample(["main.py", "utils.py", "config.py", "README.md", "tests", "data.csv"], random.randint(2, 5))}
    elif tool == "get_stock_price":
        return {"price": round(random.uniform(50, 1000), 2), "change": f"{random.choice(['+', '-'])}{random.uniform(0.1, 5.0):.1f}%"}
    elif tool == "translate":
        return {"translated": "Hola mundo"}
    elif tool == "summarize":
        return {"summary": "Key points: neural networks scale with data, data quality matters more than quantity."}
    return {}


def _gen_partial(tool, args):
    """Generate partial/truncated data."""
    clean = _gen_clean(tool, args)
    if isinstance(clean, dict):
        # Return only half the keys
        keys = list(clean.keys())
        partial_keys = keys[:max(1, len(keys)//2)]
        return {k: clean[k] for k in partial_keys}
    return clean


class ObservationMutator:
    """Layer 3 — Mutate observations with realistic environment outputs."""

    def __init__(self, seed=42, failure_rate=0.35):
        self.rng = random.Random(seed)
        self.failure_rate = failure_rate
        self.modes = list(OBSERVATION_MODES.keys())

    def mutate(self, sample, n_variants=2):
        """Generate variants with different observation outcomes."""
        asst = sample["messages"][-1]["content"]
        segments = parse_tool_calls(asst)
        if not segments:
            return []

        results = []
        for _ in range(n_variants):
            new_asst = asst
            # Mutate each clean segment independently
            for seg in reversed(segments):
                if not seg.call_data:
                    continue
                tool = seg.call_data["name"]
                args = seg.call_data["args"]

                # Choose mode — bias toward failure for realism
                mode = self._pick_mode(seg.is_failure)
                obs_result = OBSERVATION_MODES[mode](tool, args)
                obs_str = json.dumps(obs_result) if isinstance(obs_result, dict) else json.dumps(obs_result)

                new_seg_text = f"<|tool_call|>{seg.call_str}<|observe|>{obs_str}"
                new_asst = apply_positional(new_asst, seg.start, seg.end, new_seg_text)

            results.append({
                "domain": "tool_caller",
                "type": sample["type"],
                "messages": [
                    {"role": "user", "content": sample["messages"][0]["content"]},
                    {"role": "assistant", "content": new_asst},
                ]
            })
        return results

    def _pick_mode(self, currently_failure):
        if currently_failure:
            # Keep failures as failures but vary the mode
            return self.rng.choice(["timeout", "partial", "permission_denied", "rate_limit", "corrupt", "conflicting"])
        # For clean segments, occasionally inject failure
        if self.rng.random() < self.failure_rate:
            return self.rng.choice(self.modes)
        return "clean"
