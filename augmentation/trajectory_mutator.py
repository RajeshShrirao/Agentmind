"""
Layers 2, 4, 5 — Tool Order, Retry/Recovery, and Planner Style Mutation.
"""

import random
import json
import re
from copy import deepcopy
from .core import (
    TOOL_NAMES, TOOL_DEFS, parse_tool_calls,
    rebuild_from_segments, apply_positional, Segment,
)

# ── Planner cognitive styles ────────────────────────────────────

PLANNER_STYLES = {
    "cautious": {
        "plan_prefix": ["Let me verify step by step:", "I'll proceed carefully:", "Checking each step:"],
        "tool_modifier": lambda t: t,
        "extra_verify": True,
    },
    "fast_executor": {
        "plan_prefix": ["Running:", "Executing:", "Starting:"],
        "tool_modifier": lambda t: t,
        "extra_verify": False,
    },
    "chain_planner": {
        "plan_prefix": ["Chain:", "Pipeline:", "Sequence:"],
        "tool_modifier": lambda t: t,
        "extra_verify": False,
    },
    "recursive_decomposer": {
        "plan_prefix": ["Breaking down:", "Decomposing:", "Sub-steps:"],
        "tool_modifier": lambda t: t,
        "extra_verify": False,
    },
    "defensive": {
        "plan_prefix": ["Safety check — ", "Validating before:", "Sanity check first:"],
        "tool_modifier": lambda t: t,
        "extra_verify": True,
        "add_validation": True,
    },
    "verbose": {
        "plan_prefix": ["Detailed plan:\n", "Full breakdown:\n", "Step-by-step:\n"],
        "tool_modifier": lambda t: t,
        "extra_verify": False,
    },
    "minimalist": {
        "plan_prefix": ["> ", "→ ", "• "],
        "tool_modifier": lambda t: t,
        "extra_verify": False,
    },
}

SCRATCH_MESSAGES = [
    "That didn't work. Let me try a different approach.",
    "Unexpected result. Retrying with adjusted parameters.",
    "Hmm, that failed. Switching to alternative method.",
    "Error detected. Attempting recovery sequence.",
    "Let me reconsider and try another way.",
    "The tool didn't respond. Fallback strategy engaged.",
    "Partial data received. Trying supplementary query.",
]


class TrajectoryMutator:
    """Mutate tool ordering, inject retries, vary planner styles."""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def mutate_tool_order(self, sample):
        """Layer 2 — Reorder tools in multi-tool trajectories."""
        asst = sample["messages"][-1]["content"]
        structure = self._parse_structure(asst)
        segments = structure["segments"]

        if len(segments) < 3:
            return []  # Only meaningful for 3+ segments

        # Generate alternative orderings that maintain logical consistency
        alternatives = self._generate_orders(segments)
        results = []
        for alt_segments in alternatives[:2]:  # Max 2 variants
            new_text = rebuild_from_segments(asst, alt_segments)
            results.append({
                "domain": "tool_caller",
                "type": sample["type"],
                "messages": [
                    {"role": "user", "content": sample["messages"][0]["content"]},
                    {"role": "assistant", "content": new_text},
                ]
            })
        return results

    def mutate_retry(self, sample):
        """Layer 4 — Inject retry/recovery patterns into clean segments."""
        asst = sample["messages"][-1]["content"]
        if "<|scratch|>" in asst:
            return []  # Already has recovery

        segments = parse_tool_calls(asst)
        if not segments:
            return []

        # Pick a random clean segment to inject failure + retry
        clean_segs = [s for s in segments if not s.is_failure and s.call_data]
        if not clean_segs:
            return []

        target = self.rng.choice(clean_segs)
        fallback_tool = self._pick_fallback(target.call_data["name"])
        scratch_msg = self.rng.choice(SCRATCH_MESSAGES)

        if fallback_tool:
            # Fallback to different tool
            fallback_args = self._generate_fallback_args(fallback_tool, target.call_data)
            fallback_call = json.dumps({"name": fallback_tool, "args": fallback_args})
            retry_text = (
                f"<|tool_call|>{target.call_str}<|observe|>{target.observe_str}"
                f"<|scratch|>{scratch_msg}"
                f"<|tool_call|>{fallback_call}<|observe|>{target.observe_str}"
            )
            new_text = apply_positional(asst, target.start, target.end, retry_text)
        else:
            # Retry same tool with different args
            new_args = self._mutate_args(target.call_data)
            retry_call = json.dumps({"name": target.call_data["name"], "args": new_args})
            retry_text = (
                f"<|tool_call|>{target.call_str}<|observe|>{'{\"error\": \"timeout\", \"retry\": true}'}"
                f"<|scratch|>{scratch_msg}"
                f"<|tool_call|>{retry_call}<|observe|>{target.observe_str}"
            )
            new_text = apply_positional(asst, target.start, target.end, retry_text)

        return [{
            "domain": "tool_caller",
            "type": sample["type"],
            "messages": [
                {"role": "user", "content": sample["messages"][0]["content"]},
                {"role": "assistant", "content": new_text},
            ]
        }]

    def mutate_planner_style(self, sample):
        """Layer 5 — Vary the planning prefix/style."""
        asst = sample["messages"][-1]["content"]
        if "<|plan|>" not in asst:
            return [sample]  # No plan to vary

        style = self.rng.choice(list(PLANNER_STYLES.values()))
        prefix = self.rng.choice(style["plan_prefix"])

        # Replace plan prefix
        plan_match = re.search(r'<\|plan\|>(.*?)(?=<\|tool_call\|>|\Z)', asst, re.DOTALL)
        if not plan_match:
            return [sample]

        old_plan = plan_match.group(1)
        new_plan = f"<|plan|>{prefix} {old_plan.strip()}"
        new_asst = apply_positional(asst, plan_match.start(), plan_match.end(), new_plan)

        # Add extra verification step for cautious/defensive styles
        if style.get("extra_verify") and self.rng.random() < 0.3:
            verify_call = json.dumps({"name": "web_search", "args": {"query": "verify data", "max_results": 3}})
            new_asst += f"<|tool_call|>{verify_call}<|observe|>{'{\"results\": [{\"title\": \"verified\"}]}'}"

        return [{
            "domain": "tool_caller",
            "type": sample["type"],
            "messages": [
                {"role": "user", "content": sample["messages"][0]["content"]},
                {"role": "assistant", "content": new_asst},
            ]
        }]

    # ── Internals ───────────────────────────────────────────────

    def _parse_structure(self, text):
        """Full structure parse including scratch/plan/think regions."""
        segments = parse_tool_calls(text)
        plans = []
        scratches = []
        for m in re.finditer(r'<\|plan\|>(.*?)(?=<\|tool_call\|>|\Z)', text, re.DOTALL):
            plans.append({"start": m.start(), "end": m.end(), "text": m.group(1)})
        for m in re.finditer(r'<\|scratch\|>(.*?)(?=<\|tool_call\|>|<\|plan\|>|$)', text, re.DOTALL):
            scratches.append({"start": m.start(), "end": m.end(), "text": m.group(1)})
        return {"segments": segments, "plans": plans, "scratches": scratches}

    def _generate_orders(self, segments):
        """Generate alternative valid tool orderings."""
        import itertools
        tools = [s for s in segments if s.call_data]
        if len(tools) < 3:
            return [segments]

        # Generate 2-3 alternative permutations
        orders = []
        tried = set()
        for _ in range(5):
            perm = list(tools)
            self.rng.shuffle(perm)
            key = tuple(s.call_data["name"] for s in perm)
            if key not in tried:
                tried.add(key)
                orders.append(perm)
            if len(orders) >= 2:
                break
        return orders if orders else [segments]

    def _pick_fallback(self, tool_name):
        """Pick a semantically related fallback tool."""
        fallback_map = {
            "web_search": ["search_arxiv", None],
            "read_file": ["list_directory", None],
            "get_weather": [None],
            "search_arxiv": ["web_search", None],
            "execute_sql": [None],
            "run_python": [None],
            "send_email": [None],
            "get_stock_price": ["web_search", None],
        }
        options = fallback_map.get(tool_name, [None])
        return self.rng.choice(options)

    def _generate_fallback_args(self, fallback_tool, original_call):
        """Generate args for fallback tool based on original context."""
        if fallback_tool == "web_search":
            return {"query": f"research {original_call.call_data['name']}", "max_results": 5}
        if fallback_tool == "search_arxiv":
            return {"query": original_call.call_data.get("query", "machine learning"), "days": 30}
        if fallback_tool == "list_directory":
            return {"path": "/home/user/project"}
        return {}

    def _mutate_args(self, call_data):
        """Generate slightly different args for retry."""
        args = dict(call_data["args"])
        tool_name = call_data["name"]
        if tool_name == "get_weather" and "city" in args:
            args["city"] = args["city"] + ", alternative source"
        elif tool_name == "web_search":
            args["query"] = f"{args.get('query', 'retry')} — fallback"
            args["max_results"] = args.get("max_results", 5) * 2
        elif tool_name == "execute_sql":
            args["query"] = args.get("query", "").replace("SELECT", "SELECT /* retry */ ")
        elif tool_name == "read_file" and "path" in args:
            args["path"] = "/backup" + args["path"]
        return args
