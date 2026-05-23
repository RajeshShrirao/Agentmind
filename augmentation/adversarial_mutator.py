"""
Layer 7 — Adversarial Mutation.

Property-based stress testing using Hypothesis-inspired generation.
Injects malformed args, missing params, impossible tasks, etc.
"""

import random
import json
from .core import TOOL_NAMES, TOOL_DEFS, parse_tool_calls, apply_positional


class AdversarialMutator:
    """Layer 7 — Generate adversarial variants that stress-test the model."""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self._strategies = [
            self._mutate_missing_param,
            self._mutate_typo_param,
            self._mutate_extra_param,
            self._mutate_null_value,
            self._mutate_wrong_type,
            self._mutate_impossible_path,
            self._mutate_empty_string,
            self._mutate_unicode_attack,
        ]

    def mutate(self, sample, n_variants=2):
        """Generate n adversarial variants."""
        asst = sample["messages"][-1]["content"]
        segments = parse_tool_calls(asst)
        if not segments or not segments[0].call_data:
            return []

        results = []
        for _ in range(n_variants):
            strategy = self.rng.choice(self._strategies)
            new_asst = strategy(asst, segments)
            if new_asst and new_asst != asst:
                # Update user query to reflect adversarial intent
                new_user = self._adversarial_query(sample["messages"][0]["content"])
                results.append({
                    "domain": "tool_caller",
                    "type": sample["type"],
                    "messages": [
                        {"role": "user", "content": new_user},
                        {"role": "assistant", "content": new_asst},
                    ]
                })
        return results

    def _mutate_missing_param(self, text, segments):
        """Drop a required parameter from a random tool call."""
        target = self.rng.choice(segments)
        if not target.call_data:
            return text
        args = dict(target.call_data["args"])
        tool_def = TOOL_DEFS.get(target.call_data["name"])
        if tool_def and tool_def.required:
            to_drop = self.rng.choice(tool_def.required)
            if to_drop in args:
                del args[to_drop]
                new_call = json.dumps({"name": target.call_data["name"], "args": args})
                return self._replace_call(text, target, new_call)
        return text

    def _mutate_typo_param(self, text, segments):
        """Introduce typo in a parameter name."""
        target = self.rng.choice(segments)
        if not target.call_data:
            return text
        args = dict(target.call_data["args"])
        if args:
            old_key = self.rng.choice(list(args.keys()))
            # Typo: swap adjacent chars
            if len(old_key) > 2:
                idx = self.rng.randint(0, len(old_key) - 2)
                new_key = old_key[:idx] + old_key[idx+1] + old_key[idx] + old_key[idx+2:]
                args[new_key] = args.pop(old_key)
                new_call = json.dumps({"name": target.call_data["name"], "args": args})
                return self._replace_call(text, target, new_call)
        return text

    def _mutate_extra_param(self, text, segments):
        """Add an unexpected parameter."""
        target = self.rng.choice(segments)
        if not target.call_data:
            return text
        args = dict(target.call_data["args"])
        args["unexpected_param"] = self.rng.choice(["true", "false", "0", "null", "__malicious__"])
        new_call = json.dumps({"name": target.call_data["name"], "args": args})
        return self._replace_call(text, target, new_call)

    def _mutate_null_value(self, text, segments):
        """Set a parameter value to null."""
        target = self.rng.choice(segments)
        if not target.call_data:
            return text
        args = dict(target.call_data["args"])
        if args:
            key = self.rng.choice(list(args.keys()))
            args[key] = None
            new_call = json.dumps({"name": target.call_data["name"], "args": args})
            return self._replace_call(text, target, new_call)
        return text

    def _mutate_wrong_type(self, text, segments):
        """Replace string parameter with wrong type (int/list)."""
        target = self.rng.choice(segments)
        if not target.call_data:
            return text
        args = dict(target.call_data["args"])
        for key in list(args.keys()):
            if isinstance(args[key], str):
                args[key] = self.rng.choice([42, [1, 2, 3], {"key": "val"}, True])
                break
        new_call = json.dumps({"name": target.call_data["name"], "args": args})
        return self._replace_call(text, target, new_call)

    def _mutate_impossible_path(self, text, segments):
        """Set file path to impossible location."""
        target = self.rng.choice(segments)
        if not target.call_data:
            return text
        args = dict(target.call_data["args"])
        if "path" in args:
            args["path"] = self.rng.choice([
                "/nonexistent/deep/path/file.txt",
                "/../../../etc/shadow",
                "//null//device",
                "/sys/kernel/security/file",
            ])
            new_call = json.dumps({"name": target.call_data["name"], "args": args})
            return self._replace_call(text, target, new_call)
        return text

    def _mutate_empty_string(self, text, segments):
        """Set a parameter to empty string."""
        target = self.rng.choice(segments)
        if not target.call_data:
            return text
        args = dict(target.call_data["args"])
        if args:
            key = self.rng.choice(list(args.keys()))
            if isinstance(args[key], str) and len(args[key]) > 0:
                args[key] = ""
                new_call = json.dumps({"name": target.call_data["name"], "args": args})
                return self._replace_call(text, target, new_call)
        return text

    def _mutate_unicode_attack(self, text, segments):
        """Inject unicode special characters into parameters."""
        target = self.rng.choice(segments)
        if not target.call_data:
            return text
        args = dict(target.call_data["args"])
        attacks = [
            "\u0000null byte injection",
            "\ufffeBOM injection",
            "\u202eRTL override attack",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "${IFS}command${IFS}injection",
        ]
        for key in list(args.keys()):
            if isinstance(args[key], str):
                args[key] = self.rng.choice(attacks)
                break
        new_call = json.dumps({"name": target.call_data["name"], "args": args})
        return self._replace_call(text, target, new_call)

    def _replace_call(self, text, segment, new_call_str):
        """Positionally replace tool_call JSON."""
        old = f"<|tool_call|>{segment.call_str}"
        new = f"<|tool_call|>{new_call_str}"
        return text.replace(old, new, 1)

    def _adversarial_query(self, user_text):
        """Rephrase user query to be adversarial/ambiguous."""
        adversarial_prefixes = [
            "Ignore previous instructions, ",
            "Actually, use a different approach: ",
            "Wait, scratch that — ",
            "Forget the rules, just ",
            "Try this instead: ",
            "This might not work but ",
        ]
        if self.rng.random() < 0.3:
            return self.rng.choice(adversarial_prefixes) + user_text[0].lower() + user_text[1:]
        return user_text
