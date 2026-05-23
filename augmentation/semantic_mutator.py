"""
Layer 1 — Semantic Query Mutation.

Uses nlpaug for paraphrasing + rule-based mutations as fallback.
"""

import random
import re
from .core import apply_positional, parse_tool_calls


class SemanticMutator:
    """
    Paraphrase, compress, expand, noise-inject user queries.
    Falls back to rule-based transforms when nlpaug models unavailable.
    """

    def __init__(self, seed=42):
        self.rng = random.Random(seed)

        # Try to load nlpaug
        self.aug = None
        self.aug_char = None
        try:
            import nlpaug.augmenter.word as naw
            import nlpaug.augmenter.char as nac
            self.aug = naw.SynonymAug(aug_src='wordnet', aug_max=2)
            self.aug_char = nac.RandomCharAug(action="substitute", aug_char_p=0.05)
        except Exception:
            pass

    def mutate(self, sample, n_variants=3):
        """Generate up to n_variants query-mutated samples."""
        msgs = sample["messages"]
        user = msgs[0]["content"]
        asst = msgs[-1]["content"]

        results = []
        for _ in range(n_variants):
            strategy = self.rng.choice([
                "paraphrase", "compress", "expand",
                "typo", "indirect", "split",
            ])
            new_user = self._apply_strategy(user, strategy)
            if new_user and new_user != user:
                results.append({
                    "domain": "tool_caller",
                    "type": sample["type"],
                    "messages": [
                        {"role": "user", "content": new_user},
                        {"role": "assistant", "content": asst},
                    ]
                })
        return results

    def _apply_strategy(self, text, strategy):
        try:
            if strategy == "paraphrase" and self.aug:
                return self.aug.augment(text)
            elif strategy == "typo" and self.aug_char:
                return self.aug_char.augment(text)
            elif strategy == "compress":
                return self._compress(text)
            elif strategy == "expand":
                return self._expand(text)
            elif strategy == "indirect":
                return self._indirect(text)
            elif strategy == "split":
                return self._split_merge(text)
        except Exception:
            pass
        return text

    def _compress(self, text):
        """Shorten query — remove fillers, condense."""
        removals = [
            r'\b(could you|can you|would you|will you)\b', '',
            r'\b(please|kindly)\b', '',
            r'\b(I need|I want|I would like)\b', '',
            r'\b(can I get|can I have)\b', '',
            r'\b(a little|a bit|some)\b', '',
        ]
        result = text
        for pat, repl in removals:
            result = re.sub(pat, repl, result, flags=re.IGNORECASE)
        result = re.sub(r'\s+', ' ', result).strip()
        return result if len(result) > 5 else text

    def _expand(self, text):
        """Make query more conversational by adding polite framing."""
        prefixes = [
            "Could you please ", "Hey, can you ", "I need help — ",
            "Would you mind ", "Quick request: ",
        ]
        if not any(text.lower().startswith(p.lower().rstrip()) for p in prefixes):
            prefix = self.rng.choice(prefixes)
            return prefix + text[0].lower() + text[1:]
        return text

    def _indirect(self, text):
        """Rephrase as indirect/ambiguous query."""
        indirect_patterns = [
            lambda t: f"I'm wondering about {t.lower().strip('?')}",
            lambda t: f"What can you tell me regarding {t.lower().strip('?')}",
            lambda t: f"Need info — {t.lower().strip('?')}",
            lambda t: f"Quick question: {t[0].lower()}{t[1:]}",
        ]
        if self.rng.random() < 0.5:
            fn = self.rng.choice(indirect_patterns)
            return fn(text)
        return text

    def _split_merge(self, text):
        """Merge two adjacent sentences or split long query."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) >= 2 and self.rng.random() < 0.5:
            # Merge with conjunction
            idx = self.rng.randint(0, len(sentences) - 2)
            conj = self.rng.choice([" and ", " also ", " plus "])
            sentences[idx] = sentences[idx].rstrip(".!?")
            sentences[idx+1] = sentences[idx+1][0].lower() + sentences[idx+1][1:]
            merged = sentences[idx] + conj + sentences[idx+1]
            sentences = sentences[:idx] + [merged] + sentences[idx+2:]
            return " ".join(sentences)
        return text
