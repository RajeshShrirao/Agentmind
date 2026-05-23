"""
Core utilities: tool registry, parsing, validation, entropy scoring, dedup.
"""

import json
import re
import math
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


# ── Tool Registry ───────────────────────────────────────────────

@dataclass
class ToolDef:
    name: str
    params: dict  # param_name -> type
    required: list  # required param names


TOOL_DEFS = {
    "web_search": ToolDef("web_search", {"query": str, "max_results": int}, ["query"]),
    "read_file": ToolDef("read_file", {"path": str}, ["path"]),
    "write_file": ToolDef("write_file", {"path": str, "content": str}, ["path", "content"]),
    "run_python": ToolDef("run_python", {"code": str}, ["code"]),
    "get_weather": ToolDef("get_weather", {"city": str}, ["city"]),
    "search_arxiv": ToolDef("search_arxiv", {"query": str, "days": int}, ["query"]),
    "fetch_abstract": ToolDef("fetch_abstract", {"id": str}, ["id"]),
    "execute_sql": ToolDef("execute_sql", {"query": str}, ["query"]),
    "send_email": ToolDef("send_email", {"to": str, "subject": str, "body": str}, ["to", "subject", "body"]),
    "git_commit": ToolDef("git_commit", {"message": str}, ["message"]),
    "list_directory": ToolDef("list_directory", {"path": str}, ["path"]),
    "get_stock_price": ToolDef("get_stock_price", {"ticker": str}, ["ticker"]),
    "translate": ToolDef("translate", {"text": str, "target_lang": str}, ["text", "target_lang"]),
    "summarize": ToolDef("summarize", {"text": str}, ["text"]),
}

TOOL_NAMES = set(TOOL_DEFS.keys())

# ── Segment Parsing ─────────────────────────────────────────────

SEGMENT_RE = re.compile(
    r'<\|tool_call\|>(.*?)<\|observe\|>(.*?)'
    r'(?=<\|tool_call\|>|<\|scratch\|>|<\|plan\|>|<\|think_start\|>|\Z)',
    re.DOTALL
)

PLAN_RE = re.compile(r'<\|plan\|>(.*?)(?=<\|tool_call\|>|\Z)', re.DOTALL)
SCRATCH_RE = re.compile(r'<\|scratch\|>(.*?)(?=<\|tool_call\|>|<\|plan\|>|<\|think_start\|>|\Z)', re.DOTALL)
THINK_RE = re.compile(r'<\|think_start\|>(.*?)<\|think_end\|>', re.DOTALL)


@dataclass
class Segment:
    call_str: str
    observe_str: str
    call_data: Optional[dict] = None
    is_failure: bool = False
    start: int = 0
    end: int = 0


def parse_tool_calls(assistant_text):
    """Parse assistant text into ordered list of Segments."""
    segments = []
    for m in SEGMENT_RE.finditer(assistant_text):
        raw_call = m.group(1).strip()
        observe = m.group(2).strip()
        call_data = _try_json(raw_call)
        is_failure = _is_failure_observe(observe)
        segments.append(Segment(
            call_str=raw_call,
            observe_str=observe,
            call_data=call_data,
            is_failure=is_failure,
            start=m.start(),
            end=m.end(),
        ))
    return segments


def parse_all_structure(assistant_text):
    """Parse plans, tool segments, scratch blocks, think blocks."""
    segments = parse_tool_calls(assistant_text)
    plans = [(m.start(), m.end(), m.group(1)) for m in PLAN_RE.finditer(assistant_text)]
    scratches = [(m.start(), m.end(), m.group(1)) for m in SCRATCH_RE.finditer(assistant_text)]
    thinks = [(m.start(), m.end(), m.group(1), m.group(2)) for m in THINK_RE.finditer(assistant_text)]
    return {
        "segments": segments,
        "plans": plans,
        "scratches": scratches,
        "thinks": thinks,
        "has_scratch": bool(scratches),
        "has_plan": bool(plans),
        "has_think": bool(thinks),
    }


def rebuild_from_segments(text, segments):
    """Rebuild assistant text by replacing each segment in reverse order."""
    result = text
    for seg in reversed(segments):
        new_seg = f"<|tool_call|>{seg.call_str}<|observe|>{seg.observe_str}"
        result = result[:seg.start] + new_seg + result[seg.end:]
    return result


def apply_positional(text, start, end, replacement):
    """Replace text[start:end] with replacement."""
    return text[:start] + replacement + text[end:]


def _try_json(s):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


def _is_failure_observe(observe_str):
    data = _try_json(observe_str)
    if isinstance(data, dict):
        if "error" in data or "warning" in data:
            return True
        if data.get("status") in ("partial", "timeout", "error"):
            return True
    if isinstance(data, str) and ("error" in data.lower() or "unexpected" in data.lower()):
        return True
    return False


# ── Validation ──────────────────────────────────────────────────

def validate_sample(sample, strict=True):
    """Return (is_valid, reason) tuple."""
    try:
        if not isinstance(sample, dict):
            return False, "not a dict"
        if sample.get("domain") != "tool_caller":
            return False, "wrong domain"
        if sample.get("type") not in ("tool_single", "tool_multi"):
            return False, f"bad type: {sample.get('type')}"

        msgs = sample.get("messages", [])
        if not isinstance(msgs, list) or len(msgs) < 2:
            return False, "need 2+ messages"
        if msgs[0].get("role") != "user" or msgs[-1].get("role") != "assistant":
            return False, "wrong roles"

        user = msgs[0].get("content", "")
        asst = msgs[-1].get("content", "")
        if not isinstance(user, str) or not isinstance(asst, str):
            return False, "content not string"
        if not user.strip():
            return False, "empty user"
        if not asst.strip():
            return False, "empty assistant"

        if "<|tool_call|>" not in asst or "<|observe|>" not in asst:
            return False, "missing tool_call/observe"

        segments = parse_tool_calls(asst)
        if not segments:
            return False, "no valid segments"

        for seg in segments:
            if seg.call_data is None:
                continue  # malformed JSON mode is OK
            if "name" not in seg.call_data or "args" not in seg.call_data:
                return False, f"missing name/args in: {seg.call_str[:60]}"
            if seg.call_data["name"] not in TOOL_NAMES:
                return False, f"unknown tool: {seg.call_data['name']}"
            args = seg.call_data["args"]
            if not isinstance(args, dict):
                return False, "args not dict"
            if strict:
                tool = TOOL_DEFS[seg.call_data["name"]]
                for req in tool.required:
                    if req not in args:
                        return False, f"missing required param '{req}' for {tool.name}"

        call_count = len([s for s in segments if not s.is_failure])
        if call_count == 0:
            return False, "all calls are failures (no success)"

        return True, "OK"
    except Exception as e:
        return False, f"validation exception: {e}"


# ── Entropy Scoring ─────────────────────────────────────────────

class EntropyScorer:
    """Score sample diversity via n-gram entropy, tool variety, structure."""

    def __init__(self, ngram_n=3):
        self.ngram_n = ngram_n

    def score(self, sample):
        msgs = sample["messages"]
        user = msgs[0]["content"]
        asst = msgs[-1]["content"]

        # Token entropy (character-level trigrams)
        char_ngrams = [user[i:i+self.ngram_n] for i in range(len(user)-self.ngram_n+1)]
        freq = Counter(char_ngrams)
        total = sum(freq.values())
        char_entropy = -sum((c/total) * math.log2(c/total) for c in freq.values()) if total > 0 else 0

        # Tool diversity
        segments = parse_tool_calls(asst)
        tools_used = set()
        for seg in segments:
            if seg.call_data:
                tools_used.add(seg.call_data["name"])
        tool_diversity = len(tools_used) / max(len(TOOL_NAMES), 1)

        # Structure complexity bonus
        structure_bonus = 0
        if "<|plan|>" in asst:
            structure_bonus += 0.2
        if "<|scratch|>" in asst:
            structure_bonus += 0.15
        if "<|think_start|>" in asst:
            structure_bonus += 0.1
        if len(segments) > 3:
            structure_bonus += 0.1 * min(len(segments) / 5, 1)

        # Query length diversity (avoid degenerate short queries)
        length_factor = min(len(user) / 30, 1.0) * 0.1

        return char_entropy * 0.4 + tool_diversity * 0.3 + structure_bonus * 0.2 + length_factor * 0.1

    def filter(self, samples, threshold=2.5):
        """Filter out low-entropy samples."""
        scored = [(self.score(s), s) for s in samples]
        kept = [s for score, s in scored if score >= threshold]
        return kept, [s for score, s in scored if score < threshold]


class DuplicateDetector:
    """Semantic dedup via normalized content hash."""

    def __init__(self):
        self.seen = set()

    def _normalize(self, text):
        return re.sub(r'\s+', ' ', text.strip()).lower()

    def _content_hash(self, sample):
        user = self._normalize(sample["messages"][0]["content"])
        asst = self._normalize(sample["messages"][-1]["content"])
        return hashlib.md5((user + asst).encode()).hexdigest()

    def is_duplicate(self, sample):
        h = self._content_hash(sample)
        if h in self.seen:
            return True
        self.seen.add(h)
        return False

    def filter(self, samples):
        unique = []
        for s in samples:
            if not self.is_duplicate(s):
                unique.append(s)
        return unique
