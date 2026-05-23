"""
Augment tool_caller seeds → 26K training samples via programmatic variation.

Usage:
  python augment_tool_caller_data.py
  python augment_tool_caller_data.py --seeds data/apprentice_tool_caller.jsonl --multiplier 26 --out data/apprentice_tool_caller.jsonl

Strategy:
  For each seed, parse tool calls, then generate variants by:
    1. Swapping entities (cities, tickers, emails, paths, etc.)
    2. Re-rolling tool args via string substitution (preserves structure)
    3. Injecting adversarial failures (30% rate) on clean seeds
    4. Updating user query via entity substitution
"""

import json
import os
import random
import re
import argparse

random.seed(42)
os.makedirs("data", exist_ok=True)

# ── Entity pools ────────────────────────────────────────────────
CITIES = [
    "Tokyo", "London", "Paris", "Berlin", "Sydney", "Mumbai", "Seoul",
    "Singapore", "Dubai", "Barcelona", "Amsterdam", "Toronto", "Chicago",
    "San Francisco", "Los Angeles", "Boston", "Shanghai", "Hong Kong",
    "Melbourne", "Vancouver", "Rome", "Madrid", "Lisbon", "Stockholm",
]

TICKERS = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "META", "TSLA", "AMD",
    "INTC", "CRM", "ORCL", "IBM", "NFLX", "ADBE", "PYPL", "UBER",
    "SPOT", "SNAP", "SQ", "PANW",
]

EMAIL_LOCAL = ["alice", "bob", "carol", "dave", "eve", "frank", "grace",
               "hello", "support", "team", "admin", "info", "contact", "hi"]
EMAIL_DOMAIN = ["company.com", "startup.io", "example.org", "corp.net",
                "business.co", "lab.edu", "service.io", "mail.com"]

FILE_PATHS = [
    "/src/main.py", "/project/config.json", "/home/user/README.md",
    "/var/log/app.log", "/opt/data/results.csv", "/src/utils/helpers.py",
    "/docs/api.md", "/tests/test_unit.py", "/scripts/deploy.sh",
    "/data/input.csv", "/home/user/report.txt", "/config/settings.yaml",
    "/project/requirements.txt", "/src/models/model.py",
]

DIRECTORIES = [
    "/src", "/project", "/home", "/opt/data", "/var/log", "/config",
    "/docs", "/tests", "/scripts", "/data", "/home/user/project",
    "/var/www", "/etc/app",
]

GIT_MESSAGES = [
    "fix: resolve memory leak in data pipeline",
    "feat: add caching layer for API responses",
    "refactor: clean up import statements",
    "docs: update README with setup instructions",
    "test: add unit tests for MambaBlock",
    "perf: optimize SSM scan",
    "fix: correct off-by-one error in boundary check",
    "feat: add new visualization module",
    "chore: update dependencies",
    "style: format code according to PEP8",
]

SQL_QUERIES = [
    "SELECT COUNT(*) FROM users",
    "SELECT * FROM orders WHERE status = 'pending'",
    "SELECT AVG(price) FROM products WHERE category = 'electronics'",
    "SELECT name, email FROM customers WHERE signup_date > '2024-01-01'",
    "SELECT department, COUNT(*) FROM employees GROUP BY department",
    "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name",
    "SELECT * FROM logs WHERE timestamp > NOW() - INTERVAL 7 DAY",
    "SELECT product, SUM(quantity) FROM sales GROUP BY product ORDER BY SUM(quantity) DESC",
    "SELECT * FROM inventory WHERE stock < 10",
]

CODE_SNIPPETS = [
    "import numpy as np; print(np.mean([1,2,3,4,5]))",
    "print(sum(x*x for x in range(10)))",
    "import json; data = {'key': 'value'}; print(json.dumps(data, indent=2))",
    "from pathlib import Path; print(Path('.').absolute())",
    "import pandas as pd; df = pd.DataFrame({'a': [1,2,3]}); print(df.mean())",
    "import math; print(math.factorial(10))",
    "print(' '.join([str(i) for i in range(1, 11)]))",
]

ARXIV_IDS = [
    "2405.12345", "2406.23456", "2407.34567", "2408.45678", "2409.56789",
    "2310.12345", "2310.67890", "2301.12345", "2305.54321", "2303.98765",
]

ARXIV_QUERIES = [
    "state space models", "multi-agent RL", "LoRA fine-tuning",
    "retrieval augmented generation", "tool-use LLMs", "reasoning in language models",
    "transformer architecture", "Mamba SSM", "diffusion models", "RLHF",
]

WEB_QUERIES = [
    "latest AI developments 2025", "machine learning best practices",
    "Python performance optimization", "distributed systems design patterns",
    "cloud computing trends", "cybersecurity news", "data engineering tools",
    "natural language processing advances", "computer vision breakthroughs",
    "LLM evaluation benchmarks",
]

TEXTS = [
    "Neural networks and their applications in NLP. Transformers revolutionized with self-attention. Pre-training on large corpora followed by fine-tuning became the dominant paradigm. Models grew from 100M to over 1T parameters.",
    "Multi-agent systems show 30% improvement on complex tasks over single-agent baselines. Agents specialize in subtasks and communicate via protocols. Coordination mechanisms include shared memory and message passing.",
    "LoRA achieves 90% of full fine-tuning performance at 1% of parameter cost. By freezing pretrained weights and injecting rank decomposition matrices, LoRA enables efficient task adaptation.",
]

SUMMARIES = [
    "Neural networks excel at pattern recognition. Scale improves performance. Data quality matters more than quantity.",
    "Multi-agent approaches show 30% improvement over single-agent baselines on complex tasks.",
    "LoRA achieves 90% of full fine-tuning performance at 1% of the parameter cost.",
]

TRANSLATION_TEXTS = [
    ("Hello world, how are you today?", "es", "Hola mundo, ¿cómo estás hoy?"),
    ("Thank you for your help with this project", "fr", "Merci pour votre aide sur ce projet"),
    ("What is the weather like in your city?", "de", "Wie ist das Wetter in deiner Stadt?"),
    ("Please send me the report by Friday", "ja", "金曜日までにレポートを送ってください"),
    ("Good morning, nice to meet you", "zh", "早上好，很高兴认识你"),
    ("Can you help me with this task?", "hi", "क्या आप इस कार्य में मेरी मदद कर सकते हैं?"),
]

# ── Failure modes ───────────────────────────────────────────────
FAILURE_MODES = {
    "timeout": {"error": "timeout", "retry": True, "message": "Service did not respond within 30s"},
    "partial_success": {"status": "partial", "data": None, "message": "Only partial results available"},
    "malformed_json": "unexpected response from <<tool>> unable to parse",
    "contradictory": {"warning": "Conflicting data from sources", "retry": True, "message": "Source A and Source B disagree"},
    "hidden_variable": {"error": "missing_context", "message": "Request requires prior result not found"},
}

SCRATCH_MESSAGES = [
    "The tool didn't respond. Let me retry with different parameters.",
    "Got an unexpected response. Trying again with a different approach.",
    "Something went wrong. Let me try an alternative method.",
    "Error detected. Attempting recovery with a more specific query.",
    "This doesn't look right. Let me double-check and retry.",
    "Hmm, that didn't work. Let me try a different way.",
]


# ── Helpers ─────────────────────────────────────────────────────

def try_load_json(s):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


def generate_args_for_tool(tool_name):
    if tool_name == "web_search":
        return {"query": random.choice(WEB_QUERIES), "max_results": random.choice([5, 10, 20])}
    elif tool_name == "read_file":
        return {"path": random.choice(FILE_PATHS)}
    elif tool_name == "write_file":
        return {"path": random.choice(FILE_PATHS), "content": random.choice([
            "# Report\nKey findings...", "import mlx\nprint('hello')",
            "{\"setting\": \"value\", \"enabled\": true}", "print('Hello World')",
        ])}
    elif tool_name == "run_python":
        return {"code": random.choice(CODE_SNIPPETS)}
    elif tool_name == "get_weather":
        return {"city": random.choice(CITIES)}
    elif tool_name == "search_arxiv":
        return {"query": random.choice(ARXIV_QUERIES), "days": random.choice([7, 14, 30, 90])}
    elif tool_name == "fetch_abstract":
        return {"id": random.choice(ARXIV_IDS)}
    elif tool_name == "execute_sql":
        return {"query": random.choice(SQL_QUERIES)}
    elif tool_name == "send_email":
        local = random.choice(EMAIL_LOCAL)
        domain = random.choice(EMAIL_DOMAIN)
        return {
            "to": f"{local}@{domain}",
            "subject": random.choice(["Update", "Report", "Summary", "Findings", "Alert", "Status"]),
            "body": random.choice(["Progress report attached.", "Analysis complete.",
                                    "Results ready for review.", "Please find the details below."]),
        }
    elif tool_name == "git_commit":
        return {"message": random.choice(GIT_MESSAGES)}
    elif tool_name == "list_directory":
        return {"path": random.choice(DIRECTORIES)}
    elif tool_name == "get_stock_price":
        return {"ticker": random.choice(TICKERS)}
    elif tool_name == "translate":
        text, lang, _ = random.choice(TRANSLATION_TEXTS)
        return {"text": text, "target_lang": lang}
    elif tool_name == "summarize":
        return {"text": random.choice(TEXTS)}
    return {}


def generate_observe_for_tool(tool_name, args):
    if tool_name == "web_search":
        return {"results": [{"title": random.choice(WEB_QUERIES), "url": "https://example.com"}]}
    elif tool_name == "read_file":
        return {"content": random.choice(CODE_SNIPPETS)}
    elif tool_name == "write_file":
        return {"success": True}
    elif tool_name == "run_python":
        return {"stdout": random.choice(["42\n", "285\n", "[1, 2, 3]\n"]), "stderr": ""}
    elif tool_name == "get_weather":
        return {"temp": random.randint(10, 35), "condition": random.choice(["sunny", "cloudy", "rainy", "windy"])}
    elif tool_name == "search_arxiv":
        return {"results": [{"id": random.choice(ARXIV_IDS), "title": random.choice(ARXIV_QUERIES)}]}
    elif tool_name == "fetch_abstract":
        return {"abstract": "This paper presents a novel approach to the problem of efficient sequence modeling using structured state space architectures."}
    elif tool_name == "execute_sql":
        return {"rows": [{"COUNT(*)": random.randint(100, 50000)}]}
    elif tool_name == "send_email":
        return {"success": True, "message_id": f"msg_{random.randint(100000, 999999)}"}
    elif tool_name == "git_commit":
        return {"success": True, "commit_hash": ''.join(random.choice('abcdef0123456789') for _ in range(7))}
    elif tool_name == "list_directory":
        files = random.sample(["main.py", "utils.py", "config.py", "README.md", "tests",
                                "Dockerfile", "data.csv", "models.py"], random.randint(3, 6))
        return {"files": files}
    elif tool_name == "get_stock_price":
        return {"price": round(random.uniform(50, 1000), 2), "change": f"{random.choice(['+', '-'])}{random.uniform(0.1, 5.0):.1f}%"}
    elif tool_name == "translate":
        _, _, result = random.choice(TRANSLATION_TEXTS)
        return {"translated": result}
    elif tool_name == "summarize":
        return {"summary": random.choice(SUMMARIES)}
    return {}


# ── Augmentation ───────────────────────────────────────────────

def augment_sample(sample, adversarial_rate=0.3):
    """
    Generate one variant from a seed sample.
    Preserves original structure (scratch, plan, think boundaries).
    Only swaps tool args and optionally injects adversarial failure on clean seeds.
    """
    msgs = sample["messages"]
    user_text = msgs[0]["content"]
    assistant_text = msgs[-1]["content"]

    # Find all tool_call+observe segments
    # Capture everything between <|tool_call|> and <|observe|> as call_str
    # (not regex-bounded by braces, since tool args may contain nested JSON)
    segment_pattern = re.compile(
        r'<\|tool_call\|>(.*?)<\|observe\|>(.*?)(?=<\|tool_call\|>|<\|scratch\|>|<\|plan\|>|<\|think_start\|>|\Z)',
        re.DOTALL
    )
    segments = list(segment_pattern.finditer(assistant_text))
    
    if not segments:
        return None

    query_replacements = {}
    new_content = assistant_text
    
    # Process segments in reverse so position offsets don't change
    for seg in reversed(segments):
        full_match = seg.group(0)
        call_str = seg.group(1).strip()
        observe_str = seg.group(2)
        
        call_data = try_load_json(call_str)
        if not call_data or "name" not in call_data or "args" not in call_data:
            continue
        
        tool_name = call_data["name"]
        old_args = call_data["args"]
        new_args = generate_args_for_tool(tool_name)
        new_call_str = json.dumps({"name": tool_name, "args": new_args})
        
        # Track entity replacements for user query
        for key, old_val in old_args.items():
            if isinstance(old_val, str) and len(old_val) > 2 and key in new_args:
                new_val = new_args[key]
                if isinstance(new_val, str) and old_val.lower() != new_val.lower():
                    query_replacements[old_val] = new_val

        # Check if observe is a failure (preserve it)
        obs_data = try_load_json(observe_str)
        is_failure = bool(isinstance(obs_data, dict) and ("error" in obs_data or "warning" in obs_data)) or isinstance(obs_data, str)

        if is_failure:
            # Keep failure observe, just update the call JSON
            new_segment = full_match.replace(call_str, new_call_str, 1)
        else:
            # Update both call and observe
            new_observe = generate_observe_for_tool(tool_name, new_args)
            new_observe_str = json.dumps(new_observe)
            new_segment = f"<|tool_call|>{new_call_str}<|observe|>{new_observe_str}"

        new_content = new_content[:seg.start()] + new_segment + new_content[seg.end():]

    # Optionally inject adversarial failure on FIRST tool call of clean seeds
    has_scratch = "<|scratch|>" in assistant_text
    inject_adversarial = random.random() < adversarial_rate and not has_scratch
    
    if inject_adversarial:
        # Positional replacement of first segment
        first_seg = segment_pattern.search(new_content)
        if first_seg:
            first_call = first_seg.group(1).strip()
            first_observe = first_seg.group(2)
            first_call_data = try_load_json(first_call)

            if first_call_data:
                failure_mode = random.choice(list(FAILURE_MODES.keys()))
                failure_result = FAILURE_MODES[failure_mode]
                failure_json = json.dumps(failure_result) if isinstance(failure_result, dict) else json.dumps(failure_result)
                scratch_msg = random.choice(SCRATCH_MESSAGES)
                retry_call = json.dumps({"name": first_call_data["name"], "args": first_call_data["args"]})

                old_seg = first_seg.group(0)
                new_seg = (
                    f"<|tool_call|>{first_call}<|observe|>{failure_json}"
                    f"<|scratch|>{scratch_msg}"
                    f"<|tool_call|>{retry_call}<|observe|>{first_observe}"
                )
                new_content = new_content[:first_seg.start()] + new_seg + new_content[first_seg.end():]

    # Update user query with entity replacements
    new_user = user_text
    for old_val, new_val in query_replacements.items():
        if old_val in new_user:
            new_user = new_user.replace(old_val, new_val, 1)

    return {
        "domain": "tool_caller",
        "type": sample["type"],
        "messages": [
            {"role": "user", "content": new_user},
            {"role": "assistant", "content": new_content},
        ]
    }


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Augment tool_caller training data")
    parser.add_argument("--seeds", default="data/apprentice_tool_caller.jsonl",
                        help="Input seed JSONL file")
    parser.add_argument("--out", default="data/apprentice_tool_caller.jsonl",
                        help="Output augmented JSONL file")
    parser.add_argument("--multiplier", type=int, default=26,
                        help="Variants per seed (default: 26 → ~26K from 1K)")
    parser.add_argument("--adversarial-rate", type=float, default=0.30,
                        help="Probability of adversarial injection on clean seeds")
    args = parser.parse_args()

    # Load seeds
    seeds = []
    with open(args.seeds) as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    print(f"Loaded {len(seeds)} seed samples")

    # Augment
    augmented = []
    skipped = 0
    for i, seed in enumerate(seeds):
        if (i + 1) % 200 == 0:
            print(f"  Processing seed {i+1}/{len(seeds)}...")
        for _ in range(args.multiplier):
            variant = augment_sample(seed, args.adversarial_rate)
            if variant:
                augmented.append(variant)
            else:
                skipped += 1

    random.shuffle(augmented)

    # Validate and write
    valid = 0
    with open(args.out, "w") as f:
        for s in augmented:
            msgs = s["messages"]
            if "<|tool_call|>" not in msgs[-1]["content"] or "<|observe|>" not in msgs[-1]["content"]:
                skipped += 1
                continue
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            valid += 1

    # Stats
    scratch = sum(1 for s in augmented if "<|scratch|>" in str(s))
    think = sum(1 for s in augmented if "<|think_start|>" in str(s))
    single = sum(1 for s in augmented if s["type"] == "tool_single")
    multi = sum(1 for s in augmented if s["type"] == "tool_multi")

    print(f"\nWritten {valid} samples to {args.out}")
    print(f"  Dropped (invalid after augmentation): {skipped}")
    print(f"  Single: {single}, Multi: {multi}")
    print(f"  <|scratch|>: {scratch} ({100 * scratch // max(valid, 1)}%)")
    print(f"  <|think_start|>: {think} ({100 * think // max(valid, 1)}%)")


if __name__ == "__main__":
    main()
