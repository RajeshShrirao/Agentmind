"""
Generate per-specialist synthetic training data for Cognitive Apprenticeship Architecture.

Outputs separate JSONL files per apprentice domain, each with adversarial variants.
Each sample includes a `domain` field for router training.

Key improvements over old design:
  - Per-domain files for sequential specialist training
  - Adversarial failure modes at configurable rate (default 30%)
  - Latent reasoning patterns via <|think_start|>...<|think_end|> boundaries
  - Diverse tool arguments (args vary per call, not hardcoded)
  - Code apprentice with realistic failures (syntax errors, runtime exceptions, memory limits)

Usage:
  python generate_scaled_synthetic.py
  # Generates data/apprentice_*.jsonl + data/router_training.jsonl
"""

import os
import json
import random

os.makedirs("data", exist_ok=True)

# ── Tool Registry (14 tools, shared across all apprentices) ─────

TOOL_DEFS = {
    "web_search": {
        "args_schema": {"query": str, "max_results": int},
        "generate_args": lambda: {"query": random.choice([
            "current AI developments", "Mamba SSM papers 2026", "multi-agent systems",
            "LLM fine-tuning best practices", "retrieval augmented generation",
            "neural scaling laws", "transformer alternatives", "reinforcement learning from human feedback"
        ]), "max_results": random.choice([5, 10, 20])},
        "result": {"results": [{"title": "Advances in AI", "url": "https://example.com"}]}
    },
    "read_file": {
        "args_schema": {"path": str},
        "generate_args": lambda: {"path": random.choice([
            "/src/main.py", "/project/config.json", "/home/user/README.md",
            "/var/log/app.log", "/opt/data/results.csv", "/src/utils/helpers.py"
        ])},
        "result": {"content": "import mlx.core as mx\n\ndef main():\n    print('Hello')"}
    },
    "write_file": {
        "args_schema": {"path": str, "content": str},
        "generate_args": lambda: {"path": random.choice([
            "/output/report.md", "/src/new_feature.py", "/config/settings.json",
            "/docs/api.md", "/tests/test_unit.py"
        ]), "content": random.choice([
            "# Report\nKey findings...",
            "import mlx\nprint('hello')",
            "{\"setting\": \"value\", \"enabled\": true}"
        ])},
        "result": {"success": True, "bytes": lambda: random.randint(128, 4096)}
    },
    "run_python": {
        "args_schema": {"code": str},
        "generate_args": lambda: {"code": random.choice([
            "import numpy as np; print(np.mean([1,2,3]))",
            "print(sum(x*x for x in range(10)))",
            "import json; data = {'key': 'value'}; print(json.dumps(data))",
            "from pathlib import Path; print(Path('.').absolute())",
            "import pandas as pd; df = pd.DataFrame({'a': [1,2,3]}); print(df.mean())",
        ])},
        "result": {"stdout": lambda: random.choice(["2.0\n", "285\n", '{"key": "value"}\n', "/home/user/project\n"]), "stderr": ""}
    },
    "get_weather": {
        "args_schema": {"city": str},
        "generate_args": lambda: {"city": random.choice([
            "San Francisco", "Tokyo", "Pune", "London", "Sydney", "Berlin",
            "Toronto", "Singapore", "Mumbai", "New York", "Paris", "Seoul"
        ])},
        "result": {"temp": lambda: random.randint(10, 40), "condition": lambda: random.choice(["sunny", "cloudy", "rainy", "foggy", "windy"])}
    },
    "search_arxiv": {
        "args_schema": {"query": str, "days": int},
        "generate_args": lambda: {"query": random.choice([
            "state space models", "multi-agent RL", "LoRA fine-tuning",
            "retrieval augmented generation", "tool-use LLMs", "reasoning in language models"
        ]), "days": random.choice([7, 14, 30, 90])},
        "result": {"results": [{"id": "2405.12345", "title": "Mamba-2: Structured State Spaces"}]}
    },
    "fetch_abstract": {
        "args_schema": {"id": str},
        "generate_args": lambda: {"id": random.choice([
            "2405.12345", "2406.23456", "2407.34567", "2408.45678", "2409.56789"
        ])},
        "result": {"abstract": lambda: random.choice([
            "We present Mamba-2, a structured state space model that achieves...",
            "This paper introduces a novel approach to multi-agent reinforcement learning...",
            "We propose a new method for efficient fine-tuning of large language models...",
            "Retrieval augmented generation with structured knowledge bases improves factual accuracy..."
        ])}
    },
    "execute_sql": {
        "args_schema": {"query": str},
        "generate_args": lambda: {"query": random.choice([
            "SELECT COUNT(*) FROM users",
            "SELECT * FROM orders WHERE status = 'pending'",
            "SELECT AVG(price) FROM products WHERE category = 'electronics'",
            "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name",
            "SELECT * FROM logs WHERE timestamp > NOW() - INTERVAL 7 DAY",
        ])},
        "result": {"rows": lambda: [{"COUNT(*)": random.randint(100, 10000)}] if "COUNT" in str(random.choice) else [{"id": 1, "name": "sample"}]}
    },
    "send_email": {
        "args_schema": {"to": str, "subject": str, "body": str},
        "generate_args": lambda: {"to": random.choice([
            "team@company.com", "user@example.com", "admin@service.com",
            "researchers@lab.org", "support@platform.io"
        ]), "subject": random.choice(["Update", "Report", "Summary", "Findings", "Alert"]),
        "body": random.choice(["Progress report attached.", "Analysis complete.", "Results ready for review."])},
        "result": {"success": True, "message_id": lambda: f"msg_{random.randint(100000, 999999)}"}
    },
    "git_commit": {
        "args_schema": {"message": str},
        "generate_args": lambda: {"message": random.choice([
            "fix: resolve memory leak in data pipeline",
            "feat: add caching layer for API responses",
            "refactor: clean up import statements",
            "docs: update README with setup instructions",
            "test: add unit tests for MambaBlock",
            "perf: optimize SSM scan with compiler flags",
        ])},
        "result": {"success": True, "commit_hash": lambda: ''.join(random.choice('abcdef0123456789') for _ in range(7))}
    },
    "list_directory": {
        "args_schema": {"path": str},
        "generate_args": lambda: {"path": random.choice([
            "/src", "/project", "/home", "/opt/data", "/var/log", "/config"
        ])},
        "result": {"files": lambda: random.sample([
            "main.py", "utils.py", "config.py", "test_main.py", "requirements.txt",
            "Dockerfile", "README.md", "data.csv", "model.py", "train.py"
        ], random.randint(3, 6))}
    },
    "get_stock_price": {
        "args_schema": {"ticker": str},
        "generate_args": lambda: {"ticker": random.choice([
            "AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "CRM"
        ])},
        "result": {"price": lambda: round(random.uniform(50, 1000), 2), "change": lambda: f"{random.choice(['+', '-'])}{random.uniform(0.1, 5.0):.1f}%"}
    },
    "translate": {
        "args_schema": {"text": str, "target_lang": str},
        "generate_args": lambda: {"text": random.choice([
            "Hello world", "Thank you very much", "What is the weather today?",
            "Please send me the report", "How does this work?", "Good morning"
        ]), "target_lang": random.choice(["es", "fr", "de", "ja", "zh", "hi"])},
        "result": {"translated": lambda: random.choice([
            "Hola mundo", "Merci beaucoup", "Wie ist das Wetter heute?",
            "Por favor envíame el informe", "Wie funktioniert das?"
        ])}
    },
    "summarize": {
        "args_schema": {"text": str},
        "generate_args": lambda: {"text": random.choice([
            "A long article about neural networks and their applications in natural language processing...",
            "Research paper on the effectiveness of multi-agent systems for complex task decomposition...",
            "Technical report comparing different approaches to model fine-tuning on limited hardware...",
        ])},
        "result": {"summary": lambda: random.choice([
            "Key points: 1) Neural networks excel at pattern recognition 2) Scale improves performance 3) Data quality matters more than quantity",
            "Summary: Multi-agent approaches show 30% improvement on complex tasks over single-agent baselines",
            "Findings: LoRA achieves 90% of full fine-tuning performance at 1% of the parameter cost"
        ])}
    },
}

CODE_TOOLS = ["run_python", "read_file", "write_file", "execute_sql", "git_commit", "list_directory"]
RESEARCH_TOOLS = ["search_arxiv", "fetch_abstract", "web_search", "summarize"]
ALL_TOOL_NAMES = list(TOOL_DEFS.keys())

# ── Adversarial failure modes ───────────────────────────────────
FAILURE_MODES = {
    "timeout": {"error": "timeout", "retry": True, "message": "Service did not respond within 30s"},
    "partial_success": {"status": "partial", "data": None, "message": "Only partial results available"},
    "malformed_json": "unexpected response from <<tool>> unable to parse",
    "contradictory": {"warning": "Conflicting data from sources", "retry": True, "message": "Source A and Source B disagree"},
    "hidden_variable": {"error": "missing_context", "message": "Request requires prior result not found"},
}

# ── Code-specific adversarial modes ─────────────────────────────
CODE_FAILURE_MODES = {
    "syntax_error": {"error": "SyntaxError", "line": lambda: random.randint(1, 20), "message": "invalid syntax"},
    "runtime_exception": {"error": lambda: random.choice(["ValueError", "TypeError", "IndexError", "KeyError"]), "message": lambda: random.choice(["list index out of range", "unsupported operand type", "division by zero"])},
    "infinite_loop": {"error": "TimeoutError", "message": "Execution timed out after 10 seconds"},
    "memory_error": {"error": "MemoryError", "message": "Process exceeded memory limit of 512MB"},
    "import_error": {"error": "ModuleNotFoundError", "message": lambda: f"No module named '{random.choice(['numpy', 'pandas', 'torch', 'transformers', 'mlx'])}'"},
}

def resolve_result(result_template):
    """Resolve lambda values in a result template to concrete values."""
    if isinstance(result_template, dict):
        return {k: resolve_result(v) for k, v in result_template.items()}
    if callable(result_template):
        return result_template()
    return result_template

def apply_failure(tool_result: dict, mode: str):
    """Wrap a clean tool result with the given failure mode."""
    if mode == "clean":
        return tool_result
    if mode == "malformed_json":
        return FAILURE_MODES["malformed_json"]
    return {**FAILURE_MODES[mode], "partial_data": tool_result}


# ── Latent reasoning helper ─────────────────────────────────────
def inject_latent_boundaries(text: str) -> str:
    """
    Wrap <|scratch|> content in <|think_start|>...<|think_end|> boundaries.
    Applied to ~50% of samples with scratch content during generation,
    so the data pipeline can use either raw or latent-wrapped variants.
    """
    import re
    pattern = re.compile(
        r"<\|scratch\|>(.*?)(?=<\|tool_call\|>|<\|observe\|>|<\|plan\|>|<\|assistant\|>|<\|user\|>|<\|system\|>|<eos>|$)",
        re.DOTALL
    )
    def replace_match(match):
        thoughts = match.group(1)
        return f"<|think_start|><|scratch|>{thoughts}<|think_end|>"
    return pattern.sub(replace_match, text)


# ── Tool Caller Apprentice ─────────────────────────────────────
def generate_tool_caller(adversarial_rate: float = 0.3, latent_rate: float = 0.5):
    """Single tool calls. Mastery of JSON formatting, all 14 tools, boundary tokens."""
    tool_name = random.choice(ALL_TOOL_NAMES)
    tool_def = TOOL_DEFS[tool_name]
    args = tool_def["generate_args"]()
    clean_result = resolve_result(tool_def["result"])

    failure_mode = "clean" if random.random() > adversarial_rate else random.choice(list(FAILURE_MODES.keys()))
    result = apply_failure(clean_result, failure_mode)
    tool_call = json.dumps({"name": tool_name, "args": args})
    observe = json.dumps(result)

    if failure_mode == "clean":
        followup = random.choice([
            "Here's what I found.",
            "Task completed. The results are above.",
            "Done. Let me know if you need more detail.",
            f"The {tool_name} tool returned successfully.",
        ])
        assistant = f"<|tool_call|>{tool_call}<|observe|>{observe}\n{followup}"
    else:
        followup = random.choice([
            "The tool didn't respond. Let me retry.",
            "Got an unexpected response. Trying again.",
            "Something went wrong. I'll use a different approach.",
            "Error detected. Attempting recovery.",
        ])
        retry_call = json.dumps({"name": tool_name, "args": args})
        retry_result = json.dumps(clean_result)
        assistant = (
            f"<|tool_call|>{tool_call}<|observe|>{observe}"
            f"<|scratch|>{followup}"
            f"<|tool_call|>{retry_call}<|observe|>{retry_result}"
            f"\nDone after retry."
        )

    # Apply latent reasoning boundaries to a subset
    if "<|scratch|>" in assistant and random.random() < latent_rate:
        assistant = inject_latent_boundaries(assistant)

    return {
        "domain": "tool_caller",
        "type": "tool_single",
        "messages": [
            {"role": "user", "content": f"Use {tool_name} to {list(args.values())[0]}."},
            {"role": "assistant", "content": assistant}
        ]
    }


# ── Planner Apprentice ─────────────────────────────────────────
def generate_planner(adversarial_rate: float = 0.3, latent_rate: float = 0.5):
    """Multi-step trajectories with <|plan|>, decomposition, dependency chaining."""
    n_steps = random.randint(2, 5)
    tool_names = random.sample(ALL_TOOL_NAMES, min(n_steps, len(ALL_TOOL_NAMES)))

    topics = ["AI agents", "Mamba SSM", "multi-agent systems", "LLM fine-tuning",
              "retrieval augmented generation", "neural scaling laws"]
    queries = [
        f"Research {random.choice(topics)} and write a report.",
        f"Find all relevant papers, summarize them, and send the summary via email.",
        f"Analyze the codebase, run tests, and commit the fixes.",
        f"Compare weather in three cities and recommend the best destination.",
        f"Search for stock market trends, analyze the data, and generate a report.",
    ]
    query = random.choice(queries)

    plan_steps = "\n".join(f"{i+1}. Use {t}" for i, t in enumerate(tool_names))
    assistant = f"<|plan|>{plan_steps}"

    for i, tool_name in enumerate(tool_names):
        tool_def = TOOL_DEFS[tool_name]
        args = tool_def["generate_args"]()
        clean_result = resolve_result(tool_def["result"])
        failure_mode = "clean" if random.random() > adversarial_rate else random.choice(list(FAILURE_MODES.keys()))
        result = apply_failure(clean_result, failure_mode)
        call = json.dumps({"name": tool_name, "args": args})

        if failure_mode == "clean":
            assistant += f"<|tool_call|>{call}<|observe|>{json.dumps(result)}"
        elif i < len(tool_names) - 1:
            retry_result = json.dumps(clean_result)
            assistant += (
                f"<|tool_call|>{call}<|observe|>{json.dumps(result)}"
                f"<|scratch|>Step {i+1} failed. Adjusting plan..."
                f"<|tool_call|>{call}<|observe|>{retry_result}"
            )
        else:
            assistant += f"<|tool_call|>{call}<|observe|>{json.dumps(result)}"

    assistant += "\n\nTask complete based on gathered information."

    # Apply latent reasoning boundaries
    if "<|scratch|>" in assistant and random.random() < latent_rate:
        assistant = inject_latent_boundaries(assistant)

    return {
        "domain": "planner",
        "type": "agent_multi",
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": assistant}
        ]
    }


# ── Recovery Apprentice ────────────────────────────────────────
RECOVERY_SCENARIOS = [
    {"label": "retry", "error": {"error": "timeout", "retry": True}},
    {"label": "retry", "error": {"error": "rate_limit", "retry_after": 2}},
    {"label": "retry", "error": {"error": "network_error", "retry": True}},
    {"label": "fallback", "error": {"error": "invalid_params", "suggestion": "use alternative endpoint"}},
    {"label": "verify", "error": {"warning": "data_may_be_stale", "age_hours": 48}},
    {"label": "rollback", "error": {"error": "corrupt_state", "rollback": True}},
    {"label": "delegate", "error": {"error": "out_of_scope", "suggested_tool": "web_search"}},
]

def generate_recovery(adversarial_rate: float = 0.4, latent_rate: float = 0.5):
    """Failure recovery with multi-round retry, fallback, verify, rollback decisions."""
    tool_name = random.choice(ALL_TOOL_NAMES)
    tool_def = TOOL_DEFS[tool_name]
    args = tool_def["generate_args"]()
    clean_result = resolve_result(tool_def["result"])
    scenario = random.choice(RECOVERY_SCENARIOS)

    call = json.dumps({"name": tool_name, "args": args})
    error_result = json.dumps(scenario["error"])
    success_result = json.dumps(clean_result)

    if scenario["label"] == "retry":
        scratch = random.choice([
            "Tool failed. Retrying with exponential backoff...",
            "Connection issue. Trying again with delay.",
            "Rate limited. Waiting and retrying.",
        ])
        assistant = (
            f"<|tool_call|>{call}<|observe|>{error_result}"
            f"<|scratch|>{scratch}"
            f"<|tool_call|>{call}<|observe|>{success_result}"
        )
    elif scenario["label"] == "fallback":
        fallback_tool = random.choice([t for t in ALL_TOOL_NAMES if t != tool_name])
        fallback_def = TOOL_DEFS[fallback_tool]
        fallback_args = fallback_def["generate_args"]()
        fallback_result = resolve_result(fallback_def["result"])
        fallback_call = json.dumps({"name": fallback_tool, "args": fallback_args})
        assistant = (
            f"<|tool_call|>{call}<|observe|>{error_result}"
            f"<|scratch|>Primary tool doesn't support this. Switching to {fallback_tool}."
            f"<|tool_call|>{fallback_call}<|observe|>{json.dumps(fallback_result)}"
        )
    elif scenario["label"] == "verify":
        contradictory_result = json.dumps({"data": "conflicting values", "confidence": 0.3})
        assistant = (
            f"<|tool_call|>{call}<|observe|>{contradictory_result}"
            f"<|scratch|>Data seems stale. Verifying with another source."
            f"<|tool_call|>{call}<|observe|>{success_result}"
        )
    elif scenario["label"] == "rollback":
        assistant = (
            f"<|tool_call|>{call}<|observe|>{error_result}"
            f"<|scratch|>State corrupted. Rolling back previous operation."
            f"<|tool_call|>{call}<|observe|>{success_result}"
        )
    else:
        assistant = (
            f"<|tool_call|>{call}<|observe|>{error_result}"
            f"<|scratch|>This requires a different tool. Delegating."
            f"<|tool_call|>{call}<|observe|>{success_result}"
        )

    if "<|scratch|>" in assistant and random.random() < latent_rate:
        assistant = inject_latent_boundaries(assistant)

    return {
        "domain": "recovery",
        "type": "recovery",
        "messages": [
            {"role": "user", "content": f"Use {tool_name} for this task."},
            {"role": "assistant", "content": assistant}
        ]
    }


# ── Code Apprentice ────────────────────────────────────────────
CODE_TASKS = [
    ("read_file", "Read the file at {path} and tell me what it does."),
    ("write_file", "Write a {lang} script that {task}."),
    ("run_python", "Run this code and tell me the output:\n{code}"),
    ("execute_sql", "Query the database: {query}"),
    ("git_commit", "Commit the changes with message: '{msg}'"),
    ("list_directory", "List all files in {path}."),
    ("run_python", "Execute this analysis: {code}"),
]

def generate_code(adversarial_rate: float = 0.3, latent_rate: float = 0.5):
    """Code-specific operations with realistic failures: syntax errors, runtime exceptions, timeouts."""
    tool_name = random.choice(CODE_TOOLS)
    tool_def = TOOL_DEFS[tool_name]
    args = tool_def["generate_args"]()
    clean_result = resolve_result(tool_def["result"])

    # Decide whether to inject a code-specific failure
    failure_mode = "clean" if random.random() > adversarial_rate else random.choice(list(CODE_FAILURE_MODES.keys()))

    if failure_mode == "clean":
        result = clean_result
        assistant_content = f"<|tool_call|>{json.dumps({'name': tool_name, 'args': args})}<|observe|>{json.dumps(result)}\nDone."
    else:
        code_failure = CODE_FAILURE_MODES[failure_mode]
        error_detail = resolve_result(code_failure)
        result = {"error": error_detail["error"], "message": error_detail["message"]}

        scratch = random.choice([
            f"Code failed with {error_detail['error']}. Debugging...",
            f"Got {error_detail['error']}. Let me fix and retry.",
            f"Error: {error_detail['message']}. Adjusting the approach.",
        ])
        retry_result = json.dumps(clean_result)
        retry_call = json.dumps({"name": tool_name, "args": args, "fix_attempt": True})
        assistant_content = (
            f"<|tool_call|>{json.dumps({'name': tool_name, 'args': args})}<|observe|>{json.dumps(result)}"
            f"<|scratch|>{scratch}"
            f"<|tool_call|>{retry_call}<|observe|>{retry_result}"
            f"\nFixed and executed successfully."
        )

    if "<|scratch|>" in assistant_content and random.random() < latent_rate:
        assistant_content = inject_latent_boundaries(assistant_content)

    # Build user query
    code_sample = random.choice([
        {"code": "import numpy as np\nprint(np.mean([1,2,3,4,5]))", "output": "3.0"},
        {"code": "print(sum(x*x for x in range(10)))", "output": "285"},
        {"code": "import json\ndata = {'key': 'value'}\nprint(json.dumps(data))", "output": '{"key": "value"}'},
        {"code": "from pathlib import Path\nprint(Path('.').absolute())", "output": "/home/user/project"},
    ])
    query_template = random.choice(CODE_TASKS)
    user_query = query_template[1].format(
        path=args.get("path", "/src"),
        lang=random.choice(["Python", "SQL", "Bash"]),
        task=random.choice(["sort an array", "parse JSON", "scrape a webpage"]),
        code=code_sample["code"],
        query=args.get("query", "SELECT * FROM users"),
        msg=args.get("message", "fix: resolve bug"),
    )

    return {
        "domain": "code",
        "type": "tool_single",
        "messages": [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": assistant_content}
        ]
    }


# ── Research Apprentice ────────────────────────────────────────
def generate_research(adversarial_rate: float = 0.3, latent_rate: float = 0.5):
    """Research workflow: search → fetch → summarize → synthesize."""
    n_steps = random.randint(2, 4)
    tool_names = random.sample(RESEARCH_TOOLS, min(n_steps, len(RESEARCH_TOOLS)))

    topics = ["Mamba SSM", "multi-agent RL", "LoRA fine-tuning", "retrieval augmented generation",
              "tool-use LLMs", "reasoning in language models", "efficient transformer architectures"]
    topic = random.choice(topics)
    query = f"Research {topic} and provide a summary of the latest findings."

    assistant = ""
    for tool_name in tool_names:
        tool_def = TOOL_DEFS[tool_name]
        args = tool_def["generate_args"]()
        clean_result = resolve_result(tool_def["result"])
        failure_mode = "clean" if random.random() > adversarial_rate else random.choice(list(FAILURE_MODES.keys()))
        result = apply_failure(clean_result, failure_mode)
        call = json.dumps({"name": tool_name, "args": args})

        if failure_mode == "clean":
            assistant += f"<|tool_call|>{call}<|observe|>{json.dumps(result)}"
        else:
            retry_result = json.dumps(clean_result)
            assistant += (
                f"<|tool_call|>{call}<|observe|>{json.dumps(result)}"
                f"<|scratch|>Source unresponsive. Trying alternative."
                f"<|tool_call|>{call}<|observe|>{retry_result}"
            )

    assistant += f"\n\n## Research Summary for {topic}\nBased on gathered information, here are the key findings."

    if "<|scratch|>" in assistant and random.random() < latent_rate:
        assistant = inject_latent_boundaries(assistant)

    return {
        "domain": "research",
        "type": "agent_multi",
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": assistant}
        ]
    }


# ── Router Training Data ───────────────────────────────────────
def generate_router_samples(samples_by_domain: dict, n_per_domain: int = 200):
    """Create labeled samples for router classifier training."""
    router_data = []
    for domain, samples in samples_by_domain.items():
        chosen = random.sample(samples, min(n_per_domain, len(samples)))
        for s in chosen:
            router_data.append({
                "domain": domain,
                "messages": s["messages"]
            })
    random.shuffle(router_data)
    return router_data


# ── Main Generation ────────────────────────────────────────────
def main():
    print("=" * 60)
    print("AgentMind — Cognitive Apprenticeship Data Factory")
    print("=" * 60)

    TARGETS = {
        "tool_caller": (3000, generate_tool_caller),
        "planner": (2000, generate_planner),
        "recovery": (2000, generate_recovery),
        "code": (1500, generate_code),
        "research": (1500, generate_research),
    }

    samples_by_domain = {}
    total_latent = 0
    total_samples = 0

    for domain, (count, generator) in TARGETS.items():
        print(f"\n[{domain}] Generating {count} samples...")
        samples = []
        latent_count = 0
        for i in range(count):
            sample = generator()
            samples.append(sample)
            # Count latent reasoning patterns
            content = str(sample["messages"])
            if "<|think_start|>" in content:
                latent_count += 1
        samples_by_domain[domain] = samples
        total_latent += latent_count
        total_samples += count

        path = f"data/apprentice_{domain}.jsonl"
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"  → {path} ({len(samples)} samples, {latent_count} with latent reasoning)")

    # Router training data
    print(f"\n[router] Generating training samples...")
    router_data = generate_router_samples(samples_by_domain, n_per_domain=300)
    path = "data/router_training.jsonl"
    with open(path, "w") as f:
        for s in router_data:
            f.write(json.dumps(s) + "\n")
    print(f"  → {path} ({len(router_data)} samples)")

    # Detailed summary
    print(f"\n{'=' * 60}")
    print(f"Total samples generated: {total_samples}")
    print(f"  With latent reasoning: {total_latent} ({100 * total_latent // max(total_samples, 1)}%)")
    print(f"Router training samples: {len(router_data)}")
    for domain, samples in samples_by_domain.items():
        # Count adversarial samples (those with <|scratch|>)
        scratch_count = sum(1 for s in samples if "<|scratch|>" in str(s["messages"]))
        think_count = sum(1 for s in samples if "<|think_start|>" in str(s["messages"]))
        print(f"  {domain}: {len(samples)} total, ~{scratch_count} adversarial ({100 * scratch_count // max(len(samples), 1)}%), {think_count} latent")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
