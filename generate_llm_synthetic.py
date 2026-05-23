"""
LLM-driven synthetic data generator for AgentMind apprenticeship training.

Uses Cerebras API (llama3.1-8b) to generate diverse, natural tool-calling
conversations. The key value over template-based data is NATURAL user queries
and realistic multi-step workflows.

Strategy:
  - LLM generates the full conversation freely
  - Tool_call JSON is validated strictly (name, args, valid tool)
  - Observe content can be natural text or JSON (accept both)
  - Post-processing wraps text observe in JSON for format consistency

Usage:
  python generate_llm_synthetic.py --samples 3000 --domains tool_caller,planner

Output: data/apprentice_*.jsonl + data/router_training.jsonl
"""

import os, json, re, sys, random, time
from typing import Optional

os.makedirs("data", exist_ok=True)

TOOL_REGISTRY = {
    "web_search": {
        "description": "Search the web for current information on any topic",
        "params": {"query": {"type": "string", "required": True}, "max_results": {"type": "integer", "required": False}},
    },
    "read_file": {
        "description": "Read the contents of a file from disk",
        "params": {"path": {"type": "string", "required": True}},
    },
    "write_file": {
        "description": "Write content to a file on disk",
        "params": {"path": {"type": "string", "required": True}, "content": {"type": "string", "required": True}},
    },
    "run_python": {
        "description": "Execute Python code in a sandboxed environment",
        "params": {"code": {"type": "string", "required": True}},
    },
    "get_weather": {
        "description": "Get current weather conditions and temperature for a city",
        "params": {"city": {"type": "string", "required": True}},
    },
    "search_arxiv": {
        "description": "Search arxiv.org for academic papers matching a query",
        "params": {"query": {"type": "string", "required": True}, "days": {"type": "integer", "required": False}},
    },
    "fetch_abstract": {
        "description": "Fetch the abstract of an arxiv paper by its ID",
        "params": {"id": {"type": "string", "required": True}},
    },
    "execute_sql": {
        "description": "Execute a SQL query against the database",
        "params": {"query": {"type": "string", "required": True}},
    },
    "send_email": {
        "description": "Send an email to a recipient with subject and body",
        "params": {"to": {"type": "string", "required": True}, "subject": {"type": "string", "required": True}, "body": {"type": "string", "required": True}},
    },
    "git_commit": {
        "description": "Commit staged changes to git with a commit message",
        "params": {"message": {"type": "string", "required": True}},
    },
    "list_directory": {
        "description": "List all files and directories at a given path",
        "params": {"path": {"type": "string", "required": True}},
    },
    "get_stock_price": {
        "description": "Get the current stock price and daily change for a ticker symbol",
        "params": {"ticker": {"type": "string", "required": True}},
    },
    "translate": {
        "description": "Translate text from one language to another",
        "params": {"text": {"type": "string", "required": True}, "target_lang": {"type": "string", "required": True}},
    },
    "summarize": {
        "description": "Summarize a long text into concise key points",
        "params": {"text": {"type": "string", "required": True}},
    },
}

CODE_TOOLS = {"run_python", "read_file", "write_file", "execute_sql", "git_commit", "list_directory"}
RESEARCH_TOOLS = {"search_arxiv", "fetch_abstract", "web_search", "summarize"}

# Realistic result templates for each tool (used during post-processing)
TOOL_RESULTS = {
    "web_search": lambda a: {"results": [{"title": "Result about " + a.get("query", "topic"), "url": "https://example.com"}]},
    "read_file": lambda a: {"content": "File content for " + a.get("path", "/path")},
    "write_file": lambda a: {"success": True, "bytes": random.randint(50, 5000)},
    "run_python": lambda a: {"stdout": "Output from Python execution\n", "stderr": ""},
    "get_weather": lambda a: {"temp": random.randint(5, 40), "condition": random.choice(["sunny", "cloudy", "rainy", "windy"])},
    "search_arxiv": lambda a: {"results": [{"id": "2405." + str(random.randint(10000, 99999)), "title": "Paper on " + a.get("query", "topic")}]},
    "fetch_abstract": lambda a: {"abstract": "This paper presents a novel approach to " + random.choice(["state space models", "multi-agent systems", "efficient transformers", "reinforcement learning"])},
    "execute_sql": lambda a: {"rows": random.randint(1, 100), "columns": ["id", "name", "value"]},
    "send_email": lambda a: {"success": True, "message_id": "msg_" + ''.join(random.choices("0123456789abcdef", k=8))},
    "git_commit": lambda a: {"success": True, "commit_hash": ''.join(random.choices("0123456789abcdef", k=7))},
    "list_directory": lambda a: {"files": random.sample(["main.py", "utils.py", "config.py", "README.md", "test.py", "data.json", "Dockerfile"], random.randint(3, 5))},
    "get_stock_price": lambda a: {"price": round(random.uniform(50, 1000), 2), "change": f"{random.choice(['+', '-'])}{random.uniform(0.1, 5.0):.1f}%"},
    "translate": lambda a: {"translated": random.choice(["Bonjour le monde", "Hola mundo", "Hallo Welt", "Ciao mondo"]), "detected_lang": "en"},
    "summarize": lambda a: {"summary": "Key findings: 1) The research shows promising results 2) Further validation needed 3) Practical applications in deployment"},
}

ERROR_RESULTS = {
    "timeout": {"error": "timeout", "retry": True, "message": "Service did not respond within 30s"},
    "rate_limit": {"error": "rate_limit", "retry_after": 2, "message": "Rate limit exceeded"},
    "network_error": {"error": "network_error", "retry": True, "message": "Connection reset by peer"},
    "partial_data": {"warning": "partial_results", "data": None, "message": "Service returned incomplete data"},
    "invalid_params": {"error": "invalid_params", "message": "Required parameter missing or invalid"},
}


def make_client():
    from cerebras.cloud.sdk import Cerebras
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        api_key = "csk-c2fe5k5hk529dxdd3f4px8m8tct8dmfec8m689cmkp2pwr6m"
    return Cerebras(api_key=api_key)


def format_tools() -> str:
    lines = ["Available tools:"]
    for name, info in sorted(TOOL_REGISTRY.items()):
        params = ", ".join(f"{k}: {v['type']}{' (required)' if v['required'] else ''}" for k, v in info['params'].items())
        lines.append(f"  - {name}: {info['description']} | args: {params}")
    return "\n".join(lines)


# ── Prompts ────────────────────────────────────────────────────

BASE_PROMPT_INTRODUCTION = """You are generating training data for an AI assistant that calls tools.

The assistant communicates using special tokens:
  <|tool_call|> - marks the start of a tool call (must be followed by JSON)
  <|tool_call|>{"name": "...", "args": {...}}<|observe|> - tool call + result
  <|observe|> - marks the start of a tool result 
  <|scratch|> - internal reasoning when things go wrong
  <|plan|> - multi-step plan listing

Output format - ONE VALID JSON OBJECT per response, no markdown, no explanation."""

TOOL_CALLER_SYSTEM_PROMPT = f"""{BASE_PROMPT_INTRODUCTION}

{format_tools()}

JSON structure:
{{"domain": "tool_caller", "type": "tool_single", "messages": [
  {{"role": "user", "content": "NATURAL user query here"}},
  {{"role": "assistant", "content": "<|tool_call|>{{"name": "...", "args": {{...}}}}<|observe|>tool result here\nAssistant's natural response wrapping the result."}}
]}}

KEY RULES:
- User queries must be NATURAL: questions, commands, requests - like a real person
- NEVER use patterns like "Use X to do Y" - this is template garbage
- Tool_call JSON MUST have "name" and "args" matching the tool's schema
- For multi-step: chain <|tool_call|>...<|observe|>...<|tool_call|>...<|observe|>...
- For errors: <|tool_call|>...<|observe|>error<|scratch|>reasoning<|tool_call|>...<|observe|>success
- After the last <|observe|>, include \\n followed by natural response text
- VARY tools across generations (use ALL 14 tools, not just 2-3)
- Keep assistant content under 2000 characters

Good example:
{{"domain": "tool_caller", "type": "tool_single", "messages": [{{"role": "user", "content": "What's the weather in Tokyo and how's the stock market doing today?"}}, {{"role": "assistant", "content": "<|tool_call|>{{\\"name\\": \\"get_weather\\", \\"args\\": {{\\"city\\": \\"Tokyo\\"}}}}<|observe|>Sunny, 28°C<|tool_call|>{{\\"name\\": \\"get_stock_price\\", \\"args\\": {{\\"ticker\\": \\"^N225\\"}}}}<|observe|>Nikkei up 1.2%\\nTokyo is sunny and 28°C while the Nikkei is up 1.2%. A good day overall!"}}]}}"""

PLANNER_SYSTEM_PROMPT = f"""{BASE_PROMPT_INTRODUCTION}

{format_tools()}

JSON structure:
{{"domain": "planner", "type": "agent_multi", "messages": [
  {{"role": "user", "content": "NATURAL multi-step request"}},
  {{"role": "assistant", "content": "<|plan|>1. Step one\\n2. Step two<|tool_call|>...<|observe|>...<|tool_call|>...<|observe|>...\\n## Summary\\nWhat was accomplished."}}
]}}

KEY RULES:
- ALWAYS start assistant with <|plan|> listing 2-4 steps
- Execute each step via <|tool_call|>...<|observe|>...
- End with \\n## Summary\\n wrapping up results
- For failures: insert <|scratch|>reasoning between steps"""

RECOVERY_SYSTEM_PROMPT = f"""{BASE_PROMPT_INTRODUCTION}

{format_tools()}

JSON structure:
{{"domain": "recovery", "type": "recovery", "messages": [
  {{"role": "user", "content": "request that will trigger an error"}},
  {{"role": "assistant", "content": "<|tool_call|>...<|observe|>error details<|scratch|>Analyzing failure.<|tool_call|>...<|observe|>success result\\nRecovery response."}}
]}}

KEY RULES:
- First call MUST FAIL then recover
- Include <|scratch|> reasoning between failure and recovery
- Make failures realistic: timeouts, errors, contradictory data, missing info"""

CODE_SYSTEM_PROMPT = f"""{BASE_PROMPT_INTRODUCTION}

{format_tools()}

JSON structure:
{{"domain": "code", "type": "tool_single", "messages": [
  {{"role": "user", "content": "coding task"}},
  {{"role": "assistant", "content": "<|tool_call|>...<|observe|>...\\nResponse about the code."}}
]}}

KEY RULES:
- Use code tools: run_python, read_file, write_file, execute_sql, git_commit, list_directory
- Realistic tasks: debugging, refactoring, writing scripts, analyzing code
- For errors: include <|scratch|>debugging before retrying"""

RESEARCH_SYSTEM_PROMPT = f"""{BASE_PROMPT_INTRODUCTION}

{format_tools()}

JSON structure:
{{"domain": "research", "type": "agent_multi", "messages": [
  {{"role": "user", "content": "research question"}},
  {{"role": "assistant", "content": "<|tool_call|>...<|observe|>...<|tool_call|>...<|observe|>...\\n## Research Summary\\nSynthesized findings."}}
]}}

KEY RULES:
- Use research tools: search_arxiv, fetch_abstract, web_search, summarize
- Chain tools naturally (search -> fetch -> summarize)
- End with \\n## Research Summary\\n synthesizing findings"""


DOMAIN_CONFIGS = {
    "tool_caller": {"prompt": TOOL_CALLER_SYSTEM_PROMPT, "count": 3000, "temperature": 0.8, "max_tokens": 2048},
    "planner": {"prompt": PLANNER_SYSTEM_PROMPT, "count": 2000, "temperature": 0.8, "max_tokens": 3072},
    "recovery": {"prompt": RECOVERY_SYSTEM_PROMPT, "count": 2000, "temperature": 0.9, "max_tokens": 2048},
    "code": {"prompt": CODE_SYSTEM_PROMPT, "count": 1500, "temperature": 0.7, "max_tokens": 2048},
    "research": {"prompt": RESEARCH_SYSTEM_PROMPT, "count": 1500, "temperature": 0.8, "max_tokens": 3072},
}


SIGNPOSTS = {
    "tool_caller": [
        "Generate a sample where the user asks about weather in multiple cities, requiring multiple get_weather calls.",
        "Generate a sample where the user needs file operations: read a config, modify it, write it back.",
        "Generate a sample about research: searching arxiv and fetching abstracts.",
        "Generate a sample about stocks or financial data with get_stock_price.",
        "Generate a sample about data analysis using run_python or execute_sql.",
        "Generate a sample where a tool call fails and the assistant recovers with retry.",
        "Generate a sample about translating text to another language.",
        "Generate a sample about sending an email to someone.",
        "Generate a sample involving web_search for current information.",
        "Generate a sample that chains 2-3 different tools in sequence.",
        "Generate a sample involving git operations.",
        "Generate a sample where the assistant needs to summarize content.",
    ],
    "planner": [
        "Generate a planner: research a topic via search_arxiv, fetch abstracts, write a report.",
        "Generate a planner: check weather in 3 cities, compare, recommend a destination.",
        "Generate a planner: analyze stock data, generate insights, email the results.",
        "Generate a planner: review a codebase, run tests, commit fixes.",
        "Generate a planner: search web, summarize findings, save to file.",
        "Generate a planner where one step fails and the plan adapts dynamically.",
    ],
    "recovery": [
        "Generate a recovery: tool times out and assistant retries successfully.",
        "Generate a recovery: API rate-limits and assistant waits before retry.",
        "Generate a recovery: contradictory data from two sources, assistant verifies.",
        "Generate a recovery: Python code has a runtime error, assistant debugs and fixes.",
        "Generate a recovery: SQL query has syntax error, assistant fixes and re-runs.",
        "Generate a recovery: network error prevents file access, assistant finds alternative.",
    ],
    "code": [
        "Generate a code sample: debug a Python script that crashes on import.",
        "Generate a code sample: refactor a file and commit the changes.",
        "Generate a code sample: write a new function, run tests, fix bugs.",
        "Generate a code sample: analyze a slow SQL query and optimize it.",
        "Generate a code sample: list files, read a key file, explain structure.",
        "Generate a code sample: fix a bug, run the code, verify output.",
    ],
    "research": [
        "Generate a research sample about multi-agent systems: search + fetch + summarize.",
        "Generate a research sample comparing LLM fine-tuning methods.",
        "Generate a research sample about Mamba/SSM architectures.",
        "Generate a research sample about retrieval-augmented generation.",
        "Generate a research sample about efficient transformer architectures.",
        "Generate a research sample about RLHF and alignment.",
    ],
}


# ── Validation ────────────────────────────────────────────────

def postprocess_observe(asst: str) -> str:
    """
    Post-process assistant content to wrap plain-text observe content in JSON.
    Searches for <|observe|> followed by non-JSON text and wraps it.
    """
    def fix_observe(m):
        prefix = m.group(1)  # <|observe|>
        content = m.group(2)
        stripped = content.lstrip()
        if stripped.startswith("{"):
            return m.group(0)
        # Text content - wrap in JSON
        json_result = json.dumps({"result": stripped.strip()[:200]})
        return f"<|observe|>{json_result}"
    
    pattern = re.compile(r'(<\|observe\|>)(.*?)(?=<\|tool_call\|>|<\|scratch\|>|<\|plan\|>\n|<\|assistant\|>|<\|user\|>|<\|system\|>|$)', re.DOTALL)
    return pattern.sub(fix_observe, asst)


def validate_sample(sample: dict) -> tuple[bool, str]:
    """Validate - strict on tool_call JSON, lenient on observe."""
    if not isinstance(sample, dict):
        return False, "not a dict"
    if "domain" not in sample or sample["domain"] not in DOMAIN_CONFIGS:
        return False, "invalid/missing domain"
    if "type" not in sample:
        return False, "missing type"
    if "messages" not in sample or not isinstance(sample["messages"], list) or len(sample["messages"]) < 2:
        return False, "need >=2 messages"
    for msg in sample["messages"]:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            return False, "invalid message format"
        if not isinstance(msg["content"], str) or len(msg["content"]) < 5:
            return False, "content too short"

    asst = sample["messages"][-1]["content"]
    if "<|tool_call|>" not in asst:
        return False, "missing <|tool_call|>"
    if "<|observe|>" not in asst:
        return False, "missing <|observe|>"

    # Validate tool_call JSON (strict)
    calls = re.findall(r'<\|tool_call\|>(.*?)<\|observe\|>', asst, re.DOTALL)
    if not calls:
        return False, "no tool_call content found"
    for call_str in calls:
        stripped = call_str.strip()
        if not stripped.startswith("{"):
            return False, f"tool_call not JSON: {stripped[:80]}"
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as e:
            return False, f"invalid tool call JSON: {e}"
        if "name" not in parsed or "args" not in parsed:
            return False, "tool call missing name or args"
        if parsed["name"] not in TOOL_REGISTRY:
            return False, f"unknown tool: {parsed['name']}"

    return True, ""


# ── LLM calls ─────────────────────────────────────────────────

def call_llm(client, messages: list, temperature: float = 0.8, max_tokens: int = 2048) -> Optional[str]:
    try:
        resp = client.chat.completions.create(
            model="llama3.1-8b",
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"  ⚠️  LLM call failed: {e}")
        return None


def extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # Try code blocks
    for pat in [r'```json\s*\n(.*?)\n\s*```', r'```\s*\n(.*?)\n\s*```']:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                continue
    # Greedy JSON extraction
    brace_depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    start = -1
    return None


def generate_samples(client, domain: str, n: int) -> list:
    config = DOMAIN_CONFIGS[domain]
    signposts = SIGNPOSTS.get(domain, [])
    samples = []
    attempts = 0
    max_attempts = max(n * 3, 200)
    last_error = ""
    fail_streak = 0

    print(f"\n[{domain}] Generating {n} samples via Cerebras (llama3.1-8b)...")

    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        signpost = random.choice(signposts) if signposts else ""
        user_prompt = f"{signpost}\n\nGenerate exactly one valid JSON object. Output ONLY the JSON, no markdown."

        text = call_llm(client, [
            {"role": "system", "content": config["prompt"]},
            {"role": "user", "content": user_prompt},
        ], temperature=config["temperature"], max_tokens=config["max_tokens"])

        if not text:
            fail_streak += 1
            if fail_streak > 5:
                time.sleep(3)
                fail_streak = 0
            continue

        fail_streak = 0
        sample = extract_json(text)
        if not sample:
            last_error = "extract failed"
            continue

        valid, error = validate_sample(sample)
        if valid:
            # Post-process observe content
            sample["messages"][-1]["content"] = postprocess_observe(sample["messages"][-1]["content"])
            samples.append(sample)
            if len(samples) % 20 == 0:
                rate = len(samples) / max(attempts, 1) * 100
                print(f"  [{time.strftime('%H:%M:%S')}] {len(samples)}/{n} ({rate:.0f}% success)")
        else:
            last_error = error

    if len(samples) < n:
        print(f"  ⚠️  Only {len(samples)}/{n} after {attempts} attempts (last error: {last_error})")
        from generate_scaled_synthetic import generate_tool_caller, generate_planner, generate_recovery, generate_code, generate_research
        FALLBACK_MAP = {
            "tool_caller": generate_tool_caller, "planner": generate_planner,
            "recovery": generate_recovery, "code": generate_code, "research": generate_research,
        }
        fn = FALLBACK_MAP.get(domain)
        if fn:
            print(f"  Filling {n - len(samples)} with template fallback...")
            for _ in range(n - len(samples)):
                samples.append(fn(adversarial_rate=0.3, latent_rate=0.5))

    return samples


def generate_router_samples(samples_by_domain: dict, n_per_domain: int = 200) -> list:
    router_data = []
    for domain, samples in samples_by_domain.items():
        chosen = random.sample(samples, min(n_per_domain, len(samples)))
        for s in chosen:
            router_data.append({"domain": domain, "messages": s["messages"]})
    random.shuffle(router_data)
    return router_data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--domains", type=str, default="tool_caller,planner,recovery,code,research")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--no-fallback", action="store_true")
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",")]
    print("=" * 60)
    print("AgentMind — LLM-Driven Synthetic Data Factory")
    print(f"  Model: llama3.1-8b via Cerebras")
    print(f"  Domains: {domains}")
    print("=" * 60)

    client = make_client()
    samples_by_domain = {}
    total = 0
    t0 = time.time()

    base = args.samples
    counts = {
        "tool_caller": base, "planner": int(base * 0.67), "recovery": int(base * 0.67),
        "code": int(base * 0.5), "research": int(base * 0.5),
    }

    for domain in domains:
        if domain not in DOMAIN_CONFIGS:
            print(f"  Unknown domain: {domain}")
            continue
        n = counts.get(domain, 1000)
        samples = generate_samples(client, domain, n)
        samples_by_domain[domain] = samples
        total += len(samples)

        path = f"data/apprentice_{domain}.jsonl"
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"  → {path} ({len(samples)} samples)")

    print(f"\n[router] Generating training samples...")
    router_data = generate_router_samples(samples_by_domain, n_per_domain=300)
    path = "data/router_training.jsonl"
    with open(path, "w") as f:
        for s in router_data:
            f.write(json.dumps(s) + "\n")
    print(f"  → {path} ({len(router_data)} samples)")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Total: {total} samples in {elapsed:.0f}s")
    for domain, samples in samples_by_domain.items():
        scratch = sum(1 for s in samples if "<|scratch|>" in str(s["messages"]))
        print(f"  {domain}: {len(samples)} samples, {scratch} with scratch")
    print("=" * 60)


if __name__ == "__main__":
    main()
