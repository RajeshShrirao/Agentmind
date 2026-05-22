"""
[DEPRECATED] Generate synthetic agent training data using Cerebras API.

This script is superseded by generate_scaled_synthetic.py which produces
per-apprentice data with adversarial modes and latent reasoning patterns.

Kept for reference and Cerebras diversity augmentation.
To use Cerebras for additional diversity on top of scaled synthetic:
  python generate_synthetic.py
  # Produces data/synthetic_agents.jsonl (legacy format)
  # This file can be merged into corpus.txt for tokenizer retraining only.

For apprentice training data, use:
  python generate_scaled_synthetic.py
"""

import os
import json
import random
from cerebras.cloud.sdk import Cerebras

client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

TOOLS = [
    {"name": "web_search", "args": {"query": "current developments in AI agents"}},
    {"name": "read_file", "args": {"path": "/src/main.py"}},
    {"name": "write_file", "args": {"path": "/output/summary.md", "content": "# Summary\nKey findings..."}},
    {"name": "run_python", "args": {"code": "import numpy as np; print(np.mean([1,2,3]))"}},
    {"name": "get_weather", "args": {"city": "San Francisco"}},
    {"name": "search_arxiv", "args": {"query": "state space models", "days": 7}},
    {"name": "fetch_abstract", "args": {"id": "2405.12345"}},
    {"name": "execute_sql", "args": {"query": "SELECT COUNT(*) FROM users"}},
    {"name": "send_email", "args": {"to": "team@company.com", "subject": "Update", "body": "Progress report..."}},
    {"name": "git_commit", "args": {"message": "fix: resolve memory leak in data pipeline"}},
    {"name": "list_directory", "args": {"path": "/src"}},
    {"name": "get_stock_price", "args": {"ticker": "AAPL"}},
    {"name": "translate", "args": {"text": "Hello world", "target_lang": "es"}},
    {"name": "summarize", "args": {"text": "Long article text..."}},
]

USER_QUERIES = [
    "Search for the latest papers on Mamba SSM and summarize the key findings.",
    "Find all Python files in the project and count total lines of code.",
    "Check the weather in Tokyo, Pune, and San Francisco and compare them.",
    "Run the test suite and fix any failures you find.",
    "Search arxiv for papers about AI agents published in the last week.",
    "Read the README.md file and create a summary of the project structure.",
    "Execute this Python script and tell me the output.",
    "Find the stock price of AAPL and compare it with last month.",
    "Search the web for current best practices in LLM fine-tuning.",
    "Write a Python function that calculates fibonacci numbers efficiently.",
    "Get the current weather in three cities and recommend the best one to visit.",
    "Find all TODO comments in the codebase and create a task list.",
    "Search for recent news about quantum computing breakthroughs.",
    "Run a SQL query to find the top 10 users by activity.",
    "Fetch the abstract of paper 2405.12345 and explain it in simple terms.",
]

def generate_instruction():
    """Type 1: Plain instruction following."""
    query = random.choice(USER_QUERIES)
    responses = [
        f"Based on my analysis, here are the key points:\n\n1. The topic involves several important concepts\n2. Recent developments show significant progress\n3. The implications are far-reaching",
        f"I've processed your request. Here's what I found:\n\nThe information you requested is available and shows interesting patterns. Let me know if you need more details.",
        f"Here's a comprehensive response:\n\n• Point one: The primary finding relates to efficiency improvements\n• Point secondary: Secondary effects include better scalability\n• Point three: Future work should focus on optimization",
    ]
    return {
        "type": "instruction",
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": random.choice(responses)}
        ]
    }

def generate_single_tool():
    """Type 2: Single tool call."""
    tool = random.choice(TOOLS)
    tool_call = json.dumps({"name": tool["name"], "args": tool["args"]})
    results = {
        "web_search": {"results": [{"title": "Advances in AI Agent Architecture", "url": "https://arxiv.org/abs/2405.12345"}]},
        "read_file": {"content": "import mlx.core as mx\n\ndef main():\n    print('Hello World')"},
        "write_file": {"success": True, "bytes": 1024},
        "run_python": {"stdout": "2.0\n", "stderr": ""},
        "get_weather": {"temp": random.randint(15, 35), "condition": random.choice(["sunny", "cloudy", "rainy"])},
        "search_arxiv": {"results": [{"id": "2405.12345", "title": "Mamba: Linear-Time Sequence Modeling"}]},
        "fetch_abstract": {"abstract": "We present Mamba, a selective state space model that achieves..."},
        "execute_sql": {"rows": [{"COUNT(*)": 1542}]},
        "send_email": {"success": True, "message_id": "msg_abc123"},
        "git_commit": {"success": True, "commit_hash": "a1b2c3d"},
        "list_directory": {"files": ["main.py", "utils.py", "config.py"]},
        "get_stock_price": {"price": 189.50, "change": "+2.3%"},
        "translate": {"translated": "Hola mundo"},
        "summarize": {"summary": "Key points: 1... 2... 3..."}
    }
    result = results.get(tool["name"], {"status": "ok"})
    observe = json.dumps(result)

    followups = [
        f"Here are the results:\n{json.dumps(result, indent=2)}",
        f"The tool returned successfully. Based on the output, I can confirm the operation completed.",
        f"Task completed. The results show {json.dumps(result)}"
    ]

    tool_key = tool["args"].get("query", tool["args"].get("city", tool["args"].get("path", "process this request")))
    return {
        "type": "tool_single",
        "messages": [
            {"role": "user", "content": f"Use {tool['name']} to {tool_key}."},
            {"role": "assistant", "content": f"<|tool_call|>{tool_call}<|observe|>{observe}\n{random.choice(followups)}"}
        ]
    }

def generate_multi_step():
    """Type 3: Multi-step agentic trajectory."""
    n_steps = random.randint(2, 4)
    tools = random.sample(TOOLS, min(n_steps, len(TOOLS)))
    query = random.choice(USER_QUERIES)

    plan_steps = "\n".join(f"{i+1}. Use {t['name']}" for i, t in enumerate(tools))
    assistant_content = f"<|plan|>{plan_steps}"

    for tool in tools:
        tool_call = json.dumps({"name": tool["name"], "args": tool["args"]})
        result = {"status": "success", "data": f"Results from {tool['name']}"}
        observe = json.dumps(result)
        assistant_content += f"<|tool_call|>{tool_call}<|observe|>{observe}"

    assistant_content += "\n\nBased on all the gathered information, I can now provide a comprehensive answer."

    return {
        "type": "agent_multi",
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def generate_recovery():
    """Type 4: Failure recovery."""
    tool = random.choice(TOOLS)
    tool_call = json.dumps({"name": tool["name"], "args": tool["args"]})
    error_result = json.dumps({"error": "timeout", "retry": True, "message": "Service temporarily unavailable"})
    success_result = json.dumps({"status": "success", "data": f"Results from {tool['name']} (retry)"})

    return {
        "type": "recovery",
        "messages": [
            {"role": "user", "content": f"Try to {tool['name']} with {tool['args']}"},
            {"role": "assistant", "content": f"<|tool_call|>{tool_call}<|observe|>{error_result}<|scratch|>Tool failed with timeout. Retrying with exponential backoff...<|tool_call|>{tool_call}<|observe|>{success_result}\n\nSuccessfully retrieved the data after retry."}
        ]
    }

def generate_with_cerebras():
    """Use Cerebras API to generate more diverse samples."""
    prompt = """Generate exactly 3 JSON training samples for an AI agent model. Each sample must be a valid JSON object with this exact structure:
{"type": "tool_single", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Rules:
- Types can be: "instruction", "tool_single", "agent_multi", "recovery"
- For tool calls, use format: <|tool_call|>{"name": "tool_name", "args": {...}}<|observe|>{"result": "..."}
- For multi-step, include <|plan|> at start
- For recovery, include <|scratch|> for reasoning
- Available tools: web_search, read_file, write_file, run_python, get_weather, search_arxiv, fetch_abstract, execute_sql, send_email, git_commit, list_directory, get_stock_price, translate, summarize
- Make queries realistic and varied
- Output ONLY valid JSON array, no markdown, no explanation"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3.1-8b",
            max_completion_tokens=4096,
            temperature=0.7,
        )
        content = response.choices[0].message.content
        # Try to parse JSON array
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        samples = json.loads(content)
        return samples if isinstance(samples, list) else []
    except Exception as e:
        print(f"Cerebras generation error: {e}")
        return []

def main():
    print("=" * 60)
    print("AgentMind — Legacy Synthetic Generator (Cerebras)")
    print("=" * 60)
    print("NOTE: This script is DEPRECATED. Use generate_scaled_synthetic.py")
    print("for apprenticeship-compatible per-domain data with adversarial modes.\n")

    os.makedirs("data", exist_ok=True)
    output_path = "data/synthetic_agents.jsonl"

    print("Generating synthetic agent training data (legacy format)...")

    samples = []

    print("[1/4] Generating instruction samples...")
    for _ in range(500):
        samples.append(generate_instruction())

    print("[2/4] Generating single tool call samples...")
    for _ in range(500):
        samples.append(generate_single_tool())

    print("[3/4] Generating multi-step agent samples...")
    for _ in range(500):
        samples.append(generate_multi_step())

    print("[4/4] Generating failure recovery samples...")
    for _ in range(200):
        samples.append(generate_recovery())

    print("\nTrying Cerebras API for additional diverse samples...")
    cerebras_samples = generate_with_cerebras()
    if cerebras_samples:
        print(f"  → Got {len(cerebras_samples)} samples from Cerebras")
        samples.extend(cerebras_samples)
    else:
        print("  → Cerebras generation skipped (using rule-based only)")

    random.shuffle(samples)

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nGenerated {len(samples)} samples → {output_path}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"\nThis data is in legacy format (no domain field).")
    print(f"For apprenticeship training, use data/apprentice_*.jsonl from:")
    print(f"  python generate_scaled_synthetic.py")

if __name__ == "__main__":
    main()
