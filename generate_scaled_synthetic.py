"""
Generate 10K+ synthetic agent training samples at scale.
Uses template-based generation for bulk + Cerebras API for diverse samples.
Respects Cerebras rate limit: 40 req/min (1.5s delay between requests).

Usage: python generate_scaled_synthetic.py
Output: data/scaled_synthetic.jsonl
"""

import os
import json
import random
import time
from cerebras.cloud.sdk import Cerebras

CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "csk-c2fe5k5hk529dxdd3f4px8m8tct8dmfec8m689cmkp2pwr6m")
client = Cerebras(api_key=CEREBRAS_API_KEY)

os.makedirs("data", exist_ok=True)

# ── Tool Registry ────────────────────────────────────────────
TOOLS = {
    "web_search": {"args": {"query": "current AI developments"}, "result": {"results": [{"title": "Advances in AI", "url": "https://example.com"}]}},
    "read_file": {"args": {"path": "/src/main.py"}, "result": {"content": "import mlx.core as mx\n\ndef main():\n    print('Hello')"}},
    "write_file": {"args": {"path": "/output/report.md", "content": "# Report\nKey findings..."}, "result": {"success": True, "bytes": 1024}},
    "run_python": {"args": {"code": "import numpy as np; print(np.mean([1,2,3]))"}, "result": {"stdout": "2.0\n", "stderr": ""}},
    "get_weather": {"args": {"city": "San Francisco"}, "result": {"temp": 18, "condition": "cloudy"}},
    "search_arxiv": {"args": {"query": "state space models", "days": 7}, "result": {"results": [{"id": "2405.12345", "title": "Mamba-2: Efficient State Space Models"}]}},
    "fetch_abstract": {"args": {"id": "2405.12345"}, "result": {"abstract": "We present Mamba-2, a structured state space model that achieves..."}},
    "execute_sql": {"args": {"query": "SELECT COUNT(*) FROM users"}, "result": {"rows": [{"COUNT(*)": 1542}]}},
    "send_email": {"args": {"to": "team@company.com", "subject": "Update", "body": "Progress report..."}, "result": {"success": True, "message_id": "msg_abc123"}},
    "git_commit": {"args": {"message": "fix: resolve memory leak"}, "result": {"success": True, "commit_hash": "a1b2c3d"}},
    "list_directory": {"args": {"path": "/src"}, "result": {"files": ["main.py", "utils.py", "config.py"]}},
    "get_stock_price": {"args": {"ticker": "AAPL"}, "result": {"price": 189.50, "change": "+2.3%"}},
    "translate": {"args": {"text": "Hello world", "target_lang": "es"}, "result": {"translated": "Hola mundo"}},
    "summarize": {"args": {"text": "Long article text..."}, "result": {"summary": "Key points: 1... 2... 3..."}}
}

# ── Query Templates ─────────────────────────────────────────
USER_QUERIES = [
    "Search for the latest papers on {topic} and summarize the key findings.",
    "Find all Python files in the project and count total lines of code.",
    "Check the weather in {city1}, {city2}, and {city3} and compare them.",
    "Run the test suite and fix any failures you find.",
    "Search arxiv for papers about {topic} published in the last week.",
    "Read the {filename} file and create a summary of its contents.",
    "Execute this Python script and tell me the output.",
    "Get the stock price of {ticker} and compare it with last month.",
    "Search the web for current best practices in {topic}.",
    "Write a Python function that calculates {task} efficiently.",
    "Translate the following text to {language}: {text}",
    "Summarize this article in 3 bullet points: {article}",
    "Find all TODO comments in the codebase and create a task list.",
    "Search for recent news about {topic}.",
    "Run a SQL query to find {query_target}.",
    "Fetch the abstract of paper {paper_id} and explain it simply.",
    "Commit the current changes with message: {commit_msg}",
    "List all files in the {directory} directory.",
    "Send an email to {email} with subject '{subject}' and body '{body}'.",
]

TOPICS = ["AI agents", "SSMs", "multi-agent systems", "LLM fine-tuning", "reinforcement learning", "transformer architectures", "state space models", "retrieval augmented generation"]
CITIES = ["Tokyo", "Pune", "San Francisco", "London", "Berlin", "Sydney", "Mumbai", "New York", "Paris", "Singapore"]
TICKERS = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA", "META", "AMZN"]
LANGUAGES = ["Spanish", "French", "German", "Japanese", "Chinese", "Hindi"]
FILENAMES = ["README.md", "main.py", "config.yaml", "requirements.txt", "setup.py", "Dockerfile"]

# ── Response Templates ──────────────────────────────────────
ASSISTANT_RESPONSES = [
    "Based on my analysis, here are the key findings:\n\n1. The research shows significant progress in this area\n2. Recent developments indicate promising directions\n3. The implications are far-reaching for the field",
    "I've processed your request. Here's what I found:\n\nThe information shows interesting patterns and trends. Let me know if you need more details.",
    "Here's a comprehensive response:\n\n• Point one: The primary finding relates to efficiency improvements\n• Point two: Secondary effects include better scalability\n• Point three: Future work should focus on optimization",
    "Task completed successfully. The results have been compiled and are ready for review.",
    "I've gathered all the requested information. Here's a summary of the findings.",
]

# ── Sample Generators ───────────────────────────────────────
def generate_instruction():
    """Type 1: Plain instruction following."""
    query = random.choice(USER_QUERIES).format(
        topic=random.choice(TOPICS),
        city1=random.choice(CITIES), city2=random.choice(CITIES), city3=random.choice(CITIES),
        ticker=random.choice(TICKERS), language=random.choice(LANGUAGES),
        text="Hello world", article="Recent developments show...", 
        query_target="top users by activity", paper_id="2405.12345",
        commit_msg="fix: update dependencies", directory="/src",
        email="team@company.com", subject="Update", body="Progress report",
        filename=random.choice(FILENAMES), task="factorial numbers"
    )
    return {
        "type": "instruction",
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": random.choice(ASSISTANT_RESPONSES)}
        ]
    }

def generate_single_tool():
    """Type 2: Single tool call."""
    tool_name = random.choice(list(TOOLS.keys()))
    tool = TOOLS[tool_name]
    tool_call = json.dumps({"name": tool_name, "args": tool["args"]})
    observe = json.dumps(tool["result"])
    
    followup = random.choice(ASSISTANT_RESPONSES)
    return {
        "type": "tool_single",
        "messages": [
            {"role": "user", "content": f"Use {tool_name} to process this request."},
            {"role": "assistant", "content": f"<|tool_call|>{tool_call}<|observe|>{observe}\n{followup}"}
        ]
    }

def generate_multi_step():
    """Type 3: Multi-step agentic trajectory."""
    n_steps = random.randint(2, 5)
    tool_names = random.sample(list(TOOLS.keys()), min(n_steps, len(TOOLS)))
    query = random.choice(USER_QUERIES).format(
        topic=random.choice(TOPICS), city1=random.choice(CITIES), 
        city2=random.choice(CITIES), city3=random.choice(CITIES),
        ticker=random.choice(TICKERS), language=random.choice(LANGUAGES),
        text="Hello world", article="Recent developments...",
        query_target="top users", paper_id="2405.12345",
        commit_msg="feat: add new feature", directory="/src",
        email="user@example.com", subject="Report", body="Summary",
        filename=random.choice(FILENAMES), task="sorting algorithm"
    )
    
    plan_steps = "\n".join(f"{i+1}. Use {t}" for i, t in enumerate(tool_names))
    assistant_content = f"<|plan|>{plan_steps}"
    
    for tool_name in tool_names:
        tool = TOOLS[tool_name]
        tool_call = json.dumps({"name": tool_name, "args": tool["args"]})
        observe = json.dumps(tool["result"])
        assistant_content += f"<|tool_call|>{tool_call}<|observe|>{observe}"
    
    assistant_content += "\n\nBased on all gathered information, here's my comprehensive analysis."
    
    return {
        "type": "agent_multi",
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def generate_recovery():
    """Type 4: Failure recovery."""
    tool_name = random.choice(list(TOOLS.keys()))
    tool = TOOLS[tool_name]
    tool_call = json.dumps({"name": tool_name, "args": tool["args"]})
    
    errors = [
        {"error": "timeout", "retry": True, "message": "Service temporarily unavailable"},
        {"error": "rate_limit", "retry_after": 2, "message": "Too many requests"},
        {"error": "invalid_params", "message": "Missing required field: city"},
        {"error": "network_error", "retry": True, "message": "Connection refused"},
    ]
    error_result = json.dumps(random.choice(errors))
    success_result = json.dumps({"status": "success", "data": f"Results from {tool_name} (retry)"})
    
    return {
        "type": "recovery",
        "messages": [
            {"role": "user", "content": f"Try to {tool_name} with {tool['args']}"},
            {"role": "assistant", "content": f"<|tool_call|>{tool_call}<|observe|>{error_result}<|scratch|>Tool failed. Retrying with exponential backoff...<|tool_call|>{tool_call}<|observe|>{success_result}\n\nSuccessfully retrieved the data after retry."}
        ]
    }

def generate_latent_reasoning():
    """Type 5: Latent reasoning with think_start/think_end."""
    tool_name = random.choice(list(TOOLS.keys()))
    tool = TOOLS[tool_name]
    tool_call = json.dumps({"name": tool_name, "args": tool["args"]})
    observe = json.dumps(tool["result"])
    
    return {
        "type": "latent",
        "messages": [
            {"role": "user", "content": f"Analyze this carefully before responding: Use {tool_name}"},
            {"role": "assistant", "content": f"<|think_start|>I need to consider the best approach...<|think_end|><|tool_call|>{tool_call}<|observe|>{observe}\n\nBased on careful analysis, here are the results."}
        ]
    }

def generate_cerebras_batch(count: int, batch_size: int = 3):
    """Generate diverse samples using Cerebras API with rate limiting."""
    samples = []
    prompts = [
        """Generate exactly 3 JSON training samples for an AI agent. Each must be valid JSON with this structure:
{"type": "tool_single", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Rules:
- Types: "instruction", "tool_single", "agent_multi", "recovery"
- Tool format: <|tool_call|>{"name": "tool_name", "args": {...}}<|observe|>{"result": "..."}
- Multi-step: include <|plan|> at start
- Recovery: include <|scratch|> for reasoning
- Tools: web_search, read_file, write_file, run_python, get_weather, search_arxiv, fetch_abstract, execute_sql, send_email, git_commit, list_directory, get_stock_price, translate, summarize
- Output ONLY valid JSON array, no markdown""",
        
        """Generate exactly 3 complex multi-step agent trajectories as JSON. Structure:
{"type": "agent_multi", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Requirements:
- User asks for something requiring 3-5 tool calls
- Assistant starts with <|plan|> listing steps
- Each tool call: <|tool_call|>{"name": "...", "args": {...}}<|observe|>{"result": "..."}
- End with comprehensive answer
- Use realistic queries about coding, research, data analysis
- Output ONLY valid JSON array""",
        
        """Generate exactly 3 failure recovery scenarios as JSON. Structure:
{"type": "recovery", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Requirements:
- Tool call fails with error in <|observe|>
- Agent uses <|scratch|> to reason about failure
- Agent retries with different approach
- Eventually succeeds
- Tools: web_search, run_python, get_stock_price, search_arxiv, execute_sql
- Output ONLY valid JSON array"""
    ]
    
    generated = 0
    while generated < count:
        prompt = random.choice(prompts)
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3.1-8b",
                max_completion_tokens=4096,
                temperature=0.7,
            )
            content = response.choices[0].message.content
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            batch = json.loads(content)
            if isinstance(batch, list):
                samples.extend(batch)
                generated += len(batch)
                print(f"  → Generated {len(batch)} samples (total: {generated})")
        except Exception as e:
            print(f"  → Error: {e}")
        
        # Rate limit: 40 req/min = 1.5s per request
        if generated < count:
            time.sleep(1.5)
    
    return samples[:count]

# ── Main Generation ─────────────────────────────────────────
def main():
    print("Generating scaled synthetic agent training data...")
    print("Target: 12,000 samples")
    
    samples = []
    
    # Template-based generation (fast, bulk)
    print("\n[1/6] Generating instruction samples (3000)...")
    for _ in range(3000):
        samples.append(generate_instruction())
    
    print("[2/6] Generating single tool call samples (2500)...")
    for _ in range(2500):
        samples.append(generate_single_tool())
    
    print("[3/6] Generating multi-step agent samples (3000)...")
    for _ in range(3000):
        samples.append(generate_multi_step())
    
    print("[4/6] Generating failure recovery samples (2000)...")
    for _ in range(2000):
        samples.append(generate_recovery())
    
    print("[5/6] Generating latent reasoning samples (1000)...")
    for _ in range(1000):
        samples.append(generate_latent_reasoning())
    
    # Cerebras generation (slower, high-quality)
    print("\n[6/6] Generating Cerebras samples (500, rate-limited)...")
    cerebras_samples = generate_cerebras_batch(500)
    samples.extend(cerebras_samples)
    
    # Shuffle and write
    random.shuffle(samples)
    
    output_path = "data/scaled_synthetic.jsonl"
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nGenerated {len(samples)} samples → {output_path}")
    print(f"Size: {size_mb:.1f} MB")
    
    # Type distribution
    types = {}
    for s in samples:
        t = s.get("type", "unknown")
        types[t] = types.get(t, 0) + 1
    print("\nType distribution:")
    for t, c in sorted(types.items()):
        print(f"  {t}: {c}")

if __name__ == "__main__":
    main()
