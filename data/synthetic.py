"""
Tool call trajectory generator for AgentMind training data.
Produces diverse synthetic agent trajectories with realistic patterns.

Usage: 
    from data.synthetic import generate_dataset
    generate_dataset(5000, "data/synthetic_agents.jsonl")
"""

import json
import random
from typing import Callable

# ── Tool Registry ────────────────────────────────────────────
SYNTHETIC_TOOLS = {
    "web_search": {
        "description": "Search the web for current information",
        "args": {"query": str},
        "mock_result": lambda args: {"results": [{"title": f"Result for {args['query']}", "url": "https://example.com"}]}
    },
    "read_file": {
        "description": "Read a file from disk",
        "args": {"path": str},
        "mock_result": lambda args: {"content": f"<file content of {args['path']}>"}
    },
    "write_file": {
        "description": "Write content to a file",
        "args": {"path": str, "content": str},
        "mock_result": lambda args: {"success": True, "bytes": len(args["content"])}
    },
    "run_python": {
        "description": "Execute Python code",
        "args": {"code": str},
        "mock_result": lambda args: {"stdout": "42\n", "stderr": ""}
    },
    "get_weather": {
        "description": "Get current weather for a city",
        "args": {"city": str},
        "mock_result": lambda args: {"temp": random.randint(20, 40), "condition": "sunny"}
    },
    "search_arxiv": {
        "description": "Search arxiv for papers",
        "args": {"query": str, "days": int},
        "mock_result": lambda args: {"results": [{"id": "2405.12345", "title": f"Paper about {args['query']}"}]}
    },
    "fetch_abstract": {
        "description": "Fetch paper abstract by ID",
        "args": {"id": str},
        "mock_result": lambda args: {"abstract": f"Abstract for paper {args['id']}..."}
    },
    "execute_sql": {
        "description": "Execute SQL query",
        "args": {"query": str},
        "mock_result": lambda args: {"rows": [{"result": "data"}]}
    },
    "send_email": {
        "description": "Send an email",
        "args": {"to": str, "subject": str, "body": str},
        "mock_result": lambda args: {"success": True, "message_id": "msg_abc123"}
    },
    "git_commit": {
        "description": "Commit changes to git",
        "args": {"message": str},
        "mock_result": lambda args: {"success": True, "commit_hash": "a1b2c3d"}
    },
    "list_directory": {
        "description": "List files in directory",
        "args": {"path": str},
        "mock_result": lambda args: {"files": ["main.py", "utils.py", "config.py"]}
    },
    "get_stock_price": {
        "description": "Get current stock price",
        "args": {"ticker": str},
        "mock_result": lambda args: {"price": random.uniform(100, 500), "change": "+2.3%"}
    },
    "translate": {
        "description": "Translate text to another language",
        "args": {"text": str, "target_lang": str},
        "mock_result": lambda args: {"translated": f"Translated text in {args['target_lang']}"}
    },
    "summarize": {
        "description": "Summarize long text",
        "args": {"text": str},
        "mock_result": lambda args: {"summary": "Key points: 1... 2... 3..."}
    },
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
    "Summarize this article in 3 bullet points.",
    "Find all TODO comments in the codebase and create a task list.",
    "Search for recent news about {topic}.",
    "Run a SQL query to find the top 10 users by activity.",
    "Fetch the abstract of paper {paper_id} and explain it in simple terms.",
    "Commit the current changes with message: {commit_msg}",
    "List all files in the {directory} directory.",
    "Send an email to {email} with subject '{subject}' and body '{body}'.",
]

TOPICS = ["AI agents", "SSMs", "multi-agent systems", "LLM fine-tuning", "reinforcement learning", "transformer architectures"]
CITIES = ["Tokyo", "Pune", "San Francisco", "London", "Berlin", "Sydney", "Mumbai", "New York"]
TICKERS = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA", "META", "AMZN"]
LANGUAGES = ["Spanish", "French", "German", "Japanese", "Chinese", "Hindi"]
FILENAMES = ["README.md", "main.py", "config.yaml", "requirements.txt", "setup.py", "Dockerfile"]

# ── Response Templates ──────────────────────────────────────
ASSISTANT_RESPONSES = [
    "Based on my analysis, here are the key findings:\n\n1. The research shows significant progress\n2. Recent developments indicate promising directions\n3. The implications are far-reaching",
    "I've processed your request. Here's what I found:\n\nThe information shows interesting patterns. Let me know if you need more details.",
    "Here's a comprehensive response:\n\n• Point one: Primary finding relates to efficiency\n• Point two: Secondary effects include better scalability\n• Point three: Future work should focus on optimization",
    "Task completed successfully. The results have been compiled and are ready for review.",
    "I've gathered all the requested information. Here's a summary of the findings.",
]

# ── Sample Generators ───────────────────────────────────────
def generate_trajectory(n_steps: int = 3, inject_failure: bool = False) -> dict:
    """Generate a synthetic multi-step agentic trajectory."""
    tools = random.sample(list(SYNTHETIC_TOOLS.keys()), min(n_steps, len(SYNTHETIC_TOOLS)))
    query = random.choice(USER_QUERIES).format(
        topic=random.choice(TOPICS),
        city1=random.choice(CITIES), city2=random.choice(CITIES), city3=random.choice(CITIES),
        ticker=random.choice(TICKERS), language=random.choice(LANGUAGES),
        text="Hello world", filename=random.choice(FILENAMES),
        task="factorial numbers", paper_id="2405.12345",
        commit_msg="fix: update dependencies", directory="/src",
        email="team@company.com", subject="Update", body="Progress report"
    )

    # Build plan
    plan_steps = "\n".join(f"{i+1}. Use {t}" for i, t in enumerate(tools))
    assistant_content = f"<|plan|>{plan_steps}"

    for i, tool_name in enumerate(tools):
        tool = SYNTHETIC_TOOLS[tool_name]
        mock_args = {k: f"example_{k}" for k in tool["args"].keys()}
        call = json.dumps({"name": tool_name, "args": mock_args})

        if inject_failure and i == 0:
            error_result = json.dumps({"error": "timeout", "retry": True})
            recovery_result = json.dumps(tool["mock_result"](mock_args))
            assistant_content += (
                f"<|tool_call|>{call}"
                f"<|observe|>{error_result}"
                f"<|scratch|>Tool failed. Retrying with fallback."
                f"<|tool_call|>{call}"
                f"<|observe|>{recovery_result}"
            )
        else:
            result = json.dumps(tool["mock_result"](mock_args))
            assistant_content += f"<|tool_call|>{call}<|observe|>{result}"

    assistant_content += "\nTask complete based on gathered information."

    return {
        "type": "agent_multi" if not inject_failure else "recovery",
        "messages": [
            {"role": "user",      "content": query},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def generate_instruction() -> dict:
    """Generate simple instruction-following sample."""
    query = random.choice(USER_QUERIES).format(
        topic=random.choice(TOPICS), city1=random.choice(CITIES),
        city2=random.choice(CITIES), city3=random.choice(CITIES),
        ticker=random.choice(TICKERS), language=random.choice(LANGUAGES),
        text="Hello world", filename=random.choice(FILENAMES),
        task="sorting algorithm", paper_id="2405.12345",
        commit_msg="feat: add feature", directory="/src",
        email="user@example.com", subject="Report", body="Summary"
    )
    return {
        "type": "instruction",
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": random.choice(ASSISTANT_RESPONSES)}
        ]
    }

def generate_single_tool() -> dict:
    """Generate single tool call sample."""
    tool_name = random.choice(list(SYNTHETIC_TOOLS.keys()))
    tool = SYNTHETIC_TOOLS[tool_name]
    mock_args = {k: f"example_{k}" for k in tool["args"].keys()}
    call = json.dumps({"name": tool_name, "args": mock_args})
    result = json.dumps(tool["mock_result"](mock_args))
    
    return {
        "type": "tool_single",
        "messages": [
            {"role": "user", "content": f"Use {tool_name} to process this request."},
            {"role": "assistant", "content": f"<|tool_call|>{call}<|observe|>{result}\n{random.choice(ASSISTANT_RESPONSES)}"}
        ]
    }

def generate_dataset(n_samples: int, output_path: str):
    """Generate a complete synthetic dataset with balanced types."""
    samples = []
    
    # Distribution: 20% instruction, 25% single tool, 35% multi-step, 20% recovery
    n_instruction = int(n_samples * 0.20)
    n_single = int(n_samples * 0.25)
    n_multi = int(n_samples * 0.35)
    n_recovery = n_samples - n_instruction - n_single - n_multi
    
    for _ in range(n_instruction):
        samples.append(generate_instruction())
    
    for _ in range(n_single):
        samples.append(generate_single_tool())
    
    for _ in range(n_multi):
        samples.append(generate_trajectory(n_steps=random.randint(2, 5)))
    
    for _ in range(n_recovery):
        samples.append(generate_trajectory(n_steps=random.randint(2, 4), inject_failure=True))
    
    random.shuffle(samples)
    
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Generated {n_samples} samples → {output_path}")

# Usage:
# generate_dataset(5000, "data/synthetic_agents.jsonl")
