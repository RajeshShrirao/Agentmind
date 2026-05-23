"""
Layer 6 — Graph-based Task Generation.

Uses NetworkX to generate dependency DAGs → linearized multi-tool trajectories.
"""

import random
import json
import networkx as nx
from .core import TOOL_NAMES, TOOL_DEFS

# ── Tool category clusters ──────────────────────────────────────

RESEARCH_CLUSTER = ["web_search", "search_arxiv", "fetch_abstract", "summarize"]
CODE_CLUSTER = ["read_file", "write_file", "run_python", "execute_sql", "git_commit", "list_directory"]
CONSUMER_CLUSTER = ["get_weather", "get_stock_price", "translate", "web_search"]
COMMS_CLUSTER = ["send_email", "summarize"]

# ── Example DAG templates ───────────────────────────────────────

TEMPLATES = {
    "research_report": {
        "cluster": RESEARCH_CLUSTER,
        "edges": [
            ("web_search", "search_arxiv"),
            ("search_arxiv", "fetch_abstract"),
            ("fetch_abstract", "summarize"),
        ],
        "query_template": "Research {topic} and summarize findings",
    },
    "code_verify_commit": {
        "cluster": CODE_CLUSTER,
        "edges": [
            ("list_directory", "read_file"),
            ("read_file", "run_python"),
            ("run_python", "git_commit"),
        ],
        "query_template": "Check {project}, run tests, and commit fixes",
    },
    "weather_email": {
        "cluster": ["get_weather", "translate", "send_email"],
        "edges": [
            ("get_weather", "translate"),
            ("translate", "send_email"),
        ],
        "query_template": "Get weather in {city}, translate, and email it",
    },
    "stock_research": {
        "cluster": ["get_stock_price", "web_search", "summarize"],
        "edges": [
            ("get_stock_price", "web_search"),
            ("web_search", "summarize"),
        ],
        "query_template": "Check {ticker} stock and research market news",
    },
    "data_pipeline": {
        "cluster": ["execute_sql", "run_python", "write_file", "git_commit"],
        "edges": [
            ("execute_sql", "run_python"),
            ("run_python", "write_file"),
            ("write_file", "git_commit"),
        ],
        "query_template": "Query database, analyze with Python, save results, and commit",
    },
    "deep_research": {
        "cluster": RESEARCH_CLUSTER + COMMS_CLUSTER,
        "edges": [
            ("web_search", "web_search"),  # parallel searches
            ("web_search", "search_arxiv"),
            ("web_search", "fetch_abstract"),
            ("search_arxiv", "summarize"),
            ("fetch_abstract", "summarize"),
            ("summarize", "send_email"),
        ],
        "query_template": "Deep research on {topic}, compile findings, email report",
    },
    "full_stack_query": {
        "cluster": ["list_directory", "read_file", "execute_sql", "run_python", "summarize", "send_email"],
        "edges": [
            ("list_directory", "read_file"),
            ("read_file", "execute_sql"),
            ("execute_sql", "run_python"),
            ("run_python", "summarize"),
            ("summarize", "send_email"),
        ],
        "query_template": "Audit {project}, query database, analyze, summarize, and email",
    },
}

TOPICS = [
    "Mamba SSM", "RLHF", "multi-modal learning", "quantum ML",
    "transformer alternatives", "diffusion models", "sparse attention",
    "knowledge distillation", "retrieval augmented generation",
]

CITIES = ["Tokyo", "London", "Paris", "Berlin", "Sydney", "Mumbai", "Seoul", "Toronto"]
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "META", "TSLA"]
PROJECTS = ["agentmind", "hyperframes", "mamba", "loRA", "cognitive-app"]
EMAILS = ["team@company.com", "user@example.com", "admin@service.com", "researcher@lab.org"]


class GraphGenerator:
    """Layer 6 — Generate entirely new multi-tool trajectories from DAGs."""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def generate_batch(self, n_samples=50):
        """Generate n_samples of graph-based multi-tool trajectories."""
        results = []
        for _ in range(n_samples):
            sample = self._generate_one()
            if sample:
                results.append(sample)
        return results

    def _generate_one(self):
        """Generate one multi-tool trajectory from a random template."""
        template_name = self.rng.choice(list(TEMPLATES.keys()))
        tmpl = TEMPLATES[template_name]

        # Build DAG
        G = nx.DiGraph()
        for src, dst in tmpl["edges"]:
            if src == dst:
                # Self-loop = multiple calls to same tool (parallel-ish)
                continue
            G.add_edge(src, dst)

        # Handle self-loop edges (multi-call same tool)
        multi_calls = {}
        for src, dst in tmpl["edges"]:
            if src == dst:
                multi_calls[src] = multi_calls.get(src, 0) + 1

        # Topological sort for execution order
        try:
            order = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            return None

        # Fill query template
        topic = self.rng.choice(TOPICS)
        city = self.rng.choice(CITIES)
        ticker = self.rng.choice(TICKERS)
        project = self.rng.choice(PROJECTS)
        email = self.rng.choice(EMAILS)

        query = tmpl["query_template"].format(
            topic=topic, city=city, ticker=ticker, project=project, email=email
        )

        # Build assistant trajectory
        plan_parts = []
        tool_calls = []

        for tool in order:
            count = multi_calls.get(tool, 0) + 1
            for i in range(count):
                args = self._gen_args(tool, topic=topic, city=city, ticker=ticker, project=project, email=email)
                call_json = json.dumps({"name": tool, "args": args})
                observe = self._gen_observe(tool, args)
                desc = {"web_search": "Search web", "search_arxiv": "Search arxiv", "fetch_abstract": "Get abstract",
                         "summarize": "Summarize", "read_file": "Read file", "write_file": "Write file",
                         "run_python": "Run code", "execute_sql": "Query DB", "git_commit": "Commit",
                         "list_directory": "List dir", "send_email": "Send email", "get_weather": "Get weather",
                         "get_stock_price": "Get stock price", "translate": "Translate"}.get(tool, tool)
                if count > 1:
                    desc += f" ({i+1}/{count})"
                plan_parts.append(f"  {len(plan_parts)+1}. {desc}")
                tool_calls.append((call_json, json.dumps(observe)))

        plan_text = f"<|plan|>{chr(10).join(plan_parts)}"
        asst_parts = [plan_text]
        for call_str, observe_str in tool_calls:
            asst_parts.append(f"<|tool_call|>{call_str}<|observe|>{observe_str}")
        asst_parts.append("\nDone. Results above.")

        return {
            "domain": "tool_caller",
            "type": "tool_multi",
            "messages": [
                {"role": "user", "content": query},
                {"role": "assistant", "content": "".join(asst_parts)},
            ]
        }

    def _gen_args(self, tool, **ctx):
        if tool == "web_search":
            return {"query": ctx.get("topic", "AI"), "max_results": self.rng.choice([5, 10])}
        elif tool == "search_arxiv":
            return {"query": ctx.get("topic", "machine learning"), "days": self.rng.choice([7, 30])}
        elif tool == "fetch_abstract":
            return {"id": f"24{self.rng.randint(1,9)}.{self.rng.randint(10000,99999)}"}
        elif tool == "summarize":
            return {"text": f"Research findings on {ctx.get('topic', 'AI')} show promising results in recent studies."}
        elif tool == "read_file":
            return {"path": f"/home/user/{ctx.get('project', 'project')}/main.py"}
        elif tool == "write_file":
            return {"path": f"/home/user/{ctx.get('project', 'project')}/report.md", "content": "# Analysis Results\n\nKey findings documented."}
        elif tool == "run_python":
            return {"code": "import numpy as np; print(np.mean([1,2,3,4,5]))"}
        elif tool == "execute_sql":
            return {"query": "SELECT COUNT(*) FROM users"}
        elif tool == "git_commit":
            return {"message": f"feat: update {ctx.get('project', 'project')} analysis"}
        elif tool == "list_directory":
            return {"path": f"/home/user/{ctx.get('project', 'project')}"}
        elif tool == "send_email":
            return {"to": ctx.get("email", "user@example.com"), "subject": f"Report on {ctx.get('topic', 'research')}", "body": f"Please find the {ctx.get('topic', 'research')} analysis attached."}
        elif tool == "get_weather":
            return {"city": ctx.get("city", "Tokyo")}
        elif tool == "get_stock_price":
            return {"ticker": ctx.get("ticker", "AAPL")}
        elif tool == "translate":
            return {"text": "Hello world", "target_lang": self.rng.choice(["es", "fr", "de", "ja"])}
        return {}

    def _gen_observe(self, tool, args):
        """Generate plausible observation."""
        if tool == "web_search":
            return {"results": [{"title": f"Result: {args.get('query', 'topic')}", "url": "https://arxiv.org/abs/2405.12345"}]}
        elif tool == "search_arxiv":
            return {"results": [{"id": "2405.12345", "title": f"Advances in {args.get('query', 'ML')}"}]}
        elif tool == "fetch_abstract":
            return {"abstract": "We present a novel architecture achieving state-of-the-art results on multiple benchmarks."}
        elif tool == "summarize":
            return {"summary": f"Key findings on {args.get('text', 'research')} summarized."}
        elif tool in ("read_file", "write_file"):
            return {"success": True} if tool == "write_file" else {"content": "import mlx\n\ndef main():\n    pass"}
        elif tool == "run_python":
            return {"stdout": "3.0\n", "stderr": ""}
        elif tool == "execute_sql":
            return {"rows": [{"count": 15234}]}
        elif tool == "git_commit":
            return {"success": True, "hash": "a1b2c3d"}
        elif tool == "list_directory":
            return {"files": ["main.py", "tests", "README.md", "config.json"]}
        elif tool == "send_email":
            return {"success": True}
        elif tool == "get_weather":
            return {"temp": 22, "condition": "partly cloudy"}
        elif tool == "get_stock_price":
            return {"price": 178.50, "change": "+1.2%"}
        elif tool == "translate":
            return {"translated": "Hola mundo"}
        return {}
