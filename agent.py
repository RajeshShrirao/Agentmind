"""
AgentMind AgentLoop — Router dispatch, adapter swapping, SSM state persistence.

Usage:
  python agent.py --backbone ./checkpoints/backbone --adapters ./checkpoints/adapters \\
      --router ./checkpoints/router --query "Search arxiv for SSM papers"
"""

import json, os, subprocess, argparse, urllib.request, urllib.parse
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from config import AgentMindConfig
from model.agent_lm import AgentMind
from lora import apply_lora
from init import init_agentmind
from router import TaskRouter
from tokenizer_setup import load_tokenizer, get_token_ids, hydrate_config

SYSTEM_PROMPT = """You are an AI assistant with access to tools. Follow this protocol:
1. Think step by step if needed using <|think_start|>...<|think_end|>
2. To use a tool, emit: <|tool_call|>{"name": "tool_name", "args": {...}}
3. The tool result will follow as: <|observe|>{"result": ...}
4. Continue the conversation naturally after observing the result

Available tools:
- web_search(query: str): Search the web for current information
- run_python(code: str): Execute Python code with a 10-second timeout
- read_file(path: str): Read a file from disk (max 10KB)"""


def _top_p_sampling(logits, temp=0.7, top_p=0.9):
    if temp > 0:
        logits = logits / temp
    probs = mx.softmax(logits, axis=-1)
    sorted_probs = mx.sort(probs)[..., ::-1]
    cumsum = mx.cumsum(sorted_probs, axis=-1)
    cutoff = cumsum > top_p
    cutoff[..., 1:] = cutoff[..., :-1]
    cutoff[..., 0] = False
    probs[cutoff] = 0.0
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return mx.random.categorical(mx.log(probs + 1e-10)).item()


# ── Tool implementations ────────────────────────────────

def _web_search(query: str) -> dict:
    """Search the web via DuckDuckGo lite HTML API."""
    try:
        url = "https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote(query)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        results = []
        for line in html.split("\n"):
            if 'class="result__a"' in line or 'class="result__snippet"' in line:
                results.append(line.strip())
        return {"results": results[:10], "count": len(results), "query": query}
    except Exception as e:
        return {"error": str(e), "query": query}


def _run_python(code: str) -> dict:
    """Execute Python code with a 10-second timeout."""
    try:
        proc = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=10,
        )
        return {
            "stdout": proc.stdout[:5000],
            "stderr": proc.stderr[:2000],
            "return_code": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": "timeout after 10s", "stdout": "", "stderr": "Execution timed out"}
    except Exception as e:
        return {"error": str(e)}


def _read_file(path: str) -> dict:
    """Read a file from disk (max 10KB)."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return {"error": f"File not found: {path}"}
        if not p.is_file():
            return {"error": f"Not a file: {path}"}
        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > 10_000:
            content = content[:10_000] + "\n... (truncated at 10KB)"
        return {"content": content, "path": str(p), "size_bytes": len(content)}
    except Exception as e:
        return {"error": str(e)}


DEFAULT_TOOLS = {
    "web_search": _web_search,
    "run_python": _run_python,
    "read_file": _read_file,
}

TOOL_CALL_BOUNDARY_TOKENS = {"<|observe|>", "<|end|>", "<eos>"}


# ── AgentLoop ────────────────────────────────────────────

class AgentLoop:
    def __init__(self, backbone, router, adapters: dict, tok, tools: dict, cfg):
        self.backbone = backbone
        self.router = router
        self.adapters = adapters
        self.tok = tok
        self.tools = tools
        self.cfg = cfg
        self.h_states = {}
        self.active_adapter = None

    def _select_specialist(self, hidden_state):
        logits = self.router(hidden_state[:, -1:, :])
        return self.router.select_expert(hidden_state[:, -1:, :], threshold=0.6)

    def _load_adapter(self, name):
        if self.active_adapter != name:
            if name in self.adapters:
                self.backbone.load_lora(self.adapters[name])
            self.active_adapter = name

    def _handle_tool_call(self, text: str):
        """Parse and execute a tool call embedded in generated text."""
        idx = text.find("<|tool_call|>")
        if idx == -1:
            return None
        json_str = text[idx + len("<|tool_call|>"):]
        for boundary in TOOL_CALL_BOUNDARY_TOKENS:
            if boundary in json_str:
                json_str = json_str.split(boundary)[0]
        json_str = json_str.strip()
        if not json_str:
            return None
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError:
            return {"error": f"failed to parse tool call JSON: {json_str[:200]}"}
        name = obj.get("name", "")
        args = obj.get("args", {})
        if name in self.tools:
            try:
                result = self.tools[name](**args)
                return {"result": result, "tool": name, "args": args}
            except Exception as e:
                return {"error": str(e), "tool": name, "args": args}
        return {"error": f"unknown tool: {name}"}

    def run(self, user_query, max_tokens=200, temp=0.7, top_p=0.9):
        prompt = f"<|system|>{SYSTEM_PROMPT}<|user|>{user_query}<|assistant|>"
        ids = mx.array([self.tok.encode(prompt)])

        logits, self.h_states = self.backbone.forward_with_state(ids, self.h_states)
        if hasattr(self.backbone, "last_hidden") and self.backbone.last_hidden is not None:
            specialist = self._select_specialist(self.backbone.last_hidden)
        else:
            specialist = "tool_caller"
        self._load_adapter(specialist)

        output_ids = []
        output_text = ""
        tool_call_emitted = False
        gen_pos = 0

        while gen_pos < max_tokens:
            logits, self.h_states = self.backbone.forward_with_state(ids, self.h_states)
            token = _top_p_sampling(logits[0, -1], temp=temp, top_p=top_p)
            output_ids.append(token)
            gen_pos += 1

            decoded = self.tok.decode([token])
            output_text += decoded
            ids = mx.array([[token]])

            if token == self.cfg.tool_call_id:
                tool_call_emitted = True
                continue

            if tool_call_emitted:
                if token in (self.cfg.observe_id, self.cfg.eos_id):
                    result = self._handle_tool_call(output_text)
                    if result is not None:
                        observe_text = f"<|observe|>{json.dumps(result)}"
                        output_text += observe_text
                        ids = mx.array([self.tok.encode(observe_text)])
                        self.backbone.forward_with_state(ids, self.h_states)
                        ids = mx.array([[self.cfg.eos_id]])
                    tool_call_emitted = False
                    continue
                continue

            if token == self.cfg.eos_id:
                break

        return output_text


# ── CLI ────────────────────────────────────────────────────

def load_adapters(adapters_dir: str) -> dict:
    adapters_dir = Path(adapters_dir)
    adapters = {}
    for f in sorted(adapters_dir.glob("*.safetensors")):
        name = f.stem
        loaded = mx.load(str(f))
        if "metadata" in loaded:
            del loaded["metadata"]
        adapters[name] = loaded
        print(f"  Loaded adapter: {name} ({sum(v.nbytes for v in loaded.values()) // 1024} KB)")
    return adapters


def main():
    parser = argparse.ArgumentParser(description="AgentMind Agent Loop")
    parser.add_argument("--backbone", type=str, default="./checkpoints",
                        help="Path to backbone .safetensors or directory containing backbone.safetensors")
    parser.add_argument("--adapters", type=str, default="./checkpoints/adapters",
                        help="Directory containing specialist adapter .safetensors files")
    parser.add_argument("--router", type=str, default="./checkpoints/router",
                        help="Path to router .safetensors (or dir with router.safetensors)")
    parser.add_argument("--query", type=str, default=None,
                        help="Single query to run (omit for interactive mode)")
    args = parser.parse_args()

    cfg = AgentMindConfig()

    tok = load_tokenizer("agentmind_tok.model")
    ids = get_token_ids(tok)
    hydrate_config(cfg, tok)
    print("Token IDs hydrated.")

    backbone = AgentMind(cfg)
    backbone = init_agentmind(backbone, cfg)
    apply_lora(backbone)

    backbone_path = Path(args.backbone)
    if backbone_path.is_dir():
        backbone_path = backbone_path / "backbone.safetensors"
    if backbone_path.exists():
        weights = mx.load(str(backbone_path))
        backbone.update(weights)
        print(f"Backbone loaded from {backbone_path}")
    else:
        print(f"Warning: backbone not found at {backbone_path}, using random init")

    adapters = load_adapters(args.adapters)

    router_path = Path(args.router)
    if router_path.is_dir():
        router_path = router_path / "router.safetensors"
    router = TaskRouter.load(str(router_path))

    agent = AgentLoop(backbone, router, adapters, tok, DEFAULT_TOOLS, cfg)

    if args.query:
        print(f"\nUser: {args.query}\n")
        response = agent.run(args.query)
        print(f"Assistant: {response}\n")
    else:
        print("\nInteractive mode. Type 'quit' to exit.\n")
        while True:
            try:
                query = input("User: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query or query.lower() in ("quit", "exit"):
                break
            response = agent.run(query)
            print(f"Assistant: {response}\n")


if __name__ == "__main__":
    main()
