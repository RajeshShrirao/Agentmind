"""agent.py — AgentLoop for Qwen2.5-based AgentMind.

Two modes:
  Phase A/B (manual adapter):
    python agent.py --backbone Qwen/Qwen2.5-0.5B --adapter ./tool_caller.safetensors --query "..."

  Phase C (router dispatch):
    python agent.py --backbone Qwen/Qwen2.5-0.5B --adapters ./adapters --router ./router --query "..."
"""

import json, os, subprocess, argparse, urllib.request, urllib.parse
from pathlib import Path

import mlx.core as mx
from mlx_lm import load as load_model
from mlx_lm.models.cache import make_prompt_cache

from lora import apply_lora, load_lora


SYSTEM_PROMPT = """You are an AI assistant with access to tools. Follow this protocol:
1. To use a tool, emit: <|tool_call|>{"name": "tool_name", "args": {...}}
2. The tool result will follow as: <|observe|>{"result": ...}
3. Continue the conversation naturally after observing the result

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


def _web_search(query: str) -> dict:
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


class AgentLoop:
    def __init__(self, model, tokenizer, adapter_path=None, adapters_dir=None, router=None, tools=None):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools or DEFAULT_TOOLS
        self.prompt_cache = make_prompt_cache(model)
        self.eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None

        if adapter_path:
            self.fixed_adapter = self._load_adapter_weights(adapter_path)
            load_lora(model, self.fixed_adapter)
            self.adapter_mode = "fixed"
            self.router = None
            self.adapters = None
            self.active_adapter = "fixed"
            print(f"  Agent mode: fixed adapter ({adapter_path})")
        elif adapters_dir and router:
            self.adapters = self._load_all_adapters(adapters_dir)
            self.router = router
            self.adapter_mode = "routed"
            self.active_adapter = None
            print(f"  Agent mode: routed ({len(self.adapters)} adapters)")
        else:
            raise ValueError("Provide --adapter (fixed) or --adapters + --router (routed)")

    def _load_adapter_weights(self, path):
        loaded = mx.load(str(path))
        if "metadata" in loaded:
            del loaded["metadata"]
        return loaded

    def _load_all_adapters(self, adapters_dir):
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

    def _select_specialist(self, hidden_state):
        return self.router.select_expert(hidden_state, threshold=0.6)

    def _load_adapter(self, name):
        if self.active_adapter != name:
            if name in self.adapters:
                load_lora(self.model, self.adapters[name])
                self.active_adapter = name

    def _handle_tool_call(self, text):
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

    def _forward(self, input_ids):
        return self.model(input_ids, cache=self.prompt_cache)

    def run(self, user_query, max_tokens=200, temp=0.7, top_p=0.9):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if isinstance(prompt, list):
            ids = mx.array([prompt])
        else:
            ids = mx.array([prompt])

        if self.adapter_mode == "routed" and self.router is not None:
            hidden = self.model.model(ids, cache=self.prompt_cache)
            specialist = self._select_specialist(hidden)
            self._load_adapter(specialist)
            print(f"  Router selected: {specialist}")

        output_text = ""
        tool_call_emitted = False
        gen_pos = 0

        logits = self._forward(ids)

        while gen_pos < max_tokens:
            logits_last = logits[:, -1, :]
            token = _top_p_sampling(logits_last, temp=temp, top_p=top_p)
            gen_pos += 1

            decoded = self.tokenizer.decode([token])
            output_text += decoded

            if token == self.eos_token_id:
                break

            if "<|tool_call|>" in decoded:
                tool_call_emitted = True
                logits = self._forward(mx.array([[token]]))
                continue

            if tool_call_emitted:
                if "<|observe|>" in decoded or "<|end|>" in decoded or "<eos>" in decoded:
                    result = self._handle_tool_call(output_text)
                    if result is not None:
                        observe_text = f"<|observe|>{json.dumps(result)}"
                        output_text += observe_text
                        observe_ids = self.tokenizer.encode(observe_text)
                        for tid in observe_ids:
                            logits = self._forward(mx.array([[tid]]))
                    tool_call_emitted = False
                    continue
                logits = self._forward(mx.array([[token]]))
                continue

            logits = self._forward(mx.array([[token]]))

        return output_text


def main():
    parser = argparse.ArgumentParser(description="AgentMind Agent Loop")
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="HuggingFace model ID or path")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to a single adapter .safetensors (Phase A/B)")
    parser.add_argument("--adapters", type=str, default=None,
                        help="Directory containing adapter .safetensors files (Phase C)")
    parser.add_argument("--router", type=str, default=None,
                        help="Path to router .safetensors (Phase C)")
    parser.add_argument("--query", type=str, default=None,
                        help="Single query to run (omit for interactive mode)")
    args = parser.parse_args()

    print(f"Loading backbone: {args.backbone}")
    model, tokenizer = load_model(args.backbone)

    special_tokens = [
        "<|tool_call|>", "<|plan|>", "<|memory|>", "<|scratch|>", "<|observe|>",
        "<|think_start|>", "<|think_end|>", "<|system|>", "<|user|>", "<|assistant|>",
    ]
    tokenizer._tokenizer.add_tokens(special_tokens)
    # Embedding (151936 slots) already has room for these tokens

    model = apply_lora(model)

    router = None
    if args.router:
        from router import TaskRouter
        router = TaskRouter.load(str(args.router))

    agent = AgentLoop(
        model, tokenizer,
        adapter_path=args.adapter,
        adapters_dir=args.adapters,
        router=router,
    )

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
