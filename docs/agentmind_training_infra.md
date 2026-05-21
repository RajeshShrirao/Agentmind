# AgentMind — Training Infrastructure
> Everything NOT in `agentmind_architecture.md`
> Data · Init · RoPE · LoRA · Scheduler · Full Train Loop · MTP · Latent Reasoning · Eval · Export

---

## File Map

```
agentmind/
├── data/
│   ├── pipeline.py          # dataset loading, formatting, batching
│   ├── synthetic.py         # tool call trajectory generator
│   └── formats.py           # JSONL schema definitions
├── model/
│   ├── rope.py              # RoPE for attention blocks
│   ├── conv_state.py        # conv buffer for single-step inference
│   ├── mtp_head.py          # Multi-Token Prediction auxiliary head
│   └── latent.py            # think_start / think_end training logic
├── lora.py                  # LoRALinear + model wrapping
├── init.py                  # Mamba-specific weight initialization
├── scheduler.py             # cosine LR with warmup
├── train.py                 # COMPLETE training loop
├── eval.py                  # perplexity + tool call accuracy
└── export.py                # GGUF export with custom arch map
```

---

## 1. Data Pipeline

### `data/formats.py` — JSONL Schema

Every training sample is one of four types. All share the same JSONL line format.

```python
# TYPE 1 — Plain instruction following
{
    "type": "instruction",
    "messages": [
        {"role": "system", "content": "You are a precise assistant."},
        {"role": "user",   "content": "Summarize this in 3 bullets: ..."},
        {"role": "assistant", "content": "• Point one\n• Point two\n• Point three"}
    ]
}

# TYPE 2 — Single tool call
{
    "type": "tool_single",
    "messages": [
        {"role": "user", "content": "What is the weather in Pune?"},
        {"role": "assistant", "content": "<|tool_call|>{\"name\": \"get_weather\", \"args\": {\"city\": \"Pune\"}}<|observe|>{\"temp\": 34, \"condition\": \"sunny\"}The weather in Pune is 34°C and sunny."}
    ]
}

# TYPE 3 — Multi-step agentic trajectory
{
    "type": "agent_multi",
    "messages": [
        {"role": "user", "content": "Find the top AI paper from last week and summarize it."},
        {"role": "assistant", "content": "<|plan|>1. Search arxiv\n2. Fetch abstract\n3. Summarize<|tool_call|>{\"name\": \"search_arxiv\", \"args\": {\"query\": \"AI\", \"days\": 7}}<|observe|>{\"results\": [{\"id\": \"2405.1234\", \"title\": \"Mamba-2\"}]}<|tool_call|>{\"name\": \"fetch_abstract\", \"args\": {\"id\": \"2405.1234\"}}<|observe|>{\"abstract\": \"...\"}\nMamba-2 introduces structured state spaces..."}
    ]
}

# TYPE 4 — Failure recovery
{
    "type": "recovery",
    "messages": [
        {"role": "user", "content": "Get stock price of NVDA"},
        {"role": "assistant", "content": "<|tool_call|>{\"name\": \"get_stock\", \"args\": {\"ticker\": \"NVDA\"}}<|observe|>{\"error\": \"rate_limit\", \"retry_after\": 2}<|scratch|>Tool failed. Retry with backoff.<|tool_call|>{\"name\": \"get_stock\", \"args\": {\"ticker\": \"NVDA\", \"source\": \"backup\"}}<|observe|>{\"price\": 1024.5}NVDA is trading at $1024.50."}
    ]
}
```

### `data/pipeline.py` — Dataset + DataLoader

```python
import json
import random
import mlx.core as mx
from pathlib import Path
from typing import Iterator

class AgentDataset:
    def __init__(self, paths: list[str], tokenizer, cfg, split="train"):
        self.cfg = cfg
        self.tok = tokenizer
        self.samples = []

        for path in paths:
            with open(path) as f:
                for line in f:
                    self.samples.append(json.loads(line.strip()))

        # Data mixing weights by type
        self.weights = {
            "instruction": 0.30,
            "tool_single":  0.30,
            "agent_multi":  0.25,
            "recovery":     0.15,
        }

        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * 0.95)
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def _format_sample(self, sample: dict) -> str:
        """Convert message list to flat token string."""
        text = ""
        for msg in sample["messages"]:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"<|system|>{content}"
            elif role == "user":
                text += f"<|user|>{content}"
            elif role == "assistant":
                text += f"<|assistant|>{content}<eos>"
        return text

    def _tokenize(self, text: str) -> list[int]:
        return self.tok.encode(text, add_bos=True)

    def _make_labels(self, ids: list[int], sample: dict) -> list[int]:
        """
        Only compute loss on assistant turns.
        Mask system + user tokens with -100 (ignored in loss).
        """
        labels = [-100] * len(ids)
        text = self._format_sample(sample)
        # Find assistant turn boundaries and unmask them
        # Simple heuristic: unmask everything after <|assistant|>
        assistant_id = self.cfg.tool_call_id - 1  # adjust to your token IDs
        in_assistant = False
        for i, tok_id in enumerate(ids):
            if tok_id == self.cfg.tool_call_id - 1:  # <|assistant|>
                in_assistant = True
            if in_assistant:
                labels[i] = tok_id
        return labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = self._format_sample(sample)
        ids = self._tokenize(text)

        # Truncate to max_seq_len
        ids = ids[:self.cfg.max_seq_len]
        labels = self._make_labels(ids, sample)[:self.cfg.max_seq_len]

        return ids, labels

def collate_batch(samples: list, pad_id: int = 0) -> tuple:
    """Pad a list of (ids, labels) to same length."""
    ids_list, labels_list = zip(*samples)
    max_len = max(len(x) for x in ids_list)

    ids_padded    = [x + [pad_id]  * (max_len - len(x)) for x in ids_list]
    labels_padded = [x + [-100]    * (max_len - len(x)) for x in labels_list]

    return (
        mx.array(ids_padded),
        mx.array(labels_padded)
    )

def make_dataloader(dataset: AgentDataset, batch_size: int, shuffle: bool = True) -> Iterator:
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch = [dataset[i] for i in batch_idx]
        yield collate_batch(batch)
```

### `data/synthetic.py` — Tool Call Trajectory Generator

```python
import json
import random
from typing import Callable

# Tool registry for synthetic generation
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
}

def generate_trajectory(n_steps: int = 3, inject_failure: bool = False) -> dict:
    """Generate a synthetic multi-step agentic trajectory."""
    tools = random.sample(list(SYNTHETIC_TOOLS.keys()), min(n_steps, len(SYNTHETIC_TOOLS)))
    user_queries = [
        "Research the latest developments in {topic} and write a summary",
        "Find all Python files in the project and count lines of code",
        "Check the weather in three cities and compare them",
        "Search for papers on {topic} and extract key findings",
    ]
    topics = ["AI agents", "SSMs", "multi-agent systems", "LLM fine-tuning"]
    query = random.choice(user_queries).format(topic=random.choice(topics))

    # Build plan
    plan_steps = "\n".join(f"{i+1}. Use {t}" for i, t in enumerate(tools))
    assistant_content = f"<|plan|>{plan_steps}"

    for i, tool_name in enumerate(tools):
        tool = SYNTHETIC_TOOLS[tool_name]
        # Generate mock args
        mock_args = {k: f"example_{k}" for k in tool["args"].keys()}
        call = json.dumps({"name": tool_name, "args": mock_args})

        if inject_failure and i == 0:
            # Inject error on first call, recovery on second
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

def generate_dataset(n_samples: int, output_path: str):
    with open(output_path, "w") as f:
        for i in range(n_samples):
            traj = generate_trajectory(
                n_steps=random.randint(1, 4),
                inject_failure=(random.random() < 0.15)  # 15% failure cases
            )
            f.write(json.dumps(traj) + "\n")
    print(f"Generated {n_samples} samples → {output_path}")

# Usage:
# generate_dataset(5000, "data/synthetic_agents.jsonl")
```

---

## 2. Weight Initialization

### `init.py`

```python
import mlx.core as mx
import mlx.nn as nn
import math

def init_agentmind(model, cfg):
    """
    Mamba is sensitive to initialization.
    Wrong init → training instability or silent failure.
    """
    for name, module in model.named_modules():

        # ── Standard linear layers ───────────────────────────
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * cfg.n_layers)  # scaled by depth
            module.weight = mx.random.normal(module.weight.shape) * std
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = mx.zeros(module.bias.shape)

        # ── Mamba-specific: dt_proj bias ─────────────────────
        if "dt_proj" in name and isinstance(module, nn.Linear):
            # Init bias so softplus(bias) spans [dt_min, dt_max]
            # This controls how fast the SSM forgets — critical
            dt = mx.exp(
                mx.random.uniform(
                    shape=(cfg.d_inner,),
                    low=math.log(cfg.dt_min),
                    high=math.log(cfg.dt_max)
                )
            )
            inv_dt = dt + mx.log(-mx.expm1(-dt))  # inverse softplus
            module.bias = inv_dt

        # ── Mamba-specific: A_log ────────────────────────────
        if "A_log" in name:
            # A controls long-term memory decay
            # Init as evenly spaced log values — empirically stable
            A = mx.broadcast_to(
                mx.arange(1, cfg.d_state + 1, dtype=mx.float32)[None, :],
                (cfg.d_inner, cfg.d_state)
            )
            module.data = mx.log(A)  # stored as log for numerical stability

        # ── Mamba-specific: D (skip) ─────────────────────────
        if name.endswith(".D"):
            module.data = mx.ones(module.shape)  # ones = full skip connection

        # ── Embedding ────────────────────────────────────────
        if isinstance(module, nn.Embedding):
            module.weight = mx.random.normal(module.weight.shape) * 0.02

        # ── RMSNorm ──────────────────────────────────────────
        if isinstance(module, nn.RMSNorm):
            module.weight = mx.ones(module.weight.shape)

    print("Weight initialization complete.")
    return model
```

---

## 3. RoPE for Attention Blocks

### `model/rope.py`

```python
import mlx.core as mx

def precompute_rope(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """
    Precompute RoPE sin/cos tables.
    Call once at model init, reuse at every forward pass.
    """
    assert head_dim % 2 == 0
    theta = 1.0 / (base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    positions = mx.arange(max_seq_len, dtype=mx.float32)
    freqs = mx.outer(positions, theta)          # [seq_len, head_dim/2]
    cos = mx.cos(freqs)                          # [seq_len, head_dim/2]
    sin = mx.sin(freqs)                          # [seq_len, head_dim/2]
    return cos, sin

def apply_rope(x, cos, sin, offset: int = 0):
    """
    Apply rotary embeddings to query or key tensor.
    x: [B, n_heads, seq_len, head_dim]
    """
    seq_len = x.shape[2]
    cos = cos[offset:offset + seq_len]           # [seq_len, head_dim/2]
    sin = sin[offset:offset + seq_len]

    # Split head_dim into pairs
    x1 = x[..., 0::2]   # even dims
    x2 = x[..., 1::2]   # odd dims

    # Rotate
    rotated = mx.concatenate([
        x1 * cos[None, None] - x2 * sin[None, None],
        x1 * sin[None, None] + x2 * cos[None, None],
    ], axis=-1)

    return rotated

# ── Integration into LocalAttentionBlock ──────────────────
# In attention_block.py, add to __init__:
#   cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len)
#   self.rope_cos = cos
#   self.rope_sin = sin
#
# In _local_attn(), after projecting q and k:
#   q = apply_rope(q, self.rope_cos, self.rope_sin)
#   k = apply_rope(k, self.rope_cos, self.rope_sin)
```

---

## 4. LoRA

### `lora.py`

```python
import mlx.core as mx
import mlx.nn as nn
import math

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with low-rank adapter.
    Only A and B are trained. Base weight is frozen.
    """

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        in_features  = base.weight.shape[1]
        out_features = base.weight.shape[0]
        self.scale = alpha / rank

        # Freeze base weight (not a parameter)
        self.weight = base.weight          # frozen
        self.bias   = getattr(base, "bias", None)

        # Trainable low-rank matrices
        self.A = mx.random.normal((rank, in_features)) * (1 / math.sqrt(rank))
        self.B = mx.zeros((out_features, rank))

    def __call__(self, x):
        base_out = x @ self.weight.T
        if self.bias is not None:
            base_out = base_out + self.bias
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + self.scale * lora_out

def apply_lora(model, rank: int = 16, alpha: float = 32.0, targets: list[str] = None):
    """
    Wrap target linear layers with LoRA. Freeze everything else.

    Default targets — layers that matter most for agentic behavior:
      MambaBlock:         in_proj, out_proj
      LocalAttentionBlock: q_proj, v_proj
      LM head
    """
    if targets is None:
        targets = ["in_proj", "out_proj", "q_proj", "v_proj", "lm_head"]

    # Freeze entire model first
    model.freeze()

    # Walk and replace target layers
    def _replace(module, name_parts):
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            child = getattr(module, attr_name, None)
            if isinstance(child, nn.Linear):
                if any(t in attr_name for t in targets):
                    lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                    setattr(module, attr_name, lora_layer)
            elif isinstance(child, nn.Module):
                _replace(child, name_parts + [attr_name])

    _replace(model, [])

    # Count trainable params
    total = sum(p.size for _, p in model.trainable_parameters())
    print(f"LoRA applied | Trainable params: {total:,} ({total/1e6:.2f}M)")
    return model
```

---

## 5. Conv State for Single-Step Inference

### `model/conv_state.py`

```python
import mlx.core as mx

class ConvState:
    """
    Manages the sliding conv buffer for Mamba's causal depthwise conv
    during single-token (autoregressive) inference.

    At training: process full sequence with padding.
    At inference: slide a d_conv-length buffer per token.
    """

    def __init__(self, batch_size: int, d_inner: int, d_conv: int):
        # Buffer: last (d_conv - 1) input vectors
        self.buf = mx.zeros((batch_size, d_conv - 1, d_inner))
        self.d_conv = d_conv

    def step(self, x_t, conv_weight, conv_bias):
        """
        x_t:        [B, d_inner] — current token's inner representation
        conv_weight: [d_inner, d_conv] — depthwise conv weights
        conv_bias:   [d_inner]
        Returns:    [B, d_inner] — conv output for this timestep
        """
        # Append current token to buffer
        x_t_expanded = x_t[:, None, :]                  # [B, 1, d_inner]
        window = mx.concatenate([self.buf, x_t_expanded], axis=1)  # [B, d_conv, d_inner]

        # Depthwise conv: dot each channel independently
        # conv_weight: [d_inner, d_conv]
        out = mx.sum(window * conv_weight[None, :, :].transpose(0, 2, 1), axis=1)
        out = out + conv_bias[None, :]                   # [B, d_inner]

        # Slide buffer: drop oldest, keep last (d_conv - 1)
        self.buf = window[:, 1:, :]
        return out

# ── How to use in MambaBlock.step() ──────────────────────
#
# At inference init:
#   conv_states = {i: ConvState(B, cfg.d_inner, cfg.d_conv)
#                  for i, block in enumerate(model.blocks)
#                  if isinstance(block, MambaBlock)}
#
# Per token step:
#   x_conv = conv_states[i].step(x_t, block.conv.weight, block.conv.bias)
#   x_conv = nn.silu(x_conv)
```

---

## 6. LR Scheduler

### `scheduler.py`

```python
import math

class CosineWarmupScheduler:
    """
    Linear warmup → cosine decay.
    Standard for instruction-tuned models.
    """

    def __init__(
        self,
        optimizer,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.opt = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = base_lr * min_lr_ratio
        self.step_count = 0

    def step(self):
        self.step_count += 1
        lr = self._get_lr()
        self.opt.learning_rate = lr
        return lr

    def _get_lr(self):
        s = self.step_count
        W = self.warmup_steps
        T = self.total_steps

        if s < W:
            # Linear warmup
            return self.base_lr * (s / max(1, W))
        else:
            # Cosine decay
            progress = (s - W) / max(1, T - W)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine
```

---

## 7. MTP — Multi-Token Prediction

### `model/mtp_head.py`

```python
import mlx.core as mx
import mlx.nn as nn

class MTPHead(nn.Module):
    """
    Multi-Token Prediction auxiliary loss.
    Predicts next K tokens simultaneously from each position.
    Forces the model to think ahead — improves instruction following.

    Paper: "Better & Faster Large Language Models via Multi-Token Prediction"
    """

    def __init__(self, cfg, K: int = 4):
        super().__init__()
        self.K = K  # predict K tokens ahead
        self.heads = [
            nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            for _ in range(K)
        ]
        # Shared projection to avoid parameter explosion
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def __call__(self, hidden_states):
        # hidden_states: [B, L, d_model]
        projected = self.proj(hidden_states)
        return [head(projected) for head in self.heads]  # K × [B, L, vocab]

def mtp_loss(mtp_heads_out, targets, ignore_id: int = -100, weight: float = 0.3):
    """
    Compute auxiliary MTP loss.
    Each head k predicts the token k+1 steps ahead.

    weight: how much to add to main loss (0.1–0.3 works well)
    """
    import mlx.nn as nn_ops

    B, L, V = mtp_heads_out[0].shape
    total_aux = 0.0

    for k, logits_k in enumerate(mtp_heads_out):
        # Shift targets: head k predicts position + k + 1
        shift = k + 1
        if shift >= L:
            continue

        pred   = logits_k[:, :-shift].reshape(-1, V)     # [B*(L-shift), V]
        target = targets[:, shift:].reshape(-1)            # [B*(L-shift)]

        mask = (target != ignore_id).astype(mx.float32)
        loss = nn_ops.losses.cross_entropy(pred, target, reduction='none')
        total_aux += (loss * mask).sum() / (mask.sum() + 1e-8)

    return weight * (total_aux / len(mtp_heads_out))

# ── Integration ───────────────────────────────────────────
# In AgentMind.__init__:
#   self.mtp = MTPHead(cfg, K=4)
#
# In forward():
#   mtp_logits = self.mtp(hidden_before_lm_head)
#
# In train_step():
#   main_loss = cross_entropy_loss(logits, targets)
#   aux_loss  = mtp_loss(mtp_logits, targets)
#   loss = main_loss + aux_loss
```

---

## 8. Latent Reasoning Training

### `model/latent.py`

```python
import mlx.core as mx
import mlx.nn as nn

# ── Staged Training Curriculum ────────────────────────────
#
# Stage 1 (steps 0–500):   Normal training, no latent tokens
# Stage 2 (steps 500–1000): Insert <|think_start|>..CoT..<|think_end|> boundaries
# Stage 3 (steps 1000–2000): Replace 50% of CoT tokens with latent steps
# Stage 4 (steps 2000+):    Full latent — CoT removed entirely
#
# NEVER cold-start latent reasoning — the model needs to
# learn what good reasoning looks like before hiding it.

N_LATENT_STEPS = 4   # how many silent recurrence steps before emitting token

def inject_latent_tokens(sample: dict, tokenizer, stage: int) -> dict:
    """
    Progressively replace chain-of-thought with latent boundaries.
    Call during data preprocessing, not at model forward time.
    """
    if stage < 2:
        return sample  # Stage 1: pass through unchanged

    for msg in sample["messages"]:
        if msg["role"] != "assistant":
            continue

        content = msg["content"]

        # Detect CoT markers (e.g. <|scratch|> content)
        if "<|scratch|>" in content:
            if stage == 2:
                # Wrap scratch content in latent boundary tokens
                content = content.replace(
                    "<|scratch|>",
                    "<|think_start|><|scratch|>"
                ).replace(
                    # End boundary before next structural token
                    "<|tool_call|>", "<|think_end|><|tool_call|>"
                )
            elif stage >= 3:
                # Remove scratch content entirely — model thinks silently
                import re
                content = re.sub(r"<\|think_start\|>.*?<\|think_end\|>", 
                                 "<|think_start|><|think_end|>", 
                                 content, flags=re.DOTALL)

        msg["content"] = content

    return sample

class LatentReasoningWrapper(nn.Module):
    """
    Wraps a MambaBlock to execute N silent recurrence steps
    when <|think_start|> token is detected.

    At <|think_start|>: enter latent mode
    For N steps: update hidden state without emitting tokens
    At <|think_end|>: resume normal generation
    """

    def __init__(self, mamba_block, cfg, n_steps: int = N_LATENT_STEPS):
        super().__init__()
        self.block = mamba_block
        self.n_steps = n_steps
        self.think_start_id = cfg.think_start_id
        self.think_end_id   = cfg.think_end_id

    def latent_forward(self, hidden, h_state):
        """
        Execute N silent SSM recurrence steps.
        No tokens emitted. Hidden state accumulates reasoning.
        """
        for _ in range(self.n_steps):
            # Feed last hidden state back as input (no decode step)
            hidden, h_state = self.block(hidden, h_state)
        return hidden, h_state

    def __call__(self, x, input_ids=None, h_state=None):
        if input_ids is not None:
            # Check if any token in this batch is think_start
            has_think = mx.any(input_ids == self.think_start_id)
            if has_think:
                x, h_state = self.latent_forward(x, h_state)

        return self.block(x, h_state)

def latent_loss_mask(input_ids, labels, think_start_id, think_end_id):
    """
    During latent stages, zero out loss between think_start and think_end.
    Model is not penalized for what it 'thinks' — only for what it emits.
    """
    in_latent = False
    masked_labels = labels.tolist()

    for i, tok_id in enumerate(input_ids.tolist()):
        if tok_id == think_start_id:
            in_latent = True
        if tok_id == think_end_id:
            in_latent = False
        if in_latent:
            masked_labels[i] = -100  # ignore in loss

    return mx.array(masked_labels)
```

---

## 9. Complete Training Loop

### `train.py`

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import time, json
from pathlib import Path

from config import AgentMindConfig
from model.agent_lm import AgentMind
from model.mtp_head import mtp_loss
from data.pipeline import AgentDataset, make_dataloader
from lora import apply_lora
from scheduler import CosineWarmupScheduler
from init import init_agentmind

# ── Config ────────────────────────────────────────────────

cfg = AgentMindConfig()

TRAIN_CFG = dict(
    lr            = 2e-4,
    weight_decay  = 0.01,
    warmup_steps  = 100,
    total_steps   = 3000,
    grad_clip     = 1.0,
    batch_size    = 1,       # MacBook Air — keep at 1
    grad_accum    = 8,       # effective batch = 8
    seq_len       = 2048,
    lora_rank     = 16,
    lora_alpha    = 32.0,
    eval_every    = 50,
    save_every    = 200,
    save_dir      = "./checkpoints",
    use_mtp       = True,
    mtp_weight    = 0.2,
    latent_stage  = 1,       # bump to 2, 3, 4 progressively
)

# ── Loss ──────────────────────────────────────────────────

def cross_entropy_loss(logits, targets):
    B, L, V = logits.shape
    flat_logits  = logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)
    mask = (flat_targets != -100).astype(mx.float32)
    loss = nn.losses.cross_entropy(flat_logits, mx.maximum(flat_targets, 0), reduction='none')
    return (loss * mask).sum() / (mask.sum() + 1e-8)

# ── Gradient clipping ─────────────────────────────────────

def clip_gradients(grads, max_norm: float):
    leaves = [g for _, g in tree_flatten(grads)]
    norm = mx.sqrt(sum(mx.sum(g ** 2) for g in leaves))
    scale = mx.minimum(1.0, max_norm / (norm + 1e-6))
    from mlx.utils import tree_map
    return tree_map(lambda g: g * scale, grads), norm.item()

# ── Train step ────────────────────────────────────────────

def make_train_step(model):
    def train_step(input_ids, targets):
        def loss_fn(params):
            model.update(params)
            logits, h_states = model(input_ids)
            main = cross_entropy_loss(logits, targets)

            if TRAIN_CFG["use_mtp"] and hasattr(model, "mtp"):
                # Get hidden states before lm_head for MTP
                # (store as model attribute during forward pass)
                aux = mtp_loss(model.last_hidden, targets,
                               weight=TRAIN_CFG["mtp_weight"])
                return main + aux
            return main

        loss, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
        return loss, grads

    return train_step

# ── Main training loop ────────────────────────────────────

def train():
    Path(TRAIN_CFG["save_dir"]).mkdir(exist_ok=True)

    # Model
    model = AgentMind(cfg)
    model = init_agentmind(model, cfg)
    model = apply_lora(model, rank=TRAIN_CFG["lora_rank"], alpha=TRAIN_CFG["lora_alpha"])

    # Optimizer + Scheduler
    optimizer = optim.AdamW(
        learning_rate=TRAIN_CFG["lr"],
        weight_decay=TRAIN_CFG["weight_decay"]
    )
    scheduler = CosineWarmupScheduler(
        optimizer,
        base_lr=TRAIN_CFG["lr"],
        warmup_steps=TRAIN_CFG["warmup_steps"],
        total_steps=TRAIN_CFG["total_steps"]
    )

    # Data
    from tokenizer_setup import load_tokenizer
    tok = load_tokenizer("agentmind_tok.model")
    train_ds = AgentDataset(
        ["data/instructions.jsonl", "data/synthetic_agents.jsonl"],
        tokenizer=tok, cfg=cfg, split="train"
    )
    val_ds = AgentDataset(
        ["data/instructions.jsonl"],
        tokenizer=tok, cfg=cfg, split="val"
    )

    train_step_fn = make_train_step(model)
    step = 0
    accum_loss = 0.0
    accum_grad = None
    log = []

    print(f"Training AgentMind | {sum(p.size for _,p in model.trainable_parameters()):,} trainable params")

    while step < TRAIN_CFG["total_steps"]:
        loader = make_dataloader(train_ds, batch_size=TRAIN_CFG["batch_size"])

        for input_ids, targets in loader:
            if step >= TRAIN_CFG["total_steps"]:
                break

            t0 = time.time()
            loss, grads = train_step_fn(input_ids, targets)
            mx.eval(loss, grads)

            # Gradient accumulation
            grads, grad_norm = clip_gradients(grads, TRAIN_CFG["grad_clip"])
            accum_loss += loss.item()

            if accum_grad is None:
                accum_grad = grads
            else:
                from mlx.utils import tree_map
                accum_grad = tree_map(lambda a, b: a + b, accum_grad, grads)

            if (step + 1) % TRAIN_CFG["grad_accum"] == 0:
                # Average accumulated gradients
                from mlx.utils import tree_map
                accum_grad = tree_map(lambda g: g / TRAIN_CFG["grad_accum"], accum_grad)
                optimizer.update(model, accum_grad)
                mx.eval(model.parameters(), optimizer.state)
                lr = scheduler.step()
                accum_grad = None

                avg_loss = accum_loss / TRAIN_CFG["grad_accum"]
                accum_loss = 0.0
                tok_per_sec = TRAIN_CFG["batch_size"] * TRAIN_CFG["seq_len"] / (time.time() - t0)

                print(f"step {step:4d} | loss {avg_loss:.4f} | lr {lr:.2e} | grad_norm {grad_norm:.3f} | {tok_per_sec:.0f} tok/s")
                log.append({"step": step, "loss": avg_loss, "lr": lr})

            # Eval
            if step % TRAIN_CFG["eval_every"] == 0 and step > 0:
                val_loss, tool_acc = evaluate(model, val_ds, tok, cfg)
                print(f"  ── EVAL step {step} | val_loss {val_loss:.4f} | tool_acc {tool_acc:.2%}")
                log[-1].update({"val_loss": val_loss, "tool_acc": tool_acc})

            # Save
            if step % TRAIN_CFG["save_every"] == 0 and step > 0:
                save_path = f"{TRAIN_CFG['save_dir']}/step_{step:05d}"
                Path(save_path).mkdir(exist_ok=True)
                mx.savez(f"{save_path}/weights.npz", **dict(tree_flatten(model.parameters())))
                json.dump(log, open(f"{save_path}/log.json", "w"), indent=2)
                print(f"  ── Saved checkpoint → {save_path}")

            step += 1

    print("Training complete.")

if __name__ == "__main__":
    train()
```

---

## 10. Evaluation

### `eval.py`

```python
import mlx.core as mx
import mlx.nn as nn
import json, re
from data.pipeline import make_dataloader

def compute_perplexity(model, dataset, tok, cfg, max_batches: int = 50) -> float:
    """Standard perplexity on validation set."""
    total_loss = 0.0
    total_tokens = 0
    loader = make_dataloader(dataset, batch_size=1, shuffle=False)

    for i, (input_ids, targets) in enumerate(loader):
        if i >= max_batches:
            break
        logits, _ = model(input_ids)
        B, L, V = logits.shape
        flat_logits  = logits.reshape(-1, V)
        flat_targets = targets.reshape(-1)
        mask = (flat_targets != -100).astype(mx.float32)
        loss = nn.losses.cross_entropy(flat_logits, mx.maximum(flat_targets, 0), reduction='none')
        total_loss   += (loss * mask).sum().item()
        total_tokens += mask.sum().item()

    import math
    return math.exp(total_loss / max(total_tokens, 1))

def tool_call_accuracy(model, prompts: list[str], tok, cfg) -> float:
    """
    Check if model reliably produces valid JSON after <|tool_call|>.
    A structurally valid JSON tool call = pass.
    """
    passed = 0
    TOOL_CALL_TOKEN = "<|tool_call|>"

    for prompt in prompts:
        ids = mx.array([tok.encode(prompt, add_bos=True)])
        output_ids = []

        # Generate up to 200 tokens
        for _ in range(200):
            logits, _ = model(ids)
            next_tok = mx.argmax(logits[0, -1]).item()
            output_ids.append(next_tok)
            ids = mx.array([[next_tok]])
            if next_tok == cfg.eos_id:
                break

        decoded = tok.decode(output_ids)

        # Check for valid JSON tool call
        if TOOL_CALL_TOKEN in decoded:
            after = decoded.split(TOOL_CALL_TOKEN)[-1]
            try:
                obj = json.loads(after.split("<|observe|>")[0].strip())
                if "name" in obj and "args" in obj:
                    passed += 1
            except json.JSONDecodeError:
                pass

    return passed / max(len(prompts), 1)

def format_adherence(model, prompts: list[str], tok, cfg) -> dict:
    """
    Check structural output quality:
    - Does it use <|plan|> for multi-step queries?
    - Does it use <|scratch|> for intermediate reasoning?
    - Does it terminate with EOS?
    """
    results = {"plan": 0, "scratch": 0, "eos": 0, "total": len(prompts)}

    for prompt in prompts:
        ids = mx.array([tok.encode(prompt, add_bos=True)])
        output_ids = []

        for _ in range(300):
            logits, _ = model(ids)
            next_tok = mx.argmax(logits[0, -1]).item()
            output_ids.append(next_tok)
            ids = mx.array([[next_tok]])
            if next_tok == cfg.eos_id:
                results["eos"] += 1
                break

        decoded = tok.decode(output_ids)
        if "<|plan|>" in decoded:
            results["plan"] += 1
        if "<|scratch|>" in decoded:
            results["scratch"] += 1

    return results

def evaluate(model, val_dataset, tok, cfg):
    """Combined eval — returns (val_loss, tool_acc)."""
    ppl = compute_perplexity(model, val_dataset, tok, cfg)

    test_prompts = [
        "<|user|>Search arxiv for Mamba SSM papers<|assistant|>",
        "<|user|>Get the weather in Tokyo and Pune<|assistant|>",
        "<|user|>Run the test suite and fix any failures<|assistant|>",
    ]
    tool_acc = tool_call_accuracy(model, test_prompts, tok, cfg)

    import math
    return math.log(ppl), tool_acc  # return log ppl for cleaner display
```

---

## 11. GGUF Export

### `export.py`

```python
"""
Export AgentMind to GGUF (4-bit NF4) for llama.cpp / MLX inference.

Custom architecture requires:
1. Saving weights in HuggingFace format
2. Writing a config.json that describes the hybrid architecture
3. Running mlx_lm.convert with the custom arch registered

Steps:
  python export.py --checkpoint ./checkpoints/step_03000 --out ./agentmind-4bit
"""

import json
import mlx.core as mx
from mlx.utils import tree_flatten
from pathlib import Path

def save_hf_format(model, cfg, out_dir: str):
    """Save weights + config in HuggingFace-compatible format."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save weights as safetensors
    weights = dict(tree_flatten(model.parameters()))
    mx.savez(str(out / "model.npz"), **weights)

    # HuggingFace config.json — describes the custom architecture
    hf_config = {
        "model_type": "agentmind",
        "architectures": ["AgentMindForCausalLM"],
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.d_model,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,

        # Mamba-specific
        "state_size": cfg.d_state,
        "conv_kernel": cfg.d_conv,
        "expand": cfg.expand,
        "dt_rank": cfg.dt_rank_val,

        # Hybrid-specific
        "attn_every": cfg.attn_every,
        "attn_window": cfg.attn_window,

        # Tokenizer
        "bos_token_id": cfg.bos_id,
        "eos_token_id": cfg.eos_id,
        "pad_token_id": cfg.pad_id,

        # Special agentic tokens
        "tool_call_token_id": cfg.tool_call_id,
        "observe_token_id": cfg.observe_id,
        "think_start_token_id": cfg.think_start_id,
        "think_end_token_id": cfg.think_end_id,
    }
    json.dump(hf_config, open(out / "config.json", "w"), indent=2)

    # Tokenizer files
    import shutil
    shutil.copy("agentmind_tok.model", out / "tokenizer.model")

    print(f"Saved HF format → {out_dir}")

def quantize_and_export(hf_dir: str, out_dir: str, bits: int = 4):
    """
    Convert to 4-bit MLX format using mlx_lm.
    Run after save_hf_format().
    """
    import subprocess
    cmd = [
        "python", "-m", "mlx_lm.convert",
        "--hf-path", hf_dir,
        "--mlx-path", out_dir,
        "-q",
        "--q-bits", str(bits),
        "--q-group-size", "64",    # NF4-style group quantization
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"4-bit model → {out_dir}")

# ── CLI usage ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from config import AgentMindConfig
    from model.agent_lm import AgentMind

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out",        default="./agentmind-4bit")
    parser.add_argument("--bits",       type=int, default=4)
    args = parser.parse_args()

    cfg = AgentMindConfig()
    model = AgentMind(cfg)
    weights = mx.load(f"{args.checkpoint}/weights.npz")
    model.update(weights)

    hf_dir = args.out + "_hf"
    save_hf_format(model, cfg, hf_dir)
    quantize_and_export(hf_dir, args.out, bits=args.bits)
```

---

## 12. Parallel Scan (Training Speed Upgrade)

```python
# model/parallel_scan.py
# Drop-in replacement for the sequential loop in MambaBlock._ssm()
# Reduces training time from O(L) sequential steps → O(log L) parallel steps

import mlx.core as mx

def parallel_scan_log(log_coeffs, log_values):
    """
    Log-space parallel scan for numerical stability.
    Computes: h_t = prod(a_1..a_t) * h_0 + sum_k( prod(a_{k+1}..a_t) * b_k )

    log_coeffs: [B, L, d_inner, d_state] — log of dA
    log_values:  [B, L, d_inner, d_state] — log of dB * x
    """
    # Prefix sum in log space = cumulative product in linear space
    log_prefix = mx.cumsum(log_coeffs, axis=1)

    # Each position's contribution: value * (total_prefix / local_prefix)
    # = value * exp(log_prefix_total - log_prefix_local)
    log_prefix_shifted = mx.concatenate([
        mx.zeros_like(log_prefix[:, :1]),
        log_prefix[:, :-1]
    ], axis=1)

    # Numerically stable sum using log-sum-exp
    log_h = log_prefix_shifted + log_values
    # Scan via cumulative log-sum-exp
    h = mx.cumsum(mx.exp(log_h - log_prefix), axis=1) * mx.exp(log_prefix)
    return h

# ── Swap into MambaBlock._ssm() ──────────────────────────
# Replace the for-loop with:
#
#   log_dA = dt[:,:,:,None] * A[None, None]           # [B,L,di,ds]
#   log_dB = mx.log(mx.abs(dB) + 1e-8)               # [B,L,di,ds]
#   log_x  = mx.log(mx.abs(x[:,:,:,None]) + 1e-8)    # [B,L,di,1]
#   h = parallel_scan_log(log_dA, log_dB + log_x)
#   y = mx.sum(h * C_mat[:,:,None,:], axis=-1)
```

---

## Quick Execution Order

```bash
# 1. Generate synthetic data
python -c "from data.synthetic import generate_dataset; generate_dataset(5000, 'data/synthetic_agents.jsonl')"

# 2. Train tokenizer
python -c "from tokenizer_setup import train_tokenizer; train_tokenizer('data/corpus.txt')"

# 3. Train (LoRA, stage 1)
python train.py

# 4. Bump to latent stage 2 in train.py TRAIN_CFG, continue training
# TRAIN_CFG["latent_stage"] = 2
python train.py --resume checkpoints/step_03000

# 5. Evaluate
python eval.py --checkpoint checkpoints/step_03000

# 6. Export to 4-bit
python export.py --checkpoint checkpoints/step_03000 --out agentmind-4bit

# 7. Run agent
python agent.py --model agentmind-4bit --query "your query"
```
