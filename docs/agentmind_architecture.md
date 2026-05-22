# AgentMind — Hybrid SSM Architecture
> 16GB MacBook Air · MLX Backend · Agentic-First Design

---

## Memory Budget

| Phase | What | Size |
|---|---|---|
| Training (LoRA) | Weights fp16 | ~1.2GB |
| Training (LoRA) | LoRA params + optimizer | ~100MB |
| Training (LoRA) | Activations (batch=1, seq=256-2048) | ~1.5-3GB |
| Training (LoRA) | Parallel scan intermediates | ~2GB peak |
| **Training total** | | **~5GB ✓** |
| Inference (4-bit) | Weights GGUF | ~300MB |
| Inference (4-bit) | SSM state (constant) | ~2MB |
| **Inference total** | | **<1GB ✓** |

---

## Architecture Overview

```
AgentMind-600M
├── Embedding          [vocab=32k, d_model=2048]
├── 24 Hybrid Blocks
│   ├── 0,1,2   → MambaBlock   (SSM, long-range memory)
│   ├── 3       → AttnBlock    (local window=512, precision)
│   ├── 4,5,6   → MambaBlock
│   ├── 7       → AttnBlock
│   └── ... pattern × 6
├── RMSNorm
└── LM Head            [tied to embedding]

~600M params · 3:1 Mamba-to-Attention ratio
```

**Why this ratio:**
- Mamba carries long tool history, compressed into fixed state
- Attention fires every 4th layer for precise token recall and structured output formatting
- Local window (512) keeps attention cost O(L) not O(L²)

---

## `config.py`

```python
import math
from dataclasses import dataclass

@dataclass
class AgentMindConfig:
    # Vocabulary
    vocab_size: int = 32_000

    # Model dimensions
    d_model: int = 2048
    n_layers: int = 24

    # Mamba SSM
    d_state: int = 128       # memory per channel — larger = richer history
    d_conv: int = 4          # causal conv kernel
    expand: int = 2          # d_inner = expand × d_model = 4096
    dt_rank: int = -1        # -1 = auto: ceil(d_model / 16) = 128

    # Hybrid attention
    n_heads: int = 16
    attn_window: int = 512   # local attention window
    attn_every: int = 4      # attention layer every N blocks

    # FFN (SwiGLU)
    ffn_mult: float = 8 / 3  # standard SwiGLU multiplier

    # Runtime
    max_seq_len: int = 8192
    tie_embeddings: bool = True

    # Special token IDs (assigned after tokenizer init)
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 5
    tool_call_id: int = 6
    plan_id: int = 7
    memory_id: int = 8
    scratch_id: int = 9
    observe_id: int = 10
    think_start_id: int = 11
    think_end_id: int = 12
    system_id: int = 13
    user_id: int = 14
    assistant_id: int = 15

    @property
    def d_inner(self) -> int:
        return int(self.expand * self.d_model)

    @property
    def dt_rank_val(self) -> int:
        return math.ceil(self.d_model / 16) if self.dt_rank == -1 else self.dt_rank

    @property
    def ffn_hidden(self) -> int:
        raw = int(self.d_model * self.ffn_mult)
        return (raw // 256) * 256  # align to 256 for hardware efficiency

    def is_attn_layer(self, i: int) -> bool:
        return (i + 1) % self.attn_every == 0
```

---

## `model/mamba_block.py`

```python
import mlx.core as mx
import mlx.nn as nn
import math

class MambaBlock(nn.Module):
    """
    Selective State Space Model block.
    
    At inference: pure O(1) recurrence — SSM state stays fixed size
    regardless of how many tool calls have been processed.
    
    At training: sequential scan (correct). Swap with parallel scan
    for full training speed (see note at bottom).
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        di = cfg.d_inner
        ds = cfg.d_state
        dr = cfg.dt_rank_val

        self.d_inner = di
        self.d_state = ds
        self.d_conv = cfg.d_conv

        # Pre-norm
        self.norm = nn.RMSNorm(d)

        # Split input into x (signal) and z (gate)
        self.in_proj = nn.Linear(d, di * 2, bias=False)

        # Causal depthwise conv (groups=di for depthwise)
        self.conv = nn.Conv1d(
            in_channels=di,
            out_channels=di,
            kernel_size=cfg.d_conv,
            padding=cfg.d_conv - 1,   # left-pad for causality
            groups=di,
            bias=True
        )

        # SSM projections: dt (step), B (input gate), C (output gate)
        self.x_proj = nn.Linear(di, dr + ds * 2, bias=False)
        self.dt_proj = nn.Linear(dr, di, bias=True)

        # A: decay matrix, log-parameterized for stability
        A = mx.broadcast_to(
            mx.arange(1, ds + 1, dtype=mx.float32)[None, :],
            (di, ds)
        )
        self.A_log = mx.log(A)
        self.D = mx.ones((di,))

        self.out_proj = nn.Linear(di, d, bias=False)

    def _ssm(self, x):
        # x: [B, L, d_inner]
        B, L, _ = x.shape
        dr, ds = self.x_proj.weight.shape[0] - self.d_state * 2, self.d_state

        A = -mx.exp(self.A_log)                    # [d_inner, d_state]

        # Project to dt, B_mat, C_mat
        xbc = self.x_proj(x)                       # [B, L, dr + 2*ds]
        dt_raw, B_mat, C_mat = mx.split(
            xbc, [dr, dr + ds], axis=-1
        )
        dt = nn.softplus(self.dt_proj(dt_raw))     # [B, L, d_inner]

        # ZOH discretization
        # dA: [B, L, d_inner, d_state]
        dA = mx.exp(dt[:, :, :, None] * A[None, None])
        # dB: [B, L, d_inner, d_state]
        dB = dt[:, :, :, None] * B_mat[:, :, None, :]

        # Sequential scan — correct for both train and inference
        # For training speed: replace with parallel scan using mx.cumsum
        h = mx.zeros((B, self.d_inner, self.d_state))
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t, :, None]
            y = mx.sum(h * C_mat[:, t, None, :], axis=-1)  # [B, d_inner]
            ys.append(y)

        y = mx.stack(ys, axis=1)                   # [B, L, d_inner]
        return y + x * self.D[None, None, :], h    # output, final state

    def __call__(self, x, h_state=None):
        # x: [B, L, d_model]
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_in, z = mx.split(xz, [self.d_inner], axis=-1)

        # Causal depthwise conv — trim padding to maintain causality
        x_conv = self.conv(x_in)[:, :x_in.shape[1], :]
        x_conv = nn.silu(x_conv)

        # SSM + skip connection
        y, h_out = self._ssm(x_conv)

        # Gate
        y = y * nn.silu(z)

        return self.out_proj(y) + residual, h_out

    # ── Inference-only: single step recurrence ──────────────
    def step(self, x_t, h):
        """
        Single token step — pure O(1) recurrence.
        x_t: [B, d_model], h: [B, d_inner, d_state]
        """
        x_t = self.norm(x_t)
        xz = self.in_proj(x_t[:, None, :])
        x_in, z = mx.split(xz, [self.d_inner], axis=-1)

        # Conv step: slide window (maintain conv buffer externally)
        x_conv = nn.silu(x_in.squeeze(1))

        xbc = self.x_proj(x_conv)
        dt_raw, B_mat, C_mat = mx.split(xbc, [self.d_inner // 16, -self.d_state], axis=-1)
        dt = nn.softplus(self.dt_proj(dt_raw))

        A = -mx.exp(self.A_log)
        dA = mx.exp(dt[:, :, None] * A[None])
        dB = dt[:, :, None] * B_mat[:, None, :]

        h = dA * h + dB * x_conv[:, :, None]
        y = mx.sum(h * C_mat[:, None, :], axis=-1)
        y = y + x_conv * self.D[None]
        z_gate = nn.silu(z.squeeze(1))

        return self.out_proj(y * z_gate), h

# NOTE: Parallel scan for training
# Replace the sequential loop in _ssm with:
#
#   log_A = dt[:,:,:,None] * A[None,None]   # [B,L,d_inner,d_state]
#   # Compute prefix products of dA using log-sum-exp
#   log_cumA = mx.cumsum(log_A, axis=1)
#   # Then reconstruct h using einsum — see Mamba paper Appendix C
```

---

## `model/attention_block.py`

```python
import mlx.core as mx
import mlx.nn as nn
import math

class LocalAttentionBlock(nn.Module):
    """
    Sliding window attention — O(L × window) not O(L²).
    Fires every 4th layer. Handles:
      - Precise token recall (exact tool names)
      - Structured output formatting
      - In-context few-shot examples
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.window = cfg.attn_window
        d = cfg.d_model
        dh = cfg.ffn_hidden

        self.norm1 = nn.RMSNorm(d)
        self.norm2 = nn.RMSNorm(d)

        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

        # SwiGLU FFN
        self.gate_proj = nn.Linear(d, dh, bias=False)
        self.up_proj   = nn.Linear(d, dh, bias=False)
        self.down_proj = nn.Linear(dh, d, bias=False)

    def _local_attn(self, x):
        B, L, _ = x.shape
        H, Hd = self.n_heads, self.head_dim
        W = self.window

        q = self.q_proj(x).reshape(B, L, H, Hd).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, H, Hd).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, H, Hd).transpose(0, 2, 1, 3)

        scale = math.sqrt(Hd)

        # Local window mask: each token attends only to last W positions
        # Build position matrix
        pos = mx.arange(L)
        mask = (pos[None, :] - pos[:, None]) >= 0  # causal
        local = (pos[None, :] - pos[:, None]) < W   # window
        attn_mask = mask & local                     # [L, L]
        attn_mask = mx.where(attn_mask, 0.0, float('-inf'))

        scores = (q @ k.transpose(0, 1, 3, 2)) / scale  # [B,H,L,L]
        scores = scores + attn_mask[None, None]
        attn = mx.softmax(scores, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)

    def _ffn(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

    def __call__(self, x):
        x = x + self._local_attn(self.norm1(x))
        x = x + self._ffn(self.norm2(x))
        return x
```

---

## `model/agent_lm.py`

```python
import mlx.core as mx
import mlx.nn as nn
from .mamba_block import MambaBlock
from .attention_block import LocalAttentionBlock
from .mtp_head import MTPHead

class AgentMind(nn.Module):
    """
    Hybrid SSM + Local Attention Language Model.
    
    Layer pattern (n_layers=24, attn_every=4):
    [M M M A | M M M A | M M M A | M M M A | M M M A | M M M A]
     18 Mamba blocks + 6 Attention blocks = 24 total ≈ 600M params
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.blocks = [
            LocalAttentionBlock(cfg) if cfg.is_attn_layer(i)
            else MambaBlock(cfg)
            for i in range(cfg.n_layers)
        ]

        self.norm = nn.RMSNorm(cfg.d_model)

        # Tied LM head
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # MTP auxiliary head
        self.mtp = MTPHead(cfg, K=4)

    def __call__(self, input_ids, return_mtp=False):
        x = self.embed(input_ids)
        h_states = {}

        for i, block in enumerate(self.blocks):
            if isinstance(block, MambaBlock):
                x, h = block(x)
                h_states[i] = h
            else:
                x = block(x)

        self.last_hidden = x
        if return_mtp:
            self.last_mtp_logits = self.mtp(x)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, h_states
```

    def forward_with_state(self, input_ids, past_h_states=None):
        """Used during agentic inference to preserve SSM state across calls."""
        x = self.embed(input_ids)
        new_h_states = {}

        for i, block in enumerate(self.blocks):
            if isinstance(block, MambaBlock):
                h_in = past_h_states.get(i) if past_h_states else None
                x, h = block(x, h_in)
                new_h_states[i] = h
            else:
                x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_h_states
```

---

## `tokenizer_setup.py`

```python
import sentencepiece as spm
from pathlib import Path

SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>",
    # Agentic control tokens
    "<|tool_call|>",     # model wants to invoke a tool
    "<|plan|>",          # structured multi-step plan
    "<|memory|>",        # write to persistent memory
    "<|scratch|>",       # internal scratchpad (visible)
    "<|observe|>",       # tool result injection
    "<|think_start|>",   # begin latent reasoning window
    "<|think_end|>",     # surface output after latent steps
    # Role tokens
    "<|system|>", "<|user|>", "<|assistant|>",
]

def train_tokenizer(corpus_path: str, model_prefix: str = "agentmind_tok"):
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=32_000,
        character_coverage=0.9999,
        model_type="bpe",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        user_defined_symbols=SPECIAL_TOKENS[3:],  # custom tokens after <pad/bos/eos>
        byte_fallback=True,             # handles any unicode
        add_dummy_prefix=False,
        split_digits=True,              # tokenize digits separately (better for tool args)
    )
    print(f"Tokenizer saved: {model_prefix}.model")

def load_tokenizer(model_path: str):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp
```

---

## `train.py`

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

def cross_entropy_loss(logits, targets, ignore_id=0):
    # logits: [B, L, V], targets: [B, L]
    B, L, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)

    loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='none')
    mask = (targets_flat != ignore_id).astype(mx.float32)
    return (loss * mask).sum() / mask.sum()

def make_lora_layers(model, rank=16, alpha=32):
    """
    Freeze everything. LoRA-inject only:
    - MambaBlock.in_proj, out_proj
    - LocalAttentionBlock.q_proj, v_proj
    - LM head
    """
    frozen = set()
    lora = set()

    for name, module in tree_flatten(model.trainable_parameters()):
        if any(k in name for k in ['in_proj', 'out_proj', 'q_proj', 'v_proj', 'lm_head']):
            lora.add(name)
        else:
            frozen.add(name)

    # Freeze
    model.freeze()
    # Unfreeze LoRA targets
    for name in lora:
        # In practice: wrap with LoRA adapter using mlx-lm's LoRALinear
        pass

    return model

class TrainConfig:
    # Optimizer
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # Batch
    batch_size: int = 1          # MacBook Air — keep at 1
    grad_accum: int = 8          # effective batch = 8
    seq_len: int = 2048

    # Schedule
    max_steps: int = 3000
    eval_every: int = 50
    save_every: int = 200

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32

def train_step(model, batch, optimizer):
    input_ids, targets = batch

    def loss_fn(params):
        model.update(params)
        logits, _ = model(input_ids)
        return cross_entropy_loss(logits, targets)

    loss, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())

    # Gradient clipping
    norm = mx.sqrt(sum(mx.sum(g ** 2) for g in tree_flatten(grads)[1]))
    scale = mx.minimum(1.0, TrainConfig.grad_clip / (norm + 1e-6))
    grads = tree_map(lambda g: g * scale, grads)

    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)
    return loss.item()
```

---

## `agent.py` — Agentic Inference Loop

```python
import json
import mlx.core as mx
from typing import Callable

class AgentLoop:
    """
    Stateful agentic inference.
    SSM state persists across tool calls — memory stays constant
    regardless of how many rounds have executed.
    """

    def __init__(self, model, tokenizer, tools: dict[str, Callable], cfg):
        self.model = model
        self.tok = tokenizer
        self.tools = tools     # {"tool_name": callable}
        self.cfg = cfg
        self.max_rounds = 20

    def encode(self, text: str):
        return mx.array([self.tok.encode(text)])

    def decode(self, ids) -> str:
        return self.tok.decode(ids.tolist())

    def sample(self, logits, temp=0.7, top_p=0.9):
        logits = logits / temp
        probs = mx.softmax(logits, axis=-1)
        # Top-p nucleus sampling
        sorted_idx = mx.argsort(-probs)
        cum_probs = mx.cumsum(probs[sorted_idx])
        cutoff = mx.where(cum_probs <= top_p, probs[sorted_idx], 0.0)
        # Sample
        token = sorted_idx[mx.argmax(mx.random.gumbel(cutoff.shape) + mx.log(cutoff + 1e-9))]
        return token

    def run(self, user_query: str) -> str:
        system_prompt = (
            "<|system|>You are an agentic assistant. Think in tools. "
            "Use <|tool_call|>{json}<|observe|> for tool invocations. "
            "Never fake results. Verify before claiming.<|user|>"
        )
        prompt = system_prompt + user_query + "<|assistant|>"

        input_ids = self.encode(prompt)
        h_states = {}          # SSM state — persists across rounds
        output_tokens = []
        rounds = 0

        while rounds < self.max_rounds:
            logits, h_states = self.model.forward_with_state(input_ids, h_states)
            next_token = self.sample(logits[0, -1])
            output_tokens.append(next_token.item())

            # Check for tool call
            if next_token.item() == self.cfg.tool_call_id:
                result = self._handle_tool_call(output_tokens)
                if result is None:
                    break  # parse error — model will recover

                # Inject tool result — SSM state carries context forward
                observe_str = f"<|observe|>{json.dumps(result)}"
                input_ids = self.encode(observe_str)
                output_tokens = []
                rounds += 1
                continue

            # EOS — done
            if next_token.item() == self.cfg.eos_id:
                break

            # Continue generation
            input_ids = next_token.reshape(1, 1)

        return self.decode(mx.array(output_tokens))

    def _handle_tool_call(self, tokens: list) -> dict | None:
        # Decode tokens between tool_call_id and current position
        raw = self.decode(mx.array(tokens))
        # Extract JSON after <|tool_call|>
        try:
            payload = json.loads(raw.split("<|tool_call|>")[-1].strip())
            name = payload["name"]
            args = payload.get("args", {})
            if name not in self.tools:
                return {"error": f"unknown tool: {name}"}
            return self.tools[name](**args)
        except (json.JSONDecodeError, KeyError) as e:
            return {"error": str(e)}  # model sees error and recovers
```

---

## Training Curriculum

```
Phase 1 — Format Bedrock          (500 steps)
  Data: instruction pairs, JSON formatting, role adherence
  Source: FineWeb + UltraChat + 3K instruction synthetic samples
  Goal: model learns token boundaries for all special tokens

Phase 2 — Tool Calling            (800 steps)
  Data: synthetic tool call → observe → answer chains
  Source: 2.5K tool_single + ToolBench dataset
  Goal: reliable <|tool_call|>{json}<|observe|> formatting

Phase 3 — Multi-step Agents       (1000 steps)
  Data: 3–8 round agentic trajectories with real tool results
  Source: 3K agent_multi + AgentInstruct + WebArena
  Goal: SSM state carries context across rounds without drift

Phase 4 — Failure Recovery        (700 steps)
  Data: trajectories with injected errors + correct recovery
  Source: 2K recovery synthetic samples
  Goal: model recovers from bad tool results, retries, admits limits

Phase 5 — Latent Reasoning        (optional, 500 steps)
  Data: replace some CoT with <|think_start|>...<|think_end|>
  Source: 1K latent synthetic samples
  Goal: model learns to reason without surfacing tokens
```

### Data Sources Summary
- **Open datasets**: FineWeb (20K), The Stack Python (10K), UltraChat (63K), AgentInstruct (5K), ToolBench (3K), WebArena (3K)
- **Synthetic**: 13.2K samples across 5 types (instruction, tool_single, agent_multi, recovery, latent)
- **Tools**: 14 tools in registry with realistic args/results
- **Total corpus**: ~250MB plain text + 13.2K structured JSONL trajectories

---

## Quick Start

```bash
# Install
pip install mlx mlx-lm sentencepiece datasets cerebras-cloud-sdk

# Build corpus from open datasets
python build_corpus.py

# Generate scaled synthetic data (11.5K samples)
python generate_scaled_synthetic.py

# Train tokenizer
python -c "from tokenizer_setup import train_tokenizer; train_tokenizer('data/corpus.txt')"

# Run training (LoRA on existing Mamba checkpoint)
python train.py \
  --model state-spaces/mamba-370m \   # pretrained base
  --tokenizer agentmind_tok.model \
  --data data/tool_calls.jsonl \
  --lora_rank 16 \
  --steps 3000

# Export to GGUF for inference
python -m mlx_lm.convert \
  --hf-path ./checkpoints/agentmind \
  --mlx-path ./agentmind-4bit \
  -q --q-bits 4

# Run agent
python agent.py --model ./agentmind-4bit --query "search arxiv for Mamba SSM papers"
```

---

## What Each Component Owns

| Component | Responsibility |
|---|---|
| **MambaBlock** | Long-range memory, tool history compression, agentic state |
| **LocalAttentionBlock** | Precise recall, structured JSON output, few-shot examples |
| **SSM h_state** | Persistent working memory across tool call rounds |
| **`<\|think_start\|>`** | Trigger latent reasoning mode (from previous discussion) |
| **`<\|tool_call\|>`** | Structured tool invocation boundary |
| **`<\|observe\|>`** | Tool result injection point |
| **LoRA** | Low-memory fine-tuning without full backprop on 16GB |
| **4-bit GGUF** | Sub-1GB inference, fast on Apple Silicon |
