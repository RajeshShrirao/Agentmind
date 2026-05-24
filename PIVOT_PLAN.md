# Pivot Plan: Custom Backbone → Pretrained Backbone

## Strategy

Replace the 147M random-init AgentMind (Mamba+Attention) with Qwen2.5-0.5B as the backbone. Keep the apprenticeship architecture (LoRA specialists, orchestration, router, distillation) intact — it's the real innovation.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Backbone | **Qwen2.5-0.5B** | Strong language priors, fits 16GB RAM, native mlx-lm support |
| Special tokens | **Added to tokenizer** via `add_tokens()` | Prevents BPE fragmentation, preserves token IDs across sessions |

## Why Qwen2.5-0.5B

| Requirement | Qwen2.5-0.5B | Phi-3-mini | TinyLlama |
|---|---|---|---|
| Size (fp16) | ~0.9GB | ~1.9GB | ~1.3GB |
| Fits 16GB MBA | ✅ | ✅ | ✅ |
| `mlx_lm` native | ✅ direct load | ✅ | ✅ |
| Tokenizer | 151K vocab (BPE) | 32K (BPE) | 32K (BPE) |
| Chat template | built-in | built-in | basic |
| Quality | strong for size | stronger | weaker |

Phi-3 and TinyLlama are viable alternatives; the plan abstracts the backbone selection as a config param.

## Files to Delete

| File | Reason |
|---|---|
| `model/agent_lm.py` | Whole AgentMind model replaced by `mlx_lm.load()` |
| `model/mamba_block.py` | SSM architecture gone |
| `model/attention_block.py` | Custom attention replaced by Qwen's |
| `model/mtp_head.py` | MTP heads — pretrained models don't need this |
| `model/latent.py` | Latent token injection — can be done in data pipeline instead |
| `model/hybrid_block.py` | Already empty (dead code) |
| `pretrain_backbone.py` | No more raw pretraining |
| `init.py` | Weight initialization — irrelevant for pretrained models |
| `export.py` | Custom export format — not needed |
| `tokenizer_setup.py` | Custom SentencePiece tokenizer replaced |
| `agentmind_tok.model` | Custom vocab file |
| `pretokenize.py` | Pre-tokenization format changes with new tokenizer |

## Files to Rewrite

### 1. `config.py` — Backbone selection, drop AgentMind config

```python
@dataclass
class AgentMindConfig:
    backbone_id: str = "Qwen/Qwen2.5-0.5B"  # HuggingFace model ID
    d_model: int = 512                        # Qwen2.5-0.5B hidden size
    vocab_size: int = 151_936                 # Qwen2.5-0.5B vocab size
    max_seq_len: int = 8192

    # Special token IDs (added via tokenizer.add_tokens())
    pad_id: int = -1
    bos_id: int = -1
    eos_id: int = -1
    tool_call_id: int = -1
    observe_id: int = -1
    think_start_id: int = -1
    think_end_id: int = -1
    user_id: int = -1
    assistant_id: int = -1
    system_id: int = -1
```

**Changes**:
- Remove all Mamba/Attention architecture params (d_state, d_conv, expand, dt_rank, d_head, attn_window, etc.)
- `tokenizer_setup.py` config hydration replaced by `tokenizer.add_tokens()`
- `backbone_id` makes the backbone configurable at runtime

### 2. `lora.py` — Target Qwen layer names instead of AgentMind

Target layers change from:
```
["in_proj", "out_proj", "o_proj", "q_proj", "v_proj", "lm_head"]
```
to:
```
["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

These are the standard transformer Linear layers in Qwen2.5. The LoRALinear class itself doesn't change — just the target names.

**Change**: line 38 of `lora.py` — update `targets` default.

### 3. `data/pipeline.py` — Replace format_sample with chat template

**Current**: `format_sample()` builds strings like `<|system|>content<|user|>content<|assistant|>content<eos>`

**New**: Use the backbone's chat template (e.g., Qwen's `<|im_start|>user\n...<|im_end|>\n`) with special tokens added after `<|im_end|>`:

```
<|im_start|>system
You are an AI assistant with tools...<|im_end|>
<|im_start|>user
Search arxiv for Mamba papers<|im_end|>
<|im_start|>assistant
<|tool_call|>{"name": "web_search", "args": {"query": "Mamba SSM"}}<|im_end|>
```

**Approach**:
1. Tokenize using the backbone's native tokenizer + chat template
2. Append special tokens manually *after* template tokenization
3. Or: pre-pend special tokens as separate messages and let the template handle them

**Implementation** — in `training_utils.py`, replace `format_sample()`:

```python
def format_sample(sample: dict, tokenizer=None, cfg=None) -> str:
    """Apply chat template then inject agent special tokens."""
    messages = sample["messages"]
    # Apply backbone's chat template (e.g. Qwen's im_start/im_end)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return text
```

Then encode normally. Labels still mask non-assistant tokens.

### 4. `train.py` — Remove MTP, simplify model creation

**Changes**:
- `build_training_model()`: replace `AgentMind()` + `init_agentmind()` with `mlx_lm.load(cfg.backbone_id)`
- `make_train_step()`: remove MTP head loss, remove `return_mtp` flag
- `TRAIN_CFG`: remove `use_mtp`, `mtp_weight`, `mtp_start`
- `_build_teacher_models()`: same model loading logic
- `distill_backbone()`: remove MTP from loss computation

The `train_specialist()` function body stays the same — it already uses LoRALinear wrappers and only trains A/B weights.

### 5. `training_orchestrator.py` — Update model init, remove pretrain load

**Changes**:
- Remove `AgentMind()` + `init_agentmind()` + `apply_lora()` — replaced by `mlx_lm.load()` then `apply_lora()`
- Remove backbone loading from `backbone.npz` / `.safetensors` — the model is now downloaded/locally cached via HF
- `run_round()` stays the same — calls `train_specialist()` and `distill_backbone()` identically

### 6. `agent.py` — Replace forward_with_state with KV cache generation

**Current**: Custom `forward_with_state()` maintains SSM `h_states` dict across turns.

**New**: Use `mlx_lm.generate()` or `mlx_lm.stream_generate()` with KV cache. The `h_states` concept becomes `cache` in mlx-lm's generation API.

**Changes**:
- `AgentLoop.__init__()`: store `cache` instead of `h_states`
- `AgentLoop.run()`: use `mlx_lm.utils.generate_step()` with past KV cache
- Router dispatch: backbone last hidden state comes from the last token's logits
- Specialist switching: clear and rebuild KV cache
- `_handle_tool_call()`: stays the same (parses JSON from output)

Approximate code shape:
```python
from mlx_lm.utils import generate_step
from mlx_lm.models.qwen2 import Qwen2Model

class AgentLoop:
    def __init__(self, model, tokenizer, ...):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = []

    def _generate(self, prompt_ids, max_tokens=200, temp=0.7):
        self.cache = []
        y = prompt_ids
        for i in range(max_tokens):
            logits, self.cache = generate_step(y, self.model, self.cache, temp=temp)
            y = mx.argmax(logits, keepdims=True)
            token = y.item()
            yield token
            if token == self.eos_id:
                break
```

### 7. `router.py` — Use backbone's last hidden state (no forward_with_state)

**Current**: `router.train()` calls `backbone.forward_with_state(ids, {})` then reads `backbone.last_hidden[:, -1, :]`.

**New**: Run backbone forward pass with `return_hidden=True` or access the last layer's hidden state.

With mlx-lm, the model returns `(logits, cache)` from `__call__`. To get hidden states, we need to modify the forward path slightly:

```python
class BackboneWithHidden:
    def __init__(self, model):
        self.model = model
    def __call__(self, x):
        # Forward all layers, capture last hidden before lm_head
        # mlx-lm models return (logits, cache) from __call__
        # We need to intercept the last hidden state
        logits, cache = self.model(x)  # logits shape: (1, L, V)
        # Qwen2.5's lm_head is separate — we can forward without it
        return logits, cache
```

Actually, for the router we just need one forward pass to get the last-token hidden state. Simplest approach: forward the backbone without the lm_head. The mlx-lm `Qwen2Model` has a method for this.

Alternative: use the `logits` to approximate — but router training expects hidden states not logits. Let me check how hard it is.

In mlx-lm's Qwen2 model:
```python
class Qwen2Model(nn.Module):
    def __call__(self, x, cache=None):
        # embeddings → transformer layers → norm → return hidden
        ...
```

So we can use `model.model` (the inner transformer) to get hidden states directly.

### 8. `interactive_test.py` — Update model loading

Replace:
```python
from model.agent_lm import AgentMind
from init import init_agentmind
```
with:
```python
from mlx_lm import load as load_model
model, tokenizer = load_model("Qwen/Qwen2.5-0.5B")
```

### 9. `eval.py` — Remove forward_with_state, use KV cache generation

Same pattern as `agent.py`: replace SSM state generation with KV cache generation.

## Files that Stay (minor or no changes)

| File | Notes |
|---|---|
| `training_utils.py` | `cross_entropy_loss`, `clip_gradients`, `clone_tree`, helpers — no model dependency |
| `scheduler.py` | `CosineWarmupScheduler` — optimizer-agnostic |
| `monitor.py` | `ResourceScheduler`, `print_hw` — no model dependency |
| `stats_logger.py` | Logging — no model dependency |
| `decode.py` | Tool call validation — text-only, no model dependency |
| `generate_scaled_synthetic.py` | Data generation — no model dependency |
| `data/` (jsonl files) | Data format changes minimally (just template) |
| `router.py` (core class) | `TaskRouter` classifier — only the forward pass changes |
| `training_orchestrator.py` (flow) | Orchestration loop, round mgmt — only model init changes |

## Phase 1: Quick Proof (Day 1)

Goal: a running agent loop with tool protocol.

1. Install Qwen2.5-0.5B locally
2. Rewrite `config.py` — remove AgentMind arch, add `backbone_id`
3. Create `model_loader.py` — `load_backbone(backbone_id)` → model + tokenizer with special tokens added
4. Rewrite `lora.py` target list
5. Rewrite `agent.py` — KV cache generation, no more SSM state
6. Test: `python agent.py` with a single query

**Output**: working agent that can generate tool calls with the pretrained backbone.

## Phase 2: Training (Days 1-2)

Goal: train tool_caller specialist via LoRA.

1. Rewrite `data/pipeline.py` `format_sample()` — use chat template
2. Rewrite `train.py` — remove MTP, simplify
3. Rewrite `training_orchestrator.py` — new model init
4. Run: `python training_orchestrator.py --rounds 1`
5. Test: generate with the trained specialist

**Output**: tool_caller specialist that reliably emits `<|tool_call|>{"name": "...", "args": {...}}`.

## Phase 3: Full Stack (Days 2-3)

Goal: all specialists + router + distillation.

1. Train remaining specialists (planner, recovery, code, research)
2. Train router on cached hidden states
3. Run distillation
4. Full eval

**Output**: full apprenticeship pipeline on pretrained backbone.

## Risk Checklist

| Risk | Mitigation |
|---|---|
| Qwen 151K vocab → embedding layer ~155M params | Tied embeddings so lm_head shares. ~2.1GB total in fp16, fits 16GB RAM |
| Special tokens get fragmented by BPE tokenizer | Added via `tokenizer.add_tokens()`. Qwen's tokenizer adds them as single tokens at high IDs (similar to current approach) |
| `add_tokens()` grows the embedding matrix | Resized via `model.resize_vocab(n)` — standard HuggingFace pattern. Adds ~3K new embeddings for 14 tokens |
| Chat template strips `<|tool_call|>` | Apply template first, then append special tokens to the formatted string. Labels mask non-assistant turns |
| KV cache for 0.5B model uses more memory than SSM state | Qwen2.5-0.5B in fp16: ~900MB weights + ~100MB KV cache per 2K tokens. Fits 16GB |
| LoRA on 0.5B is ~6M trainable params (vs 2.36M on AgentMind) | 3× more but still tiny relative to backbone. Training is ~2× slower per step |
| Qwen's generation is O(L²) without KV cache | KV cache makes it O(L × d_model × n_layers). For 512-token seq, ~25M FLOPs vs Mamba's ~5M — slower but acceptable |
| Router needs hidden states, not logits | Use `model.model(input_ids, cache)` to get last hidden before lm_head |
| `backbone.unfreeze()` in distillation unfreezes 0.5B | Potentially large grads. Use low lr (1e-5) and gradient clipping |
