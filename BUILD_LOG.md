# AgentMind — Build Log

> Hybrid SSM + Local Attention Language Model for Agentic AI
> Target: ~600M params · 16GB MacBook Air · MLX Backend

---

## Phase 1: Project Scaffolding

### Directory Structure
```
agentmind/
├── data/
│   ├── __init__.py
│   ├── formats.py          # JSONL schema definitions
│   ├── pipeline.py         # Dataset loading, formatting, batching
│   └── synthetic.py        # Tool call trajectory generator
├── model/
│   ├── __init__.py
│   ├── mamba_block.py      # Selective State Space Model block
│   ├── attention_block.py  # Sliding window local attention
│   ├── hybrid_block.py     # (reserved)
│   ├── agent_lm.py         # Full AgentMind model
│   ├── rope.py             # Rotary Position Embeddings
│   ├── conv_state.py       # Conv buffer for single-step inference
│   ├── mtp_head.py         # Multi-Token Prediction auxiliary head
│   └── latent.py           # think_start / think_end training logic
├── config.py               # AgentMindConfig dataclass
├── lora.py                 # LoRALinear + model wrapping
├── init.py                 # Mamba-specific weight initialization
├── scheduler.py            # Cosine LR with warmup
├── train.py                # Complete training loop
├── eval.py                 # Perplexity + tool call accuracy
├── export.py               # GGUF export with custom arch map
├── agent.py                # Agentic inference loop
└── requirements.txt        # Dependencies
```

### Dependencies (`requirements.txt`)
```
mlx
mlx-lm
sentencepiece
datasets
orjson
msgspec
transformers
tqdm
numpy
```

---

## Phase 2: Core Configuration

### `config.py` — AgentMindConfig
- **Vocabulary**: 32,000 tokens
- **Model**: d_model=2048, n_layers=24
- **Mamba SSM**: d_state=128, d_conv=4, expand=2, dt_rank=auto (128)
- **Attention**: 16 heads, local window=512, every 4th layer
- **FFN**: SwiGLU with 8/3 multiplier, aligned to 256
- **Special tokens**: 10 agentic control tokens (tool_call, plan, memory, scratch, observe, think_start/end, system, user, assistant)
- **Properties**: `d_inner`, `dt_rank_val`, `ffn_hidden`, `is_attn_layer(i)`, `param_count_estimate` (~855M raw)

---

## Phase 3: Model Components

### `model/rope.py` — Rotary Position Embeddings
- `precompute_rope(head_dim, max_seq_len, base=10000.0)` — precomputes sin/cos tables
- `apply_rope(x, cos, sin, offset=0)` — applies rotary embeddings to [B, n_heads, seq_len, head_dim]
- Verified: output shape (1, 8, 16, 128) ✓

### `model/mamba_block.py` — MambaBlock
- Pre-norm RMSNorm → split into x (signal) + z (gate)
- Causal depthwise conv (padding=d_conv-1, left-pad only)
- SSM projections: x_proj → dt, B, C matrices
- ZOH discretization: dA = exp(dt * A), dB = dt * B
- Sequential scan loop: h [B, d_inner, d_state]
- `step()` method for single-token O(1) inference
- Verified: output (1, 16, 2048), hidden state (1, 4096, 128) ✓

### `model/attention_block.py` — LocalAttentionBlock
- Sliding window attention O(L × window), not O(L²)
- RoPE integrated into q and k projections
- SwiGLU FFN (gate_proj, up_proj, down_proj)
- Precomputed RoPE tables stored as self.rope_cos, self.rope_sin
- Verified: output (1, 32, 2048) ✓

### `model/agent_lm.py` — AgentMind (Full Model)
- 24 blocks: 18 Mamba + 6 Attention (3:1 ratio)
- Pattern: [M M M A] × 6
- Tied embedding weights to lm_head
- `forward_with_state()` preserves SSM state across calls
- MTP integration: `self.mtp = MTPHead(cfg, K=4)`, stores `self.last_mtp_logits`
- Verified: logits (1, 8, 32000), 18 SSM states ✓

### `model/mtp_head.py` — Multi-Token Prediction
- `MTPHead(cfg, K=4)` — shared projection + 4 independent heads
- `mtp_loss()` — auxiliary loss, each head predicts k+1 steps ahead
- Verified: 4 outputs, each (1, 16, 32000) ✓

### `model/conv_state.py` — ConvState
- Manages sliding conv buffer for Mamba's causal depthwise conv
- `step(x_t, conv_weight, conv_bias)` — single-token conv during autoregressive inference
- Buffer: last (d_conv - 1) input vectors

---

## Phase 4: Weight Initialization

### `init.py` — init_agentmind(model, cfg)
- Standard linear layers: std = 0.02 / sqrt(2 * n_layers)
- **dt_proj bias**: log-uniform sampling between dt_min (1e-4) and dt_max (1e-1), then inverse softplus
- **A_log**: broadcast arange(1, d_state+1), stored as log
- **D**: ones (full skip connection)
- Embedding: normal * 0.02
- RMSNorm: weight = ones
- Verified: runs without errors ✓

---

## Phase 5: Training Infrastructure

### `tokenizer_setup.py` — Tokenizer Training
- SentencePiece BPE, vocab_size=32,000
- 10 special tokens registered as user_defined_symbols
- byte_fallback=True, split_digits=True
- Trained on curated corpus (see Phase 6)

### Tokenizer Results
| Token | ID |
|---|---|
| `<pad>` | 0 |
| `<bos>` | 1 |
| `<eos>` | 2 |
| `<|tool_call|>` | 4 |
| `<|plan|>` | 5 |
| `<|memory|>` | 6 |
| `<|scratch|>` | 7 |
| `<|observe|>` | 8 |
| `<|think_start|>` | 9 |
| `<|think_end|>` | 10 |
| `<|system|>` | 11 |
| `<|user|>` | 12 |
| `<|assistant|>` | 13 |

Roundtrip encoding/decoding verified ✓

---

## Phase 6: Data Pipeline

### Corpus Construction (`build_corpus.py`)
| Source | Lines | Purpose |
|---|---|---|
| FineWeb | 20,001 | General text, reasoning, instruction following |
| The Stack (Python) | 9,904 | Code structure, JSON, function patterns |
| UltraChat | 63,086 | Multi-turn dialogue, system prompts |
| **Total** | **92,991** | **189.8 MB** |

### Synthetic Data (`generate_synthetic.py`)
- Used Cerebras API (llama3.1-8b) for diverse samples
- 1,703 samples generated across 4 types:
  - `instruction` (500) — simple Q&A
  - `tool_single` (500) — one tool call chain
  - `agent_multi` (500) — 3-4 step tool chains with plans
  - `recovery` (200) — tool errors + scratch reasoning + retry
  - Cerebras-generated (3) — additional diversity

### Final Corpus
- **Size**: 190.5 MB
- **Format**: Plain text + JSONL synthetic data appended
- **Special tokens**: All present in training data

---

## Architecture Decisions

### Why 3:1 Mamba-to-Attention Ratio?
- Mamba carries long tool history, compressed into fixed state
- Attention fires every 4th layer for precise token recall and structured output
- Local window (512) keeps attention cost O(L) not O(L²)

### Why LoRA?
- 16GB MacBook Air memory constraint
- ~1.2GB weights (fp16) + ~100MB LoRA params + ~3GB activations = ~5GB total
- Targets: in_proj, out_proj, q_proj, v_proj, lm_head

### Why MTP (Multi-Token Prediction)?
- Forces model to think ahead — improves instruction following
- K=4 heads with shared projection to avoid parameter explosion
- Auxiliary loss weight: 0.2-0.3

### Memory Budget
| Phase | What | Size |
|---|---|---|
| Training (LoRA) | Weights fp16 | ~1.2GB |
| Training (LoRA) | LoRA params + optimizer | ~100MB |
| Training (LoRA) | Activations (batch=1, seq=2048) | ~3GB |
| **Training total** | | **~5GB ✓** |
| Inference (4-bit) | Weights GGUF | ~300MB |
| Inference (4-bit) | SSM state (constant) | ~2MB |
| **Inference total** | | **<1GB ✓** |

---

## Files Created/Modified

| File | Status | Description |
|---|---|---|
| `config.py` | ✅ Complete | AgentMindConfig with all properties |
| `model/rope.py` | ✅ Complete | RoPE precompute and apply |
| `model/mamba_block.py` | ✅ Complete | Full MambaBlock with step() |
| `model/attention_block.py` | ✅ Complete | LocalAttentionBlock with RoPE |
| `model/agent_lm.py` | ✅ Complete | AgentMind with MTP integration |
| `model/mtp_head.py` | ✅ Complete | MTPHead + mtp_loss |
| `model/conv_state.py` | ✅ Complete | ConvState for inference |
| `init.py` | ✅ Complete | Mamba-specific weight init |
| `tokenizer_setup.py` | ✅ Complete | SentencePiece training + loading |
| `build_corpus.py` | ✅ Complete | Multi-source corpus builder |
| `generate_synthetic.py` | ✅ Complete | Cerebras-powered synthetic data |
| `data/corpus.txt` | ✅ Built | 190.5 MB training corpus |
| `data/synthetic_agents.jsonl` | ✅ Built | 1,703 synthetic samples |
| `agentmind_tok.model` | ✅ Trained | 32K vocab BPE tokenizer |
| `agentmind_tok.vocab` | ✅ Generated | Vocabulary file |

---

## Remaining Work

### Training Infrastructure (Not Yet Implemented)
- [ ] `data/pipeline.py` — AgentDataset, collate_batch, make_dataloader
- [ ] `data/formats.py` — JSONL schema definitions
- [ ] `lora.py` — LoRALinear + apply_lora
- [ ] `scheduler.py` — CosineWarmupScheduler
- [ ] `train.py` — Complete training loop with gradient accumulation
- [ ] `eval.py` — Perplexity + tool call accuracy
- [ ] `export.py` — GGUF export
- [ ] `agent.py` — Agentic inference loop
- [ ] `model/latent.py` — Latent reasoning training logic

### Training Curriculum
1. **Phase 1** — Format Bedrock (500 steps): instruction pairs, JSON formatting
2. **Phase 2** — Tool Calling (800 steps): synthetic tool call → observe → answer
3. **Phase 3** — Multi-step Agents (1000 steps): 3-8 round trajectories
4. **Phase 4** — Failure Recovery (700 steps): error injection + recovery
5. **Phase 5** — Latent Reasoning (500 steps, optional): think_start/end training

---

## Quick Start (Next Steps)

```bash
# 1. Implement remaining files (pipeline, lora, scheduler, train, eval, export, agent)
# 2. Run training
python train.py

# 3. Evaluate
python eval.py --checkpoint checkpoints/step_03000

# 4. Export to 4-bit
python export.py --checkpoint checkpoints/step_03000 --out agentmind-4bit

# 5. Run agent
python agent.py --model agentmind-4bit --query "your query"
```
