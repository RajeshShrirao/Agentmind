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
├── pretokenize.py          # Pre-tokenize dataset for faster loading
├── requirements.txt        # Dependencies
├── BUILD_LOG.md            # This file
├── instructs.md            # Training instructions and guides
├── docs/
│   ├── agentmind_architecture.md
│   └── agentmind_training_infra.md
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
- **Model**: d_model=1024 (was 2048), n_layers=16 (was 24)
- **Mamba SSM**: d_state=16 (was 128), d_conv=4, expand=2, dt_rank=auto (64)
- **Attention**: 8 heads (was 16), local window=256 (was 512), every 4th layer
- **FFN**: SwiGLU with 8/3 multiplier, aligned to 256
- **Special tokens**: 10 agentic control tokens (tool_call, plan, memory, scratch, observe, think_start/end, system, user, assistant)
- **Properties**: `d_inner`, `dt_rank_val`, `ffn_hidden`, `is_attn_layer(i)`, `param_count_estimate` (~145M raw)
- **Config halved to fit 16GB Mac**: d_model=1024, n_layers=16, d_state=16, n_heads=8, attn_window=256

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
- Sequential scan loop for numerical stability (parallel log-space scan caused NaN from 0×inf underflow/overflow)
- `step()` method for single-token O(1) inference
- Verified: output (1, 16, 2048), hidden state (1, 1024, 16) ✓

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
| `<eos>` | 5 |
| `<|tool_call|>` | 6 |
| `<|plan|>` | 7 |
| `<|memory|>` | 8 |
| `<|scratch|>` | 9 |
| `<|observe|>` | 10 |
| `<|think_start|>` | 11 |
| `<|think_end|>` | 12 |
| `<|system|>` | 13 |
| `<|user|>` | 14 |
| `<|assistant|>` | 15 |

Roundtrip encoding/decoding verified ✓

---

## Phase 6: Data Pipeline

### Corpus Construction (`build_corpus.py`)
| Source | Lines | Purpose |
|---|---|---|
| FineWeb | 20,001 | General text, reasoning, instruction following |
| The Stack (Python) | 9,904 | Code structure, JSON, function patterns |
| UltraChat | 63,086 | Multi-turn dialogue, system prompts |
| AgentInstruct | ~5,000 | High-quality agent trajectories |
| ToolBench | ~3,000 | Tool calling patterns |
| WebArena | ~3,000 | Web navigation agent data |
| **Total** | **~104,000** | **~250MB** |

### Synthetic Data - Scaled (`generate_scaled_synthetic.py`)
- **11,500 samples** generated via template-based generation
- Rate-limited Cerebras API (40 req/min) for high-quality diversity
- 14 tools in registry with realistic args/results

| Type | Count | Purpose |
|---|---|---|
| `instruction` | 3,000 | Simple Q&A, instruction following |
| `tool_single` | 2,500 | One tool call chain |
| `agent_multi` | 3,000 | 2-5 step tool chains with `<|plan|>` |
| `recovery` | 2,000 | Tool errors + `<|scratch|>` reasoning + retry |
| `latent` | 1,000 | `<|think_start|>`...`<|think_end|>` patterns |
| **Total** | **11,500** | **6.2 MB** |

### Final Training Data
- **Corpus**: ~250 MB (general text + code + dialogue + agent datasets)
- **Synthetic JSONL**: 11,500 structured agent trajectories
- **Special tokens**: All present in both corpus and synthetic data

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
- **6M trainable params** (~1% of total)

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
|---|---|---|---|
| `config.py` | ✅ Complete | AgentMindConfig with all properties + 13 special token IDs |
| `model/rope.py` | ✅ Complete | RoPE precompute and apply |
| `model/mamba_block.py` | ✅ Complete | Full MambaBlock with parallel scan + step() |
| `model/attention_block.py` | ✅ Complete | LocalAttentionBlock with RoPE |
| `model/agent_lm.py` | ✅ Complete | AgentMind with MTP integration |
| `model/mtp_head.py` | ✅ Complete | MTPHead + mtp_loss |
| `model/conv_state.py` | ✅ Complete | ConvState for inference |
| `model/latent.py` | ✅ Complete | Latent reasoning training logic |
| `init.py` | ✅ Complete | Mamba-specific weight init |
| `tokenizer_setup.py` | ✅ Complete | SentencePiece training + loading |
| `build_corpus.py` | ✅ Complete | Multi-source corpus builder (6 datasets) |
| `generate_synthetic.py` | ✅ Complete | Cerebras-powered synthetic data |
| `generate_scaled_synthetic.py` | ✅ Complete | 11.5K samples with rate limiting |
| `pretokenize.py` | ✅ Complete | Pre-tokenize dataset to .npz for 2x faster loading |
| `data/formats.py` | ✅ Complete | JSONL schemas + validate_sample() |
| `data/synthetic.py` | ✅ Complete | 14 tools, trajectory generators |
| `data/pipeline.py` | ✅ Complete | AgentDataset (pre-tokenized + raw), collate, dataloader |
| `lora.py` | ✅ Complete | LoRALinear + apply_lora (6M trainable params) |
| `scheduler.py` | ✅ Complete | CosineWarmupScheduler |
| `train.py` | ✅ Complete | Full training loop with grad accum, clipping, seq curriculum, lazy MTP |
| `eval.py` | ✅ Complete | Perplexity, tool_call_accuracy, format_adherence |
| `data/corpus.txt` | ✅ Built | 199.7 MB training corpus |
| `data/scaled_synthetic.jsonl` | ✅ Built | 11,500 synthetic samples (6.5MB) |
| `data/train_ids.npz` | ✅ Built | Pre-tokenized training inputs (98MB) |
| `data/train_labels.npz` | ✅ Built | Pre-tokenized training labels (98MB) |
| `agentmind_tok.model` | ✅ Trained | 32K vocab BPE tokenizer (0.8MB) |
| `agentmind_tok.vocab` | ✅ Generated | Vocabulary file |
| `instructs.md` | ✅ Complete | Training instructions and guides |
| `BUILD_LOG.md` | ✅ Current | Development build log |

---

## Remaining Work

### Training Infrastructure (Not Yet Implemented)
- [ ] `export.py` — GGUF export with custom arch map
- [ ] `agent.py` — Agentic inference loop with real tools

### Pre-Training Verification (23/23 Passed ✅)
- [x] All model components implemented and smoke-tested
- [x] All training infrastructure implemented
- [x] Data pipeline loads 10,925 samples correctly
- [x] Label masking works: system/user masked (-100), assistant unmasked
- [x] Forward pass produces correct logits shape
- [x] Loss computation works (10.35 for untrained model)
- [x] All imports verified
- [x] Tokenizer trained with all 13 special tokens
- [x] LoRA applied: 6M trainable params (~1% of total)

### Known Issue: Memory Pressure on 16GB Mac
Training the 161M param model on 16GB Mac works with these mitigations:
- batch_size=1 with grad_accum=8 instead of batch_size=2
- No `mx.compile` on the train step — lazy evaluation avoids materializing all intermediates
- seq_len schedule starts at 256 instead of 512
- MTP disabled by default (enable after memory is stable)
- Pre-tokenized data avoids tokenizer overhead
- **d_state=16** reduces SSM intermediates from 18.4GB to 403MB (45× reduction)
- **Parallel scan with numerical clipping**: log_contrib ∈ [-50, 50] prevents NaN from 0×inf underflow/overflow

### Performance Optimizations Applied
| Optimization | Impact | Status |
|---|---|---|
| Parallel scan with numerical clipping | 460-495 tok/s (vs 30 raw loop, vs 534 unclipped) | ✅ |
| Pre-tokenized dataset | 2x faster (no on-the-fly tokenization) | ✅ |
| Sequence length curriculum (256→512→1024) | 4x faster early training | ✅ |
| Lazy MTP (enabled after step 500) | Saves 20% memory | ✅ |
| batch_size=1, grad_accum=8 | Prevents OOM on 16GB Mac | ✅ |
| Dataloader reuse with indices shuffle | Eliminates overhead | ✅ |
| eval_every=500 | Reduces eval bottleneck | ✅ |

### Memory Budget (Revised for 16GB Mac)
| Phase | What | Size |
|---|---|---|
| Training (LoRA) | Weights fp16 | ~1.2GB |
| Training (LoRA) | LoRA params + optimizer | ~100MB |
| Training (LoRA) | Activations (batch=1, seq=256) | ~1.5GB |
| Training (LoRA) | Parallel scan intermediates | ~2GB peak |
| **Training total** | | **~5GB ✓** |
| Inference (4-bit) | Weights GGUF | ~300MB |
| Inference (4-bit) | SSM state (constant) | ~2MB |
| **Inference total** | | **<1GB ✓** |

### Estimated Training Time
- **Before optimizations**: ~42 hours
- **After optimizations**: ~6-8 hours (5-7x speedup)
- **Hardware**: 16GB MacBook Air M-series

### Training Curriculum
1. **Phase 1** — Format Bedrock (500 steps): instruction pairs, JSON formatting
2. **Phase 2** — Tool Calling (800 steps): synthetic tool call → observe → answer
3. **Phase 3** — Multi-step Agents (1000 steps): 3-8 round trajectories
4. **Phase 4** — Failure Recovery (700 steps): error injection + recovery
5. **Phase 5** — Latent Reasoning (500 steps, optional): think_start/end training

---

## Quick Start

```bash
# Everything is ready for training
python train.py

# Or resume from latest checkpoint
python train.py --resume latest

# After training completes:
python eval.py --checkpoint /Volumes/New Volume/checkpoints/step_03000
python export.py --checkpoint /Volumes/New Volume/checkpoints/step_03000 --out agentmind-4bit
python agent.py --model agentmind-4bit --query "your query"
```
