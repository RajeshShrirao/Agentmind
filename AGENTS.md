# AgentMind — Cognitive Apprenticeship Runtime

**Qwen2.5-0.5B** backbone + LoRA specialist adapters, trained via apprenticeship rounds and routed via classifier. Pivoted from custom Mamba/Attention pretraining after substrate failure (model collapsed to `.` output — 1% data coverage at 2000 steps).

## Key commands

```bash
# Full training pipeline:
python training_orchestrator.py --rounds 1-5 --save-dir ./checkpoints
python training_orchestrator.py --rounds 1,3,5 --save-dir ./checkpoints    # selected rounds
python training_orchestrator.py --rounds 1-3 --resume ./checkpoints        # resume

# Agentic inference:
python agent.py --backbone Qwen/Qwen2.5-0.5B --adapters ./checkpoints/adapters \
    --router ./checkpoints/router --query "Search arxiv for SSM papers"

# Data generation:
python generate_scaled_synthetic.py                # synthetic per-domain JSONL
python prepare_data/run_all.py                     # HF + synthetic hybrid

# Evaluate:
python eval.py

# Engineering practices reference:
cat docs/engineering_practices.md
```

## Architecture

```
Qwen2.5-0.5B (frozen) + LoRA adapter (6M trainable) → per-domain specialist
                                                                      ↓
frozen teachers ──→ KL distillation ──→ backbone absorbs all domains
                                                                      ↓
                     Router (65K params) selects specialist at inference
```

Backbone loaded via `mlx_lm.load()`. Special tokens added via `tokenizer._tokenizer.add_tokens()` (TokenizerWrapper lacks `add_tokens()`). Embedding NOT resized — 151936 slots have room for 10 special tokens (~151665-151674). MLX `Model` lacks `resize_vocab()`.

## Round schedule (`config.py` `APPRENTICE_ROUNDS`)

| Domain | Spec steps | Seq len | Seq len schedule | Distill steps | Latent stage |
|---|---|---|---|---|---|---|
| tool_caller | 2000 | 256 | `{0: 384, 200: 512}` | 200 | 1 |
| planner | 300 | 512 | None | 150 | 2 |
| recovery | 300 | 256 | `{0: 128, 150: 256}` | 150 | 2 |
| code | 300 | 512 | None | 150 | 4 |
| research | 300 | 1024 | None | 150 | 4 |

Router only trains after ≥3 specialists exist.

## Critical gotchas

1. **Data subset per epoch** — Dataloader samples 5K indices per epoch (not full dataset). Prevents the 1%-coverage bug that killed pretraining. Set in `train_specialist()` line 697.

2. **Loss masks all tokens except assistant turns.** Labels for user/system are set to -100. Uses Qwen's chat template to identify assistant spans. **Critical**: `<|im_start|>assistant` is two tokens (151644 + 77091), not one. `make_labels()` must check `ids[i] == im_start_id and ids[i+1] == assistant_id` — the old combined-token check returns `None` (all -100 labels, zero loss).

3. **Latent reasoning stages**: Stage 1 (normal), Stage 2 (wrap CoT in think boundaries), Stage 3 (50% latent), Stage 4 (full latent). `latent_loss_mask()` zeros loss between `<|think_start|>` and `<|think_end|>`.

4. **LoRA targets** differ from old AgentMind: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`. Not `in_proj`/`out_proj` — those are Mamba-specific.

5. **Special tokens added via `tokenizer._tokenizer.add_tokens()`** — not at fixed IDs. TokenizerWrapper (from `mlx_lm.load()`) lacks `add_tokens()` — must use the underlying HF tokenizer at `._tokenizer`. Always resolve at runtime via `tokenizer.convert_tokens_to_ids()`.

6. **`forward_with_state()` removed** — replaced by KV cache persistent across turns. `agent.py` uses `mlx_lm.utils.generate_step()` with past cache.

7. **No MTP** — removed with the old architecture. Distillation uses CE + KL only.

8. **No raw pretraining** — `pretrain_backbone.py` deprecated. The backbone is always a pretrained model from HuggingFace.

9. **All data is JSONL:** `{"domain": str, "type": str, "messages": [{"role": str, "content": str}]}`. Roles: `system`, `user`, `assistant`. Assistant content uses special tokens inline.

10. **batch_size=1 throughout** (16GB RAM constraint). Effective batch via `grad_accum=8`.

11. **LoRA adapters** (~9MB `.safetensors`). `apply_lora()` wraps `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Freezes everything else. Saved via `save_adapter()`, loaded via `load_adapter()`. Reset via `reset_adapter()`.

## Special tokens

Added to Qwen's tokenizer via `add_tokens()` at runtime:
`<|tool_call|>`, `<|plan|>`, `<|memory|>`, `<|scratch|>`, `<|observe|>`,
`<|think_start|>`, `<|think_end|>`, `<|system|>`, `<|user|>`, `<|assistant|>`.
These become single tokens at high IDs (>151936). Qwen's native tokens (`<|im_start|>`, `<|im_end|>`) are used for chat template boundaries.

## Training infrastructure

- **train.py**: `train_specialist()` (API for orchestrator, LoRA only, backbone frozen), `distill_backbone()` (unfreezes backbone, CE + KL)
- **mx.compile**: Loss function compiled via `mx.compile(loss_fn)` — caches computation graph, reduces step time from ~4s to ~400ms (10x speedup). Compiled function re-traces automatically when input shapes change (seq_len schedule).
- **training_utils.py**: Shared helpers — `GradientAccumulator`, `NaNRecovery`, `BatchLogger`, `compute_loss()`, `format_sample()`
- **NaN recovery**: All training loops detect non-finite loss/gradients, rollback params + optimizer state, zero gradients, skip batch
- **CosineWarmupScheduler** (`scheduler.py`): Linear warmup → cosine decay, `min_lr_ratio=0.1`
- **ResourceScheduler** (`monitor.py`): Monitors RAM/Swap, triggers GC when thresholds exceeded
- **stats_logger.py**: Logs all training events to `logs/training.jsonl`
- **No raw pretraining**: backbone is always Qwen2.5-0.5B loaded via `mlx_lm.load()`

## Router training (`router.py`)

- Tiny classifier: `Linear(d_model=896 → 64) → ReLU → Linear(64 → n_domains)` (~65K params)
- Uses **last-position** hidden state from backbone (before lm_head)
- Falls back to `"tool_caller"` when max softmax < 0.6
- `router.train()` caches hidden states (one backbone forward per sample), then trains classifier
- Saved as `.safetensors` with domain_names in metadata

## Agent inference (`agent.py`)

- **KV cache** persists across session — NOT reset on specialist switch (replaces old SSM `h_states`)
- Router dispatch: one backbone forward → router selects specialist → load adapter → generate
- 3 built-in tools: `web_search` (DuckDuckGo), `run_python` (subprocess, 10s timeout), `read_file` (max 10KB)
- Supports both single-query (`--query`) and interactive modes

## Model architecture

- **Qwen2.5-0.5B**: 24 transformer layers, RoPE, SwiGLU FFN, 896 hidden dim, 14 heads, 151K vocab
- Loaded via `mlx_lm.load()` — no custom layers
- KV cache for O(L) generation (no more SSM state)
- **No MTP, no Mamba, no custom attention** — the old `model/` directory is deleted

## Output files

```
checkpoints/
├── adapter_config.json                  # backbone_id, lora_rank, target_modules
├── router/                              # TaskRouter state
├── adapters/
│   ├── tool_caller.safetensors          # LoRA adapter per domain
│   ├── planner.safetensors
│   ├── recovery.safetensors
│   ├── code.safetensors
│   └── research.safetensors
├── step_XXXXX/                          # intermediate checkpoints
│   ├── adapter.safetensors
│   └── log.json
```

## Dependencies

```
mlx, mlx-lm, transformers, sentencepiece, datasets, orjson, msgspec, tqdm, numpy
```

## Hardware notes

- Runs on 16GB MacBook Air. Qwen2.5-0.5B in fp16: ~900MB weights + ~100MB KV cache per 2K tokens.
- Training peaks at ~3GB RAM (backbone frozen, LoRA only).
- Swap thrashing observed with seq_len > 1024 or batch_size > 1.
- Per-step timing: ~1s at seq=128, ~4s at seq=512 (backbone forward + LoRA backward). **With `mx.compile`**: ~200ms at seq=128, ~400ms at seq=256, ~1.6s at seq=512.
- **Seq=256, grad_accum=8, mx.compile**: ~400ms/step (5000 tok/s throughput). 2000 steps in ~13 min.
- Timing formula in train.py: `tok/s = seq_len * grad_accum * steps_since_last_log / elapsed` (steps_since matters — at step 0 it's 1, at step 100 it's 100).
