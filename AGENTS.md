# AgentMind — Cognitive Apprenticeship LM

Custom ~147M hybrid Mamba SSM + local attention LM on Apple **MLX** (Apple Silicon only). Trains per-domain LoRA specialists, distills them into the backbone, and routes via a classifier.

## Key commands

```bash
# Full training pipeline:
python training_orchestrator.py --rounds 1-5 --save-dir ./checkpoints
python training_orchestrator.py --rounds 1,3,5 --save-dir ./checkpoints    # selected rounds
python training_orchestrator.py --rounds 1-3 --resume ./checkpoints        # resume

# Agentic inference:
python agent.py --backbone ./checkpoints --adapters ./checkpoints/adapters \
    --router ./checkpoints/router --query "Search arxiv for SSM papers"

# Data generation:
python generate_scaled_synthetic.py                # synthetic per-domain JSONL
python prepare_data/run_all.py                     # HF + synthetic hybrid
python pretokenize.py                              # JSONL → .npz (faster loading)

# Pretrain backbone (before apprenticeship rounds):
python pretrain_backbone.py --steps 500

# Evaluate:
python eval.py

# Engineering practices reference:
cat docs/engineering_practices.md

# Export:
python export.py --checkpoint ./checkpoints/step_N --out ./agentmind --bits 4
```

Smoke-test with `python -c "..."` commands (many exist in `instructs.md`).

## Architecture

```
backbone (147M frozen) + LoRA adapter (2.36M trainable) → per-domain specialist
                                                                     ↓
frozen teachers ──→ KL distillation ──→ backbone absorbs all domains
                                                                     ↓
                     Router (65K params) selects specialist at inference
```

**Layer pattern** (default 12 layers, attn_every=2): `[M A]` repeated (6 Mamba + 6 Attention); export uses 16 layers, attn_every=4.

## Round schedule (`config.py` `APPRENTICE_ROUNDS`)

| Domain | Spec steps | Seq len | Seq len schedule | Distill steps | Latent stage |
|---|---|---|---|---|---|
| tool_caller | 2000 | 256 | `{0: 384, 200: 512}` | 200 | 1 |
| planner | 300 | 512 | None | 150 | 2 |
| recovery | 300 | 256 | `{0: 128, 150: 256}` | 150 | 2 |
| code | 300 | 512 | None | 150 | 4 |
| research | 300 | 1024 | None | 150 | 4 |

Router only trains after ≥3 specialists exist.

## Critical gotchas

1. **Token IDs must be hydrated at runtime.** Never hardcode them. Always call `hydrate_config(cfg, tokenizer)` — otherwise comparisons against `cfg.tool_call_id`, `cfg.assistant_id`, etc. silently fail (~31900s, not small ints).

2. **Loss masks all tokens except assistant turns.** Labels for user/system are set to -100. If assistant_id doesn't match the actual `<|assistant|>` token, the entire loss computation is corrupted. Verified at runtime by `assert_token_ids_real()`.

3. **Latent reasoning stages** (`model/latent.py`):
   - Stage 1 (normal), Stage 2 (wrap CoT in think boundaries), Stage 3 (50% latent), Stage 4 (full latent — no CoT, only placeholders)
   - `latent_loss_mask()` zeros loss between `<|think_start|>` and `<|think_end|>`
   - If think_start_id/think_end_id don't match the actual tokenizer IDs, the curriculum is a no-op.

4. **MTP is enabled by default during distillation and standalone training:**
   - `TRAIN_CFG["use_mtp"] = True` — activates in standalone `train()` after step 500
   - `train_specialist()` always passes `return_mtp=False` (backbone frozen, MTP head not trainable — would add noise through random projections)
   - Orchestrator passes `mtp_weight=0.2` to `distill_backbone()` — activates after step 20 warm-up
   - `MTPHead` (K=4) IS instantiated in the model and always runs in `forward_with_state()` — no speed impact since inference is rare
   - Speed note: MTP only runs during distillation (~150-200 steps/round), NOT during specialist training (~3200 total steps). Adds ~4× LM head compute but only on a small fraction of total training.

5. **Debug generation test prompt** must use proper token IDs, not literal strings:
   `[cfg.bos_id, cfg.user_id] + tok.encode("query") + [cfg.assistant_id]`
   Raw `">>> query"` is OOD and produces meaningless eval traces.

6. **`boundary_weight=1.5`** for first 300 steps only, on `{tool_call_id, observe_id}`. Removed entirely after step 300. Too high or too long causes the model to fire `<|tool_call|>` eagerly without learning the JSON structure after it.

7. **LoRA adapters** (~9MB `.safetensors`). `apply_lora()` wraps `in_proj`, `out_proj`, `o_proj`, `q_proj`, `v_proj`, `lm_head`. `apply_lora()` freezes everything else. Saved via `save_adapter()`, loaded via `load_adapter()`. Reset between specialists via `reset_adapter()`.

8. **Sequence length curriculum** (`seq_len_schedule`) — starts at 128, grows to 256/512. Pre-filter drops samples where assistant content doesn't fit within current seq_len. Re-checks at each curriculum step.

9. **All data is JSONL:** `{"domain": str, "type": str, "messages": [{"role": str, "content": str}]}`. Roles: `system`, `user`, `assistant`. Assistant content uses special tokens `<|tool_call|>`, `<|observe|>`, `<|scratch|>`, `<|plan|>` inline.

10. **batch_size=1 throughout** (16GB RAM constraint on MacBook Air). Effective batch via `grad_accum=8`.

11. **Data pipeline does 95/5 train/val split** inline (see `data/pipeline.py` `AgentDataset`). `from_data_files()`, `from_list()`, `from_hf()` classmethod factories for flexible construction.

## Special tokens (14 total)

IDs are at high vocabulary positions (~31987–31999), resolved at runtime via SentencePiece:
`<|tool_call|>`, `<|plan|>`, `<|memory|>`, `<|scratch|>`, `<|observe|>`,
`<|think_start|>`, `<|think_end|>`, `<|system|>`, `<|user|>`, `<|assistant|>`,
plus `<pad>` (0), `<s>` (1 = BOS), `</s>` (2 = SentencePiece EOS, NOT `<eos>`), `<unk>` (3).

**Critical**: `<eos>` is user-defined at ~31998, distinct from SentencePiece's built-in `</s>` (id=2). Never use `tokenizer.eos_id()` — use `tokenizer.piece_to_id("<eos>")` instead.

## Training infrastructure

- **train.py**: `train()` (standalone), `train_specialist()` (API for orchestrator, trains LoRA only, backbone frozen), `distill_backbone()` (unfreezes backbone, CE + KL + MTP after step 20)
- **training_utils.py**: Shared helpers — `GradientAccumulator`, `NaNRecovery`, `BatchLogger`, `compute_loss()`, `freeze_backbone()`, `create_optimizer_scheduler()`
- **pretrain_backbone.py**: Initial backbone pretraining on domain data, uses seq_len schedule `{0: 128, 500: 256, 1500: 384, 2500: 512}`
- **NaN recovery**: All training loops detect non-finite loss/gradients, rollback params + optimizer state, zero gradients, skip batch
- **CosineWarmupScheduler** (`scheduler.py`): Linear warmup → cosine decay, `min_lr_ratio=0.1`
- **ResourceScheduler** (`monitor.py`): Monitors CPU/RAM/Swap in background thread, triggers GC when thresholds exceeded. Integrated into `train_specialist()`, `distill_backbone()`, and `pretrain_backbone()`.
- **stats_logger.py**: Logs all training events to `logs/training.jsonl`
- **apprentice.py**: Deleted (logic consolidated into training_orchestrator.py and train.py)

## Router training (`router.py`)

- Tiny classifier: `Linear(d_model=1024 → 64) → ReLU → Linear(64 → n_domains)` (~65K params)
- Uses **last-position** hidden state (not mean-pool) — must be consistent between train and inference
- Falls back to `"tool_caller"` when max softmax < 0.6
- `router.train()` caches hidden states (one backbone forward pass per sample), then trains classifier on cached states
- Saved as `.safetensors` with domain_names in metadata

## Agent inference (`agent.py`)

- SSM state (`h_states`) persists across the entire session — NOT reset on specialist switch
- Router dispatch: one backbone forward → router selects specialist → load adapter → generate
- 3 built-in tools: `web_search` (DuckDuckGo), `run_python` (subprocess, 10s timeout), `read_file` (max 10KB)
- Supports both single-query (`--query`) and interactive modes

## Model architecture

- **MambaBlock**: chunked compiled scan (CHUNK=16, d_state=64), O(1) recurrent `step()` for inference. SSM state contains `ssm_state` (B, d_inner, d_state) + `conv_state` (B, d_conv-1, d_inner)
- **LocalAttentionBlock**: sliding window (512), RoPE, SwiGLU FFN. Fires every `attn_every` layers
- **MTPHead**: K=4 auxiliary prediction heads, disabled when backbone frozen
- Tied embeddings (32K vocab × 1024 dim)
- **`model/hybrid_block.py`**: Empty file (dead code, not imported)

## Output files

```
checkpoints/
├── backbone.npz / backbone.safetensors  # final backbone weights
├── router/                               # TaskRouter state
├── adapters/
│   ├── tool_caller.safetensors           # LoRA adapter per domain
│   ├── planner.safetensors
│   ├── recovery.safetensors
│   ├── code.safetensors
│   └── research.safetensors
├── step_XXXXX/                           # intermediate checkpoints
│   ├── weights.npz
│   └── log.json
```

## Dependencies

```
mlx, mlx-lm, sentencepiece, datasets, orjson, msgspec, transformers, tqdm, numpy
```

## Hardware notes

- Runs on 16GB MacBook Air. RAM peaks at ~2GB during training.
- Swap thrashing observed with seq_len > 1024 or batch_size > 1.
- Mamba CHUNK=16 is the confirmed sweet spot for Metal compilation on Apple Silicon.
