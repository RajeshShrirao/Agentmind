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

# Evaluate:
python eval.py

# Export:
python export.py --checkpoint ./checkpoints/step_N --out ./agentmind --bits 4
```

## Architecture

```
backbone (147M frozen) + LoRA adapter (2.36M trainable) → per-domain specialist
                                                                      ↓
frozen teachers ──→ KL distillation ──→ backbone absorbs all domains
                                                                      ↓
                              Router (65K params) selects specialist at inference
```

**Round schedule** (training_orchestrator.py ROUNDS):
| Domain | Spec steps | Seq len | Distill steps | Latent stage |
|---|---|---|---|---|
| tool_caller | 800 | 256 | 200 | 1 |
| planner | 300 | 512 | 150 | 2 |
| recovery | 300 | 256 | 150 | 2 |
| code | 300 | 512 | 150 | 4 |
| research | 300 | 1024 | 150 | 4 |

Router only trains after ≥3 specialists exist.

## Critical gotchas

1. **Token IDs must be hydrated at runtime.** Never hardcode them. Always call:
   `hydrate_config(cfg, tokenizer)` — otherwise comparisons against `cfg.tool_call_id`, `cfg.assistant_id`, etc. silently fail. This was the source of major bugs (see REMEDIATION_PLAN.md).

2. **Loss masks all tokens except assistant turns.** Labels for user/system are set to -100. If assistant_id doesn't match the actual `<|assistant|>` token, the entire loss computation is corrupted.

3. **Latent reasoning stages** (`model/latent.py`):
   - Stage 1 (normal), Stage 2 (wrap CoT in think boundaries), Stage 3 (50% latent), Stage 4 (full latent — no CoT, only placeholders)
   - `latent_loss_mask()` zeros loss between `<|think_start|>` and `<|think_end|>`
   - If think_start_id/think_end_id don't match the actual tokenizer IDs, the curriculum is a no-op.

4. **MTP is disabled during specialist training** (backbone frozen). Only enabled during distillation after step 20.

5. **Debug generation test prompt** must use proper token IDs, not literal strings:
   ``[cfg.bos_id, cfg.user_id] + tok.encode("query") + [cfg.assistant_id]``
   Raw `">>> query"` is OOD and produces meaningless eval traces.

6. **`boundary_weight=1.5`** for first 50 steps only, on `{tool_call_id, observe_id}`. Removed entirely after step 50. Too high or too long causes the model to fire `<|tool_call|>` eagerly without learning the JSON structure after it.

7. **LoRA adapters** are ~9MB each (`.safetensors`). `apply_lora()` wraps `in_proj`, `out_proj`, `o_proj`, `q_proj`, `v_proj`, `lm_head`. Saved via `save_adapter()`, loaded via `load_adapter()`.

8. **Sequence length curriculum** (`seq_len_schedule`) — starts at 128, grows to 256/512. Pre-filter drops samples where assistant content doesn't fit within current seq_len. Re-checks at each curriculum step.

9. **All data is JSONL:** `{"domain": str, "type": str, "messages": [{"role": str, "content": str}]}`. Roles: `system`, `user`, `assistant`. Assistant content uses special tokens `<|tool_call|>`, `<|observe|>`, `<|scratch|>`, `<|plan|>` inline.

## Special tokens (14 total)

IDs are at high vocabulary positions (~31987–31999), resolved at runtime:
`<|tool_call|>`, `<|plan|>`, `<|memory|>`, `<|scratch|>`, `<|observe|>`,
`<|think_start|>`, `<|think_end|>`, `<|system|>`, `<|user|>`, `<|assistant|>`,
plus `<pad>`, `<bos>`, `<eos>`, `<unk>`.

## Model architecture

- 12 or 16 layers: `[M, M, M, A]` pattern repeated (attn_every=3 or 4)
- **MambaBlock**: chunked compiled scan (CHUNK=16), O(1) recurrent `step()` for inference
- **LocalAttentionBlock**: sliding window (256), RoPE, SwiGLU FFN
- **MTPHead**: K=4 auxiliary prediction heads, disabled when backbone frozen
- Tied embeddings (32K vocab × 1024 dim)

## Config

`AgentMindConfig` dataclass in `config.py`. Derived properties:
- `d_inner = expand * d_model` (2048)
- `dt_rank = ceil(d_model / 16)` (64)
- `ffn_hidden` aligned to 256 boundary
- `is_attn_layer(i)`: True when `(i+1) % attn_every == 0`

Default training: `lr=2e-4, warmup=100, total_steps=3000, batch_size=1, grad_accum=8`.

## Output files

```
checkpoints/
├── backbone.safetensors          # final backbone weights
├── router/                       # TaskRouter state
├── adapters/
│   ├── tool_caller.safetensors   # LoRA adapter per domain
│   ├── planner.safetensors
│   ├── recovery.safetensors
│   ├── code.safetensors
│   └── research.safetensors
├── step_XXXXX/                   # intermediate checkpoints
│   ├── weights.npz
│   └── log.json
```

## Hardware notes

- Runs on 16GB MacBook Air (confirmed by `monitor.py`). RAM peaks at ~70-80%.
- Swap thrashing observed with seq_len > 1024 or batch_size > 1.
- Mamba CHUNK=16 is the confirmed sweet spot for Metal compilation on Apple Silicon.
