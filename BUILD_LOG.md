# AgentMind — Ship's Log

> Hybrid SSM + Local Attention LM for Agentic AI  
> Target: ~600M → 145M params · 16GB MacBook Air · MLX Backend  
> Log format: diary entries keyed by commit

---

## `7b2619b` — 2026-05-21 14:44

**Scaffolding the whole thing.**

Woke up with a Mamba-shaped hole in my life. Decided to build a hybrid SSM + attention LM that can call tools and think in latent space. Dropped the skeleton:

- `config.py` — AgentMindConfig with 13 special tokens. d_model=1024, n_layers=16, d_state=16, 8 heads, window=256. Halved everything from the original pipe dream (2048→1024, 24→16, 128→16) to keep it alive on 16GB unified memory.
- `model/` — MambaBlock with sequential scan, LocalAttentionBlock with RoPE, AgentLM tying everything together with MTP heads and state-preserving forward. ConvState for inference-time conv buffer management.
- `tokenizer_setup.py` — SentencePiece BPE, 32K vocab, 10 agentic control tokens as user-defined symbols.
- Init, lora, scheduler — all stubs with signatures, waiting for flesh.

The 24-block / 600M vision in the architecture doc is aspirational. What actually compiles is 16 blocks, 145M raw params, 6M LoRA-trainable. Let's see if that's enough.

---

## `0c32680` — 2026-05-21 15:23

**Synthetic data pipeline comes alive.**

The model can't learn tool use from FineWeb. Built the synthetic data factory:

- `generate_scaled_synthetic.py` — 11,500 samples across 5 types: instruction (3K), tool_single (2.5K), agent_multi (3K), recovery (2K), latent (1K). Template-based generation with rate-limited Cerebras API (40 req/min — pain).
- `build_corpus.py` — downloads 6 open datasets (FineWeb, The Stack, UltraChat, AgentInstruct, ToolBench, WebArena). ~250MB of raw text.

Realized the synthetic data is entirely templated — tools are called with `example_arg` placeholders. The model will learn formatting, not semantics. Decided this is fine for Phase 1 (format bedrock).

- `data/synthetic.py` — 14 tools in registry. Recovery data injects failures 15% of the time. Simple but covers the token protocol.

---

## `8f08dee` — 2026-05-21 15:26

**Docs catch-up.**

Wrote up the architecture doc properly. Documented the 3:1 Mamba-to-attention ratio rationale (SSM for long-range compression, attention for precise recall every 4th layer). Training infra doc got the new data strategy. The docs are now aspirational — describing what we *want* the model to be, not what it is at 145M.

---

## `e1a2167` — 2026-05-21 15:39

**The pipeline gets real.**

- `data/pipeline.py` — AgentDataset with pre-tokenized path (`.npz`) and raw JSONL path. Label masking only trains on assistant tokens. Collate pads to max_len.
- `data/formats.py` — five JSONL schemas with `validate_sample()`. Recovery format includes `<|scratch|>` reasoning between failure and retry.
- `model/latent.py` — the latent reasoning curriculum in 4 stages: normal → insert think boundaries → truncate CoT → full latent. `LatentReasoningWrapper` can execute N silent SSM steps. Not wired into the model yet — just the training data transformation.
- `lora.py` — `LoRALinear` with rank=16, alpha=32. Targets in_proj, out_proj, q_proj, v_proj, lm_head. ~6M trainable params (~1% of total).
- `scheduler.py` — Cosine warmup with linear warmup, min_lr=10% of base.

I notice the latent wrapper is dead code — defined but never imported or wired. Something to fix before Phase 5.

---

## `af5383e` — 2026-05-21 15:41

**Build log reality check.**

Updated BUILD_LOG.md to match actual implementation. Config halved, d_state=16, attn_window=256. The doc had been living in the 600M fantasy. Brought it back to earth.

---

## `0c7e3f8` — 2026-05-21 15:49

**First training loop goes in. Found bugs immediately.**

- `train.py` — full training loop with gradient accumulation (batch=1, accum=8), clipping (max_norm=1.0), seq curriculum, lazy MTP.
- `eval.py` — perplexity + tool_call_accuracy + format_adherence. 3 test prompts, 200 token generations.

**Bug found**: label masking was checking for `assistant` (the string) but the token ID is 15. The `_make_labels` method in pipeline was splitting on the wrong token. Fixed by checking against `cfg.assistant_id`.

Data paths were also wrong — the train script was looking in the wrong directories. Hardcoded to `/Volumes/New Volume/checkpoints` for now (external SSD, since internal disk is tight).

---

## `676d389` — 2026-05-21 15:53

**23/23 pre-training checks pass.**

Every component independently verified:
- MambaBlock output shape ✓, SSM state shape ✓
- AttentionBlock with RoPE ✓
- Full model forward: logits (1, 8, 32000) ✓
- MTP heads: 4 × (1, 16, 32000) ✓
- Tokenizer round-trip ✓
- Loss: 10.35 for untrained model (high, expected)
- LoRA: 6M trainable params ✓

The model can forward. Whether it can *learn* is tomorrow's problem.

---

## `479ac8f` — 2026-05-21 16:20

**Training bugs: the great debug session.**

Three bugs, three hours:

1. **Padding mismatch** — `collate_batch` was padding to fixed max_len but the data samples were variable length. Short sequences got pad tokens at positions where the model expected real data. Mask was supposed to handle this but was off-by-one at sequence boundaries.

2. **Eval guards** — `compute_loss` was crashing on empty batches when the val set was smaller than max_batches. Added try/except with fallback to 100.0 loss.

3. **model.train/eval** — The `evaluate()` function in train.py was calling `model(input_ids)` without setting `model.eval()`. RMSNorm and dropout (if any) would behave differently. Added `model.eval()` before eval, `model.train()` after.

The training loop is fragile but running. Loss goes down — that's something.

---

## `8ec4ae3` — 2026-05-21 18:47

**Performance breakthrough: 14x speedup.**

Training was going to take 42 hours. Unacceptable.

- **Parallel scan** — replaced the sequential for-loop in `_ssm()` with a log-space parallel scan using `mx.cumsum`. 460-495 tok/s vs 30 tok/s for the naive loop. But: the parallel scan is numerically fragile. `0 × inf = NaN` when `exp(large_negative)` underflows and `exp(large_positive)` overflows simultaneously. Added clipping: `log_contrib ∈ [-50, 50]`.

  **Concern**: the cumsum-based parallel scan assumes constant recurrence coefficients. Mamba's dt is input-dependent, so dA_t varies per timestep. The scan might be silently wrong for selective SSMs. Need to validate against the sequential version.

- **Pre-tokenized data** — `pretokenize.py` converts the entire dataset to `.npz` ahead of time. 2x faster loading.

- **mx.compile** — Attempted to JIT-compile the train step. MLX's compiler doesn't play well with the dynamic shapes from sequence curriculum. Reverted. Lazy evaluation mode instead.

- **Fixed timing** — the timer was resetting per-batch, not per-accumulation-window. tok/s was inflated by 8x. Fixed.

---

## `39dd9ed` — 2026-05-21 18:54

**Curriculum learning and lazy MTP.**

Training 3000 steps at seq_len=2048 on a MacBook Air is masochism. Added:

- **Sequence curriculum**: `{0: 256, 500: 512, 1500: 1024}` — start short, grow as the model learns. 4x faster early training.
- **Lazy MTP**: MTP is memory-heavy (4 extra heads full forward). Disabled by default, enabled after step 500 when format is stable.
- **Parallel scan stability**: tightened the clip bounds. Still getting occasional NaN at long sequences.
- **batch_size=2 → 1**: The model was hitting memory pressure at batch=2 during the parallel scan (SSM intermediates blow up to ~18GB at d_state=128). Wait — d_state=16 now, intermediates are 403MB. batch=1 is overly conservative but safe.
- **Dataloader reuse**: recreate the dataloader per curriculum change instead of per step. Saves overhead.

Estimated training time dropped from 42h to ~3h. Still 0.5% of Chinchilla-optimal tokens.

---

## `926787c` — 2026-05-21 18:57

**Updated the docs to match reality.**

Training performance table in the docs was showing old numbers (42h, 30 tok/s). Updated to reflect optimizations: ~3h, 460-495 tok/s with parallel scan. Added the optimization impact table so future me knows what each knob does.

---

## `e1d4e4b` — 2026-05-22 14:02

**Major docs surgery and code cleanup.**

The architecture docs and training infra docs were referencing the old 600M design (d_model=2048, n_layers=24, d_state=128). The code had already been halved but the docs hadn't caught up. Fixed the mismatch.

Changes across 12 files:

- `config.py` — fixed property calculations to match actual values. `d_inner`, `dt_rank_val`, `ffn_hidden`, `param_count_estimate` now compute from the running config, not hardcoded assumptions.
- `model/mamba_block.py` — the `step()` method had a hardcoded split point (`self.d_inner // 16`) that's wrong when `dt_rank != d_inner / 16`. The training path computes `dr` dynamically from `x_proj.weight.shape`. The inference path still has the bug — need to fix before it crashes.
- `model/agent_lm.py` — cleaned up the MTP integration. `return_mtp` flag controls whether the MTP head fires. `forward_with_state` always runs MTP (for eval).
- `train.py` — moved from hardcoded to config-driven. Sequence length schedule, latent stage, MTP start step all controllable. Gradient clipping with proper norm computation.
- `eval.py` — added `model.eval()`/`model.train()` guards. `compute_loss` handles empty validations. `tool_call_accuracy` does argmax (deterministic), which is right for eval but the actual agent needs temperature sampling.
- `lora.py` — `LoRALinear` now handles the base weight correctly. Freeze → replace flow is order-safe.
- `tokenizer_setup.py` — eos_id=5 (not 2). The gap between SentencePiece default and our custom assignments is a footgun.

**Still broken / missing:**
- `LatentReasoningWrapper` is dead code. Never wired into `agent_lm.py`.
- `agent.py` and `export.py` are still empty stubs.
- The 11.5K synthetic samples cover formatting but not real tool semantics.

---

## `7c119cb` — 2026-05-22 14:12

**Mamba Block Parity & Correctness Patch.**

Discovered major train/inference mismatches in `model/mamba_block.py` and resolved them to mathematical and functional parity.

- **Exact Sequential Scan**: Replaced the log-space parallel scan in `_ssm()` with an exact sequential scan loop compiled via `mx.compile`. The log-space parallel scan was numerically unstable, relying on `mx.clip(log_contrib, -50.0, 50.0)` which introduced severe output mismatches on sequence lengths > 3 when comparing token-by-token recurrence against full sequences. The compiled sequential loop runs in ~0.27s for 20 runs of sequence length 2048, maintaining speed while ensuring mathematical identity.
- **Causal Conv State Handling**: Resolved the inference conv bypass. Both `__call__` and `step` now accept and return an explicit, serializable dictionary state format: `{"ssm_state": mx.array, "conv_state": mx.array}`. During `__call__`, the convolution prepends the incoming `conv_state` history buffer, slides, and stores the updated last `d_conv - 1` tokens. During `step()`, the same causal logic is applied to single tokens, sliding and updating the buffer step-by-step.
- **Dynamic Slicing**: Replaced the hardcoded split point (`self.d_inner // 16`) in `step()` with dynamic slicing based on weight shapes: `dr = self.x_proj.weight.shape[0] - self.d_state * 2`, preventing crashes on arbitrary config/weight dimensions.
- **Leading Dimension Flattening**: Removed hardcoded shape assumptions from `step()`. Flattened any leading prefix dimensions of input `x_t: [..., d_model]` to `[B, d_model]` dynamically, then reshaped output back to the original layout, supporting dynamic batches and multidimensional prefixes.
- **Missing Residual Link**: Fixed a critical bug in `step()` where the residual skip connection `+ residual` (present in `__call__()`) was completely omitted.
- **Verification Suite**: Created `test_mamba_parity.py` containing complete parity checks (single token vs multi-token, chunked sequence state propagation, and prefix dimensions). Verified parity to machine epsilon precision (output max diff `2.38e-7`, state max diff `2.32e-10`).

---

## Current State

| Component | Status |
|---|---|
| Model forward | ✅ Works (145M params) |
| Training loop | ✅ Runs, ~3h for 3000 steps |
| Parallel scan | ❌ Replaced with compiled sequential loop (exact mathematical parity) |
| LoRA | ✅ 6M trainable params |
| Pre-tokenized data | ✅ 2x loading speedup |
| Tokenizer | ✅ 32K BPE with 13 special tokens |
| Synthetic data | ✅ 11.5K samples |
| Latent reasoning wrapper | ❌ Dead code |
| Inference conv state | ✅ Works (correctly updated via explicit state dictionaries in forward/step) |
| Inference dt_rank split | ✅ Works (dynamic slicing based on projection weight shapes) |
| Parity verification tests | ✅ Added complete verification suite for step vs __call__ parity |
| agent.py | ❌ Empty stub |
| export.py | ❌ Empty stub |
| Memory budget | ✅ ~5GB training, <1GB inference |

### What Keeps Me Up

- **d_state=16** is a post-it note, not a memory. 384K scalars for the entire "persistent cognition" of the system. That's about 96 bytes of information per forward pass.
- **3,000 steps × 8 effective batch × 600 avg seq_len = 14.4M tokens**. For a 145M param model, that's 0.1× Chinchilla. The model will memorize patterns, not acquire capabilities.
- **Every tool call is formatted imitation**. The model has no schema awareness, no tool registry, no structured decoding. It's a text pattern, not tool use.

---

## What's Next

1. Wire `LatentReasoningWrapper` or kill the feature
2. Run actual training and see if loss goes below 2.0
3. If training works: evaluate whether tool calls are real or hallucinated patterns
