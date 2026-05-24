# Training Throughput Optimization Brief

## Goal
10x training throughput for Qwen2.5-0.5B LoRA fine-tuning on MacBook Air 16GB (MLX backend).

## Current Baseline (uncompiled, no optimization)

| Metric | Value |
|--------|-------|
| Backbone | Qwen2.5-0.5B (500M params, 24 layers, d_model=896) |
| Training params | 8.8M LoRA (rank=16, alpha=32, 7 targets) |
| Throughput | ~580 tok/s (ALL seq lens: 128/256/384/512) |
| Step time (seq=128, grad_accum=8) | 1.7s per optimizer step |
| Step time (seq=256, grad_accum=8) | 3.5s per optimizer step |
| 2000 steps at seq=128 | ~58 min |
| RAM | ~3GB during training |
| Hardware | MacBook Air M-series, 16GB |

## Key Observation
Throughput is **flat at ~580 tok/s** regardless of sequence length. This means the bottleneck is per-token overhead (model forward+backward), not attention O(L²). The backward pass through 500M frozen params dominates.

`mx.compile` provides only **6% speedup** — the VJP backward pass is still traced dynamically.

## Architecture Context
```
LoRALinear:
  y = base(x) + scale * (x @ A.T) @ B.T    # base.weight is frozen
```

The gradient computation traces through ALL operations including `base(x)` (500M params), even though only A/B (8.8M) are trainable. MLX's VJP must visit every operation in the computation graph during backward.

## Candidates for 10x

### 1. Gradient Isolation (highest ROI if feasible)
`mx.lax.stop_gradient` or equivalent to prevent gradient flow through frozen base layers. If we can make MLX stop tracing backward through `base(x)` while still computing `dL/dA` and `dL/dB`, we skip ~90% of the backward pass.

- Modify `LoRALinear.__call__` to use a custom VJP that doesn't trace the base path
- Or use `mx.custom_function` with manual VJP for LoRALinear
- Expected: 5-8x speedup

### 2. Analytical LoRA Gradient
Compute LoRA gradients analytically without autograd:
```
dL/dB = dL/dy @ (A @ x.T)
dL/dA = B.T @ dL/dy @ x.T
```
This requires only the hidden state `x` at each LoRA layer (from forward pass) and `dL/dy` (from backward pass through subsequent layers). Still needs backward through layers after the LoRA point.

- Expected: 2-3x speedup

### 3. 4-bit Quantized Backbone
Load backbone in 4-bit to reduce memory bandwidth on forward pass. MLX supports `mx.quantize()` and `mx.dequantize()`.

- Check if `mlx_lm.load()` supports quantization in current version (0.31.1)
- Or quantize post-load with `mx.quantize(model, group_size=64, bits=4)`
- Expected: 1.5-2x speedup on forward pass

### 4. Smaller Backbone
Swap to a smaller model that's MLX-compatible:
- SmolLM2-135M (135M params, ~3.7x smaller) — check `mlx_lm` support
- Any sub-300M transformer available on HuggingFace with MLX support
- Expected: 3-4x speedup

### 5. Reduce LoRA Targets
Current: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` (7 targets, 8.8M params)
Proposed: `["q_proj", "v_proj"]` (2 targets, ~2.5M params)

This reduces both backward compute and optimizer overhead.

- Expected: 1.5-2x speedup

### 6. LoRA Rank Reduction
Current: rank=16. Test rank=4 or rank=8. Reduces trainable params by 4x or 2x.

- Expected: 1.2-1.5x speedup

### 7. Gradient Accumulation Strategy
Current: `grad_accum=8`, always 8 micro-batches per step.
Consider: `grad_accum=4` (less stable gradients but 2x fewer backward passes).

Or: gradient checkpointing to trade memory for recomputation during backward.

### 8. Optimizer Overhead
Current: AdamW with weight_decay=0.01, updates 8.8M params every 8 micro-batches.
Consider: Simpler optimizer (SGD, Adam without weight_decay), less frequent updates.

### 9. Data Pipeline
Current: `AgentDataset` with on-demand tokenization (`apply_chat_template` + `encode`), caching in `_cache` dict. 5000 samples per epoch, each tokenized once.

Check if data loading adds overhead by profiling with data loading vs. without. Pre-tokenize all data ahead of time and load from `.npz` or similar.

### 10. Batch Size
Current: `batch_size=1` (16GB RAM constraint). Check if `batch_size=2` is viable (might cause swap thrashing).

### 11. Compilation Diagnostics
`mx.compile` currently gives 6% speedup. Investigate why:
- Is the VJP function recompiling on every step?
- Are input shapes actually changing between calls?
- Does `mx.compile(mx.value_and_grad(loss_fn))` compile both forward and backward, or just forward?
- Try: compile the entire step function (including optimizer update) using `@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)` pattern from MLX examples.

## Implementation Priority

1. **Diagnose compile failure** — add trace count/hit diagnostics, try different compile patterns
2. **Try smaller backbone** (SmolLM2-135M) — largest single lever
3. **Try 4-bit quantization** — if supported, immediate 1.5x
4. **Try q_proj+v_proj only** — simple config change
5. **Custom LoRALinear VJP** — highest complexity but potentially 5-8x

## Relevant Files

```
train.py              — training loop, gradient computation
lora.py               — LoRALinear implementation, apply_lora
training_utils.py     — cross_entropy_loss, check_finite, clip_gradients
data/pipeline.py      — AgentDataset, make_dataloader, collate_batch
config.py             — APPRENTICE_ROUNDS config
training_orchestrator.py — orchestrator
```

## Test Command
```bash
python3 training_orchestrator.py --rounds 1 --save-dir ./checkpoints
```
