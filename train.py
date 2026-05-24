"""train.py — Specialist training and backbone distillation for Qwen2.5-based AgentMind.

Usage:
    from train import train_specialist, distill_backbone
    
    # Train a specialist
    weights = train_specialist(backbone, tokenizer, dataset, 'tool_caller', steps=500)
    
    # Distill into backbone
    distill_backbone(backbone, {'tool_caller': weights}, combined_data, tokenizer, steps=50)
"""

import time, json, os, random, argparse, math
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

import mlx_lm

from data.pipeline import AgentDataset, make_dataloader
from lora import apply_lora, load_lora, LoRALinear
from training_utils import (
    cross_entropy_loss, clip_gradients, check_finite, clone_tree,
    get_seq_len_from_schedule, kl_div,
    GradientAccumulator, NaNRecovery,
)
from scheduler import CosineWarmupScheduler


def make_labels(ids, tokenizer):
    labels = [-100] * len(ids)
    in_assistant = False
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_id = tokenizer.convert_tokens_to_ids("assistant")

    for i, tok_id in enumerate(ids):
        if tok_id == im_start_id and i + 1 < len(ids) and ids[i + 1] == assistant_id:
            in_assistant = True
        if in_assistant:
            labels[i] = tok_id
        if tok_id == im_end_id:
            in_assistant = False
    return labels


def _get_logits(model, input_ids):
    """Get logits from model, handling tuple returns."""
    result = model(input_ids)
    if isinstance(result, tuple):
        return result[0]
    return result


def _pretokenize_distill_data(combined_data, specialists, tokenizer, seq_len):
    """Pre-tokenize samples for distillation with domain labels.
    Pads/truncates to fixed seq_len for consistent compile shapes."""
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    samples = combined_data.samples if hasattr(combined_data, 'samples') else combined_data
    tokenized = []
    for sample in samples:
        text = tokenizer.apply_chat_template(
            sample["messages"], tokenize=False, add_generation_prompt=False
        )
        ids0 = tokenizer.encode(text)
        lbs0 = make_labels(ids0, tokenizer)
        ids = ids0[:seq_len] + [pad_id] * max(0, seq_len - len(ids0))
        lbs = lbs0[:seq_len] + [-100] * max(0, seq_len - len(lbs0))
        domain = sample.get("domain", list(specialists.keys())[0])
        tokenized.append({"ids": ids, "labels": lbs, "domain": domain})
    random.shuffle(tokenized)
    n_val = max(1, len(tokenized) // 20)
    return tokenized[n_val:], tokenized[:n_val]


def _build_teacher_models(backbone_id, specialists, lora_rank=16, lora_alpha=32.0):
    """Build frozen teacher models from saved adapter weights."""
    spec_models = {}
    for domain, adapter_weights in specialists.items():
        teacher, _ = mlx_lm.load(backbone_id)
        teacher = apply_lora(teacher, rank=lora_rank, alpha=lora_alpha)
        teacher = load_lora(teacher, adapter_weights)
        teacher.freeze()
        teacher.eval()
        spec_models[domain] = teacher
    return spec_models


def train_specialist(backbone, tokenizer, dataset, domain_name,
                     steps=500, lr=2e-4, seq_len=256,
                     seq_len_schedule=None, grad_accum=8, grad_clip=1.0):
    """Train ONLY the LoRA adapter on a domain-specific dataset.
    
    Backbone stays FROZEN throughout. Only LoRA A/B matrices are trained.
    
    Args:
        backbone: Qwen2.5 model loaded via mlx_lm.load(), with LoRA applied
        tokenizer: Qwen's AutoTokenizer
        dataset: AgentDataset or list of samples
        domain_name: Name of the domain (for logging)
        steps: Number of training steps (batches)
        lr: Learning rate
        seq_len: Sequence length
        seq_len_schedule: Dict mapping step->seq_len for curriculum
        grad_accum: Gradient accumulation steps
        grad_clip: Max gradient norm
        
    Returns:
        dict: LoRA A/B weights only
    """
    if isinstance(dataset, list):
        ds = AgentDataset(samples=dataset, tokenizer=tokenizer)
    else:
        ds = dataset
        ds.tokenizer = tokenizer

    ds.pretokenize()

    # Pre-tokenize to fixed-length .npz for consistent compile shapes
    max_dataset_len = max(seq_len, max(seq_len_schedule.values(), default=seq_len))
    ds.tokenize_to_npz(max_len=max_dataset_len)

    trainable = {k: v for k, v in backbone.trainable_parameters().items()}

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)
    warmup = min(200, max(1, steps // 10))
    scheduler = CosineWarmupScheduler(optimizer, lr, warmup, steps)

    n_total = len(ds)
    n_val = max(1, n_total // 20)
    val_indices = set(random.sample(range(n_total), n_val))
    train_indices = [i for i in range(n_total) if i not in val_indices]

    if seq_len_schedule is None:
        seq_len_schedule = {0: seq_len}

    accum = GradientAccumulator(grad_accum)
    recovery = NaNRecovery(backbone, optimizer)

    backbone.train()
    step = 0
    t_start = time.time()
    last_logged_step = 0
    nan_count = 0
    current_seq_len = 0
    n_retrace = 0

    def loss_fn(params, inp, tgt):
        backbone.update(params)
        logits = _get_logits(backbone, inp)
        return cross_entropy_loss(logits, tgt)

    # Compiled micro-step: forward + backward
    # Fixed-length inputs (from .npz pre-tokenization) ensure no retracing
    loss_and_grad = mx.value_and_grad(loss_fn)
    compiled_step = mx.compile(loss_and_grad)

    while step < steps:
        target_seq_len = get_seq_len_from_schedule(step, seq_len_schedule)
        if target_seq_len != current_seq_len:
            print(f"  [compile] seq_len {current_seq_len} -> {target_seq_len} (retrace)")
            current_seq_len = target_seq_len
            n_retrace += 1

        n_epoch = min(5000, len(train_indices))
        epoch_indices = random.sample(train_indices, n_epoch)
        loader = make_dataloader(ds, batch_size=1, max_len=current_seq_len, indices=epoch_indices)

        for input_ids, targets in loader:
            if step >= steps:
                break

            loss, grads = compiled_step(trainable, input_ids, targets)
            mx.eval(loss, grads)

            loss_finite = mx.isfinite(loss).item()
            grads_finite, _ = check_finite(grads)
            if not loss_finite or not grads_finite:
                nan_count += 1
                grads = tree_map(mx.zeros_like, grads)
                optimizer.update(trainable, grads)
                continue

            accum.add(grads, loss.item())

            if accum.is_ready:
                avg_grads, avg_loss = accum.step()
                avg_grads, grad_norm = clip_gradients(avg_grads, grad_clip)

                optimizer.update(trainable, avg_grads)
                mx.eval(list(trainable.values()), optimizer.state)

                recovery.commit(backbone, optimizer)
                lr_now = scheduler.step()
                accum.reset()

                if step % 100 == 0 or step == steps - 1 or step == 0:
                    elapsed = time.time() - t_start
                    steps_since = step - last_logged_step + 1
                    tok_per_sec = (input_ids.shape[1] * grad_accum * steps_since) / (elapsed + 1e-8)
                    print(f"[{domain_name}] step {step:3d}/{steps} "
                          f"loss {avg_loss:.4f} lr {lr_now:.2e} "
                          f"grad_norm {grad_norm:.3f} {tok_per_sec:.0f} tok/s "
                          f"retrace={n_retrace}")
                    t_start = time.time()
                    last_logged_step = step

                if step > 0 and step % 500 == 0:
                    try:
                        debug_logits = _get_logits(backbone, input_ids)
                        mx.eval(debug_logits)
                        debug_token = mx.argmax(debug_logits[:, -1, :], axis=-1).item()
                        print(f"  [debug] last_pred_token={debug_token} ({tokenizer.decode([debug_token])!r})")
                    except Exception as e:
                        print(f"  [debug] eval skipped: {e}")

                step += 1

    params = dict(tree_flatten(backbone.trainable_parameters()))
    return {k: v for k, v in params.items() if k.endswith('.A') or k.endswith('.B')}


def distill_backbone(backbone, specialists, combined_data, tokenizer,
                     beta=0.5, steps=50, lr=1e-5, seq_len=512,
                     grad_accum=8, grad_clip=1.0,
                     backbone_id="Qwen/Qwen2.5-0.5B",
                     lora_rank=16, lora_alpha=32.0):
    """Distill specialist knowledge into backbone.
    
    Unfreezes backbone, trains with CE+KL loss.
    
    Args:
        backbone: Qwen2.5 model with LoRA applied
        specialists: dict of {domain_name: adapter_weights}
        combined_data: list of samples or AgentDataset from all domains
        tokenizer: Qwen's AutoTokenizer
        beta: KL divergence weight
        steps: Number of distillation steps (batches)
        lr: Learning rate
        seq_len: Sequence length
        grad_accum: Gradient accumulation steps
        grad_clip: Max gradient norm
        backbone_id: HuggingFace model ID for loading teacher models
        lora_rank: LoRA rank for teacher models
        lora_alpha: LoRA alpha for teacher models
    """
    spec_models = _build_teacher_models(backbone_id, specialists, lora_rank, lora_alpha)

    train_samples, _ = _pretokenize_distill_data(
        combined_data, specialists, tokenizer, seq_len
    )

    backbone.unfreeze()
    trainable = {k: v for k, v in backbone.trainable_parameters().items()}

    optimizer = optim.AdamW(learning_rate=lr)
    accum = GradientAccumulator(grad_accum)
    recovery = NaNRecovery(backbone, optimizer)

    step = 0
    t_start = time.time()
    nan_count = 0

    def distill_loss_fn(params, ids, labels, spec_logits, beta):
        backbone.update(params)
        b_logits = _get_logits(backbone, ids)
        task = cross_entropy_loss(b_logits, labels)
        distill = kl_div(b_logits, spec_logits, labels)
        return task + beta * distill

    # Compiled distill step (fixed-length inputs prevent retracing)
    distill_loss_and_grad = mx.value_and_grad(distill_loss_fn)
    compiled_distill_step = mx.compile(distill_loss_and_grad)

    while step < steps:
        random.shuffle(train_samples)
        for sample in train_samples:
            if step >= steps:
                break

            ids = mx.array([sample["ids"]])
            labels = mx.array([sample["labels"]])
            domain = sample["domain"]

            spec_logits = _get_logits(spec_models[domain], ids)

            loss, grads = compiled_distill_step(trainable, ids, labels, spec_logits, beta)
            mx.eval(loss, grads)

            loss_finite = mx.isfinite(loss).item()
            grads_finite, _ = check_finite(grads)
            if not loss_finite or not grads_finite:
                nan_count += 1
                grads = tree_map(mx.zeros_like, grads)
                optimizer.update(trainable, grads)
                continue

            accum.add(grads, loss.item())

            if accum.is_ready:
                avg_grads, avg_loss = accum.step()
                avg_grads, grad_norm = clip_gradients(avg_grads, grad_clip)

                optimizer.update(trainable, avg_grads)
                mx.eval(list(trainable.values()), optimizer.state)

                recovery.commit(backbone, optimizer)
                accum.reset()

                if step % 10 == 0 or step == steps - 1:
                    elapsed = time.time() - t_start
                    print(f"[distill] step {step:3d}/{steps} "
                          f"loss {avg_loss:.4f} "
                          f"grad_norm {grad_norm:.3f} {elapsed:.1f}s")
                    t_start = time.time()

                step += 1

    backbone.freeze()
    elapsed = time.time() - t_start
    print(f"[distill] Complete ({step} steps, {elapsed:.0f}s) nan={nan_count}")
