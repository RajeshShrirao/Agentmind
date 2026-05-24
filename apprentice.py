"""
CognitiveApprentice — LoRA wrapper for Qwen2.5-0.5B backbone.

Wraps a Qwen backbone (loaded via mlx_lm.load()) with LoRA adapters on
Qwen's target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
Provides save/load/reset lifecycle and training entry point.

Usage:
    from mlx_lm import load as load_model
    from apprentice import CognitiveApprentice

    model, tokenizer = load_model('Qwen/Qwen2.5-0.5B')
    app = CognitiveApprentice(model, tokenizer, 'tool_caller')
    app.train(dataset, steps=500)
    app.save_adapter('./checkpoints/tool_caller')
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from pathlib import Path
from tqdm import tqdm

from lora import apply_lora, save_adapter, load_adapter, load_lora, reset_adapter as lora_reset
from training_utils import cross_entropy_loss, clip_gradients, check_finite
from scheduler import CosineWarmupScheduler


class CognitiveApprentice:
    def __init__(self, backbone, tokenizer, adapter_name, rank=16, alpha=32.0):

        self.backbone = backbone
        self.tokenizer = tokenizer
        self.adapter_name = adapter_name
        self.rank = rank
        self.alpha = alpha

        apply_lora(backbone, rank=rank, alpha=alpha,
            targets=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])

        total = sum(p.size for _, p in tree_flatten(backbone.trainable_parameters()))
        print(f"CognitiveApprentice '{adapter_name}' created | Trainable: {total:,} ({total/1e6:.2f}M)")

    def save_adapter(self, path):
        save_adapter(self.backbone, self.adapter_name, path)

    def load_adapter(self, path):
        load_adapter(self.backbone, path)

    def reset_adapter(self):
        lora_reset(self.backbone)

    def train(self, dataset, steps=500, lr=2e-4, seq_len=256, grad_clip=1.0):
        trainable = {k: v for k, v in self.backbone.trainable_parameters().items()}

        optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)
        warmup = min(200, max(1, steps // 10))
        scheduler = CosineWarmupScheduler(optimizer, lr, warmup, steps)

        losses = []
        pbar = tqdm(range(steps), desc=f"Training {self.adapter_name}")

        for step in pbar:
            idx = step % len(dataset)
            sample = dataset[idx]

            if isinstance(sample, dict) and 'messages' in sample:
                text = self.tokenizer.apply_chat_template(
                    sample['messages'], tokenize=False, add_generation_prompt=False
                )
                tokens = self.tokenizer.encode(text)
                tokens = tokens[:seq_len]
                input_ids = mx.array([tokens])
                targets = input_ids
            elif isinstance(sample, (tuple, list)) and len(sample) == 2:
                ids, labels = sample
                if isinstance(ids, list):
                    ids = ids[:seq_len]
                    labels = labels[:seq_len]
                input_ids = mx.array([ids])
                targets = mx.array([labels])
            else:
                input_ids = mx.array([sample[:seq_len]])
                targets = input_ids

            def loss_fn(params):
                self.backbone.update(params)
                logits, _ = self.backbone(input_ids)
                return cross_entropy_loss(logits, targets)

            loss, grads = mx.value_and_grad(loss_fn)(trainable)
            mx.eval(loss, grads)

            if not mx.isfinite(loss).item():
                grads = tree_map(mx.zeros_like, grads)
                optimizer.update(trainable, grads)
                pbar.set_postfix({"loss": "NaN"})
                continue

            grads, _ = clip_gradients(grads, grad_clip)
            optimizer.update(trainable, grads)
            mx.eval(list(trainable.values()), optimizer.state)
            lr_now = scheduler.step()

            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr_now:.2e}"})

        params = dict(tree_flatten(self.backbone.trainable_parameters()))
        return {k: v for k, v in params.items()
                if k.endswith('.A') or k.endswith('.B')}
