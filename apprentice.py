import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten, tree_map
import math
import time
from pathlib import Path

from lora import apply_lora, LoRALinear
from scheduler import CosineWarmupScheduler
from model.mtp_head import mtp_loss


def cross_entropy_loss(logits, targets):
    logits = logits[:, :-1, :]
    targets = targets[:, 1:]
    B, L, V = logits.shape
    flat_logits = logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)
    mask = (flat_targets != -100).astype(mx.float32)
    loss = nn.losses.cross_entropy(flat_logits, mx.maximum(flat_targets, 0), reduction='none')
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def kl_div(student_logits, teacher_logits, labels=None):
    student_logits = student_logits[:, :-1, :]
    teacher_logits = teacher_logits[:, :-1, :]
    student_log_probs = nn.log_softmax(student_logits, axis=-1)
    teacher_probs = nn.softmax(teacher_logits, axis=-1)
    kl_per_pos = mx.sum(teacher_probs * (mx.log(teacher_probs + 1e-10) - student_log_probs), axis=-1)
    if labels is not None:
        mask = (labels[:, 1:] != -100).astype(mx.float32)
        kl_per_pos = kl_per_pos * mask
        return kl_per_pos.sum() / (mask.sum() + 1e-8)
    return mx.mean(kl_per_pos)


def clip_gradients(grads, max_norm):
    leaves = [g for _, g in tree_flatten(grads)]
    norm = mx.sqrt(sum(mx.sum(g ** 2) for g in leaves))
    scale = mx.minimum(1.0, max_norm / (norm + 1e-6))
    return tree_map(lambda g: g * scale, grads), norm.item()


class CognitiveApprentice:
    def __init__(self, backbone, adapter_name, rank=16, alpha=32.0):
        self.adapter_name = adapter_name
        self.rank = rank
        self.alpha = alpha
        self.cfg = backbone.cfg
        apply_lora(backbone, rank=rank, alpha=alpha)
        self.backbone = backbone

    @property
    def model(self):
        return self.backbone

    def save_adapter(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path = path.with_suffix('.safetensors')

        params = dict(tree_flatten(self.backbone.trainable_parameters()))
        lora_params = {}
        for key, val in params.items():
            if key.endswith('.A') or key.endswith('.B'):
                lora_params[key] = val

        metadata = {
            "adapter_name": self.adapter_name,
            "rank": str(self.rank),
            "alpha": str(self.alpha),
        }
        mx.save_safetensors(str(path), lora_params, metadata)
        total_kb = sum(v.nbytes for v in lora_params.values()) // 1024
        print(f"[apprentice] Saved adapter '{self.adapter_name}' → {path} ({total_kb} KB)")

    def load_adapter(self, path, backbone=None):
        model = backbone or self.backbone
        path = Path(str(path))
        if path.suffix != '.safetensors':
            path = path.with_suffix('.safetensors')
        if not path.exists():
            raise FileNotFoundError(f"Adapter not found: {path}")

        loaded = mx.load(str(path))
        if 'metadata' in loaded:
            del loaded['metadata']

        nested = tree_unflatten(dict(loaded))
        model.update(nested)
        self.backbone = model
        print(f"[apprentice] Loaded adapter '{self.adapter_name}' from {path}")
        return model

    def reset_adapter(self):
        def _reset(m):
            if isinstance(m, LoRALinear):
                r = m.A.shape[0]
                in_features = m.A.shape[1]
                m.A = mx.random.normal((r, in_features)) * (1 / math.sqrt(r))
                m.B = mx.zeros_like(m.B)
            elif isinstance(m, nn.Module):
                for child in m.children().values():
                    _reset(child)
            elif isinstance(m, list):
                for item in m:
                    _reset(item)
        _reset(self.backbone)
        print(f"[apprentice] Reset adapter '{self.adapter_name}' to random init")

    def _make_labels(self, ids, sample):
        labels = [-100] * len(ids)
        in_assistant = False
        for i, tok_id in enumerate(ids):
            if tok_id == self.cfg.assistant_id:
                in_assistant = True
            if in_assistant:
                labels[i] = tok_id
            if tok_id in (self.cfg.eos_id, self.cfg.user_id, self.cfg.system_id):
                in_assistant = False
        return labels

    def _tokenize_samples(self, dataset, tokenizer, seq_len, latent_stage):
        from model.latent import inject_latent_tokens
        import copy

        tokenized = []
        for sample in dataset:
            s = copy.deepcopy(sample)
            s = inject_latent_tokens(s, tokenizer, latent_stage)
            text = ""
            for msg in s["messages"]:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    text += f"<|system|>{content}"
                elif role == "user":
                    text += f"<|user|>{content}"
                elif role == "assistant":
                    text += f"<|assistant|>{content}<eos>"
            ids = tokenizer.encode(text, add_bos=True)[:seq_len]
            labels = self._make_labels(ids, s)[:seq_len]
            tokenized.append({"ids": ids, "labels": labels, "domain": s.get("domain", self.adapter_name)})
        return tokenized

    def train(self, dataset, tokenizer=None, steps=500, lr=2e-4, seq_len=256,
              warmup=50, grad_clip=1.0, log_every=50, latent_stage=1):
        from data.pipeline import AgentDataset, make_dataloader
        from model.latent import latent_loss_mask

        if isinstance(dataset, list) and len(dataset) > 0:
            if isinstance(dataset[0], dict) and "messages" in dataset[0]:
                ds = AgentDataset.__new__(AgentDataset)
                ds.samples = dataset
                ds.cfg = self.cfg
                ds.tok = tokenizer
                ds.latent_stage = latent_stage
                ds.ids_array = None
                ds.labels_array = None
                ds.weights = {"instruction": 0.3, "tool_single": 0.3,
                              "agent_multi": 0.25, "recovery": 0.15}
                dataset = ds

        optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)
        scheduler = CosineWarmupScheduler(optimizer, lr, warmup, steps)

        trainable = {k: v for k, v in self.backbone.trainable_parameters().items()
                     if not k.startswith("last_")}

        self.backbone.train()
        step = 0
        t0 = time.time()

        while step < steps:
            loader = make_dataloader(dataset, batch_size=1, max_len=seq_len)
            for input_ids, targets in loader:
                if step >= steps:
                    break

                if latent_stage >= 3:
                    targets = latent_loss_mask(
                        input_ids, targets,
                        self.cfg.think_start_id, self.cfg.think_end_id
                    )

                def loss_fn(params):
                    self.backbone.update(params)
                    logits, _ = self.backbone(input_ids)
                    return cross_entropy_loss(logits, targets)

                loss, grads = mx.value_and_grad(loss_fn)(trainable)
                mx.eval(loss, grads)
                loss_val = loss.item()

                grads, grad_norm = clip_gradients(grads, grad_clip)
                optimizer.update(trainable, grads)
                mx.eval(self.backbone.parameters(), optimizer.state)
                lr_now = scheduler.step()

                if step % log_every == 0 or step == steps - 1:
                    tok_per_sec = (input_ids.shape[1]) / (time.time() - t0 + 1e-8)
                    print(f"[{self.adapter_name}] step {step:3d}/{steps} loss {loss_val:.4f} "
                          f"lr {lr_now:.2e} grad_norm {grad_norm:.3f} {tok_per_sec:.0f} tok/s")
                    t0 = time.time()

                step += 1

        print(f"[apprentice] '{self.adapter_name}' training complete ({step} steps)")
        return self

    def distill(self, backbone, specialists, dataset, tokenizer=None, beta=0.5, mtp_weight=0.2,
                steps=50, lr=1e-5, grad_clip=1.0, seq_len=512, log_every=10, latent_stage=1):
        from model.latent import latent_loss_mask
        import random

        tokenized = self._tokenize_samples(dataset, tokenizer, seq_len, latent_stage)

        backbone.train()
        backbone.unfreeze()
        backbone_params = {k: v for k, v in backbone.trainable_parameters().items()
                           if not k.startswith("last_")}

        optimizer = optim.AdamW(learning_rate=lr)
        step = 0
        t0 = time.time()

        while step < steps:
            random.shuffle(tokenized)
            for sample in tokenized:
                if step >= steps:
                    break
                ids = mx.array([sample["ids"]])
                labels = mx.array([sample["labels"]])
                domain = sample["domain"]

                if latent_stage >= 3:
                    labels = latent_loss_mask(
                        ids, labels,
                        self.cfg.think_start_id, self.cfg.think_end_id
                    )

                s_logits = {}
                for name, expert in specialists.items():
                    expert.model.eval()
                    s_logits[name], _ = expert.model(ids)

                def loss_fn(params):
                    backbone.update(params)
                    b_logits, _ = backbone(ids, return_mtp=True)
                    task = cross_entropy_loss(b_logits, labels)
                    distill = kl_div(b_logits, s_logits[domain], labels)
                    aux = mtp_loss(backbone.last_mtp_logits, labels, weight=mtp_weight)
                    return task + beta * distill + aux

                loss, grads = mx.value_and_grad(loss_fn)(backbone_params)
                mx.eval(loss, grads)

                grads, grad_norm = clip_gradients(grads, grad_clip)
                optimizer.update(backbone_params, grads)
                mx.eval(backbone.parameters(), optimizer.state)

                if step % log_every == 0 or step == steps - 1:
                    elapsed = time.time() - t0
                    print(f"[distill] step {step:3d}/{steps} loss {loss.item():.4f} "
                          f"grad_norm {grad_norm:.3f} {elapsed:.1f}s")
                    t0 = time.time()

                step += 1

        backbone.freeze()
        print(f"[distill] Complete ({step} steps)")
        return self
