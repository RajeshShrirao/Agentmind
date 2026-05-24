import json
import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map


def format_sample(sample: dict) -> str:
    text = ""
    for msg in sample["messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<|system|>{content}"
        elif role == "user":
            text += f"<|user|>{content}"
        elif role == "assistant":
            text += f"<|assistant|>{content}<eos>"
    return text


def make_labels(ids: list[int], cfg) -> list[int]:
    labels = [-100] * len(ids)
    in_assistant = False
    for i, tok_id in enumerate(ids):
        if tok_id == cfg.assistant_id:
            in_assistant = True
        if in_assistant:
            labels[i] = tok_id
        if tok_id == cfg.eos_id or tok_id == cfg.user_id or tok_id == cfg.system_id:
            in_assistant = False
    return labels


def cross_entropy_loss(logits, targets, boundary_weight: float = 0.0, boundary_ids: set = None):
    logits = logits[:, :-1, :]
    targets = targets[:, 1:]
    B, L, V = logits.shape
    flat_logits = logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)
    mask = (flat_targets != -100).astype(mx.float32)
    loss = nn.losses.cross_entropy(flat_logits, mx.maximum(flat_targets, 0), reduction='none')
    weighted_loss = loss * mask
    if boundary_weight > 0.0 and boundary_ids:
        boundary_mask = mx.zeros_like(mask)
        for tid in boundary_ids:
            boundary_mask = mx.maximum(boundary_mask, (flat_targets == tid).astype(mx.float32))
        boundary_mask = boundary_mask * mask
        weighted_loss = weighted_loss * (1.0 + boundary_mask * boundary_weight)
    return weighted_loss.sum() / (mask.sum() + 1e-8)


def clip_gradients(grads, max_norm: float):
    leaves = [g for _, g in tree_flatten(grads)]
    norm = mx.sqrt(sum(mx.sum(g ** 2) for g in leaves))
    scale = mx.minimum(1.0, max_norm / (norm + 1e-6))
    return tree_map(lambda g: g * scale, grads), norm.item()


def check_finite(tree):
    for k, v in tree_flatten(tree):
        if not mx.all(mx.isfinite(v)).item():
            return False, k
    return True, None


def clone_tree(tree):
    return tree_map(lambda x: mx.array(x) if isinstance(x, mx.array) else x, tree)


def get_seq_len_from_schedule(step: int, schedule: dict) -> int:
    current = 0
    for threshold, length in sorted(schedule.items()):
        if step >= threshold:
            current = length
    return current


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


def _nested_weights(flat: dict) -> dict:
    nested = {}
    for key, val in flat.items():
        parts = key.split(".")
        d = nested
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val

    def _lists(v):
        if isinstance(v, dict):
            out = {k: _lists(v) for k, v in v.items()}
            try:
                keys = sorted(out, key=int)
                ints = [int(k) for k in keys]
                if ints and ints == list(range(len(out))):
                    return [out[k] for k in keys]
            except (ValueError, TypeError):
                pass
            return out
        return v

    for skip in ["last_hidden", "last_mtp_logits"]:
        nested.pop(skip, None)

    return _lists(nested)


class GradientAccumulator:
    def __init__(self, n_steps: int = 8):
        self.n_steps_target = n_steps
        self.accum_grad = None
        self.accum_loss = 0.0
        self.n_steps = 0

    def reset(self):
        self.accum_grad = None
        self.accum_loss = 0.0
        self.n_steps = 0

    def add(self, grads, loss_val: float):
        self.n_steps += 1
        self.accum_loss += loss_val
        if self.accum_grad is None:
            self.accum_grad = grads
        else:
            self.accum_grad = tree_map(lambda a, b: a + b, self.accum_grad, grads)

    def step(self):
        if self.accum_grad is None or self.n_steps == 0:
            return None, 0.0
        avg_grads = tree_map(lambda g: g / self.n_steps, self.accum_grad)
        avg_loss = self.accum_loss / self.n_steps
        self.reset()
        return avg_grads, avg_loss

    @property
    def is_ready(self):
        return self.n_steps >= self.n_steps_target


class NaNRecovery:
    def __init__(self, model, optimizer):
        self.backup_params = clone_tree(model.parameters())
        self.backup_opt_state = clone_tree(optimizer.state)

    def check(self, loss, grads):
        if not mx.isfinite(loss).item():
            return False
        grads_finite, _ = check_finite(grads)
        return grads_finite

    def recover(self, model, optimizer, grads, accum: GradientAccumulator = None):
        grads = tree_map(mx.zeros_like, grads)
        if accum is not None:
            accum.reset()
        model.update(self.backup_params)
        optimizer.state = clone_tree(self.backup_opt_state)
        mx.eval(model.parameters(), optimizer.state)
        return tree_map(mx.zeros_like, grads)

    def commit(self, model, optimizer):
        self.backup_params = clone_tree(model.parameters())
        self.backup_opt_state = clone_tree(optimizer.state)
