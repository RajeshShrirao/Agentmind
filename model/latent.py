import mlx.core as mx
import re
import random
import numpy as np

N_LATENT_STEPS = 4


def get_latent_stage(step: int) -> int:
    if step < 500:
        return 1
    elif step < 1000:
        return 2
    elif step < 2000:
        return 3
    else:
        return 4


def inject_latent_tokens(sample: dict, tokenizer, stage: int) -> dict:
    if stage < 2:
        return sample

    effective_stage = stage
    if stage == 3:
        effective_stage = 4 if random.random() < 0.5 else 2

    for msg in sample["messages"]:
        if msg["role"] != "assistant":
            continue

        content = msg["content"]

        if "<|scratch|>" in content:
            pattern = re.compile(
                r"<\|scratch\|>(.*?)(?=<\|tool_call\|>|<\|observe\|>|<\|plan\|>|<\|assistant\|>|<\|user\|>|<\|system\|>|<eos>|$)",
                re.DOTALL
            )

            def replace_match(match):
                thoughts = match.group(1)
                if effective_stage == 2:
                    return f"<|think_start|><|scratch|>{thoughts}<|think_end|>"
                else:
                    scratch_tokens = "<|scratch|>" * N_LATENT_STEPS
                    return f"<|think_start|>{scratch_tokens}<|think_end|>"

            content = pattern.sub(replace_match, content)

        msg["content"] = content

    return sample


def latent_loss_mask(input_ids, labels, think_start_id, think_end_id):
    input_ids = mx.array(input_ids) if not isinstance(input_ids, mx.array) else input_ids
    labels = mx.array(labels) if not isinstance(labels, mx.array) else labels

    where_start = (input_ids == think_start_id).astype(mx.int32)
    where_end = (input_ids == think_end_id).astype(mx.int32)

    cum_start = mx.cumsum(where_start, axis=-1)
    cum_end = mx.cumsum(where_end, axis=-1)

    if input_ids.ndim == 1:
        shifted_start = mx.concatenate([
            mx.zeros((1,), dtype=cum_start.dtype),
            cum_start[:-1]
        ], axis=-1)
    else:
        B = input_ids.shape[0]
        shifted_start = mx.concatenate([
            mx.zeros((B, 1), dtype=cum_start.dtype),
            cum_start[:, :-1]
        ], axis=-1)

    in_region = (shifted_start > cum_end).astype(labels.dtype)
    mask_end = (input_ids == think_end_id).astype(labels.dtype)
    mask = in_region + mask_end > 0

    return mx.where(mask, mx.array(-100, dtype=labels.dtype), labels)
