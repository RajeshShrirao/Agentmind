import mlx.core as mx
import re
import random
import numpy as np

# ── Staged Training Curriculum ────────────────────────────
#
# Stage 1 (steps 0–500):   Normal training, no latent tokens
# Stage 2 (steps 500–1000): Insert <|think_start|>..CoT..<|think_end|> boundaries
# Stage 3 (steps 1000–2000): Replace 50% of CoT tokens with latent steps
# Stage 4 (steps 2000+):    Full latent — CoT removed entirely
#
# NEVER cold-start latent reasoning — the model needs to
# learn what good reasoning looks like before hiding it.

N_LATENT_STEPS = 4   # how many <|scratch|> placeholder tokens to insert when stripping CoT (stage 4)

def get_latent_stage(step: int) -> int:
    """Get the latent stage based on the current step."""
    if step < 500:
        return 1
    elif step < 1000:
        return 2
    elif step < 2000:
        return 3
    else:
        return 4

def inject_latent_tokens(sample: dict, tokenizer, stage: int) -> dict:
    """
    Progressively replace chain-of-thought with latent boundaries.
    Call during data preprocessing, not at model forward time.
    """
    if stage < 2:
        return sample  # Stage 1: pass through unchanged

    effective_stage = stage
    if stage == 3:
        # In Stage 3, we interpolate by randomly treating the sample
        # as Stage 4 (latent reasoning placeholder) with 50% probability,
        # and Stage 2 (explicit CoT wrapped in boundaries) with 50% probability.
        effective_stage = 4 if random.random() < 0.5 else 2

    for msg in sample["messages"]:
        if msg["role"] != "assistant":
            continue

        content = msg["content"]

        # Detect CoT markers (e.g. <|scratch|> content)
        if "<|scratch|>" in content:
            # Use regex to robustly capture scratch/thought content up to the next structural tag
            pattern = re.compile(
                r"<\|scratch\|>(.*?)(?=<\|tool_call\|>|<\|observe\|>|<\|plan\|>|<\|assistant\|>|<\|user\|>|<\|system\|>|<eos>|$)",
                re.DOTALL
            )
            
            def replace_match(match):
                thoughts = match.group(1)
                if effective_stage == 2:
                    return f"<|think_start|><|scratch|>{thoughts}<|think_end|>"
                else:  # stage >= 4
                    scratch_tokens = "<|scratch|>" * N_LATENT_STEPS
                    return f"<|think_start|>{scratch_tokens}<|think_end|>"

            content = pattern.sub(replace_match, content)

        msg["content"] = content

    return sample

def latent_loss_mask(input_ids, labels, think_start_id, think_end_id):
    """
    During latent stages, zero out loss between think_start and think_end.
    Model is not penalized for what it 'thinks' — only for what it emits.
    Supports both 1D and 2D arrays/tensors (MLX, NumPy or lists).
    """
    is_mlx = isinstance(labels, mx.array)
    
    if is_mlx:
        ids_np = np.array(input_ids)
        labels_np = np.array(labels)
    else:
        ids_np = np.asarray(input_ids)
        labels_np = np.asarray(labels)

    if ids_np.ndim == 1:
        in_latent = False
        for i in range(len(ids_np)):
            tok_id = ids_np[i]
            if tok_id == think_start_id:
                in_latent = True
                continue
            if tok_id == think_end_id:
                labels_np[i] = -100
                in_latent = False
                continue
            if in_latent:
                labels_np[i] = -100
    elif ids_np.ndim == 2:
        B, L = ids_np.shape
        for b in range(B):
            in_latent = False
            for i in range(L):
                tok_id = ids_np[b, i]
                if tok_id == think_start_id:
                    in_latent = True
                    continue
                if tok_id == think_end_id:
                    labels_np[b, i] = -100
                    in_latent = False
                    continue
                if in_latent:
                    labels_np[b, i] = -100
    else:
        raise ValueError(f"Unsupported input dimension: {ids_np.ndim}")

    return mx.array(labels_np) if is_mlx else labels_np
