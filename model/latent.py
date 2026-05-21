import mlx.core as mx
import mlx.nn as nn

# ── Staged Training Curriculum ────────────────────────────
#
# Stage 1 (steps 0–500):   Normal training, no latent tokens
# Stage 2 (steps 500–1000): Insert <|think_start|>..CoT..<|think_end|> boundaries
# Stage 3 (steps 1000–2000): Replace 50% of CoT tokens with latent steps
# Stage 4 (steps 2000+):    Full latent — CoT removed entirely
#
# NEVER cold-start latent reasoning — the model needs to
# learn what good reasoning looks like before hiding it.

N_LATENT_STEPS = 4   # how many silent recurrence steps before emitting token

def inject_latent_tokens(sample: dict, tokenizer, stage: int) -> dict:
    """
    Progressively replace chain-of-thought with latent boundaries.
    Call during data preprocessing, not at model forward time.
    """
    if stage < 2:
        return sample  # Stage 1: pass through unchanged

    for msg in sample["messages"]:
        if msg["role"] != "assistant":
            continue

        content = msg["content"]

        # Detect CoT markers (e.g. <|scratch|> content)
        if "<|scratch|>" in content:
            if stage == 2:
                # Wrap scratch content in latent boundary tokens
                content = content.replace(
                    "<|scratch|>",
                    "<|think_start|><|scratch|>"
                ).replace(
                    # End boundary before next structural token
                    "<|tool_call|>", "<|think_end|><|tool_call|>"
                )
            elif stage >= 3:
                # Remove scratch content entirely — model thinks silently
                import re
                content = re.sub(r"<\|think_start\|>.*?<\|think_end\|>", 
                                 "<|think_start|><|think_end|>", 
                                 content, flags=re.DOTALL)

        msg["content"] = content

    return sample

class LatentReasoningWrapper(nn.Module):
    """
    Wraps a MambaBlock to execute N silent recurrence steps
    when <|think_start|> token is detected.

    At <|think_start|>: enter latent mode
    For N steps: update hidden state without emitting tokens
    At <|think_end|>: resume normal generation
    """

    def __init__(self, mamba_block, cfg, n_steps: int = N_LATENT_STEPS):
        super().__init__()
        self.block = mamba_block
        self.n_steps = n_steps
        self.think_start_id = cfg.think_start_id
        self.think_end_id   = cfg.think_end_id

    def latent_forward(self, hidden, h_state):
        """
        Execute N silent SSM recurrence steps.
        No tokens emitted. Hidden state accumulates reasoning.
        """
        for _ in range(self.n_steps):
            # Feed last hidden state back as input (no decode step)
            hidden, h_state = self.block(hidden, h_state)
        return hidden, h_state

    def __call__(self, x, input_ids=None, h_state=None):
        if input_ids is not None:
            # Check if any token in this batch is think_start
            has_think = mx.any(input_ids == self.think_start_id)
            if has_think:
                x, h_state = self.latent_forward(x, h_state)

        return self.block(x, h_state)

def latent_loss_mask(input_ids, labels, think_start_id, think_end_id):
    """
    During latent stages, zero out loss between think_start and think_end.
    Model is not penalized for what it 'thinks' — only for what it emits.
    """
    in_latent = False
    masked_labels = labels.tolist()

    for i, tok_id in enumerate(input_ids.tolist()):
        if tok_id == think_start_id:
            in_latent = True
        if tok_id == think_end_id:
            in_latent = False
        if in_latent:
            masked_labels[i] = -100  # ignore in loss

    return mx.array(masked_labels)
