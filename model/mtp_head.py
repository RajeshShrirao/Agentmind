import mlx.core as mx
import mlx.nn as nn

class MTPHead(nn.Module):
    """
    Multi-Token Prediction auxiliary loss.
    Predicts next K tokens simultaneously from each position.
    Forces the model to think ahead — improves instruction following.

    Paper: "Better & Faster Large Language Models via Multi-Token Prediction"
    """

    def __init__(self, cfg, K: int = 4):
        super().__init__()
        self.K = K  # predict K tokens ahead
        self.heads = [
            nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            for _ in range(K)
        ]
        # Shared projection to avoid parameter explosion
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def __call__(self, hidden_states):
        # hidden_states: [B, L, d_model]
        projected = self.proj(hidden_states)
        return [head(projected) for head in self.heads]  # K × [B, L, vocab]

def mtp_loss(mtp_heads_out, targets, ignore_id: int = -100, weight: float = 0.3):
    """
    Compute auxiliary MTP loss.
    Each head k predicts the token k+1 steps ahead.

    weight: how much to add to main loss (0.1–0.3 works well)
    """
    import mlx.nn as nn_ops

    B, L, V = mtp_heads_out[0].shape
    total_aux = 0.0

    for k, logits_k in enumerate(mtp_heads_out):
        # Shift targets: head k predicts position + k + 1
        shift = k + 1
        if shift >= L:
            continue

        pred   = logits_k[:, :-shift].reshape(-1, V)     # [B*(L-shift), V]
        target = targets[:, shift:].reshape(-1)            # [B*(L-shift)]

        mask = (target != ignore_id).astype(mx.float32)
        loss = nn_ops.losses.cross_entropy(pred, target, reduction='none')
        total_aux += (loss * mask).sum() / (mask.sum() + 1e-8)

    return weight * (total_aux / len(mtp_heads_out))
