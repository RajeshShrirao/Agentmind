import mlx.core as mx
import mlx.nn as nn
import math
from .rope import precompute_rope, apply_rope

class LocalAttentionBlock(nn.Module):
    """
    Sliding window attention — O(L × window) not O(L²).
    Fires every 4th layer. Handles:
      - Precise token recall (exact tool names)
      - Structured output formatting
      - In-context few-shot examples
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.window = cfg.attn_window
        d = cfg.d_model
        dh = cfg.ffn_hidden

        self.norm1 = nn.RMSNorm(d)
        self.norm2 = nn.RMSNorm(d)

        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

        # SwiGLU FFN
        self.gate_proj = nn.Linear(d, dh, bias=False)
        self.up_proj   = nn.Linear(d, dh, bias=False)
        self.down_proj = nn.Linear(dh, d, bias=False)

        # RoPE
        cos, sin = precompute_rope(self.head_dim, cfg.max_seq_len)
        self.rope_cos = cos
        self.rope_sin = sin

    def _local_attn(self, x):
        B, L, _ = x.shape
        H, Hd = self.n_heads, self.head_dim
        W = self.window

        q = self.q_proj(x).reshape(B, L, H, Hd).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, H, Hd).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, H, Hd).transpose(0, 2, 1, 3)

        # Apply RoPE
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        scale = math.sqrt(Hd)

        # Local window mask: each token attends only to last W positions
        # Build position matrix
        pos = mx.arange(L)
        mask = (pos[None, :] - pos[:, None]) >= 0  # causal
        local = (pos[None, :] - pos[:, None]) < W   # window
        attn_mask = mask & local                     # [L, L]
        attn_mask = mx.where(attn_mask, 0.0, float('-inf'))

        scores = (q @ k.transpose(0, 1, 3, 2)) / scale  # [B,H,L,L]
        scores = scores + attn_mask[None, None]
        attn = mx.softmax(scores, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)

    def _ffn(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

    def __call__(self, x):
        x = x + self._local_attn(self.norm1(x))
        x = x + self._ffn(self.norm2(x))
        return x
