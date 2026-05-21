import mlx.core as mx

def precompute_rope(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """
    Precompute RoPE sin/cos tables.
    Call once at model init, reuse at every forward pass.
    """
    assert head_dim % 2 == 0
    theta = 1.0 / (base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    positions = mx.arange(max_seq_len, dtype=mx.float32)
    freqs = mx.outer(positions, theta)          # [seq_len, head_dim/2]
    cos = mx.cos(freqs)                          # [seq_len, head_dim/2]
    sin = mx.sin(freqs)                          # [seq_len, head_dim/2]
    return cos, sin

def apply_rope(x, cos, sin, offset: int = 0):
    """
    Apply rotary embeddings to query or key tensor.
    x: [B, n_heads, seq_len, head_dim]
    """
    seq_len = x.shape[2]
    cos = cos[offset:offset + seq_len]           # [seq_len, head_dim/2]
    sin = sin[offset:offset + seq_len]

    # Split head_dim into pairs
    x1 = x[..., 0::2]   # even dims
    x2 = x[..., 1::2]   # odd dims

    # Rotate
    rotated = mx.concatenate([
        x1 * cos[None, None] - x2 * sin[None, None],
        x1 * sin[None, None] + x2 * cos[None, None],
    ], axis=-1)

    return rotated
