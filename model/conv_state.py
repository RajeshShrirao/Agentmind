import mlx.core as mx

class ConvState:
    """
    Manages the sliding conv buffer for Mamba's causal depthwise conv
    during single-token (autoregressive) inference.

    At training: process full sequence with padding.
    At inference: slide a d_conv-length buffer per token.
    """

    def __init__(self, batch_size: int, d_inner: int, d_conv: int):
        # Buffer: last (d_conv - 1) input vectors
        self.buf = mx.zeros((batch_size, d_conv - 1, d_inner))
        self.d_conv = d_conv

    def step(self, x_t, conv_weight, conv_bias):
        """
        x_t:        [B, d_inner] — current token's inner representation
        conv_weight: [d_inner, d_conv] — depthwise conv weights
        conv_bias:   [d_inner]
        Returns:    [B, d_inner] — conv output for this timestep
        """
        # Append current token to buffer
        x_t_expanded = x_t[:, None, :]                  # [B, 1, d_inner]
        window = mx.concatenate([self.buf, x_t_expanded], axis=1)  # [B, d_conv, d_inner]

        # Depthwise conv: dot each channel independently
        # conv_weight: [d_inner, d_conv]
        out = mx.sum(window * conv_weight[None, :, :].transpose(0, 2, 1), axis=1)
        out = out + conv_bias[None, :]                   # [B, d_inner]

        # Slide buffer: drop oldest, keep last (d_conv - 1)
        self.buf = window[:, 1:, :]
        return out
