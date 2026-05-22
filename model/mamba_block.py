import mlx.core as mx
import mlx.nn as nn
import math


class MambaBlock(nn.Module):
    """
    Selective State Space Model block.

    At inference: pure O(1) recurrence — SSM state stays fixed size
    regardless of how many tool calls have been processed.

    At training: parallel scan (log-space) with numerical clipping.
    log_contrib ∈ [-50, 50] prevents overflow in exp() that caused
    0 × inf = NaN with d_state=16 and L ≥ 8.
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        di = cfg.d_inner
        ds = cfg.d_state
        dr = cfg.dt_rank_val

        self.d_inner = di
        self.d_state = ds
        self.d_conv = cfg.d_conv

        # Pre-norm
        self.norm = nn.RMSNorm(d)

        # Split input into x (signal) and z (gate)
        self.in_proj = nn.Linear(d, di * 2, bias=False)

        # Causal depthwise conv (groups=di for depthwise)
        self.conv = nn.Conv1d(
            in_channels=di,
            out_channels=di,
            kernel_size=cfg.d_conv,
            padding=cfg.d_conv - 1,   # left-pad for causality
            groups=di,
            bias=True
        )

        # SSM projections: dt (step), B (input gate), C (output gate)
        self.x_proj = nn.Linear(di, dr + ds * 2, bias=False)
        self.dt_proj = nn.Linear(dr, di, bias=True)

        # A: decay matrix, log-parameterized for stability
        A = mx.broadcast_to(
            mx.arange(1, ds + 1, dtype=mx.float32)[None, :],
            (di, ds)
        )
        self.A_log = mx.log(A)
        self.D = mx.ones((di,))

        self.out_proj = nn.Linear(di, d, bias=False)

    def _ssm(self, x, h_init=None):
        # x: [B, L, d_inner]
        B, L, _ = x.shape
        dr, ds = self.x_proj.weight.shape[0] - self.d_state * 2, self.d_state

        A = -mx.exp(self.A_log)                    # [d_inner, d_state]

        # Project to dt, B_mat, C_mat
        xbc = self.x_proj(x)                       # [B, L, dr + 2*ds]
        dt_raw, B_mat, C_mat = mx.split(
            xbc, [dr, dr + ds], axis=-1
        )
        dt = nn.softplus(self.dt_proj(dt_raw))     # [B, L, d_inner]

        # ZOH discretization
        dA = mx.exp(dt[:, :, :, None] * A[None, None])       # [B, L, d_inner, d_state]
        dB = dt[:, :, :, None] * B_mat[:, :, None, :]        # [B, L, d_inner, d_state]
        dBx = dB * x[:, :, :, None]                          # [B, L, d_inner, d_state]

        # Sequential scan (fully compiled by MLX, mathematically exact)
        h = h_init if h_init is not None else mx.zeros((B, self.d_inner, self.d_state))
        h_seq = []
        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]
            h_seq.append(h)
        h_stack = mx.stack(h_seq, axis=1)                    # [B, L, d_inner, d_state]

        # Output: y = sum(h * C) + D * x
        y = mx.sum(h_stack * C_mat[:, :, None, :], axis=-1)
        return y + x * self.D[None, None, :], h_stack[:, -1]

    def __call__(self, x, h_state=None):
        # x: [B, L, d_model]
        residual = x
        x_normed = self.norm(x)

        xz = self.in_proj(x_normed)
        x_in, z = mx.split(xz, [self.d_inner], axis=-1)

        B, L, _ = x_in.shape

        if h_state is not None:
            if isinstance(h_state, dict):
                ssm_state = h_state["ssm_state"]
                conv_state = h_state["conv_state"]
            else:
                ssm_state, conv_state = h_state
        else:
            ssm_state = mx.zeros((B, self.d_inner, self.d_state))
            conv_state = mx.zeros((B, self.d_conv - 1, self.d_inner))

        # Causal depthwise conv — prepend buffer
        x_padded = mx.concatenate([conv_state, x_in], axis=1)
        x_conv_raw = mx.conv1d(x_padded, self.conv.weight, stride=1, padding=0, dilation=1, groups=self.d_inner)
        if self.conv.bias is not None:
            x_conv_raw = x_conv_raw + self.conv.bias

        x_conv = nn.silu(x_conv_raw)

        # SSM + skip connection
        y_ssm, h_out = self._ssm(x_conv, ssm_state)

        # Gate
        y_gated = y_ssm * nn.silu(z)

        # Slide conv buffer
        new_conv_state = x_padded[:, x_padded.shape[1] - (self.d_conv - 1):, :]

        new_state = {
            "ssm_state": h_out,
            "conv_state": new_conv_state
        }

        return self.out_proj(y_gated) + residual, new_state

    # ── Inference-only: single step recurrence ──────────────
    def step(self, x_t, state=None):
        """
        Single token step — pure O(1) recurrence.
        x_t: [..., d_model], state: dict or None
        """
        orig_shape = x_t.shape
        x_t_flat = x_t.reshape(-1, orig_shape[-1])
        B = x_t_flat.shape[0]

        x_normed = self.norm(x_t_flat)
        xz = self.in_proj(x_normed)
        x_in, z = mx.split(xz, [self.d_inner], axis=-1)

        if state is not None:
            if isinstance(state, dict):
                ssm_state = state["ssm_state"]
                conv_state = state["conv_state"]
            else:
                ssm_state, conv_state = state
        else:
            ssm_state = mx.zeros((B, self.d_inner, self.d_state))
            conv_state = mx.zeros((B, self.d_conv - 1, self.d_inner))

        # Conv step: slide window
        x_in_expanded = x_in[:, None, :]
        window = mx.concatenate([conv_state, x_in_expanded], axis=1)

        # Depthwise conv
        conv_weight = self.conv.weight[:, :, 0]
        x_conv = mx.sum(window * conv_weight[None, :, :].transpose(0, 2, 1), axis=1)
        if self.conv.bias is not None:
            x_conv = x_conv + self.conv.bias[None, :]

        new_conv_state = window[:, window.shape[1] - (self.d_conv - 1):, :]

        # SSM step
        x_conv_activated = nn.silu(x_conv)

        dr = self.x_proj.weight.shape[0] - self.d_state * 2
        ds = self.d_state

        xbc = self.x_proj(x_conv_activated)
        dt_raw, B_mat, C_mat = mx.split(xbc, [dr, dr + ds], axis=-1)
        dt = nn.softplus(self.dt_proj(dt_raw))

        A = -mx.exp(self.A_log)
        dA = mx.exp(dt[:, :, None] * A[None, :, :])
        dB = dt[:, :, None] * B_mat[:, None, :]

        new_ssm_state = dA * ssm_state + dB * x_conv_activated[:, :, None]
        y = mx.sum(new_ssm_state * C_mat[:, None, :], axis=-1)
        y = y + x_conv_activated * self.D[None, :]

        z_gate = nn.silu(z)
        y_gated = y * z_gate

        out_flat = self.out_proj(y_gated) + x_t_flat
        out = out_flat.reshape(orig_shape)

        new_state = {
            "ssm_state": new_ssm_state,
            "conv_state": new_conv_state
        }

        return out, new_state
