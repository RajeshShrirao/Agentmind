import mlx.core as mx
import mlx.nn as nn
import math

class MambaBlock(nn.Module):
    """
    Selective State Space Model block.
    
    At inference: pure O(1) recurrence — SSM state stays fixed size
    regardless of how many tool calls have been processed.
    
    At training: sequential scan (correct). Swap with parallel scan
    for full training speed (see note at bottom).
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

    def _ssm(self, x):
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
        # dA: [B, L, d_inner, d_state]
        dA = mx.exp(dt[:, :, :, None] * A[None, None])
        # dB: [B, L, d_inner, d_state]
        dB = dt[:, :, :, None] * B_mat[:, :, None, :]

        # Sequential scan — correct for both train and inference
        # For training speed: replace with parallel scan using mx.cumsum
        h = mx.zeros((B, self.d_inner, self.d_state))
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t, :, None]
            y = mx.sum(h * C_mat[:, t, None, :], axis=-1)  # [B, d_inner]
            ys.append(y)

        y = mx.stack(ys, axis=1)                   # [B, L, d_inner]
        return y + x * self.D[None, None, :], h    # output, final state

    def __call__(self, x, h_state=None):
        # x: [B, L, d_model]
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_in, z = mx.split(xz, [self.d_inner], axis=-1)

        # Causal depthwise conv — trim padding to maintain causality
        x_conv = self.conv(x_in)[:, :x_in.shape[1], :]
        x_conv = nn.silu(x_conv)

        # SSM + skip connection
        y, h_out = self._ssm(x_conv)

        # Gate
        y = y * nn.silu(z)

        return self.out_proj(y) + residual, h_out

    # ── Inference-only: single step recurrence ──────────────
    def step(self, x_t, h):
        """
        Single token step — pure O(1) recurrence.
        x_t: [B, d_model], h: [B, d_inner, d_state]
        """
        x_t = self.norm(x_t)
        xz = self.in_proj(x_t[:, None, :])
        x_in, z = mx.split(xz, [self.d_inner], axis=-1)

        # Conv step: slide window (maintain conv buffer externally)
        x_conv = nn.silu(x_in.squeeze(1))

        xbc = self.x_proj(x_conv)
        dt_raw, B_mat, C_mat = mx.split(xbc, [self.d_inner // 16, -self.d_state], axis=-1)
        dt = nn.softplus(self.dt_proj(dt_raw))

        A = -mx.exp(self.A_log)
        dA = mx.exp(dt[:, :, None] * A[None])
        dB = dt[:, :, None] * B_mat[:, None, :]

        h = dA * h + dB * x_conv[:, :, None]
        y = mx.sum(h * C_mat[:, None, :], axis=-1)
        y = y + x_conv * self.D[None]
        z_gate = nn.silu(z.squeeze(1))

        return self.out_proj(y * z_gate), h
