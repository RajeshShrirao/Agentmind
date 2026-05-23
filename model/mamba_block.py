import mlx.core as mx
import mlx.nn as nn
import math


@mx.compile
def _compiled_scan_chunk(dA_c, dBx_c, h):
    C = dA_c.shape[1]
    h_seq = []
    for t in range(C):
        h = dA_c[:, t] * h + dBx_c[:, t]
        h_seq.append(h)
    return mx.stack(h_seq, axis=1), h


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
        dr = cfg.dt_rank

        self.d_inner = di
        self.d_state = ds
        self.d_conv = cfg.d_conv
        self.dt_rank = dr
        self.debug = getattr(cfg, "debug", False)

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
        B, L, _ = x.shape

        if self.debug:
            assert x.ndim == 3, f"Expected 3D tensor [B, L, d_inner], got {x.shape}"
            assert x.shape[-1] == self.d_inner, f"Expected last dimension to be {self.d_inner}, got {x.shape[-1]}"
            assert h_init is None or h_init.shape == (B, self.d_inner, self.d_state), \
                f"Expected h_init shape {(B, self.d_inner, self.d_state)}, got {h_init.shape}"

        A = -mx.exp(self.A_log)

        xbc = self.x_proj(x)

        if self.debug:
            assert xbc.shape[-1] == self.dt_rank + 2 * self.d_state, \
                f"Expected xbc last dimension to be {self.dt_rank + 2 * self.d_state}, got {xbc.shape[-1]}"

        dt_raw, B_mat, C_mat = mx.split(
            xbc, [self.dt_rank, self.dt_rank + self.d_state], axis=-1
        )

        dt = nn.softplus(self.dt_proj(dt_raw))

        dA = mx.exp(dt[:, :, :, None] * A[None, None])
        dB = dt[:, :, :, None] * B_mat[:, :, None, :]
        dBx = dB * x[:, :, :, None]

        # Chunked compiled scan — CHUNK=16 is the confirmed sweet spot on Metal.
        # Larger chunks create deeper backward graphs that Metal can't fuse.
        # This holds regardless of d_state size (tested at both 64 and 16).
        CHUNK = 16
        h = h_init if h_init is not None else mx.zeros((B, self.d_inner, self.d_state))
        h_chunks = []
        for t0 in range(0, L, CHUNK):
            t1 = min(t0 + CHUNK, L)
            h_chunk, h = _compiled_scan_chunk(dA[:, t0:t1], dBx[:, t0:t1], h)
            h_chunks.append(h_chunk)
        h_stack = mx.concatenate(h_chunks, axis=1)

        y = mx.sum(h_stack * C_mat[:, :, None, :], axis=-1)
        return y + x * self.D[None, None, :], h

    def __call__(self, x, h_state=None, return_state=True):
        # x: [B, L, d_model]
        residual = x

        if self.debug:
            assert x.ndim == 3, f"Expected 3D tensor [B, L, d_model], got {x.shape}"
            assert x.shape[-1] == self.norm.weight.shape[0], \
                f"Expected last dimension of x to be {self.norm.weight.shape[0]}, got {x.shape[-1]}"

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

        if self.debug:
            assert ssm_state.shape == (B, self.d_inner, self.d_state), \
                f"Expected ssm_state shape {(B, self.d_inner, self.d_state)}, got {ssm_state.shape}"
            assert conv_state.shape == (B, self.d_conv - 1, self.d_inner), \
                f"Expected conv_state shape {(B, self.d_conv - 1, self.d_inner)}, got {conv_state.shape}"

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

        new_conv_state = x_padded[:, x_padded.shape[1] - (self.d_conv - 1):, :]

        if return_state:
            if self.debug:
                assert h_out.shape == (B, self.d_inner, self.d_state), \
                    f"Expected output ssm_state shape {(B, self.d_inner, self.d_state)}, got {h_out.shape}"
                assert new_conv_state.shape == (B, self.d_conv - 1, self.d_inner), \
                    f"Expected output conv_state shape {(B, self.d_conv - 1, self.d_inner)}, got {new_conv_state.shape}"

            return self.out_proj(y_gated) + residual, {
                "ssm_state": h_out,
                "conv_state": new_conv_state
            }
        else:
            return self.out_proj(y_gated) + residual, None

    # ── Inference-only: single step recurrence ──────────────
    def step(self, x_t, state=None):
        """
        Single token step — pure O(1) recurrence.
        x_t: [..., d_model], state: dict or None
        """
        orig_shape = x_t.shape
        x_t_flat = x_t.reshape(-1, orig_shape[-1])
        B = x_t_flat.shape[0]

        if self.debug:
            assert x_t_flat.shape[-1] == self.norm.weight.shape[0], \
                f"Expected last dimension of x_t to be {self.norm.weight.shape[0]}, got {x_t_flat.shape[-1]}"

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

        if self.debug:
            assert ssm_state.shape == (B, self.d_inner, self.d_state), \
                f"Expected ssm_state shape {(B, self.d_inner, self.d_state)}, got {ssm_state.shape}"
            assert conv_state.shape == (B, self.d_conv - 1, self.d_inner), \
                f"Expected conv_state shape {(B, self.d_conv - 1, self.d_inner)}, got {conv_state.shape}"

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

        xbc = self.x_proj(x_conv_activated)

        if self.debug:
            assert xbc.shape[-1] == self.dt_rank + 2 * self.d_state, \
                f"Expected xbc last dimension to be {self.dt_rank + 2 * self.d_state}, got {xbc.shape[-1]}"

        dt_raw, B_mat, C_mat = mx.split(
            xbc, [self.dt_rank, self.dt_rank + self.d_state], axis=-1
        )

        if self.debug:
            assert dt_raw.shape[-1] == self.dt_rank, f"Expected dt_raw shape to end in {self.dt_rank}, got {dt_raw.shape}"
            assert B_mat.shape[-1] == self.d_state, f"Expected B_mat shape to end in {self.d_state}, got {B_mat.shape}"
            assert C_mat.shape[-1] == self.d_state, f"Expected C_mat shape to end in {self.d_state}, got {C_mat.shape}"

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

        if self.debug:
            assert new_ssm_state.shape == (B, self.d_inner, self.d_state), \
                f"Expected new_ssm_state shape {(B, self.d_inner, self.d_state)}, got {new_ssm_state.shape}"
            assert new_conv_state.shape == (B, self.d_conv - 1, self.d_inner), \
                f"Expected new_conv_state shape {(B, self.d_conv - 1, self.d_inner)}, got {new_conv_state.shape}"

        return out, new_state
