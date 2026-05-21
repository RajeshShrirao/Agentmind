import mlx.core as mx
import mlx.nn as nn
import math

def init_agentmind(model, cfg):
    """
    Mamba is sensitive to initialization.
    Wrong init → training instability or silent failure.
    """
    dt_min = getattr(cfg, "dt_min", 1e-4)
    dt_max = getattr(cfg, "dt_max", 1e-1)

    for name, module in model.named_modules():

        # ── Standard linear layers ───────────────────────────
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * cfg.n_layers)  # scaled by depth
            module.weight = mx.random.normal(module.weight.shape) * std
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = mx.zeros(module.bias.shape)

        # ── Mamba-specific: dt_proj bias ─────────────────────
        if "dt_proj" in name and isinstance(module, nn.Linear):
            # Init bias so softplus(bias) spans [dt_min, dt_max]
            # This controls how fast the SSM forgets — critical
            dt = mx.exp(
                mx.random.uniform(
                    shape=(cfg.d_inner,),
                    low=math.log(dt_min),
                    high=math.log(dt_max)
                )
            )
            inv_dt = dt + mx.log(-mx.expm1(-dt))  # inverse softplus
            module.bias = inv_dt

        # ── Mamba-specific: A_log ────────────────────────────
        if "A_log" in name:
            # A controls long-term memory decay
            # Init as evenly spaced log values — empirically stable
            A = mx.broadcast_to(
                mx.arange(1, cfg.d_state + 1, dtype=mx.float32)[None, :],
                (cfg.d_inner, cfg.d_state)
            )
            module.data = mx.log(A)  # stored as log for numerical stability

        # ── Mamba-specific: D (skip) ─────────────────────────
        if name.endswith(".D"):
            module.data = mx.ones(module.shape)  # ones = full skip connection

        # ── Embedding ────────────────────────────────────────
        if isinstance(module, nn.Embedding):
            module.weight = mx.random.normal(module.weight.shape) * 0.02

        # ── RMSNorm ──────────────────────────────────────────
        if isinstance(module, nn.RMSNorm):
            module.weight = mx.ones(module.weight.shape)

    print("Weight initialization complete.")
    return model
