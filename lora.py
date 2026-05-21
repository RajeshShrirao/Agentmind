import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import math

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with low-rank adapter.
    Only A and B are trained. Base weight is frozen.
    """

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        in_features  = base.weight.shape[1]
        out_features = base.weight.shape[0]
        self.scale = alpha / rank

        # Freeze base weight (not a parameter)
        self.weight = base.weight          # frozen
        self.bias   = getattr(base, "bias", None)

        # Trainable low-rank matrices
        self.A = mx.random.normal((rank, in_features)) * (1 / math.sqrt(rank))
        self.B = mx.zeros((out_features, rank))

    def __call__(self, x):
        base_out = x @ self.weight.T
        if self.bias is not None:
            base_out = base_out + self.bias
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + self.scale * lora_out

def apply_lora(model, rank: int = 16, alpha: float = 32.0, targets: list[str] = None):
    """
    Wrap target linear layers with LoRA. Freeze everything else.

    Default targets — layers that matter most for agentic behavior:
      MambaBlock:         in_proj, out_proj
      LocalAttentionBlock: q_proj, v_proj
      LM head
    """
    if targets is None:
        targets = ["in_proj", "out_proj", "q_proj", "v_proj", "lm_head"]

    # Freeze entire model first
    model.freeze()

    # Walk and replace target layers
    def _replace(module):
        for name in module.children():
            child = getattr(module, name)
            if isinstance(child, nn.Linear):
                if any(t in name for t in targets):
                    lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                    setattr(module, name, lora_layer)
            elif isinstance(child, nn.Module):
                _replace(child)

    _replace(model)

    # Count trainable params
    total = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"LoRA applied | Trainable params: {total:,} ({total/1e6:.2f}M)")
    return model
