import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import math
from pathlib import Path

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with low-rank adapter.
    Only A and B are trained. Base weight is frozen.
    """

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.scale = alpha / rank
        self.base = base

        in_features  = base.weight.shape[1]
        out_features = base.weight.shape[0]

        # Trainable low-rank matrices
        self.A = mx.random.normal((rank, in_features)) * (1 / math.sqrt(rank))
        self.B = mx.zeros((out_features, rank))

    def __call__(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

def apply_lora(model, rank: int = 16, alpha: float = 32.0, targets: list[str] = None):
    """
    Wrap target linear layers with LoRA. Freeze everything else.

    Default targets — layers that matter most for agentic behavior:
      MambaBlock:         in_proj, out_proj
      LocalAttentionBlock: q_proj, v_proj, o_proj
      LM head
    """
    if targets is None:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Freeze entire model first
    model.freeze()

    # Walk and replace target layers recursively
    def _replace(module):
        if isinstance(module, nn.Module):
            for name, child in list(module.children().items()):
                if isinstance(child, nn.Linear):
                    if any(t in name for t in targets):
                        lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                        setattr(module, name, lora_layer)
                else:
                    _replace(child)
        elif isinstance(module, list):
            for item in module:
                _replace(item)
        elif isinstance(module, dict):
            for item in module.values():
                _replace(item)

    _replace(model)

    # Count trainable params
    total = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"LoRA applied | Trainable params: {total:,} ({total/1e6:.2f}M)")
    return model


def load_lora(model, adapter_weights: dict):
    '''
    Load LoRA A/B weights into an MLX model with LoRALinear layers.
    adapter_weights: dict of {"layer_name.A": mx.array, "layer_name.B": ...}

    The model already has LoRALinear layers applied from apply_lora().
    This just updates A and B matrices of existing LoRALinear layers.
    Works with Qwen2.5 or any MLX model using the same target module names.
    '''
    nested = tree_unflatten(adapter_weights)
    model.update(nested)
    loaded_params = sum(v.nbytes for v in adapter_weights.values()) // 1024
    print(f"[lora] Loaded {len(adapter_weights)} LoRA parameter tensors ({loaded_params} KB)")
    return model


def save_adapter(model, adapter_name, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{adapter_name}.safetensors"

    params = dict(tree_flatten(model.trainable_parameters()))
    lora_params = {}
    target_modules = set()
    for key, val in params.items():
        if key.endswith('.A') or key.endswith('.B'):
            lora_params[key] = val
            target_modules.add(key.split('.')[-2])

    rank = None
    alpha = None
    def _find_lora(module):
        nonlocal rank, alpha
        if isinstance(module, LoRALinear):
            rank = module.A.shape[0]
            alpha = int(round(module.scale * rank))
            return True
        if isinstance(module, nn.Module):
            for child in module.children().values():
                if _find_lora(child):
                    return True
        elif isinstance(module, list):
            for item in module:
                if _find_lora(item):
                    return True
        elif isinstance(module, dict):
            for item in module.values():
                if _find_lora(item):
                    return True
        return False
    _find_lora(model)

    metadata = {
        "lora_rank": str(rank or "?"),
        "lora_alpha": str(alpha or "?"),
        "target_modules": ",".join(sorted(target_modules)),
    }
    mx.save_safetensors(str(path), lora_params, metadata)
    total_kb = sum(v.nbytes for v in lora_params.values()) // 1024
    print(f"[lora] Saved adapter '{adapter_name}' \u2192 {path} ({total_kb} KB)")

def load_adapter(model, adapter_path):
    path = Path(adapter_path)
    if not path.exists():
        raise FileNotFoundError(f"Adapter not found: {path}")

    loaded = mx.load(str(path))
    if 'metadata' in loaded:
        del loaded['metadata']

    nested = tree_unflatten(dict(loaded))
    model.update(nested)
    total_kb = sum(v.nbytes for v in loaded.values()) // 1024
    print(f"[lora] Loaded adapter from {path} ({total_kb} KB)")
    return model


def reset_adapter(model):
    def _reset(module):
        if isinstance(module, LoRALinear):
            r = module.A.shape[0]
            in_features = module.A.shape[1]
            module.A = mx.random.normal((r, in_features)) * (1 / math.sqrt(r))
            module.B = mx.zeros_like(module.B)
        elif isinstance(module, nn.Module):
            for child in module.children().values():
                _reset(child)
        elif isinstance(module, list):
            for item in module:
                _reset(item)
        elif isinstance(module, dict):
            for item in module.values():
                _reset(item)
    _reset(model)
    print("[lora] Reset all LoRA adapters to random init")
