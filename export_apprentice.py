"""
export_apprentice.py — Multi-adapter export for Qwen2.5-based AgentMind.

Exports three artifacts:
  1. Backbone: weights + config.json + tokenizer files
  2. Adapters: one .safetensors per specialist
  3. Router: weights + router_config.json

Usage:
  python export_apprentice.py \
    --backbone ./checkpoints/backbone \
    --adapters ./checkpoints/adapters \
    --out ./apprentice-system \
    --bits 4
  
  python export_apprentice.py --adapters ./checkpoints/adapters --out ./apprentice-system
  
  python export_apprentice.py --backbone ./checkpoints/backbone --out ./apprentice-system
"""

import argparse
import json, os, shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten


def export_system(backbone_path=None, adapters_dir=None, router_path=None,
                  output_dir="./apprentice-system", bits=4):
    """Export the complete AgentMind system.
    
    Args:
        backbone_path: Path to backbone .safetensors file or directory
        adapters_dir: Directory containing adapter .safetensors files
        router_path: Path to router .safetensors file
        output_dir: Output directory
        bits: Quantization bits (default 4)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Export backbone
    if backbone_path:
        _export_backbone(backbone_path, output_dir, bits)
    
    # 2. Export adapters
    if adapters_dir:
        _export_adapters(adapters_dir, output_dir)
    
    # 3. Export router
    if router_path:
        _export_router(router_path, output_dir)
    
    _print_summary(output_dir)
    return output_dir


def _export_backbone(backbone_path, output_dir, bits=4):
    backbone_dir = output_dir / "backbone"
    backbone_dir.mkdir(parents=True, exist_ok=True)
    
    backbone_path = Path(backbone_path)
    if backbone_path.is_dir():
        # Copy all files from the backbone directory
        for f in backbone_path.iterdir():
            if f.is_file():
                shutil.copy2(f, backbone_dir / f.name)
        print(f"  Copied backbone from {backbone_path} -> {backbone_dir}")
    elif backbone_path.exists() and backbone_path.suffix == '.safetensors':
        # Single safetensors file
        shutil.copy2(backbone_path, backbone_dir / backbone_path.name)
        print(f"  Copied backbone weights -> {backbone_dir / backbone_path.name}")
    
    print(f"  Backbone exported to {backbone_dir}")


def _export_adapters(adapters_dir, output_dir):
    adapters_dir = Path(adapters_dir)
    out_adapters = output_dir / "adapters"
    out_adapters.mkdir(parents=True, exist_ok=True)
    
    domain_names = ['tool_caller', 'planner', 'recovery', 'code', 'research']
    for domain in domain_names:
        src = adapters_dir / f"{domain}.safetensors"
        if src.exists():
            shutil.copy2(src, out_adapters / src.name)
            size_kb = src.stat().st_size // 1024
            print(f"  Adapter '{domain}': {src.name} ({size_kb} KB)")
        else:
            print(f"  Warning: adapter '{domain}' not found at {src}")
    
    print(f"  Adapters exported to {out_adapters}")


def _export_router(router_path, output_dir):
    router_path = Path(router_path)
    if router_path.is_dir():
        router_path = router_path / "router.safetensors"
    
    if router_path.exists():
        # Load router to extract metadata
        loaded = mx.load(str(router_path))
        metadata = loaded.get("metadata", {})
        if "metadata" in loaded:
            del loaded["metadata"]
        
        # Save router weights
        shutil.copy2(router_path, output_dir / "router.safetensors")
        
        # Write router config
        import json as json_module
        domain_names = json_module.loads(metadata.get("domain_names", "[]"))
        router_config = {
            "domain_names": domain_names,
            "hidden_size": loaded.get("classifier.layers.0.weight", mx.zeros((1,))).shape[1],
            "num_domains": len(domain_names),
            "fallback_threshold": 0.6,
        }
        with open(output_dir / "router_config.json", "w") as f:
            json_module.dump(router_config, f, indent=2)
        
        print(f"  Router exported to {output_dir / 'router.safetensors'}")
        print(f"  Router config -> {output_dir / 'router_config.json'}")
    else:
        print(f"  Warning: router not found at {router_path}")


def _print_summary(output_dir):
    print(f"\n{'='*60}")
    print(f"  Export Summary")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}")
    
    backbone_dir = output_dir / "backbone"
    if backbone_dir.exists():
        print(f"  Backbone: {backbone_dir}")
    
    adapters_dir = output_dir / "adapters"
    if adapters_dir.exists():
        for f in sorted(adapters_dir.glob("*.safetensors")):
            size_kb = f.stat().st_size // 1024
            print(f"  Adapter: {f.name} ({size_kb} KB)")
    
    router_path = output_dir / "router.safetensors"
    if router_path.exists():
        size_kb = router_path.stat().st_size // 1024
        print(f"  Router: {router_path.name} ({size_kb} KB)")
    
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Export AgentMind System (backbone + adapters + router)")
    parser.add_argument("--backbone", type=str, default=None,
                        help="Path to backbone .safetensors or directory")
    parser.add_argument("--adapters", type=str, default=None,
                        help="Directory containing adapter .safetensors files")
    parser.add_argument("--router", type=str, default=None,
                        help="Path to router .safetensors or directory")
    parser.add_argument("--out", type=str, default="./apprentice-system",
                        help="Output directory")
    parser.add_argument("--bits", type=int, default=4,
                        help="Quantization bits (default: 4)")
    args = parser.parse_args()
    
    export_system(
        backbone_path=args.backbone,
        adapters_dir=args.adapters,
        router_path=args.router,
        output_dir=args.out,
        bits=args.bits,
    )


if __name__ == "__main__":
    main()
