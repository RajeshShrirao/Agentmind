"""
Dataset Expander — CLI entrypoint.

Orchestrates all 8 augmentation layers:
  1. Semantic Query Mutation
  2. Tool Order Mutation
  3. Observation Mutation
  4. Retry / Recovery Mutation
  5. Planner Style Mutation
  6. Graph-based Generation
  7. Adversarial Mutation
  8. Environment Simulation

Usage:
  python -m augmentation.dataset_expander --input data/apprentice_tool_caller.jsonl --target 100000 --output data/apprentice_tool_caller_100k.jsonl
  python -m augmentation.dataset_expander --input seeds.jsonl --target 50000 --entropy-threshold 2.5 --workers 4
"""

import json
import os
import sys
import time
import random
import argparse
from multiprocessing import Pool, cpu_count

from .core import validate_sample, EntropyScorer, DuplicateDetector
from .semantic_mutator import SemanticMutator
from .trajectory_mutator import TrajectoryMutator
from .observation_mutator import ObservationMutator
from .graph_generator import GraphGenerator
from .adversarial_mutator import AdversarialMutator
from .environment_generator import EnvironmentGenerator


class DatasetExpander:
    """
    Multi-layer dataset augmentation orchestrator.

    Loads seeds, runs layers sequentially, validates, deduplicates,
    scores entropy, and writes final dataset at target size.
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.seed = self.config.get("seed", 42)
        random.seed(self.seed)

        # Layer configs with default ratios (% of target contributed by each layer)
        self.layer_ratios = self.config.get("layer_ratios", {
            "semantic": 0.08,
            "tool_order": 0.05,
            "observation": 0.15,
            "retry": 0.12,
            "planner_style": 0.05,
            "graph": 0.30,
            "adversarial": 0.10,
            "environment": 0.15,
        })

        # Instantiate mutators
        self.semantic = SemanticMutator(seed=self.seed)
        self.trajectory = TrajectoryMutator(seed=self.seed + 1)
        self.observation = ObservationMutator(seed=self.seed + 2, failure_rate=self.config.get("failure_rate", 0.35))
        self.graph = GraphGenerator(seed=self.seed + 3)
        self.adversarial = AdversarialMutator(seed=self.seed + 4)
        self.environment = EnvironmentGenerator(seed=self.seed + 5)

        # Filters
        self.scorer = EntropyScorer()
        self.dedup = DuplicateDetector()

    def expand(self, seeds, target_size, workers=1):
        """
        Expand seed dataset to target_size with multi-layer augmentation.
        Returns (final_dataset, stats).
        """
        t0 = time.time()
        pool = []
        stats = {"total_attempted": 0, "valid": 0, "dropped_dup": 0, "dropped_entropy": 0, "by_layer": {}}

        print(f"Seed count: {len(seeds)}")
        print(f"Target: {target_size}")
        print()

        # Each layer targets its ratio of the final dataset
        # Use random subset of seeds for speed

        # Layer 1: Semantic Query Mutation
        print("[Layer 1/8] Semantic Query Mutation...")
        n_target = max(1, int(target_size * self.layer_ratios["semantic"]))
        batch = self._run_layer_to_target(self.semantic.mutate, seeds, n_target)
        pool.extend(batch)
        stats["by_layer"]["semantic"] = len(batch)

        # Layer 2: Tool Order Mutation
        print("[Layer 2/8] Tool Order Mutation...")
        n_target = max(1, int(target_size * self.layer_ratios["tool_order"]))
        batch = self._run_layer_to_target_on_multi(self.trajectory.mutate_tool_order, seeds, n_target)
        pool.extend(batch)
        stats["by_layer"]["tool_order"] = len(batch)

        # Layer 3: Observation Mutation
        print("[Layer 3/8] Observation Mutation...")
        n_target = max(1, int(target_size * self.layer_ratios["observation"]))
        batch = self._run_layer_to_target(self.observation.mutate, seeds, n_target)
        pool.extend(batch)
        stats["by_layer"]["observation"] = len(batch)

        # Layer 4: Retry / Recovery Mutation
        print("[Layer 4/8] Retry / Recovery Mutation...")
        n_target = max(1, int(target_size * self.layer_ratios["retry"]))
        batch = self._run_layer_to_target_on_multi(self.trajectory.mutate_retry, seeds, n_target)
        pool.extend(batch)
        stats["by_layer"]["retry"] = len(batch)

        # Layer 5: Planner Style Mutation
        print("[Layer 5/8] Planner Style Mutation...")
        n_target = max(1, int(target_size * self.layer_ratios["planner_style"]))
        batch = self._run_layer_to_target_on_multi(self.trajectory.mutate_planner_style, seeds, n_target)
        pool.extend(batch)
        stats["by_layer"]["planner_style"] = len(batch)

        # Layer 6: Graph-based Generation
        print("[Layer 6/8] Graph-based Generation...")
        n_target = max(1, int(target_size * self.layer_ratios["graph"]))
        batch = self.graph.generate_batch(n_target)
        pool.extend(batch)
        stats["by_layer"]["graph"] = len(batch)

        # Layer 7: Adversarial Mutation
        print("[Layer 7/8] Adversarial Mutation...")
        n_target = max(1, int(target_size * self.layer_ratios["adversarial"]))
        batch = self._run_layer_to_target(self.adversarial.mutate, seeds, n_target)
        pool.extend(batch)
        stats["by_layer"]["adversarial"] = len(batch)

        # Layer 8: Environment Simulation
        print("[Layer 8/8] Environment Simulation...")
        n_target = max(1, int(target_size * self.layer_ratios["environment"]))
        batch = self._run_layer_to_target(self.environment.enrich, seeds, n_target)
        pool.extend(batch)
        stats["by_layer"]["environment"] = len(batch)

        # Include original seeds
        pool.extend(seeds)

        stats["total_attempted"] = len(pool)
        print(f"\nTotal generated (before filters): {len(pool)}")

        # Validate all
        print("Validating...")
        valid_samples = []
        for s in pool:
            ok, _ = validate_sample(s, strict=False)
            if ok:
                valid_samples.append(s)
        stats["valid"] = len(valid_samples)
        stats["dropped_validation"] = len(pool) - len(valid_samples)
        print(f"  Valid: {len(valid_samples)} (dropped {stats['dropped_validation']})")

        # Dedup
        print("Deduplicating...")
        unique = self.dedup.filter(valid_samples)
        stats["dropped_dup"] = len(valid_samples) - len(unique)
        print(f"  Unique: {len(unique)} (dropped {stats['dropped_dup']})")

        # Entropy filter
        print("Scoring entropy...")
        kept, rejected = self.scorer.filter(unique, threshold=self.config.get("entropy_threshold", 1.8))
        stats["dropped_entropy"] = len(rejected)
        print(f"  After entropy filter: {len(kept)} (dropped {stats['dropped_entropy']})")

        # Shuffle and truncate to target
        random.shuffle(kept)
        final = kept[:target_size]

        elapsed = time.time() - t0
        stats["final_size"] = len(final)
        stats["elapsed"] = elapsed

        print(f"\n{'='*50}")
        print(f"Final dataset: {len(final)} samples")
        print(f"Time: {elapsed:.1f}s ({len(final)/max(elapsed,1):.1f} samples/s)")
        print(f"{'='*50}")

        return final, stats

    def _run_layer_to_target(self, mutate_fn, seeds, n_target):
        """Run a mutation layer, generating up to n_target total variants."""
        random.shuffle(seeds)
        n_per_seed = max(1, min(3, n_target // max(len(seeds), 1) * 2))
        results = []
        for seed in seeds:
            if len(results) >= n_target:
                break
            try:
                variants = mutate_fn(seed, n_variants=n_per_seed)
                results.extend(variants)
            except Exception:
                continue
        return results[:n_target]

    def _run_layer_to_target_on_multi(self, mutate_fn, seeds, n_target):
        """Run on multi-tool seeds only, up to n_target."""
        multi = [s for s in seeds if s.get("type") == "tool_multi"]
        if not multi:
            multi = seeds
        random.shuffle(multi)
        results = []
        for seed in multi:
            if len(results) >= n_target:
                break
            try:
                variants = mutate_fn(seed)
                results.extend(variants)
            except Exception:
                continue
        return results[:n_target]


# ── CLI ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="AgentMind Dataset Expander — 25K → 100K+")
    parser.add_argument("--input", default="data/apprentice_tool_caller.jsonl", help="Input seed JSONL")
    parser.add_argument("--output", default="data/apprentice_tool_caller_100k.jsonl", help="Output path")
    parser.add_argument("--target", type=int, default=100000, help="Target sample count")
    parser.add_argument("--entropy-threshold", type=float, default=2.5, help="Min entropy score to keep")
    parser.add_argument("--failure-rate", type=float, default=0.35, help="Observation failure injection rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    parser.add_argument("--dry-run", action="store_true", help="Run stats only, no write")
    return parser.parse_args()


def print_stats(stats):
    """Print detailed layer-level stats."""
    print("\n--- Layer Contribution ---")
    for layer, count in sorted(stats.get("by_layer", {}).items(), key=lambda x: -x[1]):
        print(f"  {layer}: {count}")
    print(f"\n  Total generated: {stats.get('total_attempted', 0)}")
    print(f"  Valid: {stats.get('valid', 0)}")
    print(f"  Dropped (validation): {stats.get('dropped_validation', 0)}")
    print(f"  Dropped (duplicate): {stats.get('dropped_dup', 0)}")
    print(f"  Dropped (low entropy): {stats.get('dropped_entropy', 0)}")
    print(f"  Final: {stats.get('final_size', 0)}")
    print(f"  Time: {stats.get('elapsed', 0):.1f}s")


def main():
    args = parse_args()

    expander = DatasetExpander(config={
        "seed": args.seed,
        "failure_rate": args.failure_rate,
        "entropy_threshold": args.entropy_threshold,
    })

    # Load seeds
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    with open(args.input) as f:
        seeds = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(seeds)} seeds from {args.input}")

    if args.dry_run:
        print("Dry run — estimating layer contributions only")
        print(f"  Target: {args.target}")
        for layer, ratio in expander.layer_ratios.items():
            print(f"  {layer}: {int(args.target * ratio)}")
        return

    # Run expansion
    dataset, stats = expander.expand(seeds, args.target, workers=args.workers)

    # Write
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(dataset)} samples to {args.output}")
    print_stats(stats)


if __name__ == "__main__":
    main()
