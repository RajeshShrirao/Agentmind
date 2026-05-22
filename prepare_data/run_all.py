"""
Orchestrate all 5 apprentice domain data builders.
Outputs summary table and generates router_training.jsonl.

Usage:
  python prepare_data/run_all.py              # Full pipeline (HF + synthetic)
  python prepare_data/run_all.py --skip-hf     # Synthetic-only (no HF downloads)
"""
import sys
import os
import json
import random
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOMAINS = ["tool_caller", "planner", "recovery", "code", "research"]

SKIP_HF = "--skip-hf" in sys.argv


def main():
    print("=" * 70)
    print("AgentMind — Prepare Data (HF + Synthetic Hybrid Pipeline)")
    print("=" * 70)
    if SKIP_HF:
        print("[mode] Skipping HF downloads (synthetic-only)")

    all_results = {}
    all_samples_by_domain = {}

    for domain in DOMAINS:
        mod = importlib.import_module(f"prepare_data.{domain}")
        result = mod.main(skip_hf=SKIP_HF)
        all_results.update(result)

        path = f"data/apprentice_{domain}.jsonl"
        with open(path) as f:
            samples = [json.loads(line) for line in f]
        all_samples_by_domain[domain] = samples

    print(f"\n{'=' * 70}")
    print(f"{'Domain':<16} {'Total':>8} {'HF':>8} {'Synth':>8} {'Adv':>10} {'Latent':>8} {'%Real':>8}")
    print(f"{'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

    total_all = 0
    total_hf = 0
    total_synth = 0
    total_adv = 0
    total_latent = 0

    for domain in DOMAINS:
        r = all_results[domain]
        pct_real = 100 * r["hf"] // max(r["all"], 1)
        print(f"{domain:<16} {r['all']:>8} {r['hf']:>8} {r['synth']:>8} {r['adv']:>10} {r['latent']:>8} {pct_real:>7}%")
        total_all += r["all"]
        total_hf += r["hf"]
        total_synth += r["synth"]
        total_adv += r["adv"]
        total_latent += r["latent"]

    print(f"{'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    pct_real_all = 100 * total_hf // max(total_all, 1)
    print(f"{'TOTAL':<16} {total_all:>8} {total_hf:>8} {total_synth:>8} {total_adv:>10} {total_latent:>8} {pct_real_all:>7}%")

    # Router training data: 200 per domain, shuffled
    print(f"\n{'=' * 70}")
    print("[router] Building router training data...")
    router_data = []
    for domain, samples in all_samples_by_domain.items():
        chosen = random.sample(samples, min(200, len(samples)))
        for s in chosen:
            router_data.append({
                "domain": domain,
                "messages": s["messages"],
            })
    random.shuffle(router_data)
    path = "data/router_training.jsonl"
    with open(path, "w") as f:
        for s in router_data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  → {path} ({len(router_data)} samples, {len(router_data) // 5} per domain)")

    print(f"\n{'=' * 70}")
    print("Done. Data ready for specialist training + router.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
