import logging, sys, time, os

sys.path.insert(0, ".")

from prepare_data.base import (
    download_hf_dataset,
    convert_to_apprentice,
    combine,
    train_val_split,
    write_jsonl,
)
from prepare_data.domain_configs import DOMAIN_CONFIGS
from generate_scaled_synthetic import generate_recovery
from monitor import print_hw

logger = logging.getLogger(__name__)
DOMAIN = "recovery"
CFG = DOMAIN_CONFIGS[DOMAIN]


def main(skip_hf=False):
    t_start = time.time()
    print(f"\n{'=' * 60}")
    print(f"[{DOMAIN}] Preparing dataset (synthetic only)...")
    print(f"{'=' * 60}")
    print_hw(DOMAIN)

    print(f"  [synth] Generating {CFG['synthetic_count']} synthetic samples...")
    result = combine(
        [],
        generate_recovery,
        CFG["synthetic_count"],
        CFG["adversarial_rate"],
        domain=DOMAIN,
    )
    all_samples, n_hf, n_synth, n_adv, n_latent = result
    print_hw(f"{DOMAIN} synth")

    train, val = train_val_split(all_samples)
    out_path = f"data/apprentice_{DOMAIN}.jsonl"
    write_jsonl(all_samples, out_path)
    file_size = os.path.getsize(out_path) / (1024 * 1024)

    total_elapsed = time.time() - t_start
    pct_real = 100 * n_hf // max(len(all_samples), 1)
    print(f"  [{DOMAIN}] Done in {total_elapsed:.0f}s")
    print(f"  [{DOMAIN}] {len(all_samples)} samples ({pct_real}% real, {100-pct_real}% synth)")
    print(f"  [{DOMAIN}] adversarial={n_adv} latent={n_latent} file={file_size:.1f}MB")
    print(f"  [{DOMAIN}] → {out_path} ({len(train)} train, {len(val)} val)")
    print_hw(f"{DOMAIN} done")
    return {DOMAIN: {"all": len(all_samples), "hf": n_hf, "synth": n_synth, "adv": n_adv, "latent": n_latent}}


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
