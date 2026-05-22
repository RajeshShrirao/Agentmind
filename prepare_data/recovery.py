import logging
import sys

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

logger = logging.getLogger(__name__)
DOMAIN = "recovery"
CFG = DOMAIN_CONFIGS[DOMAIN]


def main(skip_hf=False):
    print(f"\n{'=' * 60}")
    print(f"[{DOMAIN}] Preparing dataset (synthetic only)...")
    print(f"{'=' * 60}")

    print(f"  [synth] Generating {CFG['synthetic_count']} synthetic samples...")
    result = combine(
        [],
        generate_recovery,
        CFG["synthetic_count"],
        CFG["adversarial_rate"],
        domain=DOMAIN,
    )
    all_samples, n_hf, n_synth, n_adv, n_latent = result

    train, val = train_val_split(all_samples)
    write_jsonl(all_samples, f"data/apprentice_{DOMAIN}.jsonl")

    print(f"  [{DOMAIN}] {len(all_samples)} samples (HF: {n_hf}, synth: {n_synth}, "
          f"adversarial: {n_adv}, latent: {n_latent})")
    print(f"  → data/apprentice_{DOMAIN}.jsonl ({len(train)} train, {len(val)} val)")
    return {DOMAIN: {"all": len(all_samples), "hf": n_hf, "synth": n_synth, "adv": n_adv, "latent": n_latent}}


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
