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
from generate_scaled_synthetic import generate_code

logger = logging.getLogger(__name__)
DOMAIN = "code"
CFG = DOMAIN_CONFIGS[DOMAIN]


def format_the_stack(raw):
    content = raw.get("content", "")
    if len(content) < 50:
        return None
    return {
        "messages": [
            {"role": "user", "content": f"Write Python code for the following task:\n{content[:200]}..."},
            {"role": "assistant", "content": content},
        ],
        "type": "tool_single",
    }


def format_codealpaca(raw):
    instruction = raw.get("instruction", "")
    inp = raw.get("input", "")
    output = raw.get("output", "")
    if not output or not instruction:
        return None
    user_content = instruction
    if inp:
        user_content += "\n" + inp
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ],
        "type": "tool_single",
    }


FORMAT_FN_MAP = {
    "bigcode/the-stack": format_the_stack,
    "sahil2801/CodeAlpaca-20k": format_codealpaca,
}


def main(skip_hf=False):
    print(f"\n{'=' * 60}")
    print(f"[{DOMAIN}] Preparing dataset...")
    print(f"{'=' * 60}")

    all_hf = []
    if skip_hf:
        print("  Skipping HF downloads (--skip-hf)")
    else:
        for entry in CFG["hf_datasets"]:
            ds_name, ds_config, ds_split, ds_filter, ds_max, ds_kwargs = entry
            fmt = FORMAT_FN_MAP.get(ds_name)
            if fmt is None:
                logger.warning(f"No format_fn for {ds_name}, skipping")
                continue
            print(f"  [HF] Processing {ds_name} (config={ds_config}, split={ds_split}, max={ds_max})...")
            raw = download_hf_dataset(ds_name, ds_split, ds_filter, ds_max, config=ds_config, **ds_kwargs)
            converted = list(convert_to_apprentice(raw, DOMAIN, fmt))
            print(f"  [HF] {ds_name}: {len(converted)} samples")
            all_hf.extend(converted)

    print(f"  [synth] Generating {CFG['synthetic_count']} synthetic samples...")
    result = combine(
        all_hf,
        generate_code,
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
