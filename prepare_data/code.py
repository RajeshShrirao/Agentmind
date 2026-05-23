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
from generate_scaled_synthetic import generate_code
from monitor import print_hw

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
    t_start = time.time()
    print(f"\n{'=' * 60}")
    print(f"[{DOMAIN}] Preparing dataset...")
    print(f"{'=' * 60}")
    print_hw(DOMAIN)

    all_hf = []
    if skip_hf:
        print("  [HF] Skipping HF downloads (--skip-hf)")
    else:
        for entry in CFG["hf_datasets"]:
            ds_name, ds_config, ds_split, ds_filter, ds_max, ds_kwargs = entry
            fmt = FORMAT_FN_MAP.get(ds_name)
            if fmt is None:
                logger.warning(f"No format_fn for {ds_name}, skipping")
                continue
            ds_t0 = time.time()
            print(f"  [HF] Processing {ds_name} (config={ds_config}, split={ds_split}, max={ds_max})...")
            raw = download_hf_dataset(ds_name, ds_split, ds_filter, ds_max, config=ds_config, domain=DOMAIN, **ds_kwargs)
            converted = list(convert_to_apprentice(raw, DOMAIN, fmt))
            ds_elapsed = time.time() - ds_t0
            print(f"  [HF] {ds_name}: {len(converted)} samples in {ds_elapsed:.0f}s")
            all_hf.extend(converted)
        print_hw(f"{DOMAIN} hf")

    print(f"  [synth] Generating {CFG['synthetic_count']} synthetic samples...")
    result = combine(
        all_hf,
        generate_code,
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
