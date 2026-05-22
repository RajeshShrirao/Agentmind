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
from generate_scaled_synthetic import generate_research

logger = logging.getLogger(__name__)
DOMAIN = "research"
CFG = DOMAIN_CONFIGS[DOMAIN]


def format_fineweb(raw):
    text = raw.get("text", "")
    if len(text) < 100:
        return None
    snippet = text[:500]
    return {
        "messages": [
            {"role": "user", "content": f"Research and summarize the following:\n{snippet}..."},
            {"role": "assistant", "content": f"Here is my summary of the provided text:\n{snippet[:300]}"},
        ],
        "type": "agent_multi",
    }


def format_ultrachat(raw):
    msgs = raw.get("messages", [])
    if len(msgs) < 2:
        return None
    filtered = []
    for m in msgs:
        role = m.get("role", "")
        if role in ("user", "assistant"):
            filtered.append({"role": role, "content": m.get("content", "")})
    if len(filtered) < 2:
        return None
    return {"messages": filtered, "type": "agent_multi"}


FORMAT_FN_MAP = {
    "HuggingFaceFW/fineweb": format_fineweb,
    "HuggingFaceH4/ultrachat_200k": format_ultrachat,
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
            raw = download_hf_dataset(ds_name, ds_split, ds_filter, ds_max, config=ds_config, **ds_kwargs)
            converted = list(convert_to_apprentice(raw, DOMAIN, fmt))
            print(f"  HF {ds_name}: {len(converted)} samples")
            all_hf.extend(converted)

    result = combine(
        all_hf,
        generate_research,
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
