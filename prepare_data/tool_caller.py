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
from generate_scaled_synthetic import generate_tool_caller

logger = logging.getLogger(__name__)
DOMAIN = "tool_caller"
CFG = DOMAIN_CONFIGS[DOMAIN]


def format_hermes(raw):
    conv = raw.get("conversations", [])
    if len(conv) < 2:
        return None
    role_map = {"human": "user", "gpt": "assistant", "tool": "assistant", "system": "system",
                "function_call": "assistant", "function_response": "assistant"}
    messages = []
    for t in conv:
        role = role_map.get(t.get("from", ""), t.get("from", ""))
        messages.append({"role": role, "content": t.get("value", "")})
    if len(messages) < 2:
        return None
    return {"messages": messages, "type": "tool_single"}


def format_agent_instruct(raw):
    conv = raw.get("conversations", [])
    if len(conv) < 2:
        return None
    messages = [
        {"role": "user" if t.get("from") == "human" else "assistant", "content": t.get("value", "")}
        for t in conv
    ]
    return {"messages": messages, "type": "agent_multi"}


FORMAT_FN_MAP = {
    "lambda/hermes-agent-reasoning-traces": format_hermes,
    "THUDM/AgentInstruct": format_agent_instruct,
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
        generate_tool_caller,
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
