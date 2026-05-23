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
from generate_scaled_synthetic import generate_planner
from monitor import print_hw

logger = logging.getLogger(__name__)
DOMAIN = "planner"
CFG = DOMAIN_CONFIGS[DOMAIN]


def format_agent_trove(raw):
    msgs = raw.get("messages", [])
    if len(msgs) < 3:
        return None
    messages = []
    for m in msgs:
        role = m.get("role", "")
        if role in ("user", "assistant", "system", "tool"):
            messages.append({"role": role, "content": m.get("content", "")})
    if len(messages) < 2:
        return None
    return {"messages": messages, "type": "agent_multi"}


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
    "open-thoughts/AgentTrove": format_agent_trove,
    "THUDM/AgentInstruct": format_agent_instruct,
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
            raw = download_hf_dataset(ds_name, ds_split, ds_filter, ds_max, config=ds_config, **ds_kwargs)
            converted = list(convert_to_apprentice(raw, DOMAIN, fmt))
            ds_elapsed = time.time() - ds_t0
            print(f"  [HF] {ds_name}: {len(converted)} samples in {ds_elapsed:.0f}s")
            all_hf.extend(converted)
        print_hw(f"{DOMAIN} hf")

    print(f"  [synth] Generating {CFG['synthetic_count']} synthetic samples...")
    result = combine(
        all_hf,
        generate_planner,
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
