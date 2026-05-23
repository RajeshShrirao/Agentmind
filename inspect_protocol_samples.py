"""
inspect_protocol_samples.py — Inspect training data for protocol behavior.

Randomly samples apprentice training data and prints:
  - Exact serialized assistant outputs
  - Whether <|tool_call|> occurs early or late
  - Ratio of prose tokens before tool call
  - Whether dataset teaches narration-before-action or immediate protocol execution

Usage:
  python inspect_protocol_samples.py
  python inspect_protocol_samples.py --domain tool_caller
  python inspect_protocol_samples.py --domain recovery --n 20
"""
import json, random, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import analyze_protocol, detect_prose_contamination

DOMAINS = ["tool_caller", "planner", "recovery", "code", "research"]


def load_samples(domain: str, max_samples: int = None):
    path = f"data/apprentice_{domain}.jsonl"
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return []
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    if max_samples and len(samples) > max_samples:
        samples = random.sample(samples, max_samples)
    return samples


def inspect_sample(sample: dict, idx: int):
    domain = sample.get("domain", "?")
    stype = sample.get("type", "?")
    messages = sample.get("messages", [])

    # Reconstruct full text
    full_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            full_text += f"<|system|>{content}"
        elif role == "user":
            full_text += f"<|user|>{content}"
        elif role == "assistant":
            full_text += f"<|assistant|>{content}<eos>"

    # Extract assistant portions
    assistant_parts = []
    in_asst = False
    buf = ""
    for msg in messages:
        if msg["role"] == "assistant":
            assistant_parts.append(msg["content"])

    print("=" * 65)
    print(f"Sample {idx}  |  domain={domain}  type={stype}")
    print(f"Messages: {len(messages)}  |  "
          f"System: {sum(1 for m in messages if m['role']=='system')}  "
          f"User: {sum(1 for m in messages if m['role']=='user')}  "
          f"Assistant: {sum(1 for m in messages if m['role']=='assistant')}")
    print("-" * 65)

    for i, part in enumerate(assistant_parts):
        proto = analyze_protocol(f"<|assistant|>{part}")
        prose = detect_prose_contamination(f"<|assistant|>{part}")
        has_tc = proto["contains_tool_call_token"]
        prose_before = proto["prose_before_tool_call"]
        tool_pos = proto["first_tool_call_position"]
        total_len = len(part)

        # Check: does the assistant output START with tool call or narration?
        starts_with_tool = part.lstrip().startswith("<|tool_call|>")
        starts_with_think = part.lstrip().startswith("<|think_start|>")

        print(f"  Assistant #{i}: {total_len} chars")
        print(f"    starts_with_tool_call={starts_with_tool}")
        print(f"    starts_with_think={starts_with_think}")
        print(f"    has_tool_call={has_tc}  pos={tool_pos}")
        print(f"    prose_before_tool_call={prose_before} tokens")
        print(f"    open_brace={proto['contains_open_brace']}  "
              f"close_brace={proto['contains_close_brace']}  "
              f"balanced={proto['balanced_braces']}")
        print(f"    name_key={proto['contains_name_key']}  "
              f"args_key={proto['contains_args_key']}")

        # Show first 200 chars of first assistant output
        preview = part[:300]
        print(f"    PREVIEW: {repr(preview)}")
        print()

    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inspect protocol behavior in training data")
    parser.add_argument("--domain", type=str, default=None,
                        help="Domain to inspect (default: all, random 2 each)")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of samples to inspect (default: 10)")
    args = parser.parse_args()

    if args.domain:
        domains = [args.domain]
    else:
        domains = DOMAINS

    for domain in domains:
        samples = load_samples(domain, max_samples=args.n)
        if not samples:
            print(f"[{domain}] No samples loaded.")
            continue
        print(f"\n{'#' * 65}")
        print(f"# Domain: {domain}  ({len(samples)} samples loaded)")
        print(f"{'#' * 65}")

        # Summary stats
        total_asst = 0
        with_tool_call = 0
        starts_with_tool = 0
        prose_before_list = []

        for i, sample in enumerate(samples):
            asst_parts = [m["content"] for m in sample["messages"] if m["role"] == "assistant"]
            for part in asst_parts:
                total_asst += 1
                text = f"<|assistant|>{part}"
                proto = analyze_protocol(text)
                if proto["contains_tool_call_token"]:
                    with_tool_call += 1
                    prose_before_list.append(proto["prose_before_tool_call"])
                    if part.lstrip().startswith("<|tool_call|>"):
                        starts_with_tool += 1

        n_inspected = min(len(samples), args.n)
        print(f"  Inspected: {n_inspected} samples, {total_asst} assistant turns")
        print(f"  With <|tool_call|>: {with_tool_call}/{total_asst} "
              f"({with_tool_call/max(total_asst,1)*100:.0f}%)")
        print(f"  Starts with tool call: {starts_with_tool}/{with_tool_call} "
              f"({starts_with_tool/max(with_tool_call,1)*100:.0f}%)")
        if prose_before_list:
            avg_prose = sum(prose_before_list) / len(prose_before_list)
            print(f"  Avg prose tokens before <|tool_call|>: {avg_prose:.1f}")
        print()

    # Full inspection of selected samples
    for domain in domains:
        samples = load_samples(domain, max_samples=max(100, args.n))
        if not samples:
            continue
        chosen = random.sample(samples, min(args.n, len(samples)))
        for i, sample in enumerate(chosen):
            inspect_sample(sample, i)


if __name__ == "__main__":
    main()
