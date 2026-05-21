"""
Build a curated training corpus for AgentMind tokenizer.
Mix of general text, code, and dialogue — optimized for agentic patterns.

Usage: python build_corpus.py
Output: data/corpus.txt (~50-100MB)
"""

from datasets import load_dataset
import os

os.environ["HF_TOKEN"] = "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"

os.makedirs("data", exist_ok=True)

output_path = "data/corpus.txt"
total_lines = 0

print("Building AgentMind training corpus...")

# 1. FineWeb — clean general text, reasoning, instruction following
print("[1/3] Downloading FineWeb (general text)...")
ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True, token=os.environ["HF_TOKEN"])
with open(output_path, "w") as f:
    for i, sample in enumerate(ds):
        if i > 20000:
            break
        text = sample.get("text", "").strip()
        if len(text) > 50:
            f.write(text + "\n")
            total_lines += 1
print(f"  → {total_lines} lines written")

# 2. The Stack (Python) — code structure, JSON, function patterns
print("[2/3] Downloading Python code (The Stack)...")
ds = load_dataset("bigcode/the-stack", data_dir="data/python", split="train", streaming=True, token=os.environ["HF_TOKEN"])
with open(output_path, "a") as f:
    for i, sample in enumerate(ds):
        if i > 10000:
            break
        content = sample.get("content", "").strip()
        if len(content) > 50:
            f.write(content + "\n")
            total_lines += 1
print(f"  → {total_lines} total lines")

# 3. UltraChat — multi-turn dialogue, system prompts
print("[3/3] Downloading UltraChat (dialogue)...")
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True, token=os.environ["HF_TOKEN"])
with open(output_path, "a") as f:
    for i, sample in enumerate(ds):
        if i > 10000:
            break
        messages = sample.get("messages", [])
        for msg in messages:
            content = msg.get("content", "").strip()
            if len(content) > 20:
                f.write(content + "\n")
                total_lines += 1
print(f"  → {total_lines} total lines")

# Report
import os
size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"\nCorpus built: {output_path}")
print(f"Size: {size_mb:.1f} MB | Lines: {total_lines}")
print("Next: python -c \"from tokenizer_setup import train_tokenizer; train_tokenizer('data/corpus.txt')\"")
