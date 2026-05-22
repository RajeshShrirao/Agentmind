"""
Build a curated training corpus for AgentMind tokenizer training.

NOTE: The Cognitive Apprenticeship architecture starts from a randomly initialized
backbone. General text pre-training is NOT used for apprenticeship training. The
backbone learns language structure through specialist (LoRA) training and distillation.

This script is kept for ONE purpose: retraining the SentencePiece tokenizer if we
need to expand the vocabulary or change the token distribution.

Usage (only if retokenizing):
  python build_corpus.py
  # Output: data/corpus.txt (~100-200MB)
  # Then: python -c "from tokenizer_setup import train_tokenizer; train_tokenizer('data/corpus.txt')"

For apprentice training data, use:
  python generate_scaled_synthetic.py
"""

from datasets import load_dataset
import os
import json

os.environ["HF_TOKEN"] = "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"

os.makedirs("data", exist_ok=True)

output_path = "data/corpus.txt"
total_lines = 0

print("=" * 60)
print("AgentMind — Corpus Builder (for tokenizer retraining only)")
print("=" * 60)
print("NOTE: This corpus is NOT used for model training in the apprenticeship architecture.")
print("It is only needed if retraining the SentencePiece tokenizer.\n")

# 1. FineWeb — clean general text, reasoning, instruction following
print("[1/7] Downloading FineWeb (general text)...")
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
print("[2/7] Downloading Python code (The Stack)...")
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
print("[3/7] Downloading UltraChat (dialogue)...")
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

# 4. AgentInstruct — high-quality agent trajectories
print("[4/7] Downloading AgentInstruct (agent trajectories)...")
try:
    ds = load_dataset("THUDM/AgentInstruct", split="train", streaming=True, token=os.environ["HF_TOKEN"])
    with open(output_path, "a") as f:
        for i, sample in enumerate(ds):
            if i > 5000:
                break
            conversations = sample.get("conversations", [])
            for turn in conversations:
                content = turn.get("value", "").strip()
                if len(content) > 20:
                    f.write(content + "\n")
                    total_lines += 1
    print(f"  → {total_lines} total lines")
except Exception as e:
    print(f"  → Skipped: {e}")

# 5. ToolBench — tool calling patterns
print("[5/7] Downloading ToolBench (tool calling)...")
try:
    ds = load_dataset("ToolBench/ToolBench", split="train", streaming=True, token=os.environ["HF_TOKEN"])
    with open(output_path, "a") as f:
        for i, sample in enumerate(ds):
            if i > 3000:
                break
            for key in ["query", "answer", "response", "content"]:
                content = sample.get(key, "")
                if isinstance(content, str) and len(content.strip()) > 20:
                    f.write(content.strip() + "\n")
                    total_lines += 1
    print(f"  → {total_lines} total lines")
except Exception as e:
    print(f"  → Skipped: {e}")

# 6. WebArena — web navigation agent data
print("[6/7] Downloading WebArena (web navigation)...")
try:
    ds = load_dataset("osunlp/WebArena", split="train", streaming=True, token=os.environ["HF_TOKEN"])
    with open(output_path, "a") as f:
        for i, sample in enumerate(ds):
            if i > 3000:
                break
            for key in ["intent", "action", "observation", "response"]:
                content = sample.get(key, "")
                if isinstance(content, str) and len(content.strip()) > 10:
                    f.write(content.strip() + "\n")
                    total_lines += 1
    print(f"  → {total_lines} total lines")
except Exception as e:
    print(f"  → Skipped: {e}")

# 7. Append existing synthetic data (apprentice format)
print("[7/7] Appending apprentice synthetic data...")
import glob
apprentice_files = glob.glob("data/apprentice_*.jsonl")
if apprentice_files:
    with open(output_path, "a") as f:
        for path in apprentice_files:
            for line in open(path):
                obj = json.loads(line)
                for msg in obj.get("messages", []):
                    content = msg.get("content", "").strip()
                    if len(content) > 10:
                        f.write(content + "\n")
                        total_lines += 1
    print(f"  → {total_lines} total lines")
else:
    print("  → No apprentice data found, checking legacy synthetic...")
    synthetic_path = "data/synthetic_agents.jsonl"
    if os.path.exists(synthetic_path):
        with open(output_path, "a") as f:
            for line in open(synthetic_path):
                obj = json.loads(line)
                for msg in obj.get("messages", []):
                    content = msg.get("content", "").strip()
                    if len(content) > 10:
                        f.write(content + "\n")
                        total_lines += 1
        print(f"  → {total_lines} total lines")
    else:
        print("  → No synthetic data found, skipping")

# Report
size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"\nCorpus built: {output_path}")
print(f"Size: {size_mb:.1f} MB | Lines: {total_lines}")
print(f"\nThis corpus is for tokenizer retraining only.")
print(f"It is NOT used as training data in the apprenticeship architecture.")
print(f"To generate training data, run: python generate_scaled_synthetic.py")
