"""
Pre-tokenize dataset for faster training.
Converts raw JSONL to tokenized .npz arrays.

Usage: python pretokenize.py
Output: data/train_ids.npz, data/train_labels.npz, data/val_ids.npz, data/val_labels.npz
"""

import json
import os
import numpy as np
from tokenizer_setup import load_tokenizer
from config import AgentMindConfig

cfg = AgentMindConfig()
tok = load_tokenizer("agentmind_tok.model")

def format_sample(sample: dict) -> str:
    text = ""
    for msg in sample["messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<|system|>{content}"
        elif role == "user":
            text += f"<|user|>{content}"
        elif role == "assistant":
            text += f"<|assistant|>{content}<eos>"
    return text

def make_labels(ids: list[int], sample: dict) -> list[int]:
    labels = [-100] * len(ids)
    in_assistant = False
    for i, tok_id in enumerate(ids):
        if tok_id == cfg.assistant_id:
            in_assistant = True
        if in_assistant:
            labels[i] = tok_id
        if tok_id == cfg.eos_id or tok_id == cfg.user_id or tok_id == cfg.system_id:
            in_assistant = False
    return labels

def pretokenize(data_files: list[str], max_len: int = 2048, split: str = "train"):
    samples = []
    for path in data_files:
        with open(path) as f:
            for line in f:
                samples.append(json.loads(line.strip()))

    if split == "train":
        split_idx = int(len(samples) * 0.95)
        samples = samples[:split_idx]
    else:
        split_idx = int(len(samples) * 0.95)
        samples = samples[split_idx:]

    all_ids = []
    all_labels = []

    for sample in samples:
        text = format_sample(sample)
        ids = tok.encode(text, add_bos=True)
        ids = ids[:max_len]
        labels = make_labels(ids, sample)[:max_len]

        # Pad to max_len
        ids = ids + [0] * (max_len - len(ids))
        labels = labels + [-100] * (max_len - len(labels))

        all_ids.append(ids)
        all_labels.append(labels)

    return np.array(all_ids, dtype=np.int32), np.array(all_labels, dtype=np.int32)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    data_files = ["data/scaled_synthetic.jsonl"]
    if os.path.exists("data/instructions.jsonl"):
        data_files.append("data/instructions.jsonl")
    if os.path.exists("data/synthetic_agents.jsonl"):
        data_files.append("data/synthetic_agents.jsonl")

    print(f"Pre-tokenizing {len(data_files)} files...")

    train_ids, train_labels = pretokenize(data_files, split="train")
    val_ids, val_labels = pretokenize(data_files, split="val")

    np.savez("data/train_ids.npz", train_ids)
    np.savez("data/train_labels.npz", train_labels)
    np.savez("data/val_ids.npz", val_ids)
    np.savez("data/val_labels.npz", val_labels)

    print(f"Train: {train_ids.shape[0]} samples, {train_ids.shape[1]} seq_len")
    print(f"Val:   {val_ids.shape[0]} samples, {val_ids.shape[1]} seq_len")
    print("Saved to data/*.npz")
