"""
data/pipeline.py — Data loading and batching for Qwen2.5-based AgentMind.

The backbone's tokenizer (Qwen's AutoTokenizer) handles formatting via
apply_chat_template(). Special tokens (<|tool_call|>, <|observe|>, etc.)
are added as user_defined_symbols and are preserved by the BPE tokenizer.
"""

import json, random
import mlx.core as mx
import numpy as np
from typing import Iterator


def make_labels(ids: list[int], tokenizer) -> list[int]:
    labels = [-100] * len(ids)
    in_assistant = False
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_id = tokenizer.convert_tokens_to_ids("assistant")

    for i, tok_id in enumerate(ids):
        if tok_id == im_start_id and i + 1 < len(ids) and ids[i + 1] == assistant_id:
            in_assistant = True
        if in_assistant:
            labels[i] = tok_id
        if tok_id == im_end_id:
            in_assistant = False
    return labels


class AgentDataset:
    def __init__(self, samples=None, tokenizer=None, npz_path=None):
        self.samples = samples or []
        self.tokenizer = tokenizer
        self._cache = {}
        self._tokenized_samples = None
        self._np_ids = None
        self._np_labels = None
        if npz_path:
            self._load_npz(npz_path)

    def _load_npz(self, path):
        data = np.load(path)
        self._np_ids = data["ids"]
        self._np_labels = data["labels"]
        print(f"  Loaded {len(self._np_ids)} pre-tokenized samples from {path}")

    @classmethod
    def from_raw(cls, path, tokenizer, pretokenize: bool = True):
        """Load samples from a JSONL file."""
        samples = []
        if isinstance(path, str):
            paths = [path]
        else:
            paths = path
        for p in paths:
            with open(p) as f:
                for line in f:
                    samples.append(json.loads(line.strip()))
        ds = cls(samples=samples, tokenizer=tokenizer)
        return ds.pretokenize() if pretokenize else ds

    def pretokenize(self):
        """Materialize tokenized samples once so training does not tokenize lazily."""
        if self._tokenized_samples is not None:
            return self
        if self.tokenizer is None:
            raise ValueError("Cannot pretokenize without a tokenizer")

        tokenized = []
        for sample in self.samples:
            if isinstance(sample, dict) and "ids" in sample and "labels" in sample:
                ids = sample["ids"]
                labels = sample["labels"]
            else:
                messages = sample["messages"]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                ids = self.tokenizer.encode(text)
                labels = make_labels(ids, self.tokenizer)
            tokenized.append((ids, labels))

        self._tokenized_samples = tokenized
        self._cache = {i: sample for i, sample in enumerate(tokenized)}
        return self

    def tokenize_to_npz(self, max_len=1024, output_path=None):
        """Tokenize all samples to fixed-length numpy arrays (padded/truncated to max_len)."""
        if self._tokenized_samples is None:
            self.pretokenize()
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        if pad_id is None:
            pad_id = 0
        ids_list = []
        labels_list = []
        for ids, labels in self._tokenized_samples:
            ids = ids[:max_len] + [pad_id] * max(0, max_len - len(ids))
            labels = labels[:max_len] + [-100] * max(0, max_len - len(labels))
            ids_list.append(ids)
            labels_list.append(labels)
        self._np_ids = np.array(ids_list, dtype=np.int32)
        self._np_labels = np.array(labels_list, dtype=np.int32)
        if output_path:
            np.savez(output_path, ids=self._np_ids, labels=self._np_labels)
            print(f"  Saved {len(self._np_ids)} pre-tokenized samples to {output_path}")
        return self

    def train_val_split(self, ratio=0.95, shuffle=True):
        indices = list(range(len(self.samples)))
        if shuffle:
            random.shuffle(indices)
        split_idx = int(len(indices) * ratio)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        train = AgentDataset(
            samples=[self.samples[i] for i in train_idx],
            tokenizer=self.tokenizer,
        )
        val = AgentDataset(
            samples=[self.samples[i] for i in val_idx],
            tokenizer=self.tokenizer,
        )
        if self._tokenized_samples is not None:
            train._tokenized_samples = [self._tokenized_samples[i] for i in train_idx]
            val._tokenized_samples = [self._tokenized_samples[i] for i in val_idx]
            train._cache = {i: sample for i, sample in enumerate(train._tokenized_samples)}
            val._cache = {i: sample for i, sample in enumerate(val._tokenized_samples)}
        return train, val

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self._np_ids is not None:
            return self._np_ids[idx], self._np_labels[idx]
        if self._tokenized_samples is not None:
            return self._tokenized_samples[idx]
        if idx in self._cache:
            return self._cache[idx]

        sample = self.samples[idx]
        messages = sample["messages"]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize
        ids = self.tokenizer.encode(text)
        labels = make_labels(ids, self.tokenizer)

        result = (ids, labels)
        self._cache[idx] = result
        return result

    def clear_cache(self):
        self._cache.clear()


def collate_batch(samples: list, pad_id: int = 0, max_len: int = 2048) -> tuple:
    ids_list, labels_list = zip(*samples)

    # Fast path: all arrays are pre-padded numpy, just need slicing and stacking
    if isinstance(ids_list[0], np.ndarray):
        n = len(ids_list)
        ids_arr = np.empty((n, max_len), dtype=np.int32)
        labels_arr = np.empty((n, max_len), dtype=np.int32)
        for i in range(n):
            ids_arr[i] = ids_list[i][:max_len]
            labels_arr[i] = labels_list[i][:max_len]
        return mx.array(ids_arr), mx.array(labels_arr)

    # Fallback: variable-length Python lists, pad dynamically
    ids_padded = [x[:max_len] + [pad_id] * (max_len - min(len(x), max_len)) for x in ids_list]
    labels_padded = [x[:max_len] + [-100] * (max_len - min(len(x), max_len)) for x in labels_list]
    return mx.array(ids_padded), mx.array(labels_padded)


def make_dataloader(dataset: AgentDataset, batch_size: int, shuffle: bool = True,
                    max_len: int = 2048, indices: list = None) -> Iterator:
    if indices is None:
        indices = list(range(len(dataset)))
    else:
        indices = list(indices)
    if shuffle:
        random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch = [dataset[i] for i in batch_idx]
        yield collate_batch(batch, max_len=max_len)
