import json
import random
import mlx.core as mx
from pathlib import Path
from typing import Iterator

class AgentDataset:
    def __init__(self, paths: list[str], tokenizer, cfg, split="train"):
        self.cfg = cfg
        self.tok = tokenizer
        self.samples = []

        for path in paths:
            with open(path) as f:
                for line in f:
                    self.samples.append(json.loads(line.strip()))

        # Data mixing weights by type
        self.weights = {
            "instruction": 0.30,
            "tool_single":  0.30,
            "agent_multi":  0.25,
            "recovery":     0.15,
        }

        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * 0.95)
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def _format_sample(self, sample: dict) -> str:
        """Convert message list to flat token string."""
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

    def _tokenize(self, text: str) -> list[int]:
        return self.tok.encode(text, add_bos=True)

    def _make_labels(self, ids: list[int], sample: dict) -> list[int]:
        """
        Only compute loss on assistant turns.
        Mask system + user tokens with -100 (ignored in loss).
        """
        labels = [-100] * len(ids)
        text = self._format_sample(sample)
        # Find assistant turn boundaries and unmask them
        # Simple heuristic: unmask everything after <|assistant|>
        assistant_id = self.cfg.tool_call_id - 1  # adjust to your token IDs
        in_assistant = False
        for i, tok_id in enumerate(ids):
            if tok_id == self.cfg.tool_call_id - 1:  # <|assistant|>
                in_assistant = True
            if in_assistant:
                labels[i] = tok_id
        return labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = self._format_sample(sample)
        ids = self._tokenize(text)

        # Truncate to max_seq_len
        ids = ids[:self.cfg.max_seq_len]
        labels = self._make_labels(ids, sample)[:self.cfg.max_seq_len]

        return ids, labels

def collate_batch(samples: list, pad_id: int = 0) -> tuple:
    """Pad a list of (ids, labels) to same length."""
    ids_list, labels_list = zip(*samples)
    max_len = max(len(x) for x in ids_list)

    ids_padded    = [x + [pad_id]  * (max_len - len(x)) for x in ids_list]
    labels_padded = [x + [-100]    * (max_len - len(x)) for x in labels_list]

    return (
        mx.array(ids_padded),
        mx.array(labels_padded)
    )

def make_dataloader(dataset: AgentDataset, batch_size: int, shuffle: bool = True) -> Iterator:
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch = [dataset[i] for i in batch_idx]
        yield collate_batch(batch)
