import json
import random
import mlx.core as mx
import numpy as np
from pathlib import Path
from typing import Iterator

class AgentDataset:
    def __init__(self, paths: list[str], tokenizer=None, cfg=None, split="train", pretokenized: bool = False):
        self.cfg = cfg
        self.tok = tokenizer
        self.samples = []
        self.ids_array = None
        self.labels_array = None
        self.latent_stage = 1
        self._cache = {}

        if pretokenized:
            # Load pre-tokenized .npz files
            ids_path = [p for p in paths if "ids" in p]
            labels_path = [p for p in paths if "labels" in p]
            if ids_path and labels_path:
                self.ids_array = np.load(ids_path[0])["arr_0"]
                self.labels_array = np.load(labels_path[0])["arr_0"]
                split_idx = int(len(self.ids_array) * 0.95)
                if split == "train":
                    self.ids_array = self.ids_array[:split_idx]
                    self.labels_array = self.labels_array[:split_idx]
                else:
                    self.ids_array = self.ids_array[split_idx:]
                    self.labels_array = self.labels_array[split_idx:]
        else:
            # Load raw JSONL files
            for path in paths:
                with open(path) as f:
                    for line in f:
                        self.samples.append(json.loads(line.strip()))

            random.shuffle(self.samples)
            split_idx = int(len(self.samples) * 0.95)
            if split == "train":
                self.samples = self.samples[:split_idx]
            else:
                self.samples = self.samples[split_idx:]

        # Data mixing weights by type
        self.weights = {
            "instruction": 0.30,
            "tool_single":  0.30,
            "agent_multi":  0.25,
            "recovery":     0.15,
        }

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
        in_assistant = False
        for i, tok_id in enumerate(ids):
            if tok_id == self.cfg.assistant_id:
                in_assistant = True
            if in_assistant:
                labels[i] = tok_id
            if tok_id == self.cfg.eos_id or tok_id == self.cfg.user_id or tok_id == self.cfg.system_id:
                in_assistant = False
        return labels

    def __len__(self):
        if self.ids_array is not None:
            return len(self.ids_array)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.ids_array is not None:
            ids = self.ids_array[idx].tolist()
            labels = self.labels_array[idx].tolist()
            return ids, labels

        cache_key = (idx, self.latent_stage)
        if cache_key in self._cache:
            return self._cache[cache_key]

        import copy
        from model.latent import inject_latent_tokens

        sample = copy.deepcopy(self.samples[idx])
        sample = inject_latent_tokens(sample, self.tok, self.latent_stage)
        text = self._format_sample(sample)
        ids = self._tokenize(text)

        # Truncate to max_seq_len
        ids = ids[:self.cfg.max_seq_len]
        labels = self._make_labels(ids, sample)[:self.cfg.max_seq_len]

        self._cache[cache_key] = (ids, labels)
        return ids, labels

def collate_batch(samples: list, pad_id: int = 0, max_len: int = 2048) -> tuple:
    """Pad a list of (ids, labels) to fixed max_len for consistent gradient shapes."""
    ids_list, labels_list = zip(*samples)

    ids_padded    = [x[:max_len] + [pad_id]  * (max_len - min(len(x), max_len)) for x in ids_list]
    labels_padded = [x[:max_len] + [-100]    * (max_len - min(len(x), max_len)) for x in labels_list]

    return (
        mx.array(ids_padded),
        mx.array(labels_padded)
    )

def make_dataloader(dataset: AgentDataset, batch_size: int, shuffle: bool = True, max_len: int = 2048, indices: list = None) -> Iterator:
    if indices is None:
        indices = list(range(len(dataset)))
    else:
        indices = list(indices)  # copy to avoid mutating caller's list
    if shuffle:
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch = [dataset[i] for i in batch_idx]
        yield collate_batch(batch, max_len=max_len)
