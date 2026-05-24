import copy
import json
import random
import mlx.core as mx
import numpy as np
from typing import Iterator

from training_utils import format_sample


class AgentDataset:
    def __init__(self, samples=None, cfg=None, tokenizer=None):
        self.samples = samples or []
        self.cfg = cfg
        self.tok = tokenizer
        self.ids_array = None
        self.labels_array = None
        self.latent_stage = 1
        self._cache = {}
        self.weights = {
            "instruction": 0.30,
            "tool_single": 0.30,
            "agent_multi": 0.25,
            "recovery": 0.15,
        }

    @classmethod
    def from_raw(cls, paths, tokenizer, cfg):
        samples = []
        for path in paths:
            with open(path) as f:
                for line in f:
                    samples.append(json.loads(line.strip()))
        return cls(samples=samples, cfg=cfg, tokenizer=tokenizer)

    @classmethod
    def from_pretokenized(cls, ids_path, labels_path):
        ds = cls()
        ds.ids_array = mx.array(np.load(ids_path)["arr_0"])
        ds.labels_array = mx.array(np.load(labels_path)["arr_0"])
        return ds

    def train_val_split(self, ratio=0.95, shuffle=True):
        if self.ids_array is not None:
            n = len(self.ids_array)
            split_idx = int(n * ratio)
            train = AgentDataset()
            train.ids_array = self.ids_array[:split_idx]
            train.labels_array = self.labels_array[:split_idx]
            train.cfg = self.cfg
            train.tok = self.tok
            val = AgentDataset()
            val.ids_array = self.ids_array[split_idx:]
            val.labels_array = self.labels_array[split_idx:]
            val.cfg = self.cfg
            val.tok = self.tok
            return train, val
        if shuffle:
            random.shuffle(self.samples)
        split_idx = int(len(self.samples) * ratio)
        train = AgentDataset(
            samples=self.samples[:split_idx], cfg=self.cfg, tokenizer=self.tok
        )
        val = AgentDataset(
            samples=self.samples[split_idx:], cfg=self.cfg, tokenizer=self.tok
        )
        return train, val

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

        from model.latent import inject_latent_tokens

        sample = copy.deepcopy(self.samples[idx])
        sample = inject_latent_tokens(sample, self.tok, self.latent_stage)
        text = format_sample(sample)
        ids = self.tok.encode(text, add_bos=True)

        ids = ids[:self.cfg.max_seq_len]
        labels = self._make_labels(ids, sample)[:self.cfg.max_seq_len]

        self._cache[cache_key] = (ids, labels)
        return ids, labels

    def _make_labels(self, ids: list[int], sample: dict) -> list[int]:
        from training_utils import make_labels
        return make_labels(ids, self.cfg)

    def clear_cache(self):
        self._cache.clear()


def collate_batch(samples: list, pad_id: int = 0, max_len: int = 2048) -> tuple:
    ids_list, labels_list = zip(*samples)
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
