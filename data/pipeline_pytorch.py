import json
import torch
from torch.utils.data import Dataset, DataLoader


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


class AgentDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=1024):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self._tokenized = None

    @classmethod
    def from_raw(cls, path, tokenizer, max_len=1024):
        samples = []
        paths = [path] if isinstance(path, str) else path
        for p in paths:
            with open(p) as f:
                for line in f:
                    samples.append(json.loads(line.strip()))
        ds = cls(samples, tokenizer, max_len)
        ds._tokenize_all()
        return ds

    def _tokenize_all(self):
        tokenized = []
        for sample in self.samples:
            messages = sample["messages"]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            ids = self.tokenizer.encode(text)
            labels = make_labels(ids, self.tokenizer)
            tokenized.append((ids, labels))
        self._tokenized = tokenized

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self._tokenized is not None:
            ids, labels = self._tokenized[idx]
        else:
            sample = self.samples[idx]
            messages = sample["messages"]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            ids = self.tokenizer.encode(text)
            labels = make_labels(ids, self.tokenizer)

        input_ids = ids[:self.max_len]
        labels = labels[:self.max_len]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def collate_fn(batch, pad_token_id=0):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def make_dataloader(dataset, batch_size, shuffle=True, max_len=None, indices=None):
    if indices is not None:
        indices = list(indices)
        subset = torch.utils.data.Subset(dataset, indices)
    else:
        subset = dataset

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=getattr(dataset.tokenizer, "pad_token_id", 0)),
    )


def get_seq_len_from_schedule(step: int, schedule: dict) -> int:
    current = 0
    for threshold, length in sorted(schedule.items()):
        if step >= threshold:
            current = length
    return current
