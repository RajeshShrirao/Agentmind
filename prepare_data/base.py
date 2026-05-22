import os
import json
import random
import logging
import time
import os as _os

from datasets import load_dataset

logger = logging.getLogger(__name__)

HF_TOKEN = _os.environ.get("HF_TOKEN", None)
# Short timeout so hung downloads fail fast
_os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "15")


class _HFDownloadError(Exception):
    pass


def _safe_iter_dataset(ds, name, max_samples=None, filter_fn=None):
    """Iterate over a streaming dataset with error handling per-batch."""
    count = 0
    errors = 0
    try:
        iterator = iter(ds)
    except Exception as e:
        logger.warning(f"  Cannot iterate {name}: {e}")
        return

    while True:
        try:
            sample = next(iterator)
        except StopIteration:
            return
        except Exception as e:
            errors += 1
            if errors > 3:
                logger.warning(f"  Too many errors reading {name}, giving up")
                return
            logger.warning(f"  Error reading {name} (attempt {errors}): {e}")
            time.sleep(1.0)
            continue
        if max_samples is not None and count >= max_samples:
            return
        if filter_fn is not None and not filter_fn(sample):
            continue
        count += 1
        yield sample


def download_hf_dataset(name, split, filter_fn=None, max_samples=None, config=None, **kwargs):
    """Download a HF dataset with streaming=True.

    Yields raw samples, optionally filtered. Handles errors gracefully.
    If the dataset is unavailable, logs a warning and yields nothing.
    config: optional sub-config name for datasets with multiple configs.
    """
    try:
        load_kwargs = {
            "path": name,
            "split": split,
            "streaming": True,
            "token": HF_TOKEN,
        }
        if config is not None:
            load_kwargs["name"] = config
        load_kwargs.update(kwargs)
        ds = load_dataset(**load_kwargs)
    except Exception as e:
        logger.warning(f"Cannot load {name} ({split}): {e}")
        return

    yield from _safe_iter_dataset(ds, name, max_samples, filter_fn)


def convert_to_apprentice(raw_samples, domain, format_fn):
    """Convert raw HF samples to apprentice-format dicts.

    format_fn: (raw_sample) -> {"messages": [...], "type": str} or None (skip)
    Yields apprentice-format dicts.
    """
    if raw_samples is None:
        return
    for raw in raw_samples:
        try:
            result = format_fn(raw)
            if result is None:
                continue
            result["domain"] = domain
            yield result
        except Exception as e:
            logger.debug(f"Format conversion skipped: {e}")
            continue


def _inject_adversarial_noise(messages, rate):
    """Apply adversarial noise to HF-derived samples at the given rate."""
    if random.random() >= rate:
        return messages

    if len(messages) < 2:
        return messages

    assistant = messages[-1]
    content = assistant.get("content", "")

    noise = random.choice([
        f"<|scratch|>Unexpected response. Let me re-check.<|tool_call|>{{\"name\": \"web_search\", \"args\": {{\"query\": \"verify result\", \"max_results\": 5}}}}<|observe|>{{\"results\": []}}\n",
        f"<|scratch|>This doesn't look right. Trying again.<|tool_call|>{{\"name\": \"read_file\", \"args\": {{\"path\": \"/tmp/debug.log\"}}}}<|observe|>{{\"content\": \"debug info\"}}\n",
        f"<|scratch|>Error detected. Adjusting plan.<|tool_call|>{{\"name\": \"run_python\", \"args\": {{\"code\": \"print('retry')\"}}}}<|observe|>{{\"stdout\": \"retry\\n\"}}\n",
    ])
    messages[-1]["content"] = content + "\n" + noise
    return messages


def _count_latent(samples):
    return sum(1 for s in samples if "<|think_start|>" in json.dumps(s.get("messages", "")))


def combine(hf_samples, synthetic_fn, n_synthetic, adversarial_rate, domain="", latent_rate=0.5):
    """Merge HF apprentice samples with synthetic generation.

    hf_samples: iterable of apprentice-format dicts from HF
    synthetic_fn: callable(adversarial_rate, latent_rate) -> sample dict
    n_synthetic: how many synthetic samples to fill
    adversarial_rate: passed through to synthetic_fn; also applied to HF subset

    Returns (all_samples, n_hf, n_synth, n_adversarial, n_latent).
    """
    try:
        hf_list = list(hf_samples) if hf_samples is not None else []
    except Exception as e:
        logger.warning(f"  Error collecting HF samples: {e}")
        hf_list = []
    n_hf_raw = len(hf_list)

    hf_adversarial = 0
    for i in range(len(hf_list)):
        old = hf_list[i].get("messages", [{}])[-1].get("content", "")
        hf_list[i]["messages"] = _inject_adversarial_noise(
            hf_list[i].get("messages", []), adversarial_rate * 0.5
        )
        new = hf_list[i].get("messages", [{}])[-1].get("content", "")
        if old != new:
            hf_adversarial += 1

    synth_list = []
    synth_adversarial = 0
    for _ in range(n_synthetic):
        sample = synthetic_fn(adversarial_rate=adversarial_rate, latent_rate=latent_rate)
        synth_list.append(sample)
        content = json.dumps(sample.get("messages", ""))
        if "<|scratch|>" in content:
            synth_adversarial += 1

    all_samples = hf_list + synth_list
    random.shuffle(all_samples)

    n_latent = _count_latent(all_samples)
    n_adversarial = hf_adversarial + synth_adversarial

    return all_samples, len(hf_list), len(synth_list), n_adversarial, n_latent


def train_val_split(samples, val_frac=0.05):
    """Split into train and validation lists."""
    shuffled = list(samples)
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


def write_jsonl(samples, path):
    """Write a list of dicts as JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
