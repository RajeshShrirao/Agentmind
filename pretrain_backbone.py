import sys, time, json, os, math, random, argparse
from pathlib import Path
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

sys.path.insert(0, ".")

from config import AgentMindConfig
from model.agent_lm import AgentMind
from data.pipeline import AgentDataset, make_dataloader
from tokenizer_setup import load_tokenizer, hydrate_config
from init import init_agentmind
from scheduler import CosineWarmupScheduler
from training_utils import (
    cross_entropy_loss, clip_gradients, check_finite, clone_tree,
    get_seq_len_from_schedule, format_sample, make_labels,
    GradientAccumulator, NaNRecovery,
)
from monitor import ResourceScheduler

DATA_FILES = [
    "data/apprentice_tool_caller.jsonl",
    "data/apprentice_planner.jsonl",
    "data/apprentice_recovery.jsonl",
    "data/apprentice_code.jsonl",
    "data/apprentice_research.jsonl",
]

SEQ_LEN_SCHEDULE = {0: 128, 500: 256, 1500: 384, 2500: 512}
GRAD_ACCUM = 8
NPZ_PREFIX = "data/backbone_pretrain"


def pretokenize_to_npz(data_files: list[str], cfg, tok, max_len: int):
    all_samples = []
    for path in data_files:
        p = Path(path)
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                all_samples.append(json.loads(line.strip()))
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * 0.95)
    train_s = all_samples[:split_idx]
    val_s = all_samples[split_idx:]

    os.makedirs("data", exist_ok=True)
    for split_name, samples in [("train", train_s), ("val", val_s)]:
        ids_list = []
        labels_list = []
        for s in samples:
            text = format_sample(s)
            ids = tok.encode(text, add_bos=True)
            ids = ids[:max_len]
            labels = make_labels(ids, cfg)[:max_len]
            ids = ids + [0] * (max_len - len(ids))
            labels = labels + [-100] * (max_len - len(labels))
            ids_list.append(ids)
            labels_list.append(labels)
        np.savez(f"{NPZ_PREFIX}_{split_name}_ids.npz", np.array(ids_list, dtype=np.int32))
        np.savez(f"{NPZ_PREFIX}_{split_name}_labels.npz", np.array(labels_list, dtype=np.int32))
        print(f"  {split_name}: {len(ids_list)} samples, {max_len} seq_len")


def main():
    parser = argparse.ArgumentParser(description="Pretrain AgentMind backbone")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = AgentMindConfig()
    tok = load_tokenizer("agentmind_tok.model")
    hydrate_config(cfg, tok)

    model = AgentMind(cfg)
    model = init_agentmind(model, cfg)
    model.train()

    train_ids_path = f"{NPZ_PREFIX}_train_ids.npz"
    train_labels_path = f"{NPZ_PREFIX}_train_labels.npz"
    if not os.path.exists(train_ids_path):
        print("Pre-tokenizing all domain data (one-time cost)...")
        pretokenize_to_npz(DATA_FILES, cfg, tok, args.seq_len)
    print("Loading pre-tokenized datasets...")
    train_ds = AgentDataset.from_pretokenized(train_ids_path, train_labels_path)
    val_ds = AgentDataset.from_pretokenized(f"{NPZ_PREFIX}_val_ids.npz", f"{NPZ_PREFIX}_val_labels.npz")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")

    all_params = model.trainable_parameters()
    n_params = sum(p.size for _, p in tree_flatten(all_params))
    print(f"Trainable params: {n_params:,} ({n_params/1e6:.1f}M)")

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)
    scheduler = CosineWarmupScheduler(optimizer, args.lr, args.warmup, args.steps)

    accum = GradientAccumulator(GRAD_ACCUM)
    recovery = NaNRecovery(model, optimizer)
    rs = ResourceScheduler(decrease_checks=3, increase_checks=15, min_seq_len=64)
    rs.record_baseline()
    step = 0
    t_start = time.time()
    nan_count = 0
    nan_logged = False
    current_seq_len = rs.get_max_seq_len(get_seq_len_from_schedule(0, SEQ_LEN_SCHEDULE))

    print(f"\n{'='*60}")
    print(f"  Pretraining backbone: {args.steps} steps")
    print(f"  Seq len schedule: {SEQ_LEN_SCHEDULE}")
    print(f"  Grad accum: {GRAD_ACCUM}")
    print(f"{'='*60}")

    while step < args.steps:
        target_seq_len = get_seq_len_from_schedule(step, SEQ_LEN_SCHEDULE)
        new_seq_len = rs.get_max_seq_len(target_seq_len)
        if new_seq_len != current_seq_len:
            current_seq_len = new_seq_len
            print(f"  ── Seq len {current_seq_len} at step {step}")

        subset_size = min(10000, len(train_ds))
        epoch_indices = random.sample(range(len(train_ds)), subset_size)
        loader = make_dataloader(train_ds, batch_size=1, max_len=current_seq_len, indices=epoch_indices, shuffle=True)

        for input_ids, targets in loader:
            if step >= args.steps:
                break

            def loss_fn(params):
                model.update(params)
                logits, _ = model(input_ids, return_mtp=False)
                return cross_entropy_loss(logits, targets)

            loss, grads = mx.value_and_grad(loss_fn)(all_params)
            loss_val = loss.item()

            if not math.isfinite(loss_val):
                nan_count += 1
                if not nan_logged:
                    print(f"  NaN at step {step}! Rolling back...")
                    nan_logged = True
                grads = tree_map(mx.zeros_like, grads)
                accum.reset()
                model.update(clone_tree(recovery.backup_params))
                optimizer.state = clone_tree(recovery.backup_opt_state)
                mx.eval(model.parameters(), optimizer.state)
                continue

            grads, grad_norm = clip_gradients(grads, 1.0)
            accum.add(grads, loss_val)

            if accum.is_ready:
                avg_grads, avg_loss = accum.step()
                avg_grads, grad_norm = clip_gradients(avg_grads, 1.0)

                accum_finite, _ = check_finite(avg_grads)
                if not accum_finite:
                    nan_count += 1
                    if not nan_logged:
                        print(f"  NaN in accumulated grads at step {step}! Rolling back...")
                        nan_logged = True
                    accum.reset()
                    model.update(clone_tree(recovery.backup_params))
                    optimizer.state = clone_tree(recovery.backup_opt_state)
                    mx.eval(model.parameters(), optimizer.state)
                    continue

                optimizer.update(all_params, avg_grads)
                mx.eval(list(all_params.values()), optimizer.state)
                recovery.commit(model, optimizer)
                accum.reset()
                nan_logged = False

            lr_now = scheduler.step()

            if step % args.log_every == 0 or step == args.steps - 1:
                elapsed = time.time() - t_start
                tok_per_s = (input_ids.shape[1] * GRAD_ACCUM) / max(elapsed / max(1, step + 1), 1e-8)
                print(f"  step {step:4d}/{args.steps} loss {loss_val:.4f} lr {lr_now:.2e} grad_norm {grad_norm:.3f} {tok_per_s:.0f} tok/s")

            if step > 0 and step % args.save_every == 0:
                ckpt_dir = save_dir / f"step_{step:05d}"
                ckpt_dir.mkdir(exist_ok=True)
                flat = dict(tree_flatten(model.parameters()))
                for skip in ["last_hidden", "last_mtp_logits"]:
                    flat.pop(skip, None)
                mx.savez(str(ckpt_dir / "weights.npz"), **flat)
                json.dump({"step": step, "loss": loss_val, "lr": lr_now}, open(ckpt_dir / "log.json", "w"))

            step += 1

    out_path = save_dir / "backbone"
    flat = dict(tree_flatten(model.parameters()))
    for skip in ["last_hidden", "last_mtp_logits"]:
        flat.pop(skip, None)
    mx.savez(str(out_path), **flat)
    written = out_path.parent / (out_path.name + ".npz")
    elapsed = time.time() - t_start
    size_mb = os.path.getsize(written) / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"  Pretrain complete: {args.steps} steps in {elapsed:.0f}s")
    print(f"  Saved backbone ({len(flat)} keys, {size_mb:.0f}MB) -> {written}")
    print(f"  NaN steps: {nan_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
