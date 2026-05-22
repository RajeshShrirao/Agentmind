import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import time, json, os, random, argparse
from pathlib import Path

from config import AgentMindConfig
from model.agent_lm import AgentMind
from model.mtp_head import mtp_loss
from data.pipeline import AgentDataset, make_dataloader
from lora import apply_lora
from scheduler import CosineWarmupScheduler
from init import init_agentmind

# ── CLI ────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--max-steps", type=int, default=None, help="Override total_steps (for testing)")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint dir to resume from")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────

cfg = AgentMindConfig()

TRAIN_CFG = dict(
    lr            = 2e-4,
    weight_decay  = 0.01,
    warmup_steps  = 100,
    total_steps   = 3000,
    grad_clip     = 1.0,
    batch_size    = 1,       # Reduced for 16GB Mac
    grad_accum    = 8,       # Maintain effective batch size
    seq_len       = 2048,
    lora_rank     = 16,
    lora_alpha    = 32.0,
    eval_every    = 500,
    save_every    = 200,
    save_dir      = "/Volumes/New Volume/checkpoints",
    use_mtp       = False,     # Disabled for memory (re-enable after step 500)
    mtp_weight    = 0.2,
    mtp_start     = 500,
    latent_stage  = 1,
    seq_len_schedule = {0: 256, 500: 512, 1500: 1024},
)

if args.max_steps is not None:
    TRAIN_CFG["total_steps"] = args.max_steps

# ── Loss ──────────────────────────────────────────────────

def cross_entropy_loss(logits, targets):
    B, L, V = logits.shape
    flat_logits  = logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)
    mask = (flat_targets != -100).astype(mx.float32)
    loss = nn.losses.cross_entropy(flat_logits, mx.maximum(flat_targets, 0), reduction='none')
    return (loss * mask).sum() / (mask.sum() + 1e-8)

# ── Gradient clipping ─────────────────────────────────────

def clip_gradients(grads, max_norm: float):
    leaves = [g for _, g in tree_flatten(grads)]
    norm = mx.sqrt(sum(mx.sum(g ** 2) for g in leaves))
    scale = mx.minimum(1.0, max_norm / (norm + 1e-6))
    return tree_map(lambda g: g * scale, grads), norm.item()

# ── Train step ────────────────────────────────────────────

def get_current_seq_len(step: int) -> int:
    """Get sequence length for current step based on curriculum schedule."""
    schedule = TRAIN_CFG["seq_len_schedule"]
    current_len = 0
    for threshold, length in sorted(schedule.items()):
        if step >= threshold:
            current_len = length
    return current_len

def make_train_step(model):
    def train_step(input_ids, targets, step):
        trainable = {k: v for k, v in model.trainable_parameters().items()
                     if not k.startswith("last_")}

        def loss_fn(params):
            model.update(params)
            use_mtp = TRAIN_CFG["use_mtp"] and step >= TRAIN_CFG["mtp_start"]
            logits, h_states = model(input_ids, return_mtp=use_mtp)
            main = cross_entropy_loss(logits, targets)

            if use_mtp and hasattr(model, "mtp"):
                aux = mtp_loss(model.last_mtp_logits, targets,
                               weight=TRAIN_CFG["mtp_weight"])
                return main + aux
            return main

        loss, grads = mx.value_and_grad(loss_fn)(trainable)
        return loss, grads

    return train_step

# ── Eval ───────────────────────────────────────────────────

from eval import evaluate as real_evaluate

def evaluate(model, val_ds, tok, cfg, max_len=None):
    """Wrapper that calls real eval from eval.py."""
    return real_evaluate(model, val_ds, tok, cfg, max_len=max_len or 512)

# ── Main training loop ────────────────────────────────────

def train():
    Path(TRAIN_CFG["save_dir"]).mkdir(exist_ok=True)

    # Model
    model = AgentMind(cfg)
    model = init_agentmind(model, cfg)
    model = apply_lora(model, rank=TRAIN_CFG["lora_rank"], alpha=TRAIN_CFG["lora_alpha"])

    # Optimizer + Scheduler
    optimizer = optim.AdamW(
        learning_rate=TRAIN_CFG["lr"],
        weight_decay=TRAIN_CFG["weight_decay"]
    )
    scheduler = CosineWarmupScheduler(
        optimizer,
        base_lr=TRAIN_CFG["lr"],
        warmup_steps=TRAIN_CFG["warmup_steps"],
        total_steps=TRAIN_CFG["total_steps"]
    )

    # Data
    from tokenizer_setup import load_tokenizer
    tok = load_tokenizer("agentmind_tok.model")

    # Use pre-tokenized data if available, otherwise use raw JSONL
    if os.path.exists("data/train_ids.npz") and os.path.exists("data/train_labels.npz"):
        train_ds = AgentDataset(
            ["data/train_ids.npz", "data/train_labels.npz"],
            cfg=cfg, split="train", pretokenized=True
        )
        val_ds = AgentDataset(
            ["data/val_ids.npz", "data/val_labels.npz"],
            cfg=cfg, split="val", pretokenized=True
        )
        print("Loaded pre-tokenized dataset")
    else:
        # Use scaled synthetic data + any instruction data if available
        data_files = ["data/scaled_synthetic.jsonl"]
        if os.path.exists("data/instructions.jsonl"):
            data_files.append("data/instructions.jsonl")
        if os.path.exists("data/synthetic_agents.jsonl"):
            data_files.append("data/synthetic_agents.jsonl")

        train_ds = AgentDataset(
            data_files,
            tokenizer=tok, cfg=cfg, split="train"
        )
        val_ds = AgentDataset(
            data_files,
            tokenizer=tok, cfg=cfg, split="val"
        )
        print(f"Loaded raw dataset from {len(data_files)} files")

    train_step_fn = make_train_step(model)
    step = 0
    accum_loss = 0.0
    accum_grad = None
    log = []

    model.train()
    print(f"Training AgentMind | {sum(p.size for _,p in tree_flatten(model.trainable_parameters())):,} trainable params")
    print(f"Sequence curriculum: {TRAIN_CFG['seq_len_schedule']}")
    print(f"Lazy MTP: enabled after step {TRAIN_CFG['mtp_start']}")

    t0 = time.time()  # Track time across entire grad_accum window
    current_seq_len = 0

    while step < TRAIN_CFG["total_steps"]:
        # Update sequence length based on curriculum
        new_seq_len = get_current_seq_len(step)
        if new_seq_len != current_seq_len:
            current_seq_len = new_seq_len
            print(f"  ── Sequence length changed to {current_seq_len} at step {step}")

        # Create dataloader with current sequence length
        loader = make_dataloader(train_ds, batch_size=TRAIN_CFG["batch_size"], max_len=current_seq_len)

        for input_ids, targets in loader:
            if step >= TRAIN_CFG["total_steps"]:
                break

            loss, grads = train_step_fn(input_ids, targets, step)
            mx.eval(loss, grads)

            loss_val = loss.item()
            if mx.isnan(loss).item():
                print(f"  ⚠️  NaN loss at step {step} — skipping batch")
                continue

            # Gradient accumulation
            grads, grad_norm = clip_gradients(grads, TRAIN_CFG["grad_clip"])
            accum_loss += loss_val

            if accum_grad is None:
                accum_grad = grads
            else:
                accum_grad = tree_map(lambda a, b: a + b, accum_grad, grads)

            if (step + 1) % TRAIN_CFG["grad_accum"] == 0:
                # Average accumulated gradients
                accum_grad = tree_map(lambda g: g / TRAIN_CFG["grad_accum"], accum_grad)
                # Only update actual trainable parameters (exclude last_hidden, last_mtp_logits)
                trainable = {k: v for k, v in model.trainable_parameters().items()
                             if not k.startswith("last_")}
                optimizer.update(trainable, accum_grad)
                mx.eval(model.parameters(), optimizer.state)
                lr = scheduler.step()
                accum_grad = None

                elapsed = time.time() - t0
                avg_loss = accum_loss / TRAIN_CFG["grad_accum"]
                accum_loss = 0.0
                tok_per_sec = (input_ids.shape[1] * TRAIN_CFG["grad_accum"]) / elapsed

                mtp_status = "ON" if step >= TRAIN_CFG["mtp_start"] else "OFF"
                print(f"step {step:4d} | loss {avg_loss:.4f} | lr {lr:.2e} | grad_norm {grad_norm:.3f} | {tok_per_sec:.0f} tok/s | seq {current_seq_len} | mtp {mtp_status}")
                log.append({"step": step, "loss": avg_loss, "lr": lr, "seq_len": current_seq_len})

                t0 = time.time()  # Reset timer for next window

            # Eval
            if step % TRAIN_CFG["eval_every"] == 0 and step > 0:
                try:
                    val_loss, tool_acc = evaluate(model, val_ds, tok, cfg, max_len=current_seq_len)
                    print(f"  ── EVAL step {step} | val_loss {val_loss:.4f} | tool_acc {tool_acc:.2%}")
                    log[-1].update({"val_loss": val_loss, "tool_acc": tool_acc})
                except Exception as e:
                    print(f"  ── EVAL skipped: {e}")

            # Save
            if step % TRAIN_CFG["save_every"] == 0 and step > 0:
                try:
                    save_path = f"{TRAIN_CFG['save_dir']}/step_{step:05d}"
                    Path(save_path).mkdir(exist_ok=True)
                    mx.savez(f"{save_path}/weights.npz", **dict(tree_flatten(model.parameters())))
                    json.dump(log, open(f"{save_path}/log.json", "w"), indent=2)
                    print(f"  ── Saved checkpoint → {save_path}")
                except Exception as e:
                    print(f"  ── Save failed: {e}")

            step += 1

    if log:
        final = log[-1]
        print(f"Training complete. Final loss: {final['loss']:.4f} at step {final['step']}")
        json.dump(log, open(f"{TRAIN_CFG['save_dir']}/log.json", "w"), indent=2)
        print(f"Log saved → {TRAIN_CFG['save_dir']}/log.json")
    else:
        print("Training complete (no steps logged).")

if __name__ == "__main__":
    train()
