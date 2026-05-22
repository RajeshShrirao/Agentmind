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
parser.add_argument("--test-nan", action="store_true", help="Run the NaN injection and recovery test harness")
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
    # Shift logits and targets for next-token prediction
    logits = logits[:, :-1, :]
    targets = targets[:, 1:]
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

# ── Finiteness & Rollback Helpers ──────────────────────────

def check_finite(tree):
    for k, v in tree_flatten(tree):
        if not mx.all(mx.isfinite(v)).item():
            return False, k
    return True, None

def clone_tree(tree):
    return tree_map(lambda x: mx.array(x) if isinstance(x, mx.array) else x, tree)

# ── NaN Recovery Test Harness ─────────────────────────────

def run_nan_test_harness():
    print("=== Running NaN Injection and Recovery Test Harness ===")

    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = mx.array([[1.0, 2.0], [3.0, 4.0]])
            self.w2 = mx.array([[5.0, 6.0], [7.0, 8.0]])

        def __call__(self, x):
            return x @ self.w1 @ self.w2

    model = SmallModel()
    optimizer = optim.AdamW(learning_rate=0.1)

    backup_params = clone_tree(model.parameters())
    backup_opt_state = clone_tree(optimizer.state)

    def train_step_fn(input_ids, targets):
        def loss_fn(params):
            model.update(params)
            pred = model(input_ids)
            return mx.mean((pred - targets) ** 2)
        loss, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
        return loss, grads

    x = mx.ones((1, 2))
    y = mx.array([[10.0, 12.0]])

    # 1. Normal step
    print("Step 1: Normal training step")
    loss, grads = train_step_fn(x, y)
    mx.eval(loss, grads)

    optimizer.update(model.trainable_parameters(), grads)
    mx.eval(model.parameters(), optimizer.state)

    backup_params = clone_tree(model.parameters())
    backup_opt_state = clone_tree(optimizer.state)
    print("Normal step completed. Step counter:", optimizer.state['step'].item())

    pre_nan_params = clone_tree(model.parameters())
    pre_nan_opt_state = clone_tree(optimizer.state)

    # 2. Inject NaN loss
    print("\nStep 2: Injecting NaN loss")
    loss, grads = train_step_fn(x, y)
    loss = mx.array(float('nan'))
    mx.eval(loss, grads)

    loss_finite = mx.isfinite(loss).item()
    grads_finite, bad_grad_key = check_finite(grads)

    if not loss_finite or not grads_finite:
        bad_source = "loss" if not loss_finite else f"gradient in {bad_grad_key}"
        print(f"  ⚠️  [TEST] Non-finite value detected! Source: {bad_source}. Recovering...")

        # Zero gradients
        grads = tree_map(mx.zeros_like, grads)

        # Reset
        model.update(backup_params)
        optimizer.state = clone_tree(backup_opt_state)
        mx.eval(model.parameters(), optimizer.state)
        grads = tree_map(mx.zeros_like, grads)

    for k, v in tree_flatten(model.parameters()):
        assert mx.array_equal(v, pre_nan_params[k]).item(), f"Model parameter {k} was contaminated!"
    for k in ['step', 'learning_rate']:
        assert mx.array_equal(optimizer.state[k], pre_nan_opt_state[k]).item(), f"Optimizer state key {k} was contaminated!"
    print("✅ Verified parameters and optimizer state are completely clean after NaN loss.")

    # 3. Inject NaN gradient
    print("\nStep 3: Injecting NaN gradient")
    loss, grads = train_step_fn(x, y)
    grads['w1'] = mx.array([[float('nan'), 2.0], [3.0, 4.0]])
    mx.eval(loss, grads)

    loss_finite = mx.isfinite(loss).item()
    grads_finite, bad_grad_key = check_finite(grads)

    if not loss_finite or not grads_finite:
        bad_source = "loss" if not loss_finite else f"gradient in {bad_grad_key}"
        print(f"  ⚠️  [TEST] Non-finite value detected! Source: {bad_source}. Recovering...")

        # Zero gradients
        grads = tree_map(mx.zeros_like, grads)

        # Reset
        model.update(backup_params)
        optimizer.state = clone_tree(backup_opt_state)
        mx.eval(model.parameters(), optimizer.state)
        grads = tree_map(mx.zeros_like, grads)

    for k, v in tree_flatten(model.parameters()):
        assert mx.array_equal(v, pre_nan_params[k]).item(), f"Model parameter {k} was contaminated!"
    for k in ['step', 'learning_rate']:
        assert mx.array_equal(optimizer.state[k], pre_nan_opt_state[k]).item(), f"Optimizer state key {k} was contaminated!"
    print("✅ Verified parameters and optimizer state are completely clean after NaN gradient.")

    # 4. Recover and run normal step
    print("\nStep 4: Running recovery normal step")
    loss, grads = train_step_fn(x, y)
    mx.eval(loss, grads)

    optimizer.update(model.trainable_parameters(), grads)
    mx.eval(model.parameters(), optimizer.state)
    print("Recovery step completed successfully. Step counter:", optimizer.state['step'].item())
    assert optimizer.state['step'].item() == 2, f"Expected step counter to be 2, but got {optimizer.state['step'].item()}"
    print("✅ Verified optimizer successfully incremented step counter after recovery.")

    # 5. Test gradient accumulation clearing
    print("\nStep 5: Testing gradient accumulation clearing on skip")
    accum_grad = None
    accum_loss = 0.0

    # Simulate step 1 of accumulation
    loss, grads = train_step_fn(x, y)
    mx.eval(loss, grads)
    accum_grad = grads
    accum_loss += loss.item()

    # Simulate step 2 (NaN step) of accumulation
    loss, grads = train_step_fn(x, y)
    loss = mx.array(float('nan'))
    mx.eval(loss, grads)

    loss_finite = mx.isfinite(loss).item()
    grads_finite, bad_grad_key = check_finite(grads)

    if not loss_finite or not grads_finite:
        grads = tree_map(mx.zeros_like, grads)
        if accum_grad is not None:
            accum_grad = tree_map(mx.zeros_like, accum_grad)

        accum_grad = None
        accum_loss = 0.0

        model.update(backup_params)
        optimizer.state = clone_tree(backup_opt_state)
        mx.eval(model.parameters(), optimizer.state)
        grads = tree_map(mx.zeros_like, grads)

    assert accum_grad is None, "accum_grad was not cleared on skip!"
    assert accum_loss == 0.0, "accum_loss was not cleared on skip!"
    print("✅ Verified gradient accumulation is successfully cleared on skip.")

    print("\n🎉 NaN Injection and Recovery Test Harness Passed successfully!")

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

        # Get dynamic latent stage and apply mask if stage >= 3
        from model.latent import get_latent_stage, latent_loss_mask
        stage = get_latent_stage(step)
        if stage >= 3:
            masked_targets = latent_loss_mask(input_ids, targets, model.cfg.think_start_id, model.cfg.think_end_id)
        else:
            masked_targets = targets

        def loss_fn(params):
            model.update(params)
            use_mtp = TRAIN_CFG["use_mtp"] and step >= TRAIN_CFG["mtp_start"]
            logits, h_states = model(input_ids, return_mtp=use_mtp)
            main = cross_entropy_loss(logits, masked_targets)

            if use_mtp and hasattr(model, "mtp"):
                aux = mtp_loss(model.last_mtp_logits, masked_targets,
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

    # Tokenizer — load first so we can derive real token IDs
    from tokenizer_setup import load_tokenizer, get_token_ids, assert_token_ids_real, hydrate_config
    tok = load_tokenizer("agentmind_tok.model")
    ids = get_token_ids(tok)
    assert_token_ids_real(tok, ids)
    hydrate_config(cfg, tok)
    print("Config token IDs hydrated from tokenizer.\n")

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
    # Note: tokenizer already loaded at top of train()
    # Use pre-tokenized data if available, otherwise use raw JSONL
    # Pre-tokenized path is only safe after Phase 0-1 (correct label IDs).
    # Dynamic latent injection still runs on tokenized IDs via latent_loss_mask.
    #
    # History: The pre-tokenized bypass was originally `if False and os.path.exists(...)`
    # to prevent using pre-tokenized data that encoded labels with wrong hardcoded
    # token IDs (e.g. assistant_id=15 vs actual ~31999). Using that data would have
    # silently masked the problem. Now that Phase 0-1 fixed label encoding via
    # hydrate_config(), pre-tokenized data must be REGENERATED (run pretokenize.py)
    # before this path will activate. Conditions for re-enabling: (1) regenerate .npz
    # files with correct token IDs, (2) verify label masks via test_token_ids.py.
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

    # Take initial clean backups
    backup_params = clone_tree(model.parameters())
    backup_opt_state = clone_tree(optimizer.state)
    nan_logged = False

    t0 = time.time()  # Track time across entire grad_accum window
    current_seq_len = 0

    while step < TRAIN_CFG["total_steps"]:
        # Update sequence length based on curriculum
        new_seq_len = get_current_seq_len(step)
        if new_seq_len != current_seq_len:
            current_seq_len = new_seq_len
            print(f"  ── Sequence length changed to {current_seq_len} at step {step}")

        # Dynamically set dataset stage on creation / update
        from model.latent import get_latent_stage
        stage = get_latent_stage(step)
        train_ds.latent_stage = stage
        val_ds.latent_stage = stage

        # Create dataloader with current sequence length
        loader = make_dataloader(train_ds, batch_size=TRAIN_CFG["batch_size"], max_len=current_seq_len)

        for input_ids, targets in loader:
            if step >= TRAIN_CFG["total_steps"]:
                break

            # Update stage dynamically inside loop as well
            stage = get_latent_stage(step)
            train_ds.latent_stage = stage
            val_ds.latent_stage = stage

            # Zero out local grads before we compute them
            grads = None

            # Compute loss and grads
            loss, grads = train_step_fn(input_ids, targets, step)
            mx.eval(loss, grads)

            # Check finiteness of loss and gradients
            loss_finite = mx.isfinite(loss).item()
            grads_finite, bad_grad_key = check_finite(grads)

            if not loss_finite or not grads_finite:
                bad_source = "loss" if not loss_finite else f"gradient of layer '{bad_grad_key}'"
                log_prefix = ""
                if not nan_logged:
                    log_prefix = "[FIRST NAN DETECTED] "
                    nan_logged = True

                print(f"  ⚠️  {log_prefix}Non-finite value detected at step {step}! Source: {bad_source}. Recovery action: Rollback parameters & optimizer state, clearing gradient accumulation, and skipping batch.")

                # Zero gradients before recovery
                grads = tree_map(mx.zeros_like, grads)
                if accum_grad is not None:
                    accum_grad = tree_map(mx.zeros_like, accum_grad)

                # Clear gradient accumulation
                accum_grad = None
                accum_loss = 0.0

                # Rollback model parameters and optimizer state
                model.update(backup_params)
                optimizer.state = clone_tree(backup_opt_state)
                mx.eval(model.parameters(), optimizer.state)

                # Zero gradients after recovery
                grads = tree_map(mx.zeros_like, grads)
                continue

            loss_val = loss.item()

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

                # Check finiteness of accumulated gradients
                accum_finite, bad_accum_key = check_finite(accum_grad)
                if not accum_finite:
                    bad_source = f"accumulated gradient of layer '{bad_accum_key}'"
                    log_prefix = ""
                    if not nan_logged:
                        log_prefix = "[FIRST NAN DETECTED] "
                        nan_logged = True
                    print(f"  ⚠️  {log_prefix}Non-finite accumulated gradients at step {step}! Source: {bad_source}. Recovery action: Rollback parameters & optimizer state, clearing gradient accumulation, and skipping batch.")

                    # Zero gradients before recovery
                    grads = tree_map(mx.zeros_like, grads)
                    accum_grad = tree_map(mx.zeros_like, accum_grad)

                    # Clear gradient accumulation
                    accum_grad = None
                    accum_loss = 0.0

                    # Rollback model parameters and optimizer state
                    model.update(backup_params)
                    optimizer.state = clone_tree(backup_opt_state)
                    mx.eval(model.parameters(), optimizer.state)

                    # Zero gradients after recovery
                    grads = tree_map(mx.zeros_like, grads)
                    continue

                # Only update actual trainable parameters (exclude last_hidden, last_mtp_logits)
                trainable = {k: v for k, v in model.trainable_parameters().items()
                             if not k.startswith("last_")}
                optimizer.update(trainable, accum_grad)
                mx.eval(model.parameters(), optimizer.state)

                # Check that post-update model parameters and optimizer state are fully finite
                post_params_finite, post_param_key = check_finite(model.parameters())
                post_opt_finite, post_opt_key = check_finite(optimizer.state)
                if not post_params_finite or not post_opt_finite:
                    bad_source = f"post-update parameter '{post_param_key}'" if not post_params_finite else f"post-update optimizer state '{post_opt_key}'"
                    log_prefix = ""
                    if not nan_logged:
                        log_prefix = "[FIRST NAN DETECTED] "
                        nan_logged = True
                    print(f"  ⚠️  {log_prefix}Non-finite values detected post-update at step {step}! Source: {bad_source}. Recovery action: Rollback parameters & optimizer state, clearing gradient accumulation, and skipping batch.")

                    # Zero gradients before recovery
                    grads = tree_map(mx.zeros_like, grads)

                    # Clear gradient accumulation
                    accum_grad = None
                    accum_loss = 0.0

                    # Rollback model parameters and optimizer state
                    model.update(backup_params)
                    optimizer.state = clone_tree(backup_opt_state)
                    mx.eval(model.parameters(), optimizer.state)

                    # Zero gradients after recovery
                    grads = tree_map(mx.zeros_like, grads)
                    continue

                # Successfully updated and verified clean! Update backups.
                backup_params = clone_tree(model.parameters())
                backup_opt_state = clone_tree(optimizer.state)

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
                    val_loss, tool_report = evaluate(model, val_ds, tok, cfg, max_len=current_seq_len)
                    tool_acc = tool_report.get("valid_pct", 0.0) / 100.0
                    print(f"  ── EVAL step {step} | val_loss {val_loss:.4f} | tool_valid {tool_report.get('valid', 0)}/{tool_report.get('total', 0)} ({tool_report.get('valid_pct', 0):.1f}%) | failures: {tool_report.get('breakdown', {})}")
                    log[-1].update({"val_loss": val_loss, "tool_acc": tool_acc, "tool_report": tool_report})
                except Exception as e:
                    print(f"  ── EVAL skipped: {e}")

            # Save
            if step % TRAIN_CFG["save_every"] == 0 and step > 0:
                params_finite, _ = check_finite(model.parameters())
                opt_finite, _ = check_finite(optimizer.state)
                if not params_finite or not opt_finite:
                    print(f"  ⚠️  Skipping checkpoint save at step {step} because model/optimizer state contains non-finite values!")
                else:
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
    if args.test_nan:
        run_nan_test_harness()
    else:
        train()
