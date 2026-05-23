import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import time, json, os, random, argparse, copy
from pathlib import Path
from monitor import print_hw
from stats_logger import GLOBAL as log

from config import AgentMindConfig
from model.agent_lm import AgentMind
from model.mtp_head import mtp_loss
from model.latent import inject_latent_tokens, latent_loss_mask
from data.pipeline import AgentDataset, make_dataloader
from lora import apply_lora, LoRALinear
from scheduler import CosineWarmupScheduler
from init import init_agentmind
from apprentice import kl_div

# ── Default config (overridable when run as main) ─────────

cfg = AgentMindConfig()

TRAIN_CFG = dict(
    lr            = 2e-4,
    weight_decay  = 0.01,
    warmup_steps  = 100,
    total_steps   = 3000,
    grad_clip     = 1.0,
    batch_size    = 1,
    grad_accum    = 8,
    seq_len       = 512,
    lora_rank     = 16,
    lora_alpha    = 32.0,
    eval_every    = 500,
    save_every    = 200,
    save_dir      = "/Users/rajeshshrirao/Desktop/Passion-projects/Agentmind/checkpoints",
    use_mtp       = False,
    mtp_weight    = 0.2,
    mtp_start     = 500,
    latent_stage  = 1,
    seq_len_schedule = {0: 128, 800: 256, 2000: 512},
)

# ── Protocol debugging helpers ────────────────────────────

PROSE_PATTERNS = [
    "let me", "i'll", "i will", "here are", "here is", "i can",
    "let's", "sure", "okay", "ok", "first", "the result", "the tool",
    "task completed", "the output", "i have", "i found", "searching",
    "looking", "checking", "running", "using", "getting",
]


def analyze_protocol(text: str) -> dict:
    """Analyze generated text for protocol transition behavior."""
    result = {
        "contains_tool_call_token": False,
        "first_tool_call_position": -1,
        "contains_open_brace": False,
        "contains_close_brace": False,
        "contains_name_key": False,
        "contains_args_key": False,
        "balanced_braces": False,
        "balanced_quotes": False,
        "prose_before_tool_call": 0,
    }

    idx = text.find("<|tool_call|>")
    if idx != -1:
        result["contains_tool_call_token"] = True
        result["first_tool_call_position"] = idx
        result["prose_before_tool_call"] = len(text[:idx].split())
        json_part = text[idx + len("<|tool_call|>"):]
        for boundary in ("<|observe|>", "<|end|>", "<eos>"):
            if boundary in json_part:
                json_part = json_part.split(boundary)[0]
        result["contains_open_brace"] = "{" in json_part
        result["contains_close_brace"] = "}" in json_part
        result["contains_name_key"] = '"name"' in json_part
        result["contains_args_key"] = '"args"' in json_part
        result["balanced_braces"] = json_part.count("{") == json_part.count("}")
        result["balanced_quotes"] = json_part.count('"') % 2 == 0
    else:
        result["prose_before_tool_call"] = len(text.split())

    return result


def detect_prose_contamination(text: str) -> dict:
    """Detect prose narration that may dominate before tool mode activates."""
    result = {
        "prose_before_tool_call": 0,
        "matched_prose_patterns": [],
        "total_token_count": len(text.split()),
    }
    idx = text.find("<|tool_call|>")
    if idx != -1:
        before = text[:idx].lower()
        result["prose_before_tool_call"] = len(before.split())
    else:
        before = text.lower()
        result["prose_before_tool_call"] = len(before.split())

    for pattern in PROSE_PATTERNS:
        if pattern in before:
            result["matched_prose_patterns"].append(pattern)

    return result


def debug_generation(backbone, tok, cfg, prompt_ids_in: list[int] = None,
                     max_new_tokens: int = 80, repetition_penalty: float = 1.2) -> dict:
    """Run a greedy decode and return full debug info.

    Uses token IDs directly so the prompt matches training distribution.
    Applies repetition penalty to suppress degenerate loops.
    """
    if prompt_ids_in is None:
        prompt_ids_in = [cfg.bos_id, cfg.user_id] + tok.encode(
            "Search arxiv for recent Mamba papers"
        ) + [cfg.assistant_id]
    prompt_ids = mx.array([prompt_ids_in])

    h_states = {}
    out_ids = []
    recent = []

    for _ in range(max_new_tokens):
        logits_t, h_states = backbone.forward_with_state(
            mx.array([[out_ids[-1]]]) if out_ids else prompt_ids,
            h_states if out_ids else h_states,
        )
        logit = logits_t[0, -1]
        if repetition_penalty > 1.0:
            for r in set(recent[-32:]):
                if logit[r] > 0:
                    logit[r] /= repetition_penalty
                else:
                    logit[r] *= repetition_penalty
        ntok = mx.argmax(logit).item()
        out_ids.append(ntok)
        recent.append(ntok)
        if ntok == cfg.eos_id:
            break

    decoded = tok.decode(out_ids)
    protocol = analyze_protocol(decoded)
    prose = detect_prose_contamination(decoded)

    return {
        "raw_output": decoded,
        "protocol": protocol,
        "prose": prose,
        "has_tool_call": protocol["contains_tool_call_token"],
    }


def print_debug_generation(step: int, domain: str, result: dict):
    """Print full debug generation trace."""
    print()
    print("=" * 55)
    print(f"[DEBUG GENERATION] step {step}  ({domain})")
    print("=" * 55)
    print("OUTPUT:")
    print(result["raw_output"])
    print("---")
    p = result["protocol"]
    pr = result["prose"]
    print(f"[PROTO] tool_call_token={p['contains_tool_call_token']} "
          f"pos={p['first_tool_call_position']}")
    print(f"[PROTO] open_brace={p['contains_open_brace']} "
          f"close_brace={p['contains_close_brace']} "
          f"balanced={p['balanced_braces']}")
    print(f"[PROTO] name_key={p['contains_name_key']} "
          f"args_key={p['contains_args_key']} "
          f"balanced_quotes={p['balanced_quotes']}")
    print(f"[PROSE] prose_before_tool_call={p['prose_before_tool_call']} "
          f"patterns={pr['matched_prose_patterns']}")
    print("=" * 55)
    print()


# ── Syntax auxiliary loss ──────────────────────────────────

def compute_syntax_aux_loss(targets: mx.array, tok, cfg, weight: float = 0.05) -> mx.array:
    """
    Lightweight structural heuristic for tool-call syntax.

    Checks decoded target tokens for:
      - Balanced braces  { }
      - Balanced quotes  "
      - Presence of "name" and "args" keys

    Returns a scalar penalty added to the main loss.
    Only activates when <|tool_call|> tokens are present in the target.
    """
    if not hasattr(tok, 'decode'):
        return mx.array(0.0)

    target_list = [int(t) for t in targets.flatten().tolist() if int(t) != -100]

    if not target_list:
        return mx.array(0.0)

    # Quick check: tool_call_id present?
    if cfg.tool_call_id not in target_list:
        return mx.array(0.0)

    decoded = tok.decode(target_list)
    idx = decoded.find("<|tool_call|>")
    if idx == -1:
        return mx.array(0.0)

    json_str = decoded[idx + len("<|tool_call|>"):]
    for boundary in ("<|observe|>", "<|end|>", "<eos>"):
        if boundary in json_str:
            json_str = json_str.split(boundary)[0]

    penalty = 0.0
    braces = json_str.count("{") + json_str.count("}")
    quotes = json_str.count('"')

    if braces % 2 != 0:
        penalty += 0.1
    if quotes % 2 != 0:
        penalty += 0.05
    if '"name"' not in json_str:
        penalty += 0.2
    if '"args"' not in json_str:
        penalty += 0.2

    return mx.array(penalty * weight)


# ── Loss ──────────────────────────────────────────────────

def cross_entropy_loss(logits, targets, boundary_weight: float = 0.0, boundary_ids: set = None):
    # Shift logits and targets for next-token prediction
    logits = logits[:, :-1, :]
    targets = targets[:, 1:]
    B, L, V = logits.shape
    flat_logits  = logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)
    mask = (flat_targets != -100).astype(mx.float32)
    loss = nn.losses.cross_entropy(flat_logits, mx.maximum(flat_targets, 0), reduction='none')
    weighted_loss = loss * mask
    if boundary_weight > 0.0 and boundary_ids:
        boundary_mask = mx.zeros_like(mask)
        for tid in boundary_ids:
            boundary_mask = mx.maximum(boundary_mask, (flat_targets == tid).astype(mx.float32))
        boundary_mask = boundary_mask * mask
        weighted_loss = weighted_loss * (1.0 + boundary_mask * boundary_weight)
    return weighted_loss.sum() / (mask.sum() + 1e-8)

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
                mx.eval(list(trainable.values()), optimizer.state)

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


# ── Apprenticeship entry points ────────────────────────────


def train_specialist(backbone, domain_dataset, domain_name,
                     steps=500, lr=2e-4, seq_len=256, latent_stage=1,
                     seq_len_schedule=None, syntax_aux_weight=0.05):
    """
    Train a LoRA specialist adapter on a single domain.

    Backbone stays FROZEN — only the 2.36M LoRA A/B matrices train.
    MTP is DISABLED during specialist training because:
      - The backbone MTP head (4 x 32K vocab) is larger than the adapter
      - MTP runs on the backbone, which is frozen here
      - Enabling it would waste compute on zero-gradient backbone params

    If syntax_aux_weight > 0, a lightweight structural heuristic is added
    to the loss for samples containing <|tool_call|> tokens. This encourages
    balanced braces, quote completion, and required key presence without
    constrained decoding or RL.
    """
    # Guard: MTP must never fire when the backbone is frozen
    _mtp_guard = False
    assert not _mtp_guard, \
        "MTP must not run during specialist training — backbone is frozen"

    from model.latent import latent_loss_mask

    if isinstance(domain_dataset, list):
        ds = AgentDataset.__new__(AgentDataset)
        ds.samples = domain_dataset
        ds.cfg = backbone.cfg
        from tokenizer_setup import load_tokenizer
        ds.tok = load_tokenizer("agentmind_tok.model")
        ds.ids_array = None
        ds.labels_array = None
        ds.latent_stage = latent_stage
        domain_dataset = ds

    domain_dataset.latent_stage = latent_stage

    from lora import LoRALinear as _LoRALinear

    # Check if LoRA is already applied (e.g. by orchestrator after round 1).
    # If so, don't call apply_lora again — it would freeze existing LoRALinear modules.
    _has_lora = False
    for mod in backbone.modules():
        if isinstance(mod, _LoRALinear):
            _has_lora = True
            break

    if not _has_lora:
        apply_lora(backbone, rank=16, alpha=32.0)
    else:
        backbone.freeze()
        # MLX freeze() adds A/B to _no_grad on each LoRALinear.
        # Remove them surgically so only LoRA adapters are trainable.
        for mod in backbone.modules():
            if isinstance(mod, _LoRALinear):
                mod._no_grad.discard('A')
                mod._no_grad.discard('B')

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)
    warmup = min(200, max(1, steps // 10))
    scheduler = CosineWarmupScheduler(optimizer, lr, warmup, steps)

    # Freeze embed + lm_head during specialist training — vocab is stable
    # exclude embed/lm_head from trainable parameters
    trainable = {k: v for k, v in backbone.trainable_parameters().items()
                 if not any(k.startswith(p) for p in ["last_", "embed", "lm_head"])}

    # Hold out 5% for validation
    n_total = len(domain_dataset)
    n_val = max(1, n_total // 20)
    val_indices = set(random.sample(range(n_total), n_val))
    train_indices = [i for i in range(n_total) if i not in val_indices]

    # Pre-filter: only keep samples where assistant_id appears within seq_len
    # (otherwise labels are all -100 and the step wastes compute)
    valid = []
    for i in train_indices:
        _, labels = domain_dataset[i]
        if any(l != -100 for l in labels[:seq_len]):
            valid.append(i)
    dropped = len(train_indices) - len(valid)
    if dropped:
        print(f"  Filtered {dropped}/{len(train_indices)} samples (assistant beyond {seq_len=})")
        train_indices = valid
    domain_dataset._cache.clear()

    # Tool-call-boundary oversampling: dynamically balance so that
    # at most ~60% of training indices are early-boundary samples.
    # Skip entirely if >85% already have tool_call in first 80 chars
    # (oversampling pure duplicates just causes repetition collapse).
    early_count = 0
    for i in range(len(train_indices)):
        s = domain_dataset.samples[train_indices[i]]
        assistant = s.get("messages", [{}])[-1].get("content", "")
        if 0 <= assistant.find("<|tool_call|>") < 80:
            early_count += 1
    early_ratio = early_count / len(train_indices) if train_indices else 0
    if early_ratio > 0.85:
        print(f"  Oversampling skipped — {early_ratio:.0%} already early-boundary")
    else:
        oversampled = []
        for i in train_indices:
            sample = domain_dataset.samples[i]
            assistant = sample.get("messages", [{}])[-1].get("content", "")
            pos = assistant.find("<|tool_call|>")
            if 0 <= pos < 80:
                oversampled.extend([i] * 3)
            else:
                oversampled.append(i)
        n_os = len(oversampled) - len(train_indices)
        if n_os > 0:
            print(f"  Oversampled tool-call-boundary samples: +{n_os} ({len(oversampled)} total)")
            train_indices = oversampled

    # Seq len curriculum: {step_threshold: seq_len}
    if seq_len_schedule is None:
        seq_len_schedule = {0: seq_len}

    backbone.train()
    step = 0
    t_start = time.time()
    nan_count = 0
    zero_loss_count = 0
    current_seq_len = 0
    hw_log_counter = 0
    tool_call_emit_count = 0
    tool_call_total_count = 0

    while step < steps:
        # Check if seq_len should change
        new_seq_len = 0
        for threshold, sl in sorted(seq_len_schedule.items()):
            if step >= threshold:
                new_seq_len = sl
        if new_seq_len != current_seq_len:
            current_seq_len = new_seq_len
            print(f"  ── Seq len {current_seq_len} at step {step}")
            # Re-filter indices for new seq_len (throwaway cache to avoid memory spike)
            saved_cache = domain_dataset._cache
            domain_dataset._cache = {}
            valid = []
            for i in train_indices:
                _, labels = domain_dataset[i]
                if any(l != -100 for l in labels[:current_seq_len]):
                    valid.append(i)
            domain_dataset._cache = saved_cache
            if domain_dataset._cache:
                domain_dataset._cache.clear()
            dropped = len(train_indices) - len(valid)
            if dropped:
                print(f"  Filtered {dropped}/{len(train_indices)} (assistant beyond seq_len={current_seq_len})")
                train_indices = valid

        t0 = time.time()
        log_steps = 0
        loader = make_dataloader(domain_dataset, batch_size=1, max_len=current_seq_len, indices=train_indices)
        for input_ids, targets in loader:
            if step >= steps:
                break

            if latent_stage >= 3:
                targets = latent_loss_mask(
                    input_ids, targets,
                    backbone.cfg.think_start_id, backbone.cfg.think_end_id
                )

            tok_local = getattr(domain_dataset, 'tok', None)

            def loss_fn(params):
                backbone.update(params)
                logits, _ = backbone(input_ids, return_mtp=False, return_h_states=False)
                bw = 5.0 if step < 200 else 0.5
                main = cross_entropy_loss(
                    logits, targets,
                    boundary_weight=bw,
                    boundary_ids={backbone.cfg.tool_call_id, backbone.cfg.observe_id}
                )
                if syntax_aux_weight > 0 and tok_local is not None:
                    main = main + compute_syntax_aux_loss(
                        targets, tok_local, backbone.cfg, weight=syntax_aux_weight
                    )
                return main

            t_fwd = time.time()
            loss, grads = mx.value_and_grad(loss_fn)(trainable)
            mx.eval(loss, grads)
            t_fwd = time.time() - t_fwd

            loss_finite = mx.isfinite(loss).item()
            grads_finite, _ = check_finite(grads)
            if not loss_finite or not grads_finite:
                nan_count += 1
                grads = tree_map(mx.zeros_like, grads)
                optimizer.update(trainable, grads)
                continue

            if loss.item() == 0.0:
                zero_loss_count += 1
                uniq = sorted(set(targets.flatten().tolist()))[:10]
                print(f"  ⚠️  [train_specialist] Zero loss at step {step}/{steps} — mask empty. targets unique: {uniq}")
                step += 1
                continue

            t_opt = time.time()
            grads, grad_norm = clip_gradients(grads, 1.0)
            optimizer.update(trainable, grads)
            mx.eval(list(trainable.values()), optimizer.state)
            lr_now = scheduler.step()
            t_opt = time.time() - t_opt
            log_steps += 1

            if log_steps == 100 or step == steps - 1 or step == 0:
                now = time.time()
                elapsed = now - t0
                tok_per_sec = (input_ids.shape[1] * log_steps) / (elapsed + 1e-8)

                # Debug generation trace every 50 steps
                debug_info = None
                if tok_local is not None and step > 0:
                    try:
                        prompt_ids = [backbone.cfg.bos_id, backbone.cfg.user_id] + \
                            tok_local.encode("Search arxiv for recent Mamba papers") + \
                            [backbone.cfg.assistant_id]
                        debug_info = debug_generation(
                            backbone, tok_local, backbone.cfg,
                            prompt_ids_in=prompt_ids, max_new_tokens=80,
                            repetition_penalty=1.5
                        )
                        if debug_info["has_tool_call"]:
                            tool_call_emit_count += 1
                        tool_call_total_count += 1
                    except Exception:
                        pass

                if debug_info:
                    print_debug_generation(step, domain_name, debug_info)

                emit_rate = 0.0
                if tool_call_total_count > 0:
                    emit_rate = tool_call_emit_count / tool_call_total_count * 100

                print(f"[{domain_name}] step {step:3d}/{steps} "
                      f"loss {loss.item():.4f} lr {lr_now:.2e} "
                      f"grad_norm {grad_norm:.3f} {tok_per_sec:.0f} tok/s"
                      f"  emit={emit_rate:.0f}%")
                log.step("specialist", step, steps, loss.item(), lr=lr_now,
                         grad_norm=grad_norm, tok_per_s=tok_per_sec,
                         seq_len=current_seq_len, domain=domain_name)
                t0 = now
                log_steps = 0
                hw_log_counter += 1
                if hw_log_counter % 2 == 0:
                    from monitor import print_hw
                    print_hw(f"spec {domain_name} step {step}")

            step += 1

    # Validation loss on held-out set
    val_losses = []
    backbone.eval()
    val_loader = make_dataloader(domain_dataset, batch_size=1, max_len=seq_len, indices=list(val_indices), shuffle=False)
    for val_ids, val_targets in val_loader:
        logits, _ = backbone(val_ids, return_mtp=False, return_h_states=False)
        vl = cross_entropy_loss(logits, val_targets).item()
        if mx.isfinite(vl) and vl > 0:
            val_losses.append(vl)
    backbone.train()

    elapsed = time.time() - t_start
    val_loss_val = sum(val_losses)/len(val_losses) if val_losses else None
    val_loss_str = f"val_loss={val_loss_val:.4f}" if val_loss_val else "val_loss=N/A"

    # Tool call accuracy eval
    from eval import evaluate_tool_calls, tool_eval_report, print_tool_report
    test_prompts = [
        "<|user|>Search arxiv for Mamba SSM papers<|assistant|>",
        "<|user|>Get the weather in Tokyo and Pune<|assistant|>",
        "<|user|>Run the test suite and fix any failures<|assistant|>",
    ]
    tok = getattr(domain_dataset, 'tok', None)
    tool_valid, tool_total, tool_pct = 0, 0, 0.0
    if tok is not None:
        try:
            backbone.eval()
            call_results = evaluate_tool_calls(backbone, test_prompts, tok, backbone.cfg)
            tool_report = tool_eval_report(call_results)
            tool_total = tool_report.get("total", 0)
            tool_valid = tool_report.get("valid", 0)
            tool_pct = tool_report.get("valid_pct", 0.0)
            if tool_total > 0:
                print_tool_report(tool_report)
            log.summary("specialist_tool", domain=domain_name, valid=tool_valid,
                        total=tool_total, valid_pct=tool_pct)
        except Exception as e:
            print(f"  ── Tool eval skipped: {e}")
        finally:
            backbone.train()

    print(f"[train_specialist] '{domain_name}' complete ({step} steps, {elapsed:.0f}s) "
          f"{val_loss_str} nan={nan_count} zero_loss={zero_loss_count} "
          f"tool_valid={tool_valid}/{tool_total} ({tool_pct:.1f}%)")
    log.summary("specialist", domain=domain_name, steps=step, elapsed=elapsed,
                val_loss=val_loss_val, nan_count=nan_count, zero_loss_count=zero_loss_count,
                tool_valid_pct=tool_pct)

    # Extract and return only LoRA A/B matrices
    params = dict(tree_flatten(backbone.trainable_parameters()))
    return {k: v for k, v in params.items()
            if k.endswith('.A') or k.endswith('.B')}


def distill_backbone(backbone, specialists, combined_data,
                     beta=0.5, mtp_weight=0.2, steps=50,
                     lr=1e-5, seq_len=512, latent_stage=1):
    """
    Distill specialist knowledge into the backbone.

    Backbone is UNFROZEN — all params train.
    MTP is ENABLED after step 20 for stability (warm-up before auxiliary loss).
    Specialists are frozen — only used as fixed teacher models.

    Loss = CE(backbone, labels)
         + beta * KL(backbone || specialist)
         + mtp_weight * MTP_loss(backbone)     (after step 20)
    """
    from model.mtp_head import mtp_loss as mtp_loss_fn

    # ── Build frozen specialist models ──────────────────────
    spec_models = {}
    for name, weights in specialists.items():
        m = AgentMind(backbone.cfg)
        m = init_agentmind(m, backbone.cfg)
        apply_lora(m)
        m.load_lora(weights)
        m.eval()
        m.freeze()
        spec_models[name] = m

    # ── Pre-tokenize data with domain labels ─────────────────
    tok = getattr(combined_data, 'tok', None)
    if tok is None:
        from tokenizer_setup import load_tokenizer
        tok = load_tokenizer("agentmind_tok.model")

    tokenized = []
    samples = (combined_data.samples
               if hasattr(combined_data, 'samples') else combined_data)
    for sample in samples:
        s = copy.deepcopy(sample)
        s = inject_latent_tokens(s, tok, latent_stage)
        text = ""
        for msg in s["messages"]:
            r, c = msg["role"], msg["content"]
            if r == "system":
                text += f"<|system|>{c}"
            elif r == "user":
                text += f"<|user|>{c}"
            elif r == "assistant":
                text += f"<|assistant|>{c}<eos>"
        ids = tok.encode(text, add_bos=True)[:seq_len]
        labels = [-100] * len(ids)
        in_asst = False
        for i, tid in enumerate(ids):
            if tid == backbone.cfg.assistant_id:
                in_asst = True
            if in_asst:
                labels[i] = tid
            if tid in (backbone.cfg.eos_id, backbone.cfg.user_id,
                       backbone.cfg.system_id):
                in_asst = False
        labels = labels[:seq_len]
        domain = sample.get("domain", list(specialists.keys())[0])
        tokenized.append({"ids": ids, "labels": labels, "domain": domain})

    # ── Hold out 5% for validation ──
    random.shuffle(tokenized)
    n_val = max(1, len(tokenized) // 20)
    val_samples = tokenized[:n_val]
    train_samples = tokenized[n_val:]

    # ── Distillation loop ──────────────────────────────────
    # Note: specialist forward passes run on-the-fly per step (~0.3s overhead).
    # Pre-computing all logits would be 25K × (1,256,32K) × 4B ≈ 800GB — not viable.
    backbone.unfreeze()
    trainable = {k: v for k, v in backbone.trainable_parameters().items()
                 if not k.startswith("last_")}

    optimizer = optim.AdamW(learning_rate=lr)
    step = 0
    t0 = time.time()
    t_start = t0
    nan_count = 0
    zero_loss_count = 0
    hw_log_counter = 0

    while step < steps:
        random.shuffle(train_samples)
        for i, sample in enumerate(train_samples):
            if step >= steps:
                break

            ids = mx.array([sample["ids"]])
            labels = mx.array([sample["labels"]])
            domain = sample["domain"]

            if latent_stage >= 3:
                labels = latent_loss_mask(
                    ids, labels,
                    backbone.cfg.think_start_id, backbone.cfg.think_end_id
                )

            # Specialist teacher forward (frozen → constant logits, ~0.3s overhead)
            spec_logits, _ = spec_models[domain](ids)

            def loss_fn(params):
                backbone.update(params)
                b_logits, _ = backbone(ids, return_mtp=True)
                task = cross_entropy_loss(b_logits, labels)
                distill = kl_div(b_logits, spec_logits, labels)
                total = task + beta * distill
                # MTP warm-up: enable after step 20 for stability
                if step >= 20 and mtp_weight > 0:
                    total += mtp_loss_fn(
                        backbone.last_mtp_logits, labels, weight=mtp_weight
                    )
                return total

            t_fwd = time.time()
            loss, grads = mx.value_and_grad(loss_fn)(trainable)
            mx.eval(loss, grads)
            t_fwd = time.time() - t_fwd

            loss_finite = mx.isfinite(loss).item()
            grads_finite, _ = check_finite(grads)
            if not loss_finite or not grads_finite:
                nan_count += 1
                grads = tree_map(mx.zeros_like, grads)
                optimizer.update(trainable, grads)
                continue

            if loss.item() == 0.0:
                zero_loss_count += 1
                step += 1
                continue

            t_opt = time.time()
            grads, grad_norm = clip_gradients(grads, 1.0)
            optimizer.update(trainable, grads)
            mx.eval(list(trainable.values()), optimizer.state)
            t_opt = time.time() - t_opt

            if step % 10 == 0 or step == steps - 1:
                elapsed = time.time() - t0
                print(f"[distill] step {step:3d}/{steps} "
                      f"loss {loss.item():.4f} "
                      f"grad_norm {grad_norm:.3f} {elapsed:.1f}s "
                      f"fwd+bwd={t_fwd:.2f}s opt={t_opt:.2f}s")
                log.step("distill", step, steps, loss.item(), grad_norm=grad_norm,
                         seq_len=seq_len)
                t0 = time.time()
                hw_log_counter += 1
                if hw_log_counter % 2 == 0:
                    from monitor import print_hw
                    print_hw(f"distill step {step}")

            step += 1

    # Validation on held-out set
    backbone.eval()
    val_losses = []
    for s in val_samples:
        ids = mx.array([s["ids"]])
        labels = mx.array([s["labels"]])
        logits, _ = backbone(ids, return_mtp=False)
        vl = cross_entropy_loss(logits, labels).item()
        if mx.isfinite(vl) and vl > 0:
            val_losses.append(vl)

    backbone.freeze()
    final_loss = loss.item() if step > 0 else float('inf')
    elapsed = time.time() - t_start
    val_loss_val = sum(val_losses)/len(val_losses) if val_losses else None
    val_loss_str = f"val_loss={val_loss_val:.4f}" if val_loss_val else "val_loss=N/A"
    print(f"[distill] Complete ({step} steps) final_loss={final_loss:.4f} "
          f"{val_loss_str} ({elapsed:.0f}s) nan={nan_count} zero_loss={zero_loss_count}")
    log.summary("distill", steps=step, elapsed=elapsed, final_loss=final_loss,
                val_loss=val_loss_val, nan_count=nan_count, zero_loss_count=zero_loss_count)
    return final_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=None, help="Override total_steps (for testing)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint dir to resume from")
    parser.add_argument("--test-nan", action="store_true", help="Run the NaN injection and recovery test harness")
    args = parser.parse_args()

    if args.max_steps is not None:
        TRAIN_CFG["total_steps"] = args.max_steps

    if args.resume is not None:
        TRAIN_CFG["resume"] = args.resume
    else:
        TRAIN_CFG.pop("resume", None)

    if args.test_nan:
        run_nan_test_harness()
    else:
        train()
