# AgentMind — Cognitive Apprenticeship Build Sequence
> Build the apprenticeship architecture step by step.
> Most of the old dense model is already built — this focuses on what's new.
> 🤖 = Claude Code handles it | 👤 = You do it manually
>
> **Convention**: Every prompt ends with a log + commit step.
> 📝 Update BUILD_LOG.md with a diary entry (challenges, decisions, results).
> 💾 Commit so every change is tracked in git history.

---

## ✅ Already Built (Skip These)

| Component | Status | Verification |
|---|---|---|
| Model architecture (MambaBlock, AttentionBlock, AgentLM) | ✅ Done | 73 tests passing |
| Config + tokenizer (32K BPE) | ✅ Done | `agentmind_tok.model` exists |
| Data pipeline (formats, dataset, dataloader) | ✅ Done | Pre-tokenized NPZ files exist |
| LoRA + init + scheduler | ✅ Done | Verified via smoke tests |
| Training loop (grad accum, NaN recovery, curriculum) | ✅ Done | `train.py` runs end-to-end |
| Eval infrastructure (perplexity, tool accuracy, format) | ✅ Done | `eval.py` has all 3 paths |
| Bug remediation (token IDs, labels, registries) | ✅ Done | 73/73 tests pass |
| Per-apprentice data generated (v1, template-only) | ✅ Done | 10K samples across 5 domains |
| Cognitive apprenticeship doc | ✅ Done | `docs/cognitive_apprenticeship.md` |

---

## Phase -1: Per-Expert Data Augmentation

> Current 10K/domain is template-only (see `generate_scaled_synthetic.py`).  
> The model learns formatting, not semantics. We need real tool trajectories.  
> Use `build_corpus.py` as reference for HF dataset downloading.  
> ⚠️ No Cerebras/API dependency — that was a bottleneck in the old design.

### 🤖 Prompt 0 — Write prepare_data/ with per-domain HF + synthetic scripts
```
Read build_corpus.py (HF dataset downloading pattern) and generate_scaled_synthetic.py (synthetic template pattern).

Create a prepare_data/ directory with one script per apprentice domain.
Each script must download domain-relevant HuggingFace datasets (streaming),
convert them to apprentice JSONL format, and fall back to synthetic templates
when real data is sparse.

Goal: 25K-50K diverse samples per domain (up from 10K template-only).

Directory structure:
prepare_data/
├── __init__.py
├── base.py                  # Shared: HF download, format conversion, train/val split, combine
├── tool_caller.py           # ToolBench, ToolACE, API-Bank → tool_call patterns
├── planner.py               # WebArena, AgentInstruct planning → multi-step trajectories
├── recovery.py              # Synthetic-only (real failure data is rare). Creative failure modes.
├── code.py                  # The Stack (Python), CodeAlpaca, BigCode → code tool calls
├── research.py              # FineWeb, UltraChat research → search→fetch→synthesize
├── run_all.py               # Orchestrates all 5, outputs summary
└── domain_configs.py        # Per-domain HF dataset list + template config

base.py requirements:
1. download_hf_dataset(name, split, filter_fn, max_samples) → iterable of dicts
   - Streaming=True, handle errors gracefully (skip unavailable datasets)
   - filter_fn: (sample) → bool, applied per-sample
   - Return generator to avoid OOM
2. convert_to_apprentice(raw_samples, domain, format_fn) -> list[dict]
   - format_fn: (raw_sample) → {"messages": [{"role": ..., "content": ...}, ...]}
   - Must inject domain field for router labels
3. combine(hf_samples, synthetic_fn, n_synthetic, adversarial_rate) → list[dict]
   - Merge HF data with synthetic fallback
   - Apply domain-appropriate adversarial rate
4. train_val_split(samples, val_frac=0.05) → (train, val)
5. write_jsonl(samples, path)

Per-domain scripts requirements:
- Each script:
  1. Imports from base.py
  2. Defines domain-specific dataset list (2-3 HF datasets each)
  3. Defines synthetic fallback_fn (uses generate_scaled_synthetic.py patterns)
  4. Defines adversarial_rate specific to domain
  5. Downloads, converts, combines, splits, writes
  6. Reports dataset composition (% real vs synthetic, adversarial rate)

domain_configs.py: structure
DOMAIN_CONFIGS = {
    "tool_caller": {
        "hf_datasets": [
            ("ToolBench/ToolBench", "train", lambda x: True, 5000),
            ("THUDM/AgentInstruct", "train", lambda x: "tool" in str(x).lower(), 3000),
        ],
        "synthetic_count": 20000,
        "adversarial_rate": 0.3,
    },
    "planner": {
        "hf_datasets": [
            ("osunlp/WebArena", "train", lambda x: len(x.get("action", "")) > 10, 3000),
            ("THUDM/AgentInstruct", "train", lambda x: "plan" in str(x).lower(), 3000),
        ],
        "synthetic_count": 25000,
        "adversarial_rate": 0.3,
    },
    "recovery": {
        "hf_datasets": [],  # No good HF data for failure recovery
        "synthetic_count": 30000,
        "adversarial_rate": 0.4,
    },
    "code": {
        "hf_datasets": [
            ("bigcode/the-stack", "train", lambda x: x.get("lang") == "python", 10000),
            ("microsoft/CodeAlpaca", "train", None, 5000),
        ],
        "synthetic_count": 15000,
        "adversarial_rate": 0.3,
    },
    "research": {
        "hf_datasets": [
            ("HuggingFaceFW/fineweb", "train", lambda x: len(x.get("text", "")) > 200, 10000),
            ("HuggingFaceH4/ultrachat_200k", "train_sft", None, 5000),
        ],
        "synthetic_count": 20000,
        "adversarial_rate": 0.3,
    },
}

Key design rules:
- NO Cerebras/OpenAI/API dependency — synthetic uses templates only
- HF datasets must use streaming=True to avoid disk blowup
- Each script must handle dataset download failures gracefully (skip, don't crash)
- Output to data/apprentice_{domain}.jsonl (overwrite old template-only versions)
- Also generate data/router_training.jsonl (sample 200 per domain, shuffled)
- Report: total samples, % real vs synthetic, adversarial count, latent count

Smoke test:
python prepare_data/run_all.py
Expected output:
  [tool_caller] 35000 samples (HF: 8000, synth: 27000, adversarial: 10500)
  [planner]     ...
  [recovery]    ...
  [code]        ...
  [research]    ...
  [router]      1000 samples (200 per domain)
```

After it passes:
- 📝 **Update BUILD_LOG.md** — diary entry: what HF datasets mapped to which domains, yield rates from HF, challenges with format conversion, any datasets that were unavailable
- 💾 **Commit** with message: `prepare_data/ — per-domain HF + synthetic data pipeline (25-50K/domain, no API dependency)`

### 🤖 Prompt 1 — Write apprentice.py
```
Read docs/cognitive_apprenticeship.md (it's in the repo).
Implement apprentice.py — the CognitiveApprentice wrapper.

It must:
1. Wrap a backbone (AgentMind-147M) with a LoRA adapter
2. Support save_adapter() / load_adapter() to serialize just the 2.36M LoRA weights
3. Support reset_adapter() to zero out the LoRA weights (fresh start for each apprentice)
4. Provide train_on_domain(dataset, steps, lr) — trains only the LoRA adapter on domain data
5. Provide distill_into_backbone(backbone, specialist_data, beta=0.5, mtp_weight=0.2) — 
   unfreezes backbone, runs task_loss + KL(backbone || specialist) + MTP aux loss

The key constraint: each apprentice adapter is 2.36M params (rank=16, alpha=32).
Adapters should be savable as standalone .safetensors files.

Implementation plan:
class CognitiveApprentice:
    def __init__(self, backbone, adapter_name, rank=16, alpha=32.0):
        # Clone backbone structure, apply LoRA, freeze backbone weights
        # Store adapter_name, rank, alpha
    
    def save_adapter(self, path):
        # Save only LoRA A/B matrices + metadata (name, rank, alpha, targets)
    
    def load_adapter(self, backbone, path):
        # Load LoRA weights and apply to a fresh backbone
    
    def reset_adapter(self):
        # Re-init A (random normal / sqrt(rank)) and B (zeros)
    
    def train_step(self, batch):
        # Single training step on the LoRA adapter (backbone frozen)
        # Returns loss
    
    def train(self, dataset, steps=500, lr=2e-4, seq_len=256):
        # Training loop: create dataloader, optimizer, scheduler
        # Uses existing train infrastructure (cross_entropy_loss, clip_gradients)
        # Returns trained adapter weights
    
    def distill(self, backbone, specialists, data, beta=0.5, mtp_weight=0.2, steps=50):
        # Unfreeze backbone
        # For each batch:
        #   b_logits, _ = backbone(batch.ids, return_mtp=True)  # MTP active
        #   s_logits = {name: specialist(batch.ids) for name, specialist in specialists}
        #   correct = batch.domain
        #   loss = CE(b_logits, batch.labels) + beta * KL(b_logits, s_logits[correct])
        #          + mtp_weight * MTP_loss(backbone.last_mtp_logits, batch.labels)
        #   grad(loss).update(backbone.params)
        # Freeze backbone

Smoke test after writing:
python -c "
from config import AgentMindConfig
from model.agent_lm import AgentMind
from apprentice import CognitiveApprentice
cfg = AgentMindConfig()
backbone = AgentMind(cfg)
app = CognitiveApprentice(backbone, 'tool_caller')
print('Apprentice created. Adapter name:', app.adapter_name)
print('Trainable params:', sum(p.size for _, p in backbone.trainable_parameters()))
"
Expected: ~2.36M trainable params.
```

After it passes:
- 📝 **Update BUILD_LOG.md** — diary entry: what CognitiveApprentice wraps, why save/load/reset/distill matter, the 2.36M constraint, any MLX surprises
- 💾 **Commit** with message: `apprentice.py — CognitiveApprentice wrapper (LoRA save/load/reset + distill)`

---

### 🤖 Prompt 2 — Write router.py
```
Read docs/cognitive_apprenticeship.md (it's in the repo).
Implement router.py — the TaskRouter for apprentice dispatch.

It must:
1. TaskRouter: a tiny classifier (d_model=1024 → 64 → n_domains = 5)
   - Takes backbone last_hidden state (pooled or [CLS]-like)
   - Outputs logits over domain names
   - ~65K params total
2. select_expert(hidden_state, threshold=0.6) -> domain_name
   - If max softmax < threshold, return "tool_caller" (fallback)
3. train(router_dataset, backbone, steps=200) -> trained router
   - For each sample: backbone forward, extract last_hidden, train router to predict domain

Implementation plan:
class TaskRouter(nn.Module):
    def __init__(self, d_model=1024, hidden=64, n_domains=5, domain_names=None):
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_domains)
        )
        self.domain_names = domain_names or []
    
    def __call__(self, hidden_state):
        # hidden_state: [B, L, d_model] — use LAST position, not mean-pool.
        # Design doc (Inference Flow section) and agent.py both extract the last
        # hidden position for routing: backbone.last_hidden[:, -1, :]. Using mean
        # would create a train/infer mismatch — router trained on mean but queried
        # at inference with the last token. Always use last position.
        if hidden_state.ndim == 3:
            last = hidden_state[:, -1, :]  # [B, d_model]
        else:
            last = hidden_state             # already [B, d_model]
        return self.classifier(last)  # [B, n_domains]
    
    def select_expert(self, hidden_state, threshold=0.6):
        # Pass the full hidden_state tensor — __call__ extracts the last position.
        logits = self(hidden_state)
        probs = mx.softmax(logits, axis=-1)
        if mx.max(probs).item() < threshold:
            return "tool_caller"  # fallback
        return self.domain_names[mx.argmax(logits, axis=-1).item()]

    def train(self, dataset, backbone, tokenizer=None, steps=200, lr=1e-3):
        # dataset: list of {"domain": str, "messages": [...]}
        # For each sample:
        #   1. Tokenize messages
        #   2. backbone.forward_with_state(ids, {}) → backbone.last_hidden
        #      Note: backbone.last_hidden is [B, L, d_model]. Pass the full tensor;
        #      __call__ will extract the last position.
        #   3. router(backbone.last_hidden) → domain_logits
        #   4. CE(domain_logits, domain_label)
        #   5. grad update router only (backbone frozen)

Smoke test:
python -c "
from router import TaskRouter
import mlx.core as mx
router = TaskRouter(d_model=1024, n_domains=5, domain_names=['tool_caller','planner','recovery','code','research'])
hidden = mx.ones((1, 16, 1024))  # [B=1, L=16, d_model=1024]
logits = router(hidden)           # internally uses hidden[:, -1, :]
print('Router output shape:', logits.shape)  # expected: (1, 5)
expert = router.select_expert(hidden)
print('Selected expert:', expert)
print('Router params: ~65K (verify: sum of param sizes)')
"

# Verify last-position extraction is consistent:
# - router.__call__: uses hidden[:, -1, :]
# - agent.py _select_specialist: self.router(hidden_state[:, -1:, :]) — passes [B,1,D]
#   both collapse to the same last token representation. No pooling mismatch.

After it passes:
- 📝 **Update BUILD_LOG.md** — diary entry: why last-position (not mean-pool) for routing, train/infer parity, 65K classifier, fallback threshold rationale
- 💾 **Commit** with message: `router.py — TaskRouter for apprentice dispatch (last-position hidden, 65K params, threshold-based fallback)`
```

---

### 🤖 Prompt 3 — Update lora.py with adapter save/load/reset
```
Read the existing lora.py.
Add three methods to the LoRALinear class or as standalone functions:

1. save_adapter(adapter_name, rank, alpha, target_modules, save_dir) 
   - Iterates model.trainable_parameters()
   - Filters to LoRA A/B weights
   - Saves as MLX .safetensors with metadata keys: lora_rank, lora_alpha, target_modules

2. load_adapter(model, adapter_path)
   - Loads LoRA weights from .safetensors
   - Applies them to a fresh backbone (model must have same architecture)
   - Returns model with loaded adapters

3. reset_adapter(model)
   - Re-initializes all LoRA A matrices (random normal / sqrt(rank))
   - Re-initializes all LoRA B matrices (zeros)
   - Does NOT touch backbone weights

These are needed during the apprenticeship loop:
- save after training each specialist
- load when running agent inference (swap adapters)
- reset between specialists to avoid cross-contamination

Test after writing:
python -c "
from config import AgentMindConfig
from model.agent_lm import AgentMind
from lora import apply_lora, save_adapter, load_adapter, reset_adapter
import tempfile, os

cfg = AgentMindConfig()
model = AgentMind(cfg)
model = apply_lora(model, rank=16, alpha=32.0)

# Save adapter
tmpdir = tempfile.mkdtemp()
save_adapter(model, 'tool_caller', tmpdir)
assert os.path.exists(f'{tmpdir}/tool_caller.safetensors')
print('Save OK')

# Reset and verify
reset_adapter(model)
print('Reset OK')

# Reload
model2 = AgentMind(cfg)
model2 = apply_lora(model2, rank=16, alpha=32.0)
load_adapter(model2, f'{tmpdir}/tool_caller.safetensors')
print('Load OK')
"

After it passes:
- 📝 **Update BUILD_LOG.md** — diary entry: adapter save/load/reset lifecycle, .safetensors format, why reset is needed between specialists
- 💾 **Commit** with message: `lora.py — adapter save/load/reset for apprenticeship lifecycle`
```

---

### 🤖 Prompt 4 — Add load_lora() to model/agent_lm.py
```
Read model/agent_lm.py. Add a load_lora() method to the AgentMind class.

This method is needed at inference time to dynamically swap adapter weights
(2.36M params) on the frozen backbone. It must be fast (<1ms for the swap).

def load_lora(self, adapter_weights: dict):
    '''
    Load LoRA A/B weights from a specialist adapter into the model.
    adapter_weights: dict of {"layer_name.A": mx.array, "layer_name.B": mx.array, ...}
    
    The model already has LoRALinear layers applied from lora.py.
    This just updates the A and B matrices of existing LoRALinear layers.
    '''
    for name, param in adapter_weights.items():
        # Walk the module tree and set the matching parameter
        # e.g., "blocks.0.in_proj.A" -> self.blocks[0].in_proj.A = param
        ...

Test:
python -c "
from config import AgentMindConfig
from model.agent_lm import AgentMind
from lora import apply_lora
cfg = AgentMindConfig()
model = AgentMind(cfg)
model = apply_lora(model)

# Get current adapter weights
adapter_weights = {k: v for k, v in model.trainable_parameters().items() 
                   if not k.startswith('last_')}

# Create new model and load
model2 = AgentMind(cfg)
model2 = apply_lora(model2)
model2.load_lora(adapter_weights)

# Verify weights match
for k in adapter_weights:
    orig = adapter_weights[k]
    loaded = dict(model2.trainable_parameters())[k]
    assert mx.all(mx.equal(orig, loaded)).item(), f'{k} mismatch'
print('load_lora verified: all weights match')
"

After it passes:
- 📝 **Update BUILD_LOG.md** — diary entry: load_lora for fast adapter swapping (<1ms), module tree walking approach, why it's separate from save/load in lora.py
- 💾 **Commit** with message: `agent_lm.py — load_lora() for sub-millisecond adapter swap`
```

---

### 🤖 Prompt 5 — Refactor train.py into reusable functions
```
Read the existing train.py. It currently has a monolithic train() function.
Refactor it into two callable entry points for the training_orchestrator:

1. train_specialist(backbone, domain_dataset, domain_name, steps=500, lr=2e-4, seq_len=256, latent_stage=1) -> adapter_weights
   - Wraps backbone in CognitiveApprentice for the given domain
   - Trains ONLY the LoRA adapter (backbone stays FROZEN throughout)
   - Returns the trained adapter weights (LoRA A/B matrices)
   - Uses existing: cross_entropy_loss, clip_gradients, CosineWarmupScheduler
   - Per-step: latent stage injection via inject_latent_tokens + latent_loss_mask

   ⚠️ MTP DISABLED during specialist training:
   Specialist adapters are 2.36M params — the backbone MTP head (4 × 32K vocab) is larger
   than the adapter. More importantly, MTP runs on the backbone, which is FROZEN here.
   Never set return_mtp=True inside train_specialist. Add an explicit assertion:
       assert not return_mtp_in_specialist, "MTP must not run during specialist training — backbone is frozen"
   MTP only fires during distill_backbone() when the backbone is unfrozen.
   Add a clear comment to this effect in the code.

   Latent stage is passed in from the orchestrator (not derived from step count).
   The orchestrator maps per-round stages explicitly (see training_orchestrator.py).

2. distill_backbone(backbone, specialists, combined_data, beta=0.5, mtp_weight=0.2, steps=50) -> None
   - Unfreezes backbone
   - For each batch: 
     - backbone forward with return_mtp=True  ← MTP ENABLED here (backbone unfrozen)
     - specialist forward for correct domain (backbone frozen for each specialist forward)
     - loss = CE(backbone, labels) + beta * KL(backbone || specialist) + mtp_weight * MTP_loss
       MTP starts after step 20 for stability (warm-up before applying MTP auxiliary loss)
   - Gradient clipping, NaN recovery (same as existing)
   - Freezes backbone after distillation
   - Uses existing: cross_entropy_loss, mtp_loss, clip_gradients, check_finite

Keep the existing monolith train() function as-is for backward compatibility,
but add the two new functions alongside it. Import what's needed:

from model.latent import get_latent_stage, inject_latent_tokens, latent_loss_mask
from model.mtp_head import mtp_loss

Test both functions with a 5-step smoke test:
python -c "
from config import AgentMindConfig
from model.agent_lm import AgentMind
from lora import apply_lora
from init import init_agentmind
from train import train_specialist, distill_backbone, cross_entropy_loss
import mlx.core as mx

cfg = AgentMindConfig()
backbone = AgentMind(cfg)
backbone = init_agentmind(backbone, cfg)

# Test train_specialist with fake data
fake_data = [{'domain': 'tool_caller', 'messages': [
    {'role': 'user', 'content': 'test'},
    {'role': 'assistant', 'content': '<|tool_call|>{\"name\": \"search\"}<|observe|>{\"ok\": true}\nDone.'}
]}]
from data.pipeline import AgentDataset
import sentencepiece as spm
tok = spm.SentencePieceProcessor()
tok.load('agentmind_tok.model')
ds = AgentDataset.__new__(AgentDataset)
ds.samples = fake_data
ds.tok = tok
ds.cfg = cfg

weights = train_specialist(backbone, ds, 'tool_caller', steps=2)
print('train_specialist OK, got', len(weights), 'weight tensors')

# Test distill_backbone
specialists = {'tool_caller': weights}
distill_backbone(backbone, specialists, ds, steps=2)
print('distill_backbone OK')
"

After it passes:
- 📝 **Update BUILD_LOG.md** — diary entry: refactoring train.py into train_specialist + distill_backbone, why monolithic function needed splitting, how latent injection + MTP were wired
- 💾 **Commit** with message: `train.py — refactored into train_specialist() + distill_backbone() for apprenticeship`
```

---

### 🤖 Prompt 6 — Write training_orchestrator.py
```
Read docs/cognitive_apprenticeship.md (it's in the repo).
Read the updated train.py (has train_specialist and distill_backbone).
Read data/pipeline.py (AgentDataset, make_dataloader).

Implement training_orchestrator.py — the round management loop.

It must run the full apprenticeship protocol:

# Per-round latent stage mapping — explicit, from the design doc table:
#   Round 1 (tool_caller):         latent stages 1→2 (basic tool calling, then wrap scratch in boundaries)
#   Round 2 (planner):             latent stages 2→3 (planned trajectories, 50% CoT → latent replacement)
#   Round 3+ (recovery/code/research): latent stage 4 (full latent — CoT removed, only think boundaries)
# This mapping MUST be explicit in the orchestrator — do NOT derive it from global step count.
# Pass latent_stage directly to train_specialist() for each round.

ROUNDS = [
    {
        "domain": "tool_caller",
        "file": "data/apprentice_tool_caller.jsonl",
        "specialist_steps": 500,
        "seq_len": 256,
        "distill_steps": 50,
        "adversarial": 0.3,
        "latent_stage": 1,  # Round 1: start at stage 1, curriculum advances to 2
    },
    {
        "domain": "planner",
        "file": "data/apprentice_planner.jsonl",
        "specialist_steps": 300,
        "seq_len": 512,
        "distill_steps": 50,
        "adversarial": 0.3,
        "latent_stage": 2,  # Round 2: start at stage 2, advances to 3
    },
    {
        "domain": "recovery",
        "file": "data/apprentice_recovery.jsonl",
        "specialist_steps": 300,
        "seq_len": 256,
        "distill_steps": 50,
        "adversarial": 0.4,
        "latent_stage": 4,  # Round 3+: full latent — backbone already understands silent reasoning
    },
    {
        "domain": "code",
        "file": "data/apprentice_code.jsonl",
        "specialist_steps": 300,
        "seq_len": 512,
        "distill_steps": 50,
        "adversarial": 0.3,
        "latent_stage": 4,  # Round 4: full latent
    },
    {
        "domain": "research",
        "file": "data/apprentice_research.jsonl",
        "specialist_steps": 300,
        "seq_len": 1024,
        "distill_steps": 50,
        "adversarial": 0.3,
        "latent_stage": 4,  # Round 5: full latent
    },
]

Then router training using data/router_training.jsonl.

The orchestration:
1. Load backbone, init, LoRA apply
2. For EACH round (rounds 1 through 5 — ALL of them):
   a. Load domain dataset with correct latent_stage from ROUNDS config
   b. train_specialist(backbone, dataset, domain, latent_stage=round_cfg["latent_stage"]) → adapter_weights
      ↳ Backbone stays FROZEN. MTP is OFF. Latent stage from per-round config.
   c. save_adapter(adapter_weights, domain, save_dir)
   d. Load ALL completed specialists so far (including this round's new one)
   e. distill_backbone(backbone, specialists, combined_data)
      ↳ THIS MUST HAPPEN AFTER EVERY SPECIALIST, not just round 1.
      ↳ Design doc explicitly requires: distill after rounds 1, 2, 3, 4, 5.
      ↳ MTP is ON during distillation (backbone unfrozen). MTP starts after distill step 20.
      ↳ After distillation: backbone now understands N domains. Future specialists start richer.
   f. Print round summary (loss, tool_acc, interference)
3. Train router:
   a. Load router_training.jsonl
   b. Create router model
   c. router.train(dataset, backbone, tokenizer=tok)
   d. Save router weights
4. Final export: backbone + all adapters + router

CLI:
python training_orchestrator.py \
  --rounds 1-5 \           # which rounds to run (1=tool_caller, 2=planner, etc.)
  --resume ./checkpoints \ # resume from saved state
  --save-dir ./checkpoints

Test with a mini run (2 steps per specialist, 1 step distill):
python -c "
from training_orchestrator import run_round
from config import AgentMindConfig
from model.agent_lm import AgentMind
from lora import apply_lora
from init import init_agentmind
import tempfile

cfg = AgentMindConfig()
backbone = AgentMind(cfg)
backbone = init_agentmind(backbone, cfg)
backbone = apply_lora(backbone)

result = run_round(backbone, domain='tool_caller', 
                   data_path='data/apprentice_tool_caller.jsonl',
                   specialist_steps=2, distill_steps=1, save_dir=tempfile.mkdtemp())
print('Round complete. Adapter saved at:', result.get('adapter_path'))
print('Distillation loss:', result.get('distill_loss'))
"

After it passes:
- 📝 **Update BUILD_LOG.md** — diary entry: orchestration loop design, per-round config, checkpoint resume strategy, end-to-end flow
- 💾 **Commit** with message: `training_orchestrator.py — round management loop for apprenticeship protocol`
```

---

### 🤖 Prompt 7 — Update eval.py with per-apprentice + interference
```
Read the existing eval.py. Add two functions:

1. evaluate_apprentice(model, adapter_weights, domain_dataset, tok, cfg) -> dict
   - Load adapter into backbone via model.load_lora()
   - Run existing metrics: compute_loss, evaluate_tool_calls, format_adherence
   - Return {"loss": float, "tool_acc": float, "format": dict}

2. test_interference(model, adapters: dict, test_fn, tok, cfg) -> (baselines, interference)
   - For each adapter: load, run test_fn, record baseline
   - For each pair (A, B): load A, test, load B, test, compute diff
   - Return baselines dict + interference dict
   - Interference > 5% triggers warning: "SPECIALIST INTERFERENCE DETECTED"

Test:
python -c "
from eval import evaluate_apprentice, test_interference
from config import AgentMindConfig
from model.agent_lm import AgentMind
from lora import apply_lora
import sentencepiece as spm

cfg = AgentMindConfig()
model = AgentMind(cfg)
tok = spm.SentencePieceProcessor()
tok.load('agentmind_tok.model')

# Quick smoke test
from data.pipeline import AgentDataset
ds = AgentDataset(['data/apprentice_tool_caller.jsonl'], tokenizer=tok, cfg=cfg, split='train')

# Dummy adapter (just use current LoRA weights)
adapters = {'tool_caller': {k: v for k, v in model.trainable_parameters().items() if not k.startswith('last_')}}
result = evaluate_apprentice(model, adapters['tool_caller'], ds, tok, cfg)
print('Apprentice eval:', result)
"

After it passes:
- 📝 **Update BUILD_LOG.md** — diary entry: per-apprentice eval metrics, interference testing, what constitutes >5% interference and why
- 💾 **Commit** with message: `eval.py — per-apprentice evaluation + interference detection`
```

---

### 🤖 Prompt 8 — Rewrite agent.py with router-aware AgentLoop
```
Read docs/cognitive_apprenticeship.md — the "Inference Flow: Router-Aware Agent Loop" section.
Read agent.py (currently empty stub).

Implement the full AgentLoop class with router dispatch + adapter swapping + SSM state persistence.

Key design:
- SSM state (h_states) persists across entire session — NOT reset on specialist switch
- Router runs on backbone hidden state (before specialist bias)
- Fallback to "tool_caller" if router confidence < 0.6
- Adapter swapping is cheap (~9MB, <1ms on Apple Unified Memory)

Implementation:
class AgentLoop:
    def __init__(self, backbone, router, adapters: dict, tok, tools: dict, cfg):
        self.backbone = backbone      # AgentMind-147M
        self.router = router          # TaskRouter instance
        self.adapters = adapters      # {"tool_caller": weights_dict, ...}
        self.tok = tok
        self.tools = tools            # {"tool_name": callable}
        self.cfg = cfg
        self.h_states = {}            # SSM state — persists entire session
        self.active_adapter = None

    def _select_specialist(self, hidden_state):
        # Pass the last-position slice [B, 1, d_model] to the router.
        # router.__call__ handles both [B, L, D] and [B, D] — it always extracts
        # the last position. Passing [:, -1:, :] (shape [B, 1, D]) is consistent
        # with how router.py's __call__ works and matches the design doc (last_hidden).
        # Do NOT use mean-pooling here — that would break the train/infer parity
        # established in router.py (see Prompt 2 reconciliation).
        logits = self.router(hidden_state[:, -1:, :])  # [B, 1, D] → router extracts last pos
        return self.router.select_expert(hidden_state[:, -1:, :], threshold=0.6)

    def _load_adapter(self, name):
        if self.active_adapter != name:
            self.backbone.load_lora(self.adapters[name])
            self.active_adapter = name

    def run(self, user_query, max_tokens=200, temp=0.7, top_p=0.9):
        # 1. Build prompt with system + user + assistant prefix
        # 2. Tokenize
        # 3. Backbone forward for router (with SSM state)
        # 4. Router dispatch
        # 5. Load specialist adapter
        # 6. Generate with forward_with_state + sampling
        # 7. Tool call handling (pause → execute → observe → continue)
        # 8. Return response string

    def sample(logits, temp, top_p):
        # Temperature + top-p nucleus sampling
        # Same as the old design

Wire 3 real tools (from original design):
- web_search(query) — DuckDuckGo lite or requests
- run_python(code) — subprocess with timeout=10s
- read_file(path) — open and read, max 10KB

Test:
python agent.py \
  --backbone ./apprentice-system-4bit/backbone \
  --adapters ./apprentice-system-4bit/adapters \
  --router ./apprentice-system-4bit/router \
  --query "Search arxiv for recent papers on Mamba SSMs and summarize the findings"
```

After it passes:
- 📝 **Update BUILD_LOG.md** — diary entry: AgentLoop with router dispatch, SSM state persistence philosophy (don't reset on specialist switch), live tool wiring
- 💾 **Commit** with message: `agent.py — router-aware AgentLoop with SSM state persistence across specialist switches`

---

## Phase 1 — Round 1: Tool Caller

### 👤 Human Step 1 — Train tool_caller specialist
```bash
cd agentmind
python training_orchestrator.py --rounds 1
```
> Trains the first specialist (tool_caller) for 500 steps at seq_len=256.
> Adversarial rate: 30%. Backbone frozen; only 2.36M LoRA params train.
> Expected: ~15 minutes on M-series MacBook Air.
> Watch for loss < 3.0 by step 200, tool_acc emerging by step 300.
>
> After completion:
> - 📝 **Update BUILD_LOG.md** — training diary: loss curve, convergence behavior, any NaNs or instabilities, tool_acc emergence
> - 💾 **Commit** with message: `round 1 — tool_caller specialist trained (loss=X, tool_acc=X%)`

### 👤 Human Step 2 — Monitor first distillation
```bash
python training_orchestrator.py --rounds 1 --distill-only
```
> Runs 50-step distillation: backbone unfrozen, trained with CE + KL(backbone || tool_caller) + MTP.
> After this, backbone understands tool protocol. Future specialists start from higher base.
>
> After completion:
> - 📝 **Update BUILD_LOG.md** — diary entry: distillation loss vs baseline, KL divergence behavior, backbone capacity observations
> - 💾 **Commit** with message: `round 1 — backbone distilled with tool_caller (distill_loss=X, beta=0.5)`

---

## Phase 2 — Rounds 2-5: Remaining Specialists

### 👤 Human Step 3 — Train remaining specialists
```bash
# Round 2: Planner (300 steps, seq_len=512)
python training_orchestrator.py --rounds 2

# Round 3: Recovery (300 steps, seq_len=256, 40% adversarial)
python training_orchestrator.py --rounds 3

# Round 4: Code (300 steps, seq_len=512)
python training_orchestrator.py --rounds 4

# Round 5: Research (300 steps, seq_len=1024)
python training_orchestrator.py --rounds 5
```
> Each specialist starts with N-1 domains already embedded in backbone.
> Fewer steps needed (300 vs 500) because tool protocol is already learned.
> Each followed by automatic 50-step distillation with MTP.
> Total: ~70 minutes across all 5 rounds + distillations.
>
> After completion:
> - 📝 **Update BUILD_LOG.md** — per-round diary: loss, tool_acc, interference measurements, any cross-domain bleed
> - 💾 **Commit** per round with message: `round N — <domain> specialist (loss=X, tool_acc=X%)`

---

## Phase 3 — Router Training

### 👤 Human Step 4 — Train the router
```bash
python training_orchestrator.py --train-router
```
> Collects backbone hidden states for 1500 samples (300 per domain).
> Trains 65K-param router classifier for 200 steps.
> Verify accuracy > 85% on holdout set.
> Fallback to "tool_caller" when confidence < 0.6.
>
> After completion:
> - 📝 **Update BUILD_LOG.md** — diary entry: router accuracy, confusion matrix (which domains get confused), fallback rate
> - 💾 **Commit** with message: `router trained — accuracy=X% on holdout, fallback threshold=0.6`

---

## Phase 4 — Export + Inference

### 🤖 Prompt 9 — Write export_apprentice.py
```
Read the "Export Format: Multi-Adapter GGUF" section in docs/cognitive_apprenticeship.md.
Write export_apprentice.py.

It must export three artifacts:
1. Backbone: weights + config.json + tokenizer.model
2. Adapters: one .safetensors per specialist
3. Router: weights + router_config.json

CLI:
python export_apprentice.py \
  --backbone ./checkpoints/round_5_backbone \
  --adapters ./checkpoints/round_5_adapters \
  --out ./apprentice-system-4bit \
  --bits 4

Test:
python -c "
from export_apprentice import export_system
export_system(
    backbone_path='./checkpoints/round_5_backbone',
    adapters_dir='./checkpoints/round_5_adapters',
    output_dir='./apprentice-system-4bit',
    bits=4
)
# Verify files exist
import os
for f in ['backbone.safetensors', 'config.json', 'tokenizer.model']:
    assert os.path.exists(f'./apprentice-system-4bit/{f}')
for domain in ['tool_caller', 'planner', 'recovery', 'code', 'research']:
    assert os.path.exists(f'./apprentice-system-4bit/adapters/{domain}.safetensors')
assert os.path.exists('./apprentice-system-4bit/router.safetensors')
print('Export verified: all 3 artifacts present')
"

After it passes:
- 📝 **Update BUILD_LOG.md** — diary entry: export format rationale (backbone + adapters, not monolithic), why multi-adapter GGUF, 4-bit quantization, inference setup
- 💾 **Commit** with message: `export_apprentice.py — multi-adapter export (backbone + 5 specialists + router)`
```

### 👤 Human Step 5 — Run end-to-end agent test
```bash
python agent.py \
  --backbone ./apprentice-system-4bit/backbone \
  --adapters ./apprentice-system-4bit/adapters \
  --router ./apprentice-system-4bit/router \
  --query "Search arxiv for recent Mamba SSM papers and summarize the key findings"
```
Expected behavior:
1. Backbone encodes query
2. Router selects "research" specialist
3. Specialist generates tool call: search_arxiv
4. Result observed
5. Summary generated
6. SSM state carries context across all turns

After completion:
- 📝 **Update BUILD_LOG.md** — diary entry: end-to-end results, router dispatch quality, generation quality, tool call success rate, any failures
- 💾 **Commit** with message: `end-to-end agent test — router=X%, tool_calls=X%, summary: <brief assessment>`

---

## Debug Prompts

### NaN loss during specialist training
```
My tool_caller specialist training is producing NaN loss from step 1.
The backbone was randomly initialized.
Check if the LoRA A matrix init (random normal / sqrt(rank)) is numerically stable.
Also check if the cross_entropy_loss handles edge cases correctly (all -100 mask).
```

### Router accuracy below 50%
```
My TaskRouter is only getting ~40% accuracy despite training for 200 steps.
The backbone hidden states might not differentiate between domains yet.
Check:
1. Are the domain datasets truly distinct? (Compare token distributions)
2. Is the backbone trained enough? (Early rounds → weak hidden state signal)
3. Is the router's hidden dimension (64) too small? Try 128.
```

### Specialist interference > 10%
```
After training 3 specialists, test_interference shows >10% degradation.
Means adapters are competing for backbone capacity.
Mitigations to try in order:
1. Reduce LoRA rank from 16 to 8
2. Increase distillation steps from 50 to 100
3. Add interference penalty to distillation loss: 
   loss = task_loss + beta * KL + mtp_aux + gamma * sum(CE(backbone, specialist_B) for B != correct)
```

### Agent inference returns wrong specialist
```
My agent always selects "tool_caller" regardless of the query.
Check:
1. Is the router loaded correctly from the exported .safetensors?
2. Does the backbone produce different hidden states for different domains?
3. Is the fallback threshold (0.6) too high? Try 0.3.
4. Run: python -c "from agent import diagnose_router; diagnose_router(backbone, router, test_queries)"
```
---

## Summary — What You Do vs Claude Code

| Step | Who | Log + Commit |
|---|---|---|---|
| Prepare per-domain HF + synthetic data (25-50K/domain) | 🤖 Claude Code | ✅ Required |
| Write apprentice.py (CognitiveApprentice) | 🤖 Claude Code | ✅ Required |
| Write router.py (TaskRouter) | 🤖 Claude Code | ✅ Required |
| Update lora.py (save/load/reset adapter) | 🤖 Claude Code | ✅ Required |
| Add load_lora() to AgentMind | 🤖 Claude Code | ✅ Required |
| Refactor train.py (train_specialist + distill_backbone) | 🤖 Claude Code | ✅ Required |
| Write training_orchestrator.py | 🤖 Claude Code | ✅ Required |
| Update eval.py (per-apprentice, interference) | 🤖 Claude Code | ✅ Required |
| Rewrite agent.py (router-aware AgentLoop) | 🤖 Claude Code | ✅ Required |
| Write export_apprentice.py | 🤖 Claude Code | ✅ Required |
| Start Round 1 training (tool_caller) | 👤 You | ✅ Record results |
| Monitor first distillation | 👤 You | ✅ Record results |
| Run Rounds 2-5 sequentially | 👤 You | ✅ Per-round log |
| Train router | 👤 You | ✅ Record accuracy |
| Run end-to-end agent test | 👤 You | ✅ Record assessment |
| Debug any issues | 🤖 Claude Code | ✅ As needed |
