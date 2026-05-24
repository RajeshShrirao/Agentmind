# AgentMind — Cognitive Apprenticeship Build Sequence
> Build the apprenticeship architecture step by step on Qwen2.5-0.5B backbone.
> Old Mamba architecture archived per PIVOT_PLAN — apprenticeship layer is new.
> 🤖 = Claude Code handles it | 👤 = You do it manually
>
> **Convention**: Every prompt ends with a log + commit step.
> 📝 Update BUILD_LOG.md with a diary entry (challenges, decisions, results).
> 💾 Commit so every change is tracked in git history.

## Strategic Post-Pivot Analysis

This pivot (custom backbone → 2.5-0.5B) is the first time the project feels strategically grounded instead of romantically ambitious.

**What this pivot gets right:** Removes the "dead substrate" problem — Qwen already has syntax, world knowledge, reasoning traces. Preserves the real innovation (specialists, distillation, router, adversarial traces, apprenticeship). Qwen2.5-0.5B is the correct scale.

**Three traps to avoid:**
1. **Distillation too early** — Prove specialization first. Phase A: Qwen backbone + single tool-caller LoRA only. No router, no distillation.
2. **Too many specialists** — Start with ONLY `tool_caller`. Tool calling is the foundation of all agency.
3. **Router complexity too early** — Manual adapter selection > learned routing until specialists visibly diverge.

**Recommended build order:** Phase A (tool-caller LoRA + tool loop) → Phase B (adversarial robustness) → Phase C (planner, router, distillation).

**Note on data:** You already have 100K+ synthetic data from `generate_scaled_synthetic.py` + `augmentation/` (see `docs/synthetic_data_strategy.md`). The Phase -1 HF dataset section below was written before this pipeline existed — consider it superseded. Skip to training.

---

## ✅ Available Assets

| Component | Status | Notes |
|---|---|---|
| Synthetic data pipeline (3-phase: seeds → augment → 8 expansion layers) | ✅ Done | `generate_scaled_synthetic.py` + `augmentation/` |
| 100K+ tool_caller dataset, per-domain datasets, router data | ✅ Done | `data/apprentice_*.jsonl`, `data/router_training.jsonl` |
| Cognitive apprenticeship design doc | ✅ Done | `docs/cognitive_apprenticeship.md` |
| MLX LoRA utilities (apply_lora, save/load/reset adapter) | ✅ Existing | `lora.py` — update target layer names for Qwen |
| Training utilities (cross_entropy_loss, clip_gradients, NaN recovery) | ✅ Existing | `training_utils.py` — no arch dependency |
| Scheduler (CosineWarmupScheduler) | ✅ Existing | `scheduler.py` — no arch dependency |
| Decode utilities (tool call validation) | ✅ Existing | `decode.py` — text-only, no arch dependency |
| Old Mamba architecture | 🗄️ Archived | Files deleted per PIVOT_PLAN, not in critical path |

---

## Phase -1: Data — Already Complete

> You already have 100K+ synthetic data from `generate_scaled_synthetic.py` + `augmentation/`. See `docs/synthetic_data_strategy.md` for quality metrics (103K samples, 0 validation errors, 14/14 tools, 46% adversarial). No HF dataset downloading needed. Skip directly to Phase A.

### 🤖 Prompt 1 — Write apprentice.py
```
Read docs/cognitive_apprenticeship.md (it's in the repo).
Read the existing lora.py (LoRALinear, apply_lora).
Read the existing training_utils.py (cross_entropy_loss, GradientAccumulator).

Implement apprentice.py — the CognitiveApprentice wrapper for Qwen2.5-0.5B.

The backbone is loaded via mlx_lm.load(), NOT a custom model class.
Qwen2.5-0.5B has d_model=512, 24 layers, 151K vocab. LoRA rank=16 gives ~6M trainable params.

It must:
1. Wrap a Qwen backbone (from mlx_lm.load()) with LoRA adapters on Qwen's target modules:
   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
2. Support save_adapter() / load_adapter() to serialize just the LoRA weights (~9MB as .safetensors)
3. Support reset_adapter() to zero out LoRA weights (fresh start for each specialist)
4. Provide train_on_domain(dataset, steps, lr) — trains ONLY LoRA, backbone frozen
5. No MTP, no latent modules — those do not exist in this architecture

Implementation plan:
class CognitiveApprentice:
    def __init__(self, backbone, tokenizer, adapter_name, rank=16, alpha=32.0):
        # backbone is already loaded via mlx_lm.load()
        # apply_lora(backbone, rank=16, alpha=32.0, targets=[q_proj, k_proj, ...])
        # Freeze backbone weights, only LoRA A/B trainable
    
    def save_adapter(self, path):
        # Save LoRA A/B matrices via mlx.save_safetensors()
        # Include metadata: backbone_id, lora_rank, lora_alpha, target_modules
    
    def load_adapter(self, backbone, path):
        # Load LoRA weights from .safetensors, apply to backbone
    
    def reset_adapter(self):
        # Re-init A (random normal / sqrt(rank)) and B (zeros)
        # Keep backbone weights untouched
    
    def train(self, dataset, steps=500, lr=2e-4, seq_len=256):
        # Training loop using mlx.optimizers.AdamW
        # Use existing cross_entropy_loss from training_utils.py
        # logits, _ = backbone(input_ids)  # standard mlx_lm forward
        # loss = cross_entropy_loss(logits, labels, ignore_index=-100)
        # Returns trained adapter weights dict

Smoke test:
python -c "
from mlx_lm import load as load_model
from apprentice import CognitiveApprentice
model, tokenizer = load_model('Qwen/Qwen2.5-0.5B')
app = CognitiveApprentice(model, tokenizer, 'tool_caller')
print('Apprentice created. Adapter name:', app.adapter_name)
print('Trainable params:', sum(p.size for _, p in model.trainable_parameters()))
"
Expected: ~6M trainable params.
```

After it passes:
- 📝 **Update BUILD_LOG.md** — diary entry: Qwen LoRA wrapper, ~6M params, save/load/reset lifecycle, no MTP
- 💾 **Commit** with message: `apprentice.py — CognitiveApprentice for Qwen2.5-0.5B (LoRA save/load/reset)`

---

### 🤖 Prompt 2 — Write router.py (used in Phase C only)
```
Read docs/cognitive_apprenticeship.md (it's in the repo).
Read PIVOT_PLAN.md section "7. router.py — Use backbone's last hidden state".

Implement router.py — the TaskRouter for apprentice dispatch.

Key architectural note: The backbone is Qwen2.5-0.5B (d_model=512) loaded via
mlx_lm.load(). Hidden states come from model.model (the inner transformer),
not from a custom forward_with_state().

It must:
1. TaskRouter: a tiny classifier (d_model=512 → 64 → n_domains = 5)
   - Takes last-position hidden state from Qwen's inner transformer
   - Outputs logits over domain names (~33K params)
2. select_expert(hidden_state, threshold=0.6) -> domain_name
   - If max softmax < threshold, return "tool_caller" (fallback)
3. train(router_dataset, backbone, tokenizer, steps=200) -> trained router
   - For each sample: tokenize, forward through model.model (inner transformer),
     extract last hidden position, train router classifier

Implementation plan:
class TaskRouter(nn.Module):
    def __init__(self, d_model=512, hidden=64, n_domains=5, domain_names=None):
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_domains)
        )
        self.domain_names = domain_names or []
    
    def __call__(self, hidden_state):
        # hidden_state: [B, L, d_model] — use LAST position
        if hidden_state.ndim == 3:
            last = hidden_state[:, -1, :]  # [B, d_model]
        else:
            last = hidden_state
        return self.classifier(last)  # [B, n_domains]
    
    def select_expert(self, hidden_state, threshold=0.6):
        logits = self(hidden_state)
        probs = mx.softmax(logits, axis=-1)
        if mx.max(probs).item() < threshold:
            return "tool_caller"
        return self.domain_names[mx.argmax(logits, axis=-1).item()]

    def train(self, dataset, backbone, tokenizer, steps=200, lr=1e-3):
        # For each sample:
        #   1. Tokenize messages via tokenizer
        #   2. Forward through model.model (inner transformer, no lm_head)
        #      last_hidden = model.model(input_ids)[:, -1, :]
        #   3. router(last_hidden) → domain_logits
        #   4. CE(domain_logits, domain_label), grad update router only

Smoke test:
python -c "
from router import TaskRouter
import mlx.core as mx
router = TaskRouter(d_model=512, n_domains=5,
    domain_names=['tool_caller','planner','recovery','code','research'])
hidden = mx.ones((1, 16, 512))
logits = router(hidden)
print('Router output shape:', logits.shape)  # (1, 5)
expert = router.select_expert(hidden)
print('Selected expert:', expert)
"

After it passes:
- 📝 **Update BUILD_LOG.md** — diary entry: d_model=512 for Qwen, hidden from model.model (not forward_with_state), ~33K params
- 💾 **Commit** with message: `router.py — TaskRouter for Qwen (last-position hidden, d_model=512, Phase C)`
```

---

### 🤖 Prompt 3 — Update lora.py with Qwen target layers + adapter save/load/reset
```
Read the existing lora.py.
Read PIVOT_PLAN.md section "2. lora.py — Target Qwen layer names instead of AgentMind".

Update target_modules default from old AgentMind layers to Qwen2.5 layers:
OLD: ["in_proj", "out_proj", "o_proj", "q_proj", "v_proj", "lm_head"]
NEW: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

These are the standard SwiGLU transformer Linear layers in Qwen2.5.
The LoRALinear class itself doesn't change — just the target names in apply_lora().

Add three methods (as standalone functions using mlx):

1. save_adapter(model, adapter_name, save_dir, rank=16, alpha=32.0, target_modules=None)
   - Iterates model.trainable_parameters(), filters to LoRA A/B weights
   - Saves as MLX .safetensors with metadata keys

2. load_adapter(model, adapter_path)
   - Loads LoRA weights from .safetensors
   - Applies to a Qwen backbone (same architecture assumed)

3. reset_adapter(model)
   - Re-init LoRA A (random normal / sqrt(rank)) and B (zeros)
   - Does NOT touch backbone weights

Test after writing:
python -c "
from mlx_lm import load as load_model
from lora import apply_lora, save_adapter, load_adapter, reset_adapter
import tempfile, os

model, tokenizer = load_model('Qwen/Qwen2.5-0.5B')
model = apply_lora(model, rank=16, alpha=32.0,
    targets=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])

tmpdir = tempfile.mkdtemp()
save_adapter(model, 'tool_caller', tmpdir)
assert os.path.exists(f'{tmpdir}/tool_caller.safetensors')
print('Save OK')

reset_adapter(model)
print('Reset OK')

model2, _ = load_model('Qwen/Qwen2.5-0.5B')
model2 = apply_lora(model2, rank=16, alpha=32.0,
    targets=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
load_adapter(model2, f'{tmpdir}/tool_caller.safetensors')
print('Load OK')
"

After it passes:
- 📝 **Update BUILD_LOG.md** — diary entry: Qwen target modules, adapter lifecycle, .safetensors format
- 💾 **Commit** with message: `lora.py — Qwen target layers + adapter save/load/reset`
```

---

### 🤖 Prompt 4 — Add load_lora() to lora.py (replaces old model/agent_lm.py)
```
NOTE: model/agent_lm.py is DELETED per PIVOT_PLAN. There is no AgentMind class anymore.
The load_lora functionality should be added to lora.py as a standalone function.

Read lora.py. Add a load_lora() function that works with any MLX model that has
LoRALinear layers applied.

This is needed at inference time to dynamically swap adapter weights (~9MB)
on the frozen Qwen backbone. Must be fast (<1ms on Apple Unified Memory).

def load_lora(model, adapter_weights: dict):
    '''
    Load LoRA A/B weights into an MLX model with LoRALinear layers.
    adapter_weights: dict of {"layer_name.A": mx.array, "layer_name.B": ...}
    
    The model already has LoRALinear layers applied from apply_lora().
    This just updates A and B matrices of existing LoRALinear layers.
    Works with Qwen2.5 or any MLX model using the same target module names.
    '''
    for name, param in adapter_weights.items():
        # Walk model module tree via dotted path
        # e.g., "model.layers.0.self_attn.q_proj.lora_a" -> find by name
        ...

Test:
python -c "
from mlx_lm import load as load_model
from lora import apply_lora, load_lora
import mlx.core as mx

model, tokenizer = load_model('Qwen/Qwen2.5-0.5B')
model = apply_lora(model, targets=['q_proj', 'k_proj', 'v_proj', 'o_proj',
    'gate_proj', 'up_proj', 'down_proj'])

# Get current adapter weights
adapter_weights = {k: v for k, v in model.trainable_parameters().items()}

# Create new model and load
model2, _ = load_model('Qwen/Qwen2.5-0.5B')
model2 = apply_lora(model2, targets=['q_proj', 'k_proj', 'v_proj', 'o_proj',
    'gate_proj', 'up_proj', 'down_proj'])
load_lora(model2, adapter_weights)

# Verify weights match
for k in adapter_weights:
    orig = adapter_weights[k]
    loaded = dict(model2.trainable_parameters())[k]
    assert mx.all(mx.equal(orig, loaded)).item(), f'{k} mismatch'
print('load_lora verified: all weights match')
"

After it passes:
- 📝 **Update BUILD_LOG.md** — diary entry: load_lora as standalone function, works with any MLX LoRA model, sub-ms swap
- 💾 **Commit** with message: `lora.py — load_lora() for fast adapter swapping on Qwen`
```

---

### 🤖 Prompt 5 — Refactor train.py into reusable functions (Qwen architecture)
```
Read the existing train.py. It currently has a monolithic train() function.
Read PIVOT_PLAN.md section "4. train.py — Remove MTP, simplify model creation".
Read training_utils.py (cross_entropy_loss, clip_gradients, etc.).

Refactor into two callable entry points. IMPORTANT: model/agent_lm.py,
model/latent.py, model/mtp_head.py, init.py are ALL DELETED. The backbone
is loaded via mlx_lm.load(), not constructed from config.

1. train_specialist(backbone, tokenizer, domain_dataset, domain_name,
                    steps=500, lr=2e-4, seq_len=256) -> adapter_weights
   - backbone is loaded via mlx_lm.load(), already has LoRA applied
   - Trains ONLY LoRA adapter (backbone stays FROZEN)
   - Uses mlx.optimizers.AdamW, CosineWarmupScheduler, cross_entropy_loss
   - No MTP (modules don't exist), no latent stage injection (modules don't exist)
   - Tokenization uses tokenizer.apply_chat_template() with assistant mask

2. distill_backbone(backbone, specialists, combined_data, beta=0.5, steps=50) -> None
   - Unfreezes backbone
   - loss = CE(backbone, labels) + beta * KL(backbone, specialist_model)
   - No MTP loss (mtp_head.py doesn't exist)
   - Gradient clipping, NaN recovery, freeze backbone after

Keep the old train() for backward compat but add the two new functions:

# Training utils already exist — use them:
from training_utils import cross_entropy_loss, clip_gradients, check_finite
from scheduler import CosineWarmupScheduler

Test:
python -c "
from mlx_lm import load as load_model
from lora import apply_lora
from train import train_specialist, distill_backbone
from data.pipeline import AgentDataset

model, tokenizer = load_model('Qwen/Qwen2.5-0.5B')
model = apply_lora(model, targets=['q_proj','k_proj','v_proj','o_proj',
    'gate_proj','up_proj','down_proj'])

fake_data = [{'domain': 'tool_caller', 'messages': [
    {'role': 'user', 'content': 'test'},
    {'role': 'assistant', 'content': '<|tool_call|>{\"name\": \"search\"}<|observe|>{\"ok\": true}\nDone.'}
]}]
ds = AgentDataset(fake_data, tokenizer=tokenizer)

weights = train_specialist(model, tokenizer, ds, 'tool_caller', steps=2)
print('train_specialist OK, got', len(weights), 'weight tensors')
"

After it passes:
- 📝 **Update BUILD_LOG.md** — diary entry: Qwen-based training, no MTP/latent, mlx_lm.load() model creation
- 💾 **Commit** with message: `train.py — refactored for Qwen (no MTP, no latent, mlx_lm.load())`
```

---

### 🤖 Prompt 6 — Write training_orchestrator.py (Qwen architecture)
```
Read docs/cognitive_apprenticeship.md (it's in the repo).
Read PIVOT_PLAN.md section "5. training_orchestrator.py".
Read train.py (train_specialist, distill_backbone).
Read data/pipeline.py (AgentDataset).

Implement training_orchestrator.py — the round management loop.

KEY DIFFERENCES FROM OLD ARCHITECTURE:
- Backbone is loaded via mlx_lm.load('Qwen/Qwen2.5-0.5B'), NOT AgentMind()
- No init_agentmind(), no AgentMindConfig — those files are deleted
- No latent_stage in round config — model/latent.py is deleted
- Distillation is opt-in (--distill-only flag), default is no distillation

ROUNDS = [
    {"domain": "tool_caller", "file": "data/apprentice_tool_caller.jsonl",
     "specialist_steps": 500, "seq_len": 256, "adversarial": 0.3},
    {"domain": "planner", "file": "data/apprentice_planner.jsonl",
     "specialist_steps": 300, "seq_len": 512, "adversarial": 0.3},
    {"domain": "recovery", "file": "data/apprentice_recovery.jsonl",
     "specialist_steps": 300, "seq_len": 256, "adversarial": 0.4},
    {"domain": "code", "file": "data/apprentice_code.jsonl",
     "specialist_steps": 300, "seq_len": 512, "adversarial": 0.3},
    {"domain": "research", "file": "data/apprentice_research.jsonl",
     "specialist_steps": 300, "seq_len": 1024, "adversarial": 0.3},
]

Orchestration:
1. Load backbone via mlx_lm.load(), apply_lora() with Qwen target modules
2. For each round:
   a. train_specialist(backbone, tokenizer, dataset, domain) → adapter_weights
   b. save_adapter(adapter_weights, domain, save_dir)
   c. If --distill-only: distill_backbone(backbone, specialists, combined_data)
   d. Print round summary
3. Router training (--train-router): train router on cached hidden states
4. Export: backbone + adapters + router

CLI:
python training_orchestrator.py \
  --rounds 1-5 --no-distill --save-dir ./checkpoints
python training_orchestrator.py --rounds 1-5 --distill-only   # Phase C
python training_orchestrator.py --train-router                  # Phase C

Test:
python -c "
from mlx_lm import load as load_model
from lora import apply_lora
from training_orchestrator import run_round
import tempfile

model, tokenizer = load_model('Qwen/Qwen2.5-0.5B')
model = apply_lora(model)
result = run_round(model, tokenizer, domain='tool_caller',
    data_path='data/apprentice_tool_caller.jsonl',
    specialist_steps=2, save_dir=tempfile.mkdtemp())
print('Adapter saved at:', result.get('adapter_path'))
"

After it passes:
- 📝 **Update BUILD_LOG.md** — diary entry: Qwen orchestration, mlx_lm.load(), no latent stage
- 💾 **Commit** with message: `training_orchestrator.py — Qwen round management (distillation opt-in)`
```

---

### 🤖 Prompt 7 — Update eval.py with per-apprentice + interference (Qwen)
```
Read the existing eval.py. Add two functions:

1. evaluate_apprentice(model, tokenizer, adapter_weights, domain_dataset) -> dict
   - Load adapter via load_lora()
   - Run: compute_loss on held-out data, evaluate_tool_calls (parse <|tool_call|> JSON),
     format_adherence (check <|tool_call|>...<|observe|> structure)
   - Return {"loss": float, "tool_acc": float, "format": dict}

2. test_interference(model, tokenizer, adapters: dict) -> (baselines, interference)
   - For each adapter: load, run test_fn, record baseline
   - For each pair: load A→test→load B→test, compute diff
   - Interference > 5% triggers warning

Note: model is loaded via mlx_lm.load(), tokenizer is Qwen's AutoTokenizer.
No AgentMindConfig, no sentencepiece, no agentmind_tok.model.

Test:
python -c "
from mlx_lm import load as load_model
from lora import apply_lora, load_lora
from eval import evaluate_apprentice, test_interference

model, tokenizer = load_model('Qwen/Qwen2.5-0.5B')
model = apply_lora(model)

from data.pipeline import AgentDataset
ds = AgentDataset('data/apprentice_tool_caller.jsonl', tokenizer=tokenizer)

adapter_weights = dict(model.trainable_parameters())
result = evaluate_apprentice(model, tokenizer, adapter_weights, ds)
print('Apprentice eval:', result)
"

After it passes:
- 📝 **Update BUILD_LOG.md** — diary entry: Qwen eval, tool call accuracy, interference detection
- 💾 **Commit** with message: `eval.py — per-apprentice evaluation for Qwen + interference detection`
```

---

### 🤖 Prompt 8 — Write agent.py with manual adapter selection (Phase A), optional router (Phase C)
```
Read docs/cognitive_apprenticeship.md.
Read agent.py (currently empty stub).

Implement AgentLoop that supports two modes:

Mode 1 (Phase A/B — manual adapter selection):
  python agent.py --backbone Qwen/Qwen2.5-0.5B --adapter ./path.safetensors --query "..."
  - Single fixed adapter loaded at startup, no router involved
  - KV cache persists across turns within the same session

Mode 2 (Phase C — router dispatch):
  python agent.py --backbone Qwen/Qwen2.5-0.5B --adapters ./dir/ --router ./path --query "..."
  - Router selects specialist per turn based on last hidden state
  - Falls back to "tool_caller" if confidence < 0.6

Key design:
- KV cache persists across entire session — NOT reset on specialist switch
- Adapter swapping is fast (~9MB, <1ms on Apple Unified Memory)
- In Phase A/B mode: no router instantiated, adapter is fixed

Implementation:
class AgentLoop:
    def __init__(self, model, tokenizer, adapter_path=None, adapters_dir=None, router=None, tools=None):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = []

        if adapter_path:
            # Phase A/B: single fixed adapter, no router
            self.fixed_adapter = load_adapter_weights(adapter_path)
            self.model.load_lora(self.fixed_adapter)
            self.adapter_mode = "fixed"
        elif adapters_dir and router:
            # Phase C: router dispatch
            self.adapters = load_all_adapters(adapters_dir)
            self.router = router
            self.adapter_mode = "routed"
            self.active_adapter = None

    def run(self, user_query, max_tokens=200, temp=0.7):
        # 1. Build prompt, tokenize
        # 2. If routed mode: forward backbone → router → load adapter
        # 3. Generate with KV cache (generate_step)
        # 4. Tool call handling: pause → execute → observe → continue
        # 5. Return response

Wire 3 real tools:
- web_search(query) — DuckDuckGo lite or requests
- run_python(code) — subprocess with timeout=10s
- read_file(path) — open and read, max 10KB

Phase A test (manual adapter):
python agent.py \
  --backbone Qwen/Qwen2.5-0.5B \
  --adapter ./checkpoints/adapters/tool_caller.safetensors \
  --query "Search arxiv for recent papers on Mamba SSMs"

Phase C test (router):
python agent.py \
  --backbone Qwen/Qwen2.5-0.5B \
  --adapters ./checkpoints/adapters \
  --router ./checkpoints/router \
  --query "Search arxiv for recent papers on Mamba SSMs"
```

After it passes:
- 📝 **Update BUILD_LOG.md** — diary entry: AgentLoop modes (fixed vs routed), KV cache persistence, tool wiring
- 💾 **Commit** with message: `agent.py — AgentLoop with manual + routed adapter selection`

---

## Phase A: Tool-Caller Specialist (Proof of Agency)

Goal: can the model reliably do one tool call, observe, and continue. No router, no distillation, no planner.

### 👤 Human Step 1 — Train tool_caller specialist
```bash
cd agentmind
python training_orchestrator.py --rounds 1
```
> Trains tool_caller LoRA for 500 steps at seq_len=256.
> Backbone frozen; LoRA only. No distillation after.
> Watch for loss < 3.0 by step 200, tool_acc emerging by step 300.
>
> After completion:
> - 📝 **Update BUILD_LOG.md** — training diary: loss curve, convergence behavior, tool_acc emergence
> - 💾 **Commit** with message: `Phase A — tool_caller specialist trained (loss=X, tool_acc=X%)`

### 👤 Human Step 2 — Run manual adapter agent test
```bash
python agent.py \
  --backbone Qwen/Qwen2.5-0.5B \
  --adapter ./checkpoints/adapters/tool_caller.safetensors \
  --query "Search arxiv for Mamba SSM papers and summarize the findings"
```
> No router yet — adapter loaded manually via `--adapter`.
> Expected: model emits `<|tool_call|>`, observes result, continues coherently.
>
> After completion:
> - 📝 **Update BUILD_LOG.md** — diary entry: tool call success rate, observation handling, continuation quality
> - 💾 **Commit** with message: `Phase A — tool_caller agent test (tool_call=X%, continuation=pass/fail)`

---

## Phase B: Adversarial Robustness

Goal: tool_caller handles malformed observations, timeouts, retries, partial failures.

### 👤 Human Step 3 — Train adversarial variants
```bash
python training_orchestrator.py --rounds 1 --adversarial-only --adversarial-rate 0.6
```
> Use the existing adversarial augmentation from `augmentation/adversarial_mutator.py` and `observation_mutator.py`.
> Train the same tool_caller adapter on failure-heavy data (60% adversarial rate).
> Tests: malformed JSON responses, timeout signals, rate limiting, partial results, permission errors.
>
> After completion:
> - 📝 **Update BUILD_LOG.md** — diary entry: adversarial training loss, which failure modes it handles, which it struggles with
> - 💾 **Commit** with message: `Phase B — tool_caller adversarial robustness (failure_modes_handled=X/8)`

---

## Phase C: Multi-Specialist + Router + Distillation

Goal: remaining specialists, then learned routing, then distillation.

### 👤 Human Step 4 — Train remaining specialists (no distillation)
```bash
# Round 2: Planner (300 steps, seq_len=512)
python training_orchestrator.py --rounds 2 --no-distill

# Round 3: Recovery (300 steps, seq_len=256)
python training_orchestrator.py --rounds 3 --no-distill

# Round 4: Code (300 steps, seq_len=512)
python training_orchestrator.py --rounds 4 --no-distill

# Round 5: Research (300 steps, seq_len=1024)
python training_orchestrator.py --rounds 5 --no-distill
```
> Each specialist trains independently with `--no-distill`.
> Verify each manually: `python agent.py --adapter ./checkpoints/adapters/{domain}.safetensors --query "..."`.
>
> After completion:
> - 📝 **Update BUILD_LOG.md** — per-round diary: loss, tool_acc, interference measurements
> - 💾 **Commit** per round with message: `Phase C — <domain> specialist trained (loss=X, tool_acc=X%)`

### 👤 Human Step 5 — Train the router
```bash
python training_orchestrator.py --train-router
```
> Only after ALL 5 specialists are trained and independently verified.
> Collects backbone hidden states for 1500 samples (300 per domain).
> Train 65K-param classifier for 200 steps. Verify accuracy > 85%.
> Fallback to "tool_caller" when confidence < 0.6.
>
> After completion:
> - 📝 **Update BUILD_LOG.md** — diary entry: router accuracy, confusion matrix, fallback rate
> - 💾 **Commit** with message: `Phase C — router trained (accuracy=X%, fallback threshold=0.6)`

### 👤 Human Step 6 — Distillation (optional, only if specialists diverge)
```bash
python training_orchestrator.py --rounds 1-5 --distill-only
```
> Only run this AFTER specialists visibly diverge and are independently proven.
> Distillation blends specialist behaviors back into backbone.
> If specialists aren't meaningfully different, skip distillation entirely.
>
> After completion:
> - 📝 **Update BUILD_LOG.md** — diary entry: distillation loss vs baseline, KL divergence, whether it improved anything
> - 💾 **Commit** with message: `Phase C — backbone distilled with N specialists (distill_loss=X)`

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
import os
for f in ['backbone.safetensors', 'config.json', 'tokenizer.model']:
    assert os.path.exists(f'./apprentice-system-4bit/{f}')
for domain in ['tool_caller', 'planner', 'recovery', 'code', 'research']:
    assert os.path.exists(f'./apprentice-system-4bit/adapters/{domain}.safetensors')
assert os.path.exists('./apprentice-system-4bit/router.safetensors')
print('Export verified: all 3 artifacts present')
"

After it passes:
- 📝 **Update BUILD_LOG.md** — diary entry: export format rationale (backbone + adapters, not monolithic)
- 💾 **Commit** with message: `export_apprentice.py — multi-adapter export (backbone + 5 specialists + router)`
```

### 👤 Human Step 7 — Run end-to-end agent test (with router)
```bash
python agent.py \
  --backbone Qwen/Qwen2.5-0.5B \
  --adapters ./checkpoints/adapters \
  --router ./checkpoints/router \
  --query "Search arxiv for recent Mamba SSM papers and summarize the key findings"
```
Expected behavior:
1. Backbone encodes query
2. Router selects specialist
3. Specialist generates tool call
4. Result observed
5. Summary generated

After completion:
- 📝 **Update BUILD_LOG.md** — diary entry: end-to-end results, router dispatch quality, tool call success rate
- 💾 **Commit** with message: `Phase C — end-to-end agent test (router=X%, tool_calls=X%)`

---

## Debug Prompts

### NaN loss during specialist training
```
My tool_caller specialist training is producing NaN loss from step 1.
The backbone is Qwen2.5-0.5B (pretrained, not random).
Check:
1. Is the LoRA init scale correct? (A: normal/sqrt(rank), B: zeros)
2. Does cross_entropy_loss handle all -100 mask correctly?
3. Is the learning rate too high for pretrained weights? Try 1e-4 instead of 2e-4.
4. Are there extreme logits from the pretrained model on new special tokens?
```

### Router accuracy below 50%
```
My TaskRouter (d_model=512) is only getting ~40% accuracy despite training for 200 steps.
Check:
1. Are the domain datasets truly distinct? (Compare token distributions)
2. Does Qwen's backbone produce differentiated hidden states for different domains?
   (Test: run a few samples, check last-hidden cosine similarity across domains)
3. Is the router's hidden dimension (64) too small? Try 128.
4. Make sure you're extracting from model.model (inner transformer), not model output.
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

### Agent inference returns wrong specialist (Phase C only)
```
My agent always selects "tool_caller" regardless of the query.
Check:
1. Is the router loaded correctly from the exported .safetensors?
2. Does Qwen's backbone produce different hidden states for different domains?
   (Run a few samples and check last-hidden cosine similarity)
3. Is the fallback threshold (0.6) too high? Try 0.3.
4. Run: diagnose_router(backbone, router, test_queries)

Note: In Phase A/B, there is no router — adapter is loaded manually via --adapter.
If the wrong adapter loads, check load_lora() for correct weight mapping.
```
---

## Summary — What You Do vs Claude Code

| Step | Who | Log + Commit |
|---|---|---|
| Synthetic data pipeline | ✅ Already done (`generate_scaled_synthetic.py` + `augmentation/`) | N/A |
| Write apprentice.py (CognitiveApprentice) | 🤖 Claude Code | ✅ Required |
| Write router.py (TaskRouter) | 🤖 Claude Code | ✅ Required (Phase C use) |
| Update lora.py (save/load/reset adapter) | 🤖 Claude Code | ✅ Required |
| Refactor train.py (train_specialist + distill_backbone) | 🤖 Claude Code | ✅ Required |
| Write training_orchestrator.py (`--no-distill` default) | 🤖 Claude Code | ✅ Required |
| Update eval.py (per-apprentice, interference) | 🤖 Claude Code | ✅ Required |
| Write agent.py (manual + routed adapter modes) | 🤖 Claude Code | ✅ Required |
| Write export_apprentice.py | 🤖 Claude Code | ✅ Required |
| **Phase A** — Train tool_caller specialist | 👤 You | ✅ Record loss + tool_acc |
| **Phase A** — Manual adapter agent test | 👤 You | ✅ Assess tool call + continuation |
| **Phase B** — Adversarial robustness training | 👤 You | ✅ Record failure modes handled |
| **Phase C** — Train remaining specialists (no distill) | 👤 You | ✅ Per-round log |
| **Phase C** — Train router | 👤 You | ✅ Record accuracy |
| **Phase C** — Distillation (optional) | 👤 You | ✅ Record if improved |
| **Phase C** — End-to-end agent test | 👤 You | ✅ Record assessment |
| Debug any issues | 🤖 Claude Code | ✅ As needed |
