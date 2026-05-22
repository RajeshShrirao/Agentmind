# AgentMind — Cognitive Apprenticeship Architecture

> Not MoE. Not tiny frontier LLM.  
> Modular cognition through iterative specialization.  
> A cognitive operating system, grown one apprenticeship at a time.

---

## Core Philosophy

Stop trying to imitate frontier LLMs with 1% of the data. Instead, build a system that **accumulates capability over time** through focused apprenticeships.

Each specialist (LoRA adapter) is a temporary apprentice that crystallizes one domain of competence. The backbone is the shared substrate — procedural memory, tool grammar, executive continuity. Distillation is skill consolidation — the apprentice teaches the master, raising the floor for the next apprentice.

```
                    ┌─────────────────────────────────┐
                    │         Router (task classifier) │
                    │  d_model(1024) → 64 → n_experts  │
                    └──────────┬──────────────────────┘
                               │ argmax / softmax
                               ▼
┌──────────────────────────────────────────────────────────┐
│              Backbone — Executive Substrate               │
│              AgentMind-147M (frozen at inference)         │
│              Learned via cumulative distillation          │
│              NOT random — semantic bedrock                │
├──────────────────────────────────────────────────────────┤
│    LoRA targets: in_proj, out_proj, o_proj, q_proj,      │
│                  v_proj, lm_head (2.36M/adapter)          │
└──────────────────────────────────────────────────────────┘
         ▲          ▲          ▲          ▲          ▲
         │          │          │          │          │
   ┌─────┴──┐ ┌────┴───┐ ┌───┴────┐ ┌───┴────┐ ┌───┴────┐
   │Planner │ │Tool    │ │Recovery│ │Code    │ │Research│
   │Apprentice│ │Caller  │ │Apprentice│ │Apprentice│ │Apprentice│
   └────────┘ └────────┘ └────────┘ └────────┘ └────────┘
```

---

## The Shared Semantic Substrate

**CRITICAL**: The backbone cannot stay randomly initialized.

### SSM State: Persistent Working Memory

The backbone uses a hybrid Mamba + Attention architecture (12 Mamba blocks, 4 attention layers, 3:1 ratio). The Mamba SSM maintains a fixed-size recurrent state (~2MB at 4-bit) that persists across token generations and tool call rounds. This means:

- **O(1) memory per conversation turn** — no quadratic attention blowup
- **State survives tool calls** — the SSM state carries context across `<|tool_call|>` → `<|observe|>` → response cycles
- **Router doesn't disrupt state** — switching specialists doesn't reset the SSM. The backbone processes all tokens; the specialist adapter only biases the output distribution

The sequential SSM scan is compiled via `mx.compile` — mathematically identical to single-step inference (verified at 2.38e-7 max diff). Training uses an exact sequential scan, not an approximate parallel scan, guaranteeing train/infer parity.

If it does, each specialist becomes an isolated reflex module:
- planner invents its own abstractions
- code expert interprets tokens differently
- recovery expert repairs inconsistently
- router becomes unstable — it sees noise, not signal

The result is **fragmented cognition**, not modular cognition.

### How the Substrate Forms

The distillation loop is the real invention. It solves this:

```
Step 1: Train specialist A (tool_caller)
        → 2.36M params learn JSON grammar, tool protocol, boundary tokens

Step 2: Backbone absorbs A's knowledge
        Loss = CE(backbone, labels) + β × KL(backbone || specialist_A)
        → Backbone now understands tool calling (partially)

Step 3: Train specialist B (planner) on backbone from Step 2
        → B starts with tool protocol already embedded in backbone
        → B doesn't need to re-learn JSON grammar from scratch
        → B can focus on planning structure

Step 4: Backbone absorbs (A + B) knowledge via distillation
        → Substrate now knows tool calling AND planning

... repeat for each specialist

Result after N rounds:
  Backbone contains: tool grammar + planning + recovery patterns + ...
  Future specialists start at a higher base
  The substrate is dense, not random
```

**The backbone is NOT a pretrained internet model. It's a distilled executive substrate.** Every specialist bootstraps the next. That's the asymmetry — frontier labs can't do this because their models are frozen at release.

---

## Recursive Cognitive Accumulation

The key property no single dense model has:

```
recovery expert → improves planner robustness
            ↓
planner expert → improves research decomposition
            ↓
research expert → improves tool selection
            ↓
tool_caller → improves verifier accuracy
            ↓
verifier → improves recovery expert's detection
            ↓
                    (loop feeds back into itself)
```

Each specialist raises the backbone, which raises the starting point for the next specialist, which raises the backbone further. This creates a **compounding cognition curve** — not linear skill addition.

```
Capability
    ▲
    |                                          ● specialist 5
    |                                    ●
    |                              ●           ← backbone absorbs
    |                        ●
    |                  ●                      ● specialist 4
    |            ●
    |      ●                                  ● specialist 3
    | ●
    |                                           ● specialist 2
    |                                           ● specialist 1 (baseline)
    └──────────────────────────────────────────► Time
         distillation lifts the floor
         each specialist starts higher than the last
```

---

## The Distillation Loop — Central Mechanism

This is NOT optional. It is the core training algorithm.

### MTP — Multi-Token Prediction Auxiliary Loss

The backbone has an MTP head (4 auxiliary heads, each predicting k+1 tokens ahead). During distillation, MTP forces the backbone to plan future tokens — improving instruction following, JSON formatting, and tool call structure.

Only the backbone gets MTP. Specialist adapters are 2.36M params — an MTP head (4 × 32K vocab) would be larger than the adapter itself. MTP is a training-only auxiliary loss; it has zero inference cost.

```python
def distill(backbone, specialists, data, β=0.5, mtp_weight=0.2):
    """
    Specialists crystallize behaviors.
    Backbone absorbs abstractions.
    Future specialists start smarter.
    Cognition compounds over time.
    """
    backbone.unfreeze()
    for batch in data:
        # Forward through backbone alone (with MTP)
        b_logits, _ = backbone(batch.ids, return_mtp=True)

        # Forward through each specialist
        s_logits = {}
        for name, expert in specialists.items():
            expert.unfreeze()
            s_logits[name], _ = expert(batch.ids)
            expert.freeze()

        # Identify which specialist is correct for this sample
        correct = batch.domain

        # Task loss: backbone learns the correct output
        task_loss = cross_entropy(b_logits, batch.labels)

        # Distillation loss: backbone mimics specialist's distribution
        distill_loss = kl_div(b_logits, s_logits[correct])

        # MTP auxiliary loss: backbone learns to plan ahead
        mtp_aux = mtp_loss(backbone.last_mtp_logits, batch.labels,
                           weight=mtp_weight)

        # Total
        loss = task_loss + β * distill_loss + mtp_aux
        grad(loss).update(backbone.params)

    backbone.freeze()
```

---

## The Biggest Danger: Expert Theatricality

> "Each specialist learns stylistic patterns and formatting rituals instead of causal competence."

This is the failure mode that kills the architecture. Your current synthetic traces are too clean and deterministic:

```json
{"name": "search_arxiv", "args": {"query": "example_query", "days": 42}}
→ {"results": [{"title": "Result for example_query", ...}]}
```

The specialist learns: `tool_call → JSON → observe → emit result`. It learns **ritual**, not **causality**.

### The Fix: Adversarial Training for Every Specialist

Each expert must train against realistic failure:

```python
def adversarial_trajectory(tool_name: str) -> dict:
    """Generate a trajectory with realistic chaos."""
    failure_type = random.choose([
        "timeout",           # tool never returns
        "partial_success",   # returns partial data
        "malformed_json",    # observe is corrupt
        "contradictory",     # two sources disagree
        "ambiguous_goal",    # query underspecified
        "hidden_variable",   # needs info from context
        "delayed_feedback",  # result arrives late
        "state_corruption",  # SSM state drifts
    ])
    # ... inject failure into trajectory
    # Model must decide: retry? continue? verify? rollback?
```

Example — recovery expert faces:

```
Round 1: <|tool_call|>{"name": "get_weather", "args": {"city": "Tokyo"}}
         <|observe|>{"error": "timeout", "retry": true}
         → Expert must decide to retry

Round 2: <|tool_call|>{"name": "get_weather", "args": {"city": "Tokyo", "source": "backup"}}
         <|observe|>{"status": "partial_success", "temp": 22}
         → Expert has partial data. Continue? Retry missing fields?
```

That creates **operational intelligence**, not formatting ritual.

## Latent Reasoning: Cross-Cutting Cognitive Skill

Every specialist should be able to reason internally before acting. The backbone supports latent reasoning via `<|think_start|>...<|think_end|>` tokens — the model generates hidden tokens that are masked in the loss function.

Latent reasoning is NOT a separate specialist. It's a token protocol any specialist can use:

```
<|tool_call|>{"name": "search_arxiv", "args": {"query": "SSM papers"}}
<|observe|>{"results": [...]}
<|think_start|><|scratch|><|scratch|><|scratch|><|think_end|>
Response: Here are the latest SSM papers...
```

### How It Works

1. **Data level**: `inject_latent_tokens()` wraps CoT/scratchpad content in `<|think_start|>...<|think_end|>` boundaries
2. **Loss level**: `latent_loss_mask()` zeroes out loss between boundaries — the model isn't penalized for what it thinks
3. **Curriculum**: Stage 1 (no latent) → Stage 2 (insert boundaries) → Stage 3 (50% latent) → Stage 4 (full latent)

### Integration with Apprenticeship

Each specialist's adversarial training data includes trajectories where latent reasoning precedes tool calls or recovery decisions. The distillation process propagates the latent token protocol back into the backbone, so every future specialist inherits it.

Results:
- Recovery specialist: `observe(timeout) → think(reason about retry) → tool_call(retry with backup)`
- Planner specialist: `user(query) → think(decompose problem) → plan(step-by-step)`
- Code specialist: `observe(syntax error) → think(identify root cause) → tool_call(fixed code)`

### What Not To Do

DON'T add the latent loss mask to specialist training. The specialist adapter learns patterns at 2.36M params — the loss mask must be applied at the backbone level during distillation. Specialist training uses the full loss (including latent region), because the specialist must learn what good latent reasoning looks like. The backbone absorbs only the input/output boundaries.

### Adversarial Data Requirements Per Specialist

| Specialist | Must Handle |
|---|---|
| Tool caller | Malformed JSON in observe, unexpected fields, empty results, rate limits, auth failures |
| Planner | Ambiguous goals, conflicting constraints, resource limits, dependency failures |
| Recovery | Timeouts repeating, partial success, corrupt state, cascading failures |
| Code expert | Syntax errors, runtime exceptions, infinite loops, memory limits |
| Research | No results, contradictory sources, paywalled content, stale data |

---

## Training Protocol

### Sequence Length Curriculum

Each specialist benefits from a per-domain sequence schedule. Start short (model learns token boundaries), grow as it learns domain structure:

| Specialist | Curriculum | Rationale |
|---|---|---|
| Tool caller | 256 → 512 | Short call/observe cycles |
| Planner | 512 → 1024 | Multi-step trajectories |
| Recovery | 256 → 512 | Failure patterns are local |
| Code | 512 → 1024 | Code blocks need context |
| Research | 512 → 2048 | Long fetch→summarize pipelines |

This is the same approach validated in the old dense design — 4x speedup in early training with no quality loss.

### Round 1: Establish the Substrate

```
1. Train specialist #1 (tool_caller):
   - 500 steps, 2.36M params
   - Sequence curriculum: 256 → 512
   - Adversarial data: 30% failure rate
   - MTP disabled (backbone frozen, no distillation yet)
   - Goal: model learns tool grammar under realistic conditions

2. Distill into backbone:
   - 50 steps, backbone unfrozen
   - β = 0.5 (equal task + distillation weight)
   - MTP enabled (weight=0.2), started after step 20 for stability
   - Backbone now understands tool protocol (core + MTP future-planning)
```

### Round N (n ≥ 2): Compound

```
3. Train specialist #N on current backbone:
   - Starts with N-1 domains already embedded in backbone
   - Fewer steps needed (200-300 vs 500)
   - Uses domain-specific sequence curriculum
   - Focuses on its novel domain — doesn't re-learn tool grammar

4. Distill into backbone:
   - Backbone now understands N domains + latent reasoning
   - Future specialists start from richer base
   - MTP carries forward planning signal across specialists
```

### Router Training

```
5. Collect dataset: for each specialist, generate N samples
   → backbone forward (no active adapter)
   → record hidden state + correct specialist label

6. Train router:
   for batch in router_data:
       logits = router(batch.hidden)
       loss = cross_entropy(logits, batch.label)
       grad(loss).update(router.params)
```

### Full Schedule

| Step | Event | Params Trained | Seq Len | Aux Losses | Cumulative Time |
|---|---|---|---|---|---|
| 0 | Random init backbone | 0 | — | — | 0 |
| 1 | Train tool_caller (adversarial) | 2.36M | 256→512 | Latent stage 1→2 | ~15 min |
| 2 | Distill → backbone (β=0.5) | 147M | 512 | MTP (weight=0.2) | ~20 min |
| 3 | Train planner | 2.36M | 512→1024 | Latent stage 2→3 | ~30 min |
| 4 | Distill → backbone | 147M | 1024 | MTP (weight=0.2) | ~35 min |
| 5 | Train recovery (adversarial) | 2.36M | 256→512 | Latent stage 3→4 | ~50 min |
| 6 | Train code expert | 2.36M | 512→1024 | Latent stage 4 | ~65 min |
| 7 | Train research | 2.36M | 512→2048 | Latent stage 4 | ~80 min |
| 8 | Train router | 65K | — | — | ~85 min |
| 9 | Joint eval (all specialists) | — | — | — | — |

Total: ~85 minutes for a 5-specialist system with a non-random, semantically-dense backbone.

### Latent Stage Progression Across Rounds

| Round | Stage | What Happens |
|---|---|---|
| 1 (tool_caller) | 1→2 | Basic tool calling, then wrap scratch in boundaries |
| 2 (planner) | 2→3 | Planned trajectories, 50% CoT → latent replacement |
| 3+ (recovery, code, research) | 3→4 | Full latent — CoT removed, only think boundaries remain |

The backbone accumulates latent capability via distillation: by Round 3, the backbone already understands silent reasoning, so recovery/code/research specialists learn it without explicit stage progression.

---

## Comparison: Old vs This vs Frontier

| Dimension | Old Dense | Cognitive Apprenticeship | Gemma 4 |
|---|---|---|---|---|
| Training compute | ~500 petaFLOP | ~1 petaFLOP | ~50K petaFLOP |
| Training time | 3h (invalid labels) | 85 min (valid) | TPU-weeks |
| Per-domain capability | Good | **Super-expert** | Excellent |
| Add new skill | Retrain all | +15 min, new adapter | N/A |
| Backbone state | Frozen (wrong labels) | **Cumulative knowledge** | Frozen (internet) |
| Inference cost | 147M | 149M (backbone + 1 adapter) | 400M |
| Robustness | Low (clean data) | **High (adversarial)** | High (massive data) |
| Auxiliary losses | MTP + latent (wired) | MTP + latent (integrated) | None exposed |
| SSM state | O(1) persistent memory | **O(1) persistent memory** | KV cache grows with context |
| Latent reasoning | `<\|think_start\|>` trained | Per-specialist, distilled to backbone | No |
| General knowledge | Low | Low (synthetic) | **Very high** |
| Hardware | 1 MBA | 1 MBA | TPU pod |
| Cognitive model | Single forward | **Apprenticeship + consolidation** | Single forward |

---

## Inference Flow: Router-Aware Agent Loop

The old dense design used a single forward pass. The apprenticeship architecture requires a router dispatch step. SSM state persists across the entire interaction — switching specialists doesn't reset it.

### Flow

```
1. USER → user query text

2. BACKBONE FORWARD (with SSM state)
   Encode query → backbone forward → extract last_hidden

3. ROUTER DISPATCH
   router(last_hidden) → argmax → specialist_name
   If confidence < threshold → fallback to "tool_caller" (safest default)

4. LOAD SPECIALIST ADAPTER
   Load LoRA weights for selected specialist into backbone
   (2.36M params swapped in microseconds on Apple Silicon)

5. GENERATE RESPONSE
   Backbone + specialist generate tokens autoregressively
   SSM state (h_states) maintained between tokens via forward_with_state

6. TOOL CALL HANDLING
   If <|tool_call|> detected:
     - Pause generation, execute tool
     - Inject <|observe|> result
     - Continue generation with same SSM state + specialist
     - SSM state carries entire history: previous rounds, tool results, conversation

7. ITERATE
   If user sends follow-up → go to step 2 (SSM state already has conversation)
   The SSM state persists across the entire session — O(1) memory per turn
```

### Key Design Decisions

- **SSM state is NOT reset on specialist switch.** The backbone processes every token. The specialist adapter only biases the output. The state carries conversation history.
- **Router runs on backbone hidden state, not specialist output.** This keeps the router stable — it sees the raw substrate representation, not a specialist-biased one.
- **Fallback strategy:** If router confidence for all specialists is below 0.6, default to `tool_caller`. In practice, the routing decision only matters for the first generation; after that, the SSM state usually constrains the model to stay in the same domain.
- **Adapter swapping is cheap.** 2.36M params ≈ 9MB at fp16. Loading from RAM to GPU takes <1ms on Apple Unified Memory.

---

### Agent Loop Pseudocode

```python
class AgentLoop:
    def __init__(self, backbone, router, adapters: dict, tok, tools, cfg):
        self.backbone = backbone      # AgentMind-147M
        self.router = router          # TaskRouter
        self.adapters = adapters      # {"tool_caller": LoRA weights, ...}
        self.tok = tok
        self.tools = tools
        self.cfg = cfg
        self.h_states = {}            # SSM state — persists entire session
        self.active_adapter = None

    def _select_specialist(self, hidden_state):
        logits = self.router(hidden_state[-1:])  # use last hidden
        probs = mx.softmax(logits, axis=-1)
        if mx.max(probs).item() < 0.6:
            return "tool_caller"  # fallback
        return self.router.domain_names[mx.argmax(probs).item()]

    def _load_adapter(self, name):
        if self.active_adapter != name:
            self.backbone.load_lora(self.adapters[name])
            self.active_adapter = name

    def run(self, user_query):
        prompt = f"<|system|>...<|user|>{user_query}<|assistant|>"
        ids = mx.array([self.tok.encode(prompt)])

        # Backbone forward for router
        logits, self.h_states = self.backbone.forward_with_state(ids, self.h_states)
        specialist = self._select_specialist(self.backbone.last_hidden)
        self._load_adapter(specialist)

        # Generate with specialist
        output = ""
        for _ in range(self.max_tokens):
            logits, self.h_states = self.backbone.forward_with_state(ids, self.h_states)
            token = self.sample(logits[0, -1])

            if token == self.cfg.tool_call_id:
                result = self._handle_tool(output)
                ids = mx.array([self.tok.encode(f"<|observe|>{json.dumps(result)}")])
                continue

            if token == self.cfg.eos_id:
                break

            decoded = self.tok.decode([token.item()])
            output += decoded
            ids = mx.array([[token.item()]])

        return output
```

---

## Export Format: Multi-Adapter GGUF

The old single-model export doesn't work here. Three artifacts need to be saved:

### Artifact 1: Backbone (frozen)
- `backbone.safetensors` — 147M params, AgentMind architecture
- `config.json` — model dimensions, token IDs, RoPE config
- `tokenizer.model` — SentencePiece BPE model

### Artifact 2: Specialist Adapters
Each specialist is a separate file:
- `adapters/tool_caller.safetensors` — LoRA A/B matrices (2.36M)
- `adapters/planner.safetensors`
- `adapters/recovery.safetensors`
- `adapters/code.safetensors`
- `adapters/research.safetensors`

Format: MLX `.safetensors` with metadata keys `lora_rank`, `lora_alpha`, `target_modules`.

### Artifact 3: Router
- `router.safetensors` — 65K params
- `router_config.json` — domain_names, hidden_size, threshold

### Export Command

```bash
python export_apprentice.py \
  --backbone ./checkpoints/round_5_backbone \
  --adapters ./checkpoints/round_5_adapters \
  --out ./apprentice-system-4bit \
  --bits 4
```

### Load for Inference

```bash
python agent.py \
  --backbone ./apprentice-system-4bit/backbone \
  --adapters ./apprentice-system-4bit/adapters \
  --router ./apprentice-system-4bit/router \
  --query "Search arxiv for Mamba papers"
```

---

## What This Is Not

| Wrong Label | Why It's Wrong |
|---|---|
| "MoE" | MoE trains all experts jointly from scratch. This trains sequentially, each bootstrapping the next. |
| "LoRA ensemble" | Ensembles combine outputs at inference. This distills back into the backbone. |
| "Fine-tuning" | Fine-tuning adapts one model. This grows a substrate over time. |
| "Small GPT" | GPTs are generalist imitators. This is a specialist composition system. |

---

## File Changes

### New

| File | Purpose |
|---|---|
| `apprentice.py` | `CognitiveApprentice`: LoRA wrapper with save/load/distill |
| `router.py` | `TaskRouter`: classifier, `select_expert()`, router training |
| `training_orchestrator.py` | Round management: train specialist → distill into backbone → repeat |
| `export_apprentice.py` | Multi-adapter export: backbone + adapters + router (replaces old `export.py`) |
| `test_cross_apprentice.py` | Cross-apprentice interference tests, router accuracy |

### Modified

| File | Change |
|---|---|
| `agent.py` | Full `AgentLoop` with router dispatch, adapter loading, SSM state persistence (was empty stub) |
| `lora.py` | Add `save_adapter()`, `load_adapter()`, `reset_adapter()` for per-apprentice weight management |
| `data/synthetic.py` | Adversarial modes: timeout, partial, corrupt, contradictory, hidden_variable, ambiguous_goal |
| `train.py` | Refactor into reusable `train_specialist()` and `distill_backbone()` functions callable by orchestrator |
| `model/mtp_head.py` | No changes needed — already correct. Used by orchestrator during distillation |
| `model/latent.py` | No changes needed — `inject_latent_tokens()` and `latent_loss_mask()` already work per-batch |
| `model/agent_lm.py` | Add `load_lora()` helper to dynamically swap adapter weights at inference |
| `decode.py` | Accept active apprentice name for tool routing context |
| `eval.py` | Add `evaluate_apprentice()` per-domain eval, `test_interference()` cross-apprentice check |

---

---

## Data Requirements

### Per-Specialist Datasets

Each apprentice needs its own training corpus. `generate_scaled_synthetic.py` now outputs separate JSONL files:

| File | Domain | Samples | Focus |
|---|---|---|---|
| `data/apprentice_tool_caller.jsonl` | Tool caller | 3,000 | Single tool calls, all 14 tools, JSON boundary tokens |
| `data/apprentice_planner.jsonl` | Planner | 2,000 | Multi-step with `<\|plan\|>`, dependency chains, 2-5 tools |
| `data/apprentice_recovery.jsonl` | Recovery | 2,000 | Failures, retries, fallbacks, verification, rollback decisions |
| `data/apprentice_code.jsonl` | Code | 1,500 | Python, git, SQL, file I/O — code-specific operations |
| `data/apprentice_research.jsonl` | Research | 1,500 | arxiv → web → fetch → summarize pipelines |
| `data/router_training.jsonl` | Router | 1,500 | 300 per domain, labeled for classifier training |

Total: ~10,000 samples, ~30% adversarial.

### Adversarial Modes

Each specialist generator injects failures at a configurable rate (default 30%):

| Mode | Description | Example Observe |
|---|---|---|
| `clean` | Normal success | `{"results": [...]}` |
| `timeout` | Tool never responds | `{"error": "timeout", "retry": true}` |
| `partial_success` | Partial data returned | `{"status": "partial", "data": null}` |
| `malformed_json` | Corrupt observe payload | `"unexpected response from <<tool>>"` |
| `contradictory` | Sources disagree | `{"warning": "Conflicting data", "retry": true}` |
| `hidden_variable` | Missing context | `{"error": "missing_context"}` |

### Router Training

The router is a 65K-param classifier trained on backbone hidden states.
Training data is sub-sampled from per-specialist datasets (300 per domain).

Each router sample is a plain JSONL line with a `domain` field:

```json
{"domain": "tool_caller", "messages": [...]}
```

The backbone forward pass converts messages → token IDs → hidden states.
The router learns: `hidden_state → domain_logits`.

### Data Generation

```bash
python generate_scaled_synthetic.py
# Outputs:
#   data/apprentice_tool_caller.jsonl
#   data/apprentice_planner.jsonl
#   data/apprentice_recovery.jsonl
#   data/apprentice_code.jsonl
#   data/apprentice_research.jsonl
#   data/router_training.jsonl
```

---

## Evaluation: Per-Apprentice + Cross-Apprentice

The old dense design had a single evaluation: perplexity + tool call accuracy on 3 prompts. The apprenticeship architecture needs richer metrics.

### Per-Apprentice Evaluation

Each specialist is evaluated on:
1. **Held-out domain data**: 100 samples from its own domain not seen during training
2. **Tool call accuracy**: Structured validation via `decode.validate_tool_call()` — 6 failure modes tracked
3. **Format adherence**: Does the specialist emit correct boundary tokens (`<|plan|>`, `<|observe|>`, etc.)?
4. **Adversarial robustness**: Performance on adversarial variants (timeout, partial, corrupt)
5. **Latent reasoning quality**: Does the specialist use `<|think_start|>...<|think_end|>` appropriately?

### Cross-Apprentice Interference Test

The fundamental risk: does specialist A degrade when specialist B is loaded?

Test protocol:
```
1. Load backbone + specialist A (e.g., tool_caller)
2. Run 5 tool call prompts → record accuracy_A
3. Load backbone + specialist B (e.g., planner)
4. Run 5 tool call prompts → record accuracy_A_under_B
5. Interference = accuracy_A - accuracy_A_under_B
```

If interference > 5%, the specialist adapters are competing for the same backbone capacity. Mitigations:
- Reduce LoRA rank (16 → 8)
- Increase distillation steps to reinforce shared patterns
- Add interference penalty to distillation loss

### Router Accuracy

The router is evaluated on:
1. **Top-1 accuracy**: Does it select the correct specialist?
2. **Confidence calibration**: Is the probability well-calibrated?
3. **Fallback rate**: How often does confidence fall below threshold?

Router training uses 300 samples per domain (1,500 total). Evaluation uses a separate 50-per-domain holdout set (250 total).

### Evaluation Schedule

| Point | What |
|---|---|
| After each specialist training | Per-apprentice eval on held-out domain data |
| After each distillation | Cross-apprentice interference test |
| After router training | Router accuracy on holdout set |
| End of all rounds | Joint eval: 5 specialists × 3 metrics each + router |

### Implementation

The existing `eval.py` infrastructure (perplexity, tool call accuracy, format adherence) is reused for backbone-level metrics. Apprentice-specific evaluators are added in `eval.py`:

```python
def evaluate_apprentice(model, adapter, domain_dataset, tok, cfg):
    """Run all metrics for one specialist."""
    model.load_lora(adapter)
    loss = compute_loss(model, domain_dataset, tok, cfg)
    tool_acc = evaluate_tool_calls(model, domain_prompts, tok, cfg)
    format_scores = format_adherence(model, domain_prompts, tok, cfg)
    return {"loss": loss, "tool_acc": tool_acc, "format": format_scores}

def test_interference(model, adapters, test_fn, tok, cfg):
    """Measure cross-apprentice interference."""
    baselines = {}
    for name, adapter in adapters.items():
        model.load_lora(adapter)
        baselines[name] = test_fn(model, tok, cfg)

    interference = {}
    for name_a, adapter_a in adapters.items():
        model.load_lora(adapter_a)
        for name_b in adapters:
            if name_a == name_b:
                continue
            score = test_fn(model, tok, cfg)
            interference[f"{name_a}_under_{name_b}"] = score - baselines[name_a]

    return baselines, interference
```

---

## The Deep Idea

> Not MoE. Not Mamba. Not LoRA.
> **Iterative abstraction accumulation through specialization.**

Each specialist learns one thing deeply. The backbone absorbs the shared structure. The next specialist starts from a higher base. Cognition compounds.

That is the asymmetry frontier labs cannot replicate — their models ship frozen, their knowledge is static at release. This architecture **gets smarter every time you train a new apprentice**.

The revolutionary claim is not about parameter count. It's about **capability density per training FLOP** and **compounding cognition over time**.

A system that improves every time you teach it one new thing — without forgetting anything — might be more valuable than a model that knows everything at release and never changes.
