# Engineering Practices for AgentMind

## 1. Invariant Enforcement via Design by Contract

The codebase has critical implicit invariants that currently live only in comments:

- `hydrate_config(cfg, tok)` must run before any `cfg.*_id` comparison
- MTP must NEVER run when backbone is frozen
- `lm_head` and `embed` must be excluded from specialist training
- `last_hidden`/`last_mtp_logits` must be stripped before saving

**Practice:** Enforce with runtime assertions at module boundaries instead of comments.

```python
class TrainingContract:
    @staticmethod
    def require_hydrated(cfg):
        assert cfg.assistant_id != -1, "Config not hydrated — call hydrate_config() first"
    
    @staticmethod
    def require_mtp_disabled(model):
        assert not hasattr(model, 'mtp') or model.mtp is None
    
    @staticmethod
    def require_no_last_keys(params):
        assert not any(k.startswith("last_") for k in params)
```

**Why:** Silent failures (e.g., comparing against `-1` token IDs that match no token, masking all loss) waste hours of training.

---

## 2. State Machine for Training Lifecycle

The apprenticeship protocol is a deterministic state machine being modeled as a flat script:

```
INIT → PRETRAIN → (SPECIALIST → DISTILL)ᵏ → ROUTER → EXPORT
```

**Practice:** Encode transitions explicitly so impossible states are unrepresentable.

```python
class TrainingPhase(Enum):
    INIT = auto()
    BACKBONE_LOADED = auto()
    LORA_APPLIED = auto()
    SPECIALIST_TRAINED = auto()
    DISTILLED = auto()
    ROUTER_TRAINED = auto()
    EXPORTED = auto()

TRANSITIONS = {
    TrainingPhase.INIT: [TrainingPhase.BACKBONE_LOADED],
    TrainingPhase.BACKBONE_LOADED: [TrainingPhase.LORA_APPLIED],
    ...
}
```

**Why:** Prevents catastrophic states like "started distillation without resetting LoRA" — the kind of bug that silently produces bad weights.

---

## 3. Strategy Pattern for Training Variants

Three training modes share ~80% of their code but diverge in key ways:

| Aspect | Pretrain | Specialist | Distill |
|---|---|---|---|
| Backbone | Trainable | Frozen | Trainable |
| MTP | No | No | After step 20 |
| Loss | CE | CE + syntax aux | CE + KL + MTP |
| LoRA | No | Yes, trainable | No |

**Practice:** Extract a `TrainingStrategy` interface:

```python
class TrainingStrategy(ABC):
    @abstractmethod
    def should_freeze_backbone(self) -> bool: ...
    @abstractmethod
    def should_enable_mtp(self, step: int) -> bool: ...
    @abstractmethod
    def compute_loss(self, logits, targets, **extra) -> mx.array: ...

class SpecialistStrategy(TrainingStrategy):
    def should_freeze_backbone(self): return True
    def should_enable_mtp(self, _): return False
```

**Why:** Adding a new training variant (RLHF, DPO, etc.) requires zero changes to the loop.

---

## 4. Structured Experiment Tracking

**Current state:** `log` dicts appended to a list, dumped to `log.json`. No experiment ID, no config hash, no artifact linking.

**Practice:** Self-describing checkpoints with full provenance:

```
checkpoints/exp_a1b2c3d4/
├── step_00001/
│   ├── weights.npz
│   ├── run.json       # ← full provenance + metrics
│   └── config.yaml    # ← frozen at creation time
├── step_00003/
└── ...
```

```python
@dataclass
class RunMetadata:
    id: str              # git-hash based
    config_hash: str     # sha256 of config serialization
    data_hash: str       # sha256 of data source
    git_commit: str
    start_time: float
    metrics: list[StepMetrics]
```

**Why:** Enables comparison between runs — compare `config_hash` and `data_hash` to determine if two runs are meaningfully different.

---

## 5. Circuit Breaker for Degradation

Current NaN recovery is reactive (fires after corruption). A circuit breaker detects systematic degradation.

```python
class DegradationCircuitBreaker:
    def __init__(self, threshold=3, window=100):
        self.failures = deque(maxlen=window)
        self.state = "CLOSED"

    def record(self, step, loss):
        if not mx.isfinite(loss) or loss > 100:
            self.failures.append(step)
        if len(self.failures) >= self.threshold \
           and (step - self.failures[0]) < self.window:
            self.state = "OPEN"
            raise TrainingDegradationError("Training collapsed")
```

**Why:** Catches genuine training collapse (diverging loss, exploding gradients), not just one-off NaN spikes.

---

## 6. Schema Enforcement at Data Boundaries

JSONL data has an implicit schema:

```json
{"domain": "tool_caller", "type": "tool_single", "messages": [...]}
```

Missing `role`/`content` keys produce silent garbage.

**Practice:** Validate every sample at load time, not deep in training:

```python
MESSAGE_SCHEMA = {
    "domain": lambda v: v in VALID_DOMAINS,
    "type": lambda v: v in SAMPLE_TYPES,
    "messages": lambda msgs: all(
        m["role"] in ("system", "user", "assistant")
        and isinstance(m.get("content"), str)
        for m in msgs
    ),
}

def validate_sample(sample):
    for key, validator in MESSAGE_SCHEMA.items():
        assert validator(sample[key]), f"Invalid {key}: {sample.get(key)}"
```

**Why:** Fail at data loading, not 3 hours into training.

---

## 7. Token-Level Contract Testing

The biggest silent bug class: **wrong token IDs**. The existing `assert_token_ids_real()` helps — extend this to a full contract test.

```python
def test_training_contract(tok, cfg, model, sample_batch):
    """Verify every invariant before training (5ms overhead)."""
    assert cfg.bos_id == tok.bos_id()
    assert cfg.eos_id == tok.piece_to_id("<eos>")
    ids, labels = sample_batch
    loss = cross_entropy_loss(model(ids)[0], labels)
    assert mx.isfinite(loss).item() and loss.item() > 0
    for k, v in model.trainable_parameters().items():
        assert not mx.any(mx.isnan(v)).item(), f"NaN in {k}"
```

Run once at startup — 5ms, prevents hours of silent corruption.

---

## 8. Resource-Aware Scheduling (Implemented)

**Problem:** macOS uses most RAM (~12GB baseline). Training adds ~2GB. Total sits at ~14/16GB. Swap is the real danger.

**Implementation in `monitor.py`:** `ResourceScheduler` tracks:
- Baseline RAM at startup (OS + other apps)
- Training memory delta (current - baseline)
- Swap usage (the real threat signal)

On sustained swap > 100MB: reduces seq_len. Recovers when swap clears.

**Pattern:** Like TCP congestion control — additive increase, multiplicative decrease.

---

## 9. Principle of Least Surprise: Detach Tied Embeddings

The model has tied embeddings (`lm_head.weight = embed.weight`). After `apply_lora()`, `lm_head` gets wrapped in `LoRALinear`, but `lm_head.weight` still references `embed.weight`.

**Spooky action at a distance:** Mutating one mutates both.

**Practice:** At LoRA application, detach the reference:

```python
if lm_head.weight is embed.weight:
    lm_head.weight = mx.array(embed.weight)  # Deep copy
```

Add a defensive `assert` that the reference is broken after `apply_lora()`.

---

## 10. Postel's Law for Configuration

**Before:** Raw dict with zero validation — typo `totl_stpes` silently ignored.

**Practice:** Frozen dataclass with `__post_init__` validation:

```python
@dataclass
class TrainingConfig:
    lr: float = 2e-4
    total_steps: int = 3000

    def __post_init__(self):
        assert self.lr > 0
        assert self.total_steps > 0
        assert self.warmup_steps < self.total_steps
        self._frozen = True  # Prevent typos after init

    def __setattr__(self, name, value):
        if getattr(self, '_frozen', False):
            raise AttributeError(f"Config frozen — can't set {name}")
        super().__setattr__(name, value)
```

**Why:** Catches typos immediately instead of producing silent NaN loss 500 steps in.

---

## Priority Matrix

| Practice | Impact | Effort | Status |
|---|---|---|---|
| Contract tests (token IDs, loss mask) | Critical | Low (1 function) | ✅ Partial (assert_token_ids_real exists) |
| Config validation + freezing | High | Low (post_init) | ✅ Phase 2 done |
| Schema enforcement on JSONL | Medium | Low (validate fn) | ❌ Not yet |
| Experiment tracking + provenance | Medium | Medium | ❌ Not yet |
| Strategy pattern for training variants | Medium | Medium | ❌ Not yet |
| Circuit breaker for degradation | Low | Low | ❌ Nice-to-have |
| Resource-aware scheduling | Medium | Low | ✅ Implemented |
| State machine for lifecycle | Low | Medium | ❌ If complexity grows |
| Detach tied embeddings | Medium | 1 line | ❌ Not yet |
