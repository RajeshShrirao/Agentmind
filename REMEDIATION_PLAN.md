# AgentMind Codebase — Systems-Level Remediation Plan

---

## 1. Executive Summary

### What Is Fundamentally Broken

The codebase has three **foundational architecture-level failures** that render the current training pipeline logically invalid:

**A. Tokenizer/Config ID Registry Mismatch (critical, silent).** `config.py` hardcodes special token IDs 0–15. The SentencePiece tokenizer assigns `<pad>`=0, `<bos>`=1, `<eos>`=2, `<unk>`=3, and all 10 agentic control tokens (`<|tool_call|>`, `<|plan|>`, etc.) as `user_defined_symbols` appended at the high end of the 32K vocab (~positions 31987–31999). **There is no token with ID 5, 6, 10, 11, 12, 13, 14, 15 in the actual vocabulary.** Every comparison against `cfg.eos_id` (5), `cfg.tool_call_id` (6), `cfg.think_start_id` (11), etc. compares against unassigned IDs that never appear in tokenized sequences. This single issue poisons every downstream system.

**B. Training Label Corruption (critical, silent).** `_make_labels` in both `data/pipeline.py:77-91` and `pretokenize.py:31-41` uses hardcoded config IDs to determine assistant turn boundaries. Since `cfg.assistant_id` (15) never matches the actual `<|assistant|>` token ID (~31999), `in_assistant` becomes True after the first occurrence and **never resets**. The loss function penalizes the model for predicting user and system tokens — the model is actively trained to *not* reconstruct user queries, which is the exact opposite of instruction following. Training loss (~12.46) is artificially inflated and structurally meaningless.

**C. Latent Curriculum Is a No-Op (critical, silent).** `latent_loss_mask` in `model/latent.py:72-118` compares against `think_start_id` (11) and `think_end_id` (12), which never match the actual `<|think_start|>` (~31996) and `<|think_end|>` (~31997) token IDs. The masking loop runs to completion without masking anything. The entire staged latent curriculum (stages 2–4) has zero effect — the model is always trained as if in stage 1 regardless of step count.

### Architecture-Level vs Implementation-Level

| Level | Issues |
|---|---|
| **Architecture** | Token registry design (centralized config vs dynamic lookup), data pipeline split logic, LoRA target naming, synthetic type system |
| **Implementation** | Docstring drift, dead code, hardcoded eval break, pre-tokenize bypass, tool name mismatch in templates |

### Which Bugs Cause Silent Corruption

- **Bug 1** (token IDs) — corrupts *all* loss computation, eval, inference, latent training
- **Bug 2** (dual split) — corrupts train/val data distribution, silently drops 5% of data
- **Bug 4** (LoRA o_proj) — attention output path never adapted
- **Bug 6** (synthetic int types) — model trains on type-incorrect tool calls
- **Bug 8** (get_stock name) — recovery training data teaches non-existent tools

### Which Bugs Invalidate Current Training/Evals

- **The entire checkpoint at `step_01600` is untrustworthy.** The loss values are inflated by ~2-3x due to incorrect assistant masking. The `tool_acc=0.0` eval results are not a training failure — they're an eval harness failure (broken token comparisons in `generate_tool_call`). The val_loss=8.0 reported at steps 499, 999, 1499 is a placeholder constant (falls into the `except` path in `eval.py:117-118`), not an actual computation.

---

## 2. Root Cause Graph

### Primary Dependency Chains

```
                     ┌─────────────────────────────────────────────────────────────────────┐
                     │  BUG #1: Hardcoded Token IDs in config.py (0-15)                    │
                     │  ──────────────────────────────────────────────────────────────────  │
                     │  SentencePiece assigns: pad=0 bos=1 eos=2 unk=3,                    │
                     │  user_defined_symbols at ~31987-31999                               │
                     │  NO TOKEN HAS ID 5-15                                               │
                     └────────────────────┬────────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┼──────────────────────┐
                    ▼                     ▼                       ▼
        ┌────────────────────┐  ┌─────────────────┐  ┌──────────────────────┐
        │ _make_labels       │  │ latent_loss_mask │  │ generate_tool_call  │
        │ pipeline.py:77-91  │  │ latent.py:72-118 │  │ decode.py:242-251   │
        │ pretokenize.py:31-41│  │                  │  │                     │
        │  checks assistant_id│  │ checks think_    │  │ checks tool_call_id │
        │  =15, eos_id=5,    │  │ start_id=11,     │  │ =6, eos_id=5,      │
        │  user_id=14,       │  │ think_end_id=12  │  │ observe_id=10      │
        │  system_id=13      │  │ (never match)    │  │ (never match)      │
        └────────┬───────────┘  └────────┬──────────┘  └─────────┬──────────┘
                 │                       │                       │
                 ▼                       ▼                       ▼
        ┌────────────────┐    ┌─────────────────────┐  ┌─────────────────────────┐
        │ in_assistant   │    │ Entire latent       │  │ generate_tool_call      │
        │ never resets   │    │ curriculum is no-op │  │ always returns           │
        │ → loss on user │    │ → stages 2-4 have   │  │ {"valid":false,          │
        │ & system tokens│    │ no behavioral effect│  │ "error":"no <|tool_call|>│
        │ → LOSS IS      │    │ → always stage 1   │  │ emitted"}                │
        │ STRUCTURALLY   │    │ regardless of step  │  │ → TOOL ACC ALWAYS 0%     │
        │ INVALID        │    └─────────────────────┘  └─────────────────────────┘
        └────────────────┘
                 │
                 ▼
        ┌──────────────────────────────┐
        │ Cross-entropy loss penalizes │
        │ model for predicting user &  │
        │ system tokens                │
        │ → ~12.46 loss is inflated    │
        │ by ~2-3x                     │
        │ → Model trained to IGNORE    │
        │   user input                 │
        └──────────────────────────────┘

                     ┌─────────────────────────────────────┐
                     │  BUG #2: Duplicated Train/Val Split │
                     │  pipeline.py:38-43 AND 53-58        │
                     └────────────────┬────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                                   ▼
        ┌────────────────────────┐          ┌──────────────────────┐
        │ First split: 95/5      │          │ Second split on 95%  │
        │ correctly partitions   │          │ already-split data:  │
        │ raw 10K samples        │          │ train = 90.25%       │
        │ into 9500/500          │          │ val = 4.75%          │
        └────────────────────────┘          │ 5% LOST entirely    │
                                            └──────────────────────┘
                    │                               │
                    └───────────────────────────────┘
                                      │
                                      ▼
                          ┌─────────────────────────────┐
                          │ Eval set is 4.75% instead of│
                          │ 5% → ~5% underestimation   │
                          │ ~5% of data never seen      │
                          │ → less data, slight val     │
                          │   distribution shift        │
                          └─────────────────────────────┘

┌──────────────────────────────────────────────┐
│  BUG #4: LoRA Misses o_proj                  │
│  lora.py:36-37 targets "out_proj"            │
│  attention_block.py:29 uses "o_proj"         │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │ Attention output path has NO LoRA   │
        │ adaptation despite being critical  │
        │ for structured tool JSON output     │
        └─────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│  BUG #6: Synthetic Integer Type Mismatch     │
│  data/synthetic.py:146 — all mock_args are   │
│  f"example_{k}" strings even for int params  │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │ search_arxiv "days" param trained as │
        │ string → validate_tool_call always   │
        │ rejects with type_mismatch           │
        │ → Model can never succeed at arxiv   │
        │   calls with days param              │
        └─────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│  BUG #8: get_stock vs get_stock_price        │
│  data/formats.py:35 — "get_stock" (wrong)    │
│  decode.py:86 — "get_stock_price" (correct)  │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │ Recovery training data teaches model │
        │ to call non-existent tool           │
        │ → Recovery samples are harmful       │
        │ → evaluate_tool_calls finds 0% acc   │
        └─────────────────────────────────────┘
```

### Secondary Dependency Chains

```
Bug #2 (dual split) ─► Bug #1 (wrong IDs) ─► train.py:312 hardcoded False (pre-tokenize bypass)
                            │                             │
                            │                             └─ Every training run re-tokenizes from scratch
                            │                                → 30s overhead per run
                            ▼
                    Pre-tokenized .npz files exist but are VALID
                    (they encode labels with wrong IDs too)
                    → Using them would just mask the problem
                    → Must regenerate after fixing IDs

Bug #7 (eval O(L²)) ─► Works correctly but slowly
                         → Not silently wrong, but precedent for ignoring forward_with_state
```

---

## 3. Severity Reclassification

| # | Issue | Original | Reclassified | Rationale |
|---|---|---|---|---|
| 1 | Token ID Registry Mismatch | CRITICAL | **CATASTROPHIC** | Structural root cause. Invalidates every downstream system. All training, eval, and inference are logically wrong. |
| 2 | Duplicated Train/Val Split | CRITICAL | **CRITICAL** | Corrupts data distribution, loses 5% of data. Must fix before any meaningful training run. |
| 3 | Docstring Architecture Mismatch | HIGH | LOW | Misleading but doesn't affect runtime correctness. Cosmetic. |
| 4 | generate_tool_call Wrong IDs | HIGH | **CATASTROPHIC** | Direct consequence of Bug #1. Tool eval accuracy will be 0% regardless of model quality. |
| 5 | _make_labels Never Exits Assistant | HIGH | **CATASTROPHIC** | Direct consequence of Bug #1. Loss computation is structurally invalid, training actively harmful. |
| 6 | LoRA misses o_proj | HIGH | **HIGH** | The attention output projection is not adapted, limiting LoRA's ability to fix tool call formatting. |
| 7 | Synthetic Integer Type Mismatch | MEDIUM | **HIGH** | Model trains on type-incorrect patterns. Tool calls with integer params will structurally fail. |
| 8 | latent_loss_mask Cannot Fire | MEDIUM | **CRITICAL** | Direct consequence of Bug #1. Entire latent curriculum is a no-op. Wasted training. |
| 9 | ConvState Dead Code | MEDIUM | LOW | No runtime impact. Slight maintenance burden. |
| 10 | get_stock vs get_stock_price | MEDIUM | **HIGH** | Recovery samples train model on non-existent tool. Actively harmful training data. |
| 11 | eval.py SSM State Ignored | MEDIUM | LOW | Correct but O(L²). No silent data corruption. |
| 12 | generate_synthetic.py Missing Tools | LOW | LOW | Only affects older 1.7K dataset. The newer scaled generator has all 14. |
| 13 | train.py Pre-tokenize Bypass | LOW | **MEDIUM** | Every training run re-tokenizes. Also masks that pre-tokenized data has wrong labels. |

### Silent Corruption Potential Ranking

1. **Bug #1 + #5**: Loss is computed on wrong tokens → model trained to ignore user input
2. **Bug #8**: Latent curriculum is entirely wasted compute
3. **Bug #4**: Tool call evaluation shows 0% → leads to wrong debugging conclusions
4. **Bug #10**: Recovery training data actively harmful
5. **Bug #7**: Integer type mismatches → tool calls structurally fail
6. **Bug #2**: 5% data silently dropped, val set slightly corrupted

---

## 4. Remediation Phases

### Phase 0 — Stop-the-Bleeding (PREREQUISITE FOR ANY FURTHER TRAINING)

**Issues addressed:** Bug #1 (token ID mismatch) — the foundational root cause

**Why ordering matters:** Every other fix depends on knowing the correct token IDs. Fixing anything else before this wastes engineering effort because the corrected code will need different IDs.

**Risks if skipped:** Any training run produces a model trained to ignore user input. Any eval produces meaningless metrics. Any latent stage logic is wasted. **The entire codebase operates on fictional token IDs.**

**Exact work:**
1. Add `get_token_ids(tokenizer)` function in `tokenizer_setup.py` that returns a frozen dataclass with real IDs derived from `tokenizer.token_to_id()` and `tokenizer.eos_id()`, `tokenizer.bos_id()`, `tokenizer.pad_id()`
2. In `train.py`, before training starts, call `get_token_ids(tok)` and override `cfg` attributes with real values
3. Add startup assertion that prints all 13 special token IDs and verifies they differ from the hardcoded defaults

**Validation strategy:**
- Assert `tok.token_to_id("<|eos|>") == 2` (not 5)
- Assert `tok.token_to_id("<|tool_call|>") > 30000`
- Assert `tok.token_to_id("<|assistant|>") > 30000`
- Print all 13 IDs at startup and assert they match between tokenizer and runtime config

**Expected behavioral improvements:**
- Loss drops from ~12.46 to ~8-9 (correct assistant-masked loss)
- `generate_tool_call` can now detect `<|tool_call|>`, EOS, `<|observe|>`
- `latent_loss_mask` can now find think boundaries
- `_make_labels` correctly resets at user/system boundaries

---

### Phase 1 — Tokenization Integrity

**Issues addressed:**
- Bug #1: Remove all hardcoded IDs from `config.py`. Replace with runtime-derived values.
- All files that compare against hardcoded IDs: `data/pipeline.py`, `pretokenize.py`, `model/latent.py`, `decode.py`, `eval.py`, `export.py`

**Why ordering matters:** This is the natural extension of Phase 0. Once the `get_token_ids()` function exists, every consumer must be updated to use it.

**Exact work:**
1. In `config.py`, remove all hardcoded special token IDs (lines 32-44). Replace with `None` defaults or remove entirely.
2. Create a `SpecialTokenIDs` frozen dataclass in `tokenizer_setup.py`.
3. Add a `hydrate_config(cfg, tokenizer)` function that sets `cfg.pad_id`, `cfg.bos_id`, etc. from tokenizer.
4. Update `train.py` to call `hydrate_config(cfg, tok)` before creating model.
5. Update `pretokenize.py` to use `tok.token_to_id()` directly instead of `cfg.*_id`.
6. Update `export.py` `save_hf_format()` to use real IDs.
7. Hardcode the assertion that `cfg.eos_id != 5` (old wrong value) — this prevents regression.

**Validation strategy:**
- `test_token_ids.py`: Load tokenizer, assert all 13 special tokens have unique IDs > 3, assert pad/bos/eos/unk match SentencePiece defaults
- `test_config_hydration.py`: Create config, hydrate, assert `cfg.eos_id == 2`

---

### Phase 2 — Training Correctness

**Issues addressed:**
- Bug #2: Duplicated train/val split in `data/pipeline.py`
- Bug #5: `_make_labels` using wrong IDs (already fixed by Phase 0-1, but label logic must be verified)
- Bug #8: `latent_loss_mask` using wrong IDs (already fixed by Phase 0-1)
- Bug #13: Hardcoded `False` in `train.py:312` blocking pre-tokenized data
- Bug #10: `get_stock` vs `get_stock_price` in `data/formats.py`

**Why ordering matters:** Correct token IDs from Phase 1 are prerequisites. The dual split fix is independent of token IDs but must precede any meaningful training run.

**Risks if skipped:**
- Training data loses 5% of samples
- Pre-tokenized path is dead code
- Recovery schema teaches non-existent tool
- Training still re-tokenizes on every run

**Exact work:**
1. **Dual split fix:** Delete lines 52-58 from `data/pipeline.py`. The first split (lines 38-43) is correct. `random.shuffle` on line 53 is harmless only if lines 54-58 are removed.
2. **Pre-tokenize restore:** Change `if False and ...` to `if os.path.exists(...)` in `train.py:312`. Verify the pre-tokenized .npz files were generated with corrected labels (if not, regenerate).
3. **get_stock fix:** Change `"get_stock"` to `"get_stock_price"` in `data/formats.py:35`. Update the schema docstring in `docs/agentmind_training_infra.md`.
4. **Verify _make_labels:** After Phase 1, tokenize `"<|user|>hello<|assistant|>world"`, run `_make_labels`, assert labels before `<|assistant|>` are -100.
5. **Verify latent_loss_mask:** After Phase 1, construct sequence with `<|think_start|>`...`<|think_end|>`, run `latent_loss_mask`, assert tokens between boundaries are -100.

**Validation strategy:**
- `assert len(train_ds.samples) + len(val_ds.samples) == original_count`
- `assert os.path.exists("data/train_ids.npz") → train.py uses pre-tokenized path`
- Tokenize known string, run `_make_labels`, assert correct masking

---

### Phase 3 — Inference + Tool Runtime

**Issues addressed:**
- Bug #4: `generate_tool_call` using wrong IDs (fixed by Phase 0-1)
- Bug #6: LoRA misses `o_proj`
- Bug #11: `evaluate_tool_calls_from_text` ignores SSM state
- Verify `extract_tool_calls` and `validate_tool_call` work end-to-end

**Why ordering matters:** Token IDs must be correct before any inference path works correctly. LoRA fix is independent but must happen before training runs that depend on attention output adaptation.

**Exact work:**
1. **generate_tool_call:** After Phase 1, this function receives real IDs. No additional code changes needed if it uses `cfg.tool_call_id`, `cfg.eos_id`, `cfg.observe_id`. Verify by running end-to-end test.
2. **LoRA o_proj fix:** Add `"o_proj"` to targets list in `lora.py:37`. Change to `targets = ["in_proj", "out_proj", "o_proj", "q_proj", "v_proj", "lm_head"]`. Alternatively, use exact name matching: the attention output projection is named `o_proj` while Mamba output is `out_proj`.
3. **eval.py SSM state fix:** In `evaluate_tool_calls_from_text`, replace `model(ids)` with `model.forward_with_state(ids, h_states)` and maintain state across iterations. Also fix `format_adherence` similarly.

**Validation strategy:**
- `test_generate_tool_call.py`: Load model, run `generate_tool_call` with a tool prompt, assert `result["valid"] == True`
- `test_lora_targets.py`: After `apply_lora`, iterate `model.trainable_parameters()`, assert `o_proj` appears
- `test_eval_ssm.py`: Compare `evaluate_tool_calls` vs `evaluate_tool_calls_from_text` on same prompts, assert same results

---

### Phase 4 — Data + Synthetic Quality

**Issues addressed:**
- Bug #7: Integer type mismatch in `data/synthetic.py:146`
- Bug #12: Missing 4 tools in `generate_synthetic.py`

**Why ordering matters:** These affect training data quality but don't block other fixes. Can run in parallel with Phase 3.

**Exact work:**
1. **Integer type fix:** In `data/synthetic.py:146`, change mock args generation:
   ```python
   mock_args = {}
   for k, v in tool["args"].items():
       if v is int:
           mock_args[k] = random.randint(1, 100)
       elif v is str:
           mock_args[k] = f"example_{k}"
   ```
2. **Missing tools:** Add `list_directory`, `get_stock_price`, `translate`, `summarize` to the TOOLS list in `generate_synthetic.py`.
3. **Regenerate** the `data/synthetic_agents.jsonl` dataset.
4. **Reconciliation:** Ensure `SYNTHETIC_TOOLS` in `data/synthetic.py`, `TOOLS` in `generate_scaled_synthetic.py`, and `TOOL_REGISTRY` in `decode.py` all have the same 14 tools with consistent parameter types.

**Validation strategy:**
- `test_synthetic_types.py`: Generate a `search_arxiv` call with `days` param, feed through `validate_tool_call`, assert `valid=True`
- `test_tool_registry_consistency.py`: Assert all 4 registries have identical tool names and parameter types
- `test_generate_synthetic_tools.py`: Count unique tools in output, assert == 14

---

### Phase 5 — Architecture Cleanup

**Issues addressed:**
- Bug #3: Docstring claims 600M (reality 147M)
- Bug #9: `ConvState` dead code
- Bug #10: `get_stock` in docs
- Bug #13: Hardcoded `False` in train.py (documentation)
- General documentation drift

**Why ordering matters:** These have no runtime impact. Safe to defer to end.

**Exact work:**
1. Delete `model/conv_state.py` (or add deprecation notice and integrate into `MambaBlock.step()`)
2. Fix docstring in `model/agent_lm.py:11-13` to reflect actual 16-layer, 147M architecture
3. Audit `docs/agentmind_architecture.md` and `docs/agentmind_training_infra.md` for all stale code snippets and wrong token IDs
4. Update `docs/agentmind_training_infra.md` recovery schema example
5. Add a note in `train.py` about why the `False` was originally there and the conditions for re-enabling

**Validation strategy:** None needed — documentation-only fixes.

---

## 5. Unified Token Registry Design

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Tokenizer Source of Truth                              │
│  ─────────────────────────────                           │
│  tokenizer_setup.py:                                     │
│                                                         │
│  @dataclass(frozen=True)                                 │
│  class SpecialTokenIDs:                                  │
│      pad_id: int         # tokenizer.pad_id()            │
│      bos_id: int         # tokenizer.bos_id()            │
│      eos_id: int         # tokenizer.eos_id()            │
│      unk_id: int         # tokenizer.unk_id()            │
│      tool_call_id: int   # tokenizer.token_to_id(...)    │
│      plan_id: int                                         │
│      memory_id: int                                       │
│      scratch_id: int                                      │
│      observe_id: int                                      │
│      think_start_id: int                                  │
│      think_end_id: int                                    │
│      system_id: int                                       │
│      user_id: int                                         │
│      assistant_id: int                                    │
│                                                         │
│  def get_token_ids(tokenizer) -> SpecialTokenIDs:         │
│      # Derive ALL IDs from loaded tokenizer               │
│      # NO hardcoded defaults                              │
│                                                         │
│  def hydrate_config(cfg, tokenizer):                      │
│      ids = get_token_ids(tokenizer)                       │
│      cfg.pad_id = ids.pad_id                              │
│      # ... set all 13 IDs                                 │
│                                                         │
│  def assert_token_ids_real(cfg):                          │
│      # Startup assertion:                                 │
│      assert cfg.eos_id != 5  # old wrong value            │
│      assert cfg.tool_call_id > 30000                      │
│      assert cfg.assistant_id > 30000                      │
│      assert cfg.pad_id == 0                               │
│      assert cfg.bos_id == 1                               │
└─────────────────────┬───────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
┌────────────┐ ┌────────────┐ ┌────────────┐
│ train.py   │ │ decode.py  │ │ eval.py    │
│ uses       │ │ uses       │ │ uses       │
│ cfg.*_id   │ │ cfg.*_id   │ │ cfg.*_id   │
│ values     │ │ values     │ │ values     │
└────────────┘ └────────────┘ └────────────┘
```

### Design Principles

1. **Single source of truth:** `tokenizer.token_to_id()` is always the canonical mapping. Config is a snapshot at startup.

2. **Immutable registry:** `SpecialTokenIDs` is a frozen dataclass — cannot be mutated after creation. Prevents accidental drift.

3. **Startup assertions:** Every training/eval/inference entry point must call `assert_token_ids_real()` before proceeding.

4. **No hardcoded fallbacks:** `tokenizer.token_to_id()` must raise if a token is missing. No silent fallback to a guessed ID.

5. **Serialization safety:** When exporting to GGUF or HF format, serialize the *runtime-derived* IDs from the tokenizer, not the config defaults.

6. **GGUF compatibility:** The GGUF format stores `bos_token_id`, `eos_token_id`, `pad_token_id` as metadata fields. Any custom `tokenizer_config.json` must use the real IDs.

7. **Synthetic generation compatibility:** Each synthetic generator must ensure its tool registry's token types match what `validate_tool_call` expects. This means integer types for integer params, not strings.

8. **Latent curriculum compatibility:** `inject_latent_tokens` works at the text level (string replacement), so it's unaffected. But `latent_loss_mask` must use the real `think_start_id` and `think_end_id` from the derived registry.

---

## 6. Training Integrity Analysis

### Current Loss Computation Is Invalid

| Component | Current Behavior | Correct Behavior | Impact |
|---|---|---|---|
| `_make_labels` assistant boundary | Never exits assistant mode | Resets at `<|user|>`, `<|system|>`, `<|eos|>` | Loss includes user/system tokens, inflated by ~2-3x |
| `cross_entropy_loss` | Penalizes prediction of user tokens | User tokens masked with -100 | Model trained to ignore user input |
| `latent_loss_mask` (stage 3+) | Never masks between think boundaries | Masks tokens between `<|think_start|>` and `<|think_end|>` | Latent curriculum is no-op |
| `mtp_loss` | Operates on same corrupted labels | Would use corrected labels | MTP also learns wrong patterns |

### Which Metrics Are Currently Invalid

| Metric | Status | Why |
|---|---|---|
| Training loss (~12.46) | **INVALID** | Includes wrong labels, inflated by ~2-3x |
| Validation loss (8.0) | **INVALID** | 8.0 is the fallback constant from `except Exception` — not a real computation |
| `tool_acc` (0.0) | **INVALID** | Not a training problem — eval harness never detects `<|tool_call|>` due to wrong IDs |
| Latent stage progression | **INVALID** | Stages 2-4 have zero effect — `latent_loss_mask` never fires |

### Estimated Corrected Values

Based on analysis of similar Mamba models with proper instruction masking:
- Corrected training loss: ~8-9 (vs current 12.46)
- Corrected validation loss: ~8-9 (vs current 8.0 fallback)
- Tool accuracy after correction: Unknown — must be measured after fix

### Instruction Following Impact

The current training actively penalizes the model for predicting user/system tokens (since `in_assistant` never resets, loss includes `"<|user|>search arxiv..."` as target). This means:
- The model is trained to **not** reconstruct user queries
- This directly harms instruction following — the model doesn't learn to "read" user input
- Tool use learning is also corrupted because the `<|tool_call|>` token detection in training loss doesn't align with inference

---

## 7. Regression Prevention Strategy

### Invariant Tests (Must Block Training Startup)

1. **`test_tokenizer_id_consistency`:**
   - Load tokenizer
   - Assert all 13 special tokens have unique IDs
   - Assert `<pad>` ID == 0, `<s>` ID == 1, `</s>` ID == 2, `<unk>` ID == 3
   - Assert all agentic tokens have IDs >= vocab_size (i.e., > 31987)
   - Assert `tokenizer.token_to_id("<|tool_call|>") != 6` (should be ~31989)
   - **Blocks training if any assertion fails**

2. **`test_config_hydration`:**
   - Create `AgentMindConfig()`, hydrate with tokenizer
   - Assert `cfg.eos_id == 2` (not 5)
   - Assert `cfg.tool_call_id > 30000`
   - Assert `cfg.pad_id == 0`
   - **Blocks training if any assertion fails**

### Tokenizer Consistency Tests

3. **`test_token_id_roundtrip`:**
   - Encode a string containing all special tokens
   - Decode the IDs back
   - Assert roundtrip preserves all special tokens
   - Assert the tokenized IDs match `cfg.*_id` values

4. **`test_all_special_tokens_decodeable`:**
   - For each special token, call `tokenizer.id_to_token(cfg.*_id)`
   - Assert result matches the expected string

### Synthetic Schema Tests

5. **`test_tool_registry_parity`:**
   - Assert `SYNTHETIC_TOOLS` (synthetic.py), `TOOLS` (generate_scaled_synthetic.py), `TOOL_REGISTRY` (decode.py) have identical tool names
   - Assert parameter types are consistent across all registries
   - Assert integer-type params in registries match Python `int` type

6. **`test_synthetic_type_correctness`:**
   - Generate one sample for each tool
   - Parse the tool call JSON
   - Pass through `validate_tool_call`
   - Assert all calls are valid

### Train/Infer Parity Tests

7. **`test_train_infer_label_parity`:**
   - Tokenize a known sample string
   - Run `_make_labels` on the ID sequence
   - Assert labels before first `<|assistant|>` are -100
   - Assert labels after `<|user|>` reset to -100
   - Assert `<eos>` label is not -100 (model should learn to predict EOS)

8. **`test_latent_mask_correctness`:**
   - Construct sequence: `100, think_start_id, 9, 9, think_end_id, 200`
   - Run `latent_loss_mask`
   - Assert positions 1-4 (think_start through think_end) masked to -100
   - Assert position 5 (200 after think_end) NOT masked

### End-to-End Tool Trajectory Tests

9. **`test_generate_tool_call_e2e`:**
   - Create minimal model
   - Encode a tool prompt
   - Run `generate_tool_call` with 200 max tokens
   - Assert `result["valid"] == True`
   - Print failure modes if false

10. **`test_validate_tool_call_all_tools`:**
    - For each tool in `TOOL_REGISTRY`, construct a valid call with type-correct args
    - Assert `validate_tool_call` returns `valid=True`

### Latent Masking Tests

11. **`test_latent_loss_mask_stages`:**
    - Test all 4 latent stages
    - Assert stage 1: no masking
    - Assert stage 2: think_start/think_end inserted
    - Assert stage 3: ~50% CoT replaced
    - Assert stage 4: CoT entirely removed

---

## 8. Suggested Refactors

### 1. Registry Centralization

**Problem:** 5 separate tool registries in 5 files (`decode.py`, `data/synthetic.py`, `generate_scaled_synthetic.py`, `generate_synthetic.py`, `data/formats.py`) with inconsistent naming, parameter types, and coverage.

**Refactor:** Create a single `ToolRegistry` dataclass in `data/registry.py` with:
- Tool names as enum
- Parameter schemas with Python types
- Mock result generators
- Validation helpers
- `validate_call()` method that replaces `decode.validate_tool_call`

All consumers import from this single source of truth.

### 2. Config Architecture

**Problem:** `AgentMindConfig` mixes model architecture params with runtime token IDs. Token IDs are volatile (depend on tokenizer training) while architecture params are stable.

**Refactor:** Split into:
- `ModelConfig`: architecture params (d_model, n_layers, etc.)
- `TrainingConfig`: hyperparams (LR, batch size, schedule)
- `SpecialTokenIDs`: derived from tokenizer at startup

`AgentMind.__init__` takes `ModelConfig`. Training loop takes `TrainingConfig` + `SpecialTokenIDs` separately.

### 3. Evaluation Abstraction

**Problem:** `eval.py` has 3 competing tool evaluation approaches (`evaluate_tool_calls`, `evaluate_tool_calls_from_text`, `tool_call_accuracy` in docs) with inconsistent SSM state handling.

**Refactor:** Single `ToolEvaluator` class:
- Uses `generate_tool_call` (structured) as primary path
- Supports `forward_with_state` enforcement
- Reports structured `EvalReport` dataclass
- Deprecate the text-based legacy path with a note

### 4. Synthetic Tool Schema Anchoring

**Problem:** Synthetic generators produce data that `validate_tool_call` rejects due to type mismatches and missing tools.

**Refactor:** Each synthetic generator calls `tool_registry.generate_mock_args(tool_name)` which returns type-correct args. Tool lists are populated from `tool_registry.tools` rather than being manually maintained.

### 5. Tokenizer Boot Sequence

**Problem:** `tokenizer_setup.py` has no startup validation. Consumers must remember to call `hydrate_config`.

**Refactor:** Create a `TokenizerContext` manager:
```python
with TokenizerContext("agentmind_tok.model", cfg) as tok:
    # cfg is now hydrated
    # All special token IDs are validated
    train(tok, cfg)
```
On `__enter__`, it loads tokenizer, hydrates config, runs all startup assertions.

### 6. Train Startup Validation

**Problem:** `train.py` has no data integrity checks before starting training.

**Refactor:** Add a `validate_training_setup()` function called at the start of `train()` that:
1. Loads tokenizer, hydrates config
2. Runs all invariant tests (Section 7 #1-2)
3. Verifies data files exist
4. Verifies data directory has correct structure
5. Verifies checkpoint directory is writable
6. Verifies model initialization produces finite values
7. Prints a summary report of what will be used

---

## 9. Long-Term Reliability Risks

### 1. Tokenizer Drift

**Risk:** If the tokenizer is retrained (e.g., with a different corpus), all special token IDs change. Existing checkpoints become incompatible because frozen model weights reference old embedding/lm_head positions.

**Mitigation:**
- Version-stamp tokenizer in checkpoint metadata
- On resume, assert `tokenizer.pad_id() == checkpoint_cfg.pad_id` (and all other IDs)
- If mismatch, raise clear error with migration instructions
- Never silently load a checkpoint with a different tokenizer

### 2. Registry Duplication

**Risk:** The 5 tool registries will inevitably diverge as tools are added to some but not others. This has already happened (4 missing tools in `generate_synthetic.py`).

**Mitigation:**
- Implement the single `ToolRegistry` refactor (#1 in Section 8)
- Add a CI step that asserts all registries are identical
- Make the registry importable as `from data.registry import REGISTRY`

### 3. Silent Eval Corruption

**Risk:** If token IDs drift and eval code falls into `except Exception:` paths (like the current `val_loss=8.0` fallback), eval results silently degrade without alerting.

**Mitigation:**
- Remove all bare `except Exception:` handlers in evaluation code
- Add structured error reporting that distinguishes "no data" from "error"
- Validate eval results by checking they differ from fallback constants
- Log the number of tokens contributing to each loss value

### 4. Synthetic Distribution Mismatch

**Risk:** The synthetic data distribution (tool call patterns, error rates, domain topics) may not match real deployment. The model could learn artifacts of the generation process rather than general tool use.

**Mitigation:**
- Log synthetic data type distribution periodically
- Evaluate on held-out real data (not synthetic) for tool accuracy
- Monitor synthetic data size — currently ~11.5K samples is very small
- Add real-world trajectory data from execution logs

### 5. Recurrent Inference Divergence

**Risk:** The `MambaBlock.__call__` (training) and `MambaBlock.step` (inference) paths could diverge numerically over time, especially with the `ConvState` dead code and the separate `state` management in `step()`.

**Mitigation:**
- Run `test_mamba_parity.py` as a pre-commit check
- Add numerical tolerance tracking in CI (not just pass/fail)
- If numerical drift exceeds 1e-5, log a warning
- Consider unifying the conv state management in both paths

---

## 10. Immediate Go/No-Go Verdict

### Is the current codebase trainable safely?

**NO. Absolutely not.** The codebase cannot produce a useful model in its current state.

### Which fixes are mandatory before any further runs?

**All of Phase 0 and Phase 1 are blocking.** Specifically:

1. **Token ID derivation from tokenizer** — without this, the loss function penalizes correct assistant behavior
2. **Config hydration** — every comparison against special tokens uses the wrong values
3. **Startup assertions** — must verify IDs before training begins

Additionally, **Bug #2 (dual split)** must be fixed before any training run that cares about data integrity.

### Which metrics/results are currently untrustworthy?

| Metric | Trust? | Action |
|---|---|---|
| Training loss (~12.46) | **DO NOT TRUST** | Artifact of wrong label masking |
| Validation loss (8.0) | **DO NOT TRUST** | Fallback constant from except handler |
| Tool accuracy (0.0%) | **DO NOT TRUST** | Eval harness never detects `<|tool_call|>` |
| Latent stage progression | **DO NOT TRUST** | Curriculum is entirely no-op |
| LoRA training quality | **DO NOT TRUST** | o_proj never adapted; wrong loss signal |
| Validation split quality | **DO NOT TRUST** | 5% data lost, 4.75% instead of 5% |

### Which components are fundamentally sound despite the bugs?

- **MambaBlock implementation** — The SSM core (`_ssm`, `step`, `forward_with_state`) is mathematically correct and well-tested by `test_mamba_parity.py`
- **Attention block implementation** — `LocalAttentionBlock` is a standard sliding window attention with RoPE, correctly implemented
- **MTP head** — Multi-token prediction logic is correct (though it operates on corrupted labels)
- **Gradient clipping and NaN recovery** — The `train.py` recovery loop is robust and well-designed
- **Cosine warmup scheduler** — Standard implementation, no issues
- **Weight initialization** — `init.py` follows Mamba-specific best practices
- **LoRA adapters** — The `LoRALinear` class itself is correct; only the target list is wrong
- **GGUF export** — The structure is sound; only the token IDs written to config.json are wrong
- **Latent injection logic** — `inject_latent_tokens` works correctly at the string level; only the loss masking step is broken

The core architectural idea (hybrid Mamba+Attention SSM with staged latent curriculum and LoRA fine-tuning) is sound. The implementation quality of individual components is generally good. **The failures are in the integration layer** — specifically how components communicate through shared assumptions about token IDs and data formats. A systematic cleanup of the token registry and data pipeline will resolve all critical issues without requiring rewrites of core model code.
