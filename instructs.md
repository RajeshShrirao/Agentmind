# AgentMind — Claude Code Prompt Sequence
> Copy-paste each prompt into Claude Code in order.
> 🤖 = Claude Code handles it | 👤 = You must do it manually

---

## Phase 0 — Project Setup

### 🤖 Prompt 1 — Scaffold project structure
```
Create the following directory structure for a project called agentmind:

agentmind/
├── data/
│   ├── __init__.py
│   ├── formats.py
│   ├── pipeline.py
│   └── synthetic.py
├── model/
│   ├── __init__.py
│   ├── mamba_block.py
│   ├── attention_block.py
│   ├── hybrid_block.py
│   ├── agent_lm.py
│   ├── rope.py
│   ├── conv_state.py
│   ├── mtp_head.py
│   └── latent.py
├── config.py
├── lora.py
├── init.py
├── scheduler.py
├── train.py
├── eval.py
├── export.py
├── agent.py
└── requirements.txt

Then create requirements.txt with:
mlx
mlx-lm
sentencepiece
datasets
orjson
msgspec
transformers
tqdm
numpy

Do not write any code yet. Just create the empty files and requirements.txt with content.
```

---

### 👤 Human Step 1 — Install dependencies
```bash
cd agentmind
pip install -r requirements.txt --break-system-packages
```
> Do this yourself. Dependency installs need your terminal.

---

### 👤 Human Step 2 — Verify MLX works on your Mac
```bash
python -c "import mlx.core as mx; print(mx.metal.is_available())"
```
> Should print `True`. If not, reinstall mlx: `pip install mlx --upgrade`.

---

## Phase 1 — Config + Model Architecture

### 🤖 Prompt 2 — Write config.py
```
Read the file agentmind_architecture.md (I will paste it below).
Implement config.py exactly as specified in the "config.py" section of that document.
Make sure all @property methods are correct.
After writing, run: python -c "from config import AgentMindConfig; c = AgentMindConfig(); print(c.param_count_estimate)"
It should print a param count without errors.

[paste contents of agentmind_architecture.md here]
```

---

### 🤖 Prompt 3 — Write model/rope.py
```
Read the "RoPE for Attention Blocks" section in agentmind_training_infra.md.
Implement model/rope.py exactly as specified.
After writing, run:
python -c "
from model.rope import precompute_rope, apply_rope
import mlx.core as mx
cos, sin = precompute_rope(128, 512)
x = mx.ones((1, 8, 16, 128))
out = apply_rope(x, cos, sin)
print('RoPE output shape:', out.shape)
"
Shape should be (1, 8, 16, 128).

[paste contents of agentmind_training_infra.md here]
```

---

### 🤖 Prompt 4 — Write model/mamba_block.py
```
Read the "model/mamba_block.py" section in agentmind_architecture.md.
Implement it fully. Pay careful attention to:
- The causal depthwise conv (padding must be d_conv - 1 on the left only)
- The ZOH discretization of A and B
- The sequential scan loop shape: h is [B, d_inner, d_state]
- The step() method for single-token inference

After writing, run this smoke test:
python -c "
import mlx.core as mx
from config import AgentMindConfig
from model.mamba_block import MambaBlock
cfg = AgentMindConfig()
block = MambaBlock(cfg)
x = mx.ones((1, 16, cfg.d_model))
out, h = block(x)
print('MambaBlock output shape:', out.shape)
print('Hidden state shape:', h.shape)
"
Expected: (1, 16, 2048) and (1, 4096, 128)
```

---

### 🤖 Prompt 5 — Write model/attention_block.py
```
Read the "model/attention_block.py" section in agentmind_architecture.md.
Implement it. Then integrate RoPE from model/rope.py into _local_attn():
- Import precompute_rope and apply_rope
- In __init__, call precompute_rope(cfg.head_dim, cfg.max_seq_len) and store as self.rope_cos, self.rope_sin
- After projecting q and k, call apply_rope on both

Smoke test:
python -c "
import mlx.core as mx
from config import AgentMindConfig
from model.attention_block import LocalAttentionBlock
cfg = AgentMindConfig()
block = LocalAttentionBlock(cfg)
x = mx.ones((1, 32, cfg.d_model))
out = block(x)
print('AttnBlock output shape:', out.shape)
"
Expected: (1, 32, 2048)
```

---

### 🤖 Prompt 6 — Write model/agent_lm.py
```
Read the "model/agent_lm.py" section in agentmind_architecture.md.
Implement the AgentMind class. It must:
- Alternate MambaBlock and LocalAttentionBlock according to cfg.is_attn_layer(i)
- Implement forward_with_state() that passes h_states across calls
- Store last hidden state as self.last_hidden before lm_head (needed for MTP)
- Tie embedding weights to lm_head if cfg.tie_embeddings is True

Smoke test:
python -c "
import mlx.core as mx
from config import AgentMindConfig
from model.agent_lm import AgentMind
cfg = AgentMindConfig()
model = AgentMind(cfg)
ids = mx.ones((1, 8), dtype=mx.int32)
logits, h = model(ids)
print('Logits shape:', logits.shape)
print('Num SSM states:', len(h))
"
Expected: (1, 8, 32000) and 18 SSM states (24 layers - 6 attention layers)
```

---

### 🤖 Prompt 7 — Write model/mtp_head.py and model/conv_state.py
```
Read agentmind_training_infra.md.
Implement both:
1. model/mtp_head.py — MTPHead class and mtp_loss function exactly as specified
2. model/conv_state.py — ConvState class exactly as specified

Then add MTPHead to AgentMind in model/agent_lm.py:
- In __init__: self.mtp = MTPHead(cfg, K=4)
- In forward(): after computing hidden states but before lm_head, run self.mtp(hidden) and store as self.last_mtp_logits

Smoke test mtp_head:
python -c "
import mlx.core as mx
from config import AgentMindConfig
from model.mtp_head import MTPHead, mtp_loss
cfg = AgentMindConfig()
head = MTPHead(cfg, K=4)
hidden = mx.ones((1, 16, cfg.d_model))
outs = head(hidden)
print('MTP outputs:', len(outs), 'each shape:', outs[0].shape)
"
```

---

### 🤖 Prompt 8 — Write init.py
```
Read the "Weight Initialization" section in agentmind_training_infra.md.
Implement init.py with init_agentmind(model, cfg).

Critical: the dt_proj bias init using log-uniform sampling between dt_min and dt_max
is the most important part. Get this right — wrong init causes training instability.

After writing, test:
python -c "
from config import AgentMindConfig
from model.agent_lm import AgentMind
from init import init_agentmind
cfg = AgentMindConfig()
model = AgentMind(cfg)
model = init_agentmind(model, cfg)
print('Init complete')
"
```

---

## Phase 2 — Data Pipeline

### 👤 Human Step 3 — Verify HF token access (optional)
`build_corpus.py` uses your HF token to access gated datasets (The Stack, AgentInstruct, etc.).
If you haven't already, ensure your token is set in the script or env var:
```bash
export HF_TOKEN="your_token_here"
```
> If you skip this, public datasets (FineWeb, UltraChat) will still download.

---

### 🤖 Prompt 9 — Write data/formats.py and data/synthetic.py
```
Read the "Data Pipeline" section in agentmind_training_infra.md.
Implement:
1. data/formats.py — just the four JSONL example schemas as constants/docstrings
   with a validate_sample(sample) function that checks the schema is correct
2. data/synthetic.py — full generate_trajectory() and generate_dataset() functions

Then generate a test dataset:
python -c "
from data.synthetic import generate_dataset
generate_dataset(100, 'data/test_synthetic.jsonl')
"
Open data/test_synthetic.jsonl and print the first 2 lines to confirm format.
```

---

### 🤖 Prompt 10 — Write build_corpus.py and generate_scaled_synthetic.py
```
Create two new data generation scripts:

1. build_corpus.py — Downloads open datasets from HF Hub:
   - FineWeb (20K lines)
   - The Stack Python (10K lines)
   - UltraChat (10K lines)
   - AgentInstruct (5K lines)
   - ToolBench (3K lines)
   - WebArena (3K lines)
   Output: data/corpus.txt (~250MB)

2. generate_scaled_synthetic.py — Generates 11.5K synthetic agent samples:
   - 3K instruction, 2.5K tool_single, 3K agent_multi, 2K recovery, 1K latent
   - Uses template-based generation for bulk + Cerebras API for diversity
   - Respects Cerebras rate limit: 40 req/min (1.5s delay between requests)
   - 14 tools in registry with realistic args/results
   Output: data/scaled_synthetic.jsonl (~6MB)

Run both scripts and verify output.
```

---

### 🤖 Prompt 11 — Write data/pipeline.py
```
Read the "data/pipeline.py" section in agentmind_training_infra.md.
Implement AgentDataset and make_dataloader.

Important: the _make_labels() function must mask system and user tokens with -100.
Only assistant turn tokens should be in the labels.

Test with the synthetic data we just generated:
python -c "
from config import AgentMindConfig
from data.pipeline import AgentDataset, make_dataloader
import sentencepiece as spm

# Use a dummy tokenizer for the test
class DummyTok:
    def encode(self, text, add_bos=False): return [1] * min(len(text.split()), 64)
    def decode(self, ids): return ' '.join(str(i) for i in ids)

cfg = AgentMindConfig()
ds = AgentDataset(['data/test_synthetic.jsonl'], DummyTok(), cfg, split='train')
loader = make_dataloader(ds, batch_size=1)
ids, labels = next(iter(loader))
print('Batch ids shape:', ids.shape)
print('Labels shape:', labels.shape)
print('Masked tokens:', (labels == -100).sum().item())
"
```

---

### 🤖 Prompt 11 — Write tokenizer_setup.py
```
Read the "tokenizer_setup.py" section in agentmind_architecture.md.
Implement it exactly. Then train the tokenizer on data/corpus.txt:

python -c "
from tokenizer_setup import train_tokenizer
train_tokenizer('data/corpus.txt', 'agentmind_tok')
"

After training, verify all special tokens are present:
python -c "
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load('agentmind_tok.model')
print('Vocab size:', sp.vocab_size())
for tok in ['<|tool_call|>', '<|plan|>', '<|observe|>', '<|think_start|>']:
    print(f'{tok} -> id:', sp.piece_to_id(tok))
"
All IDs should be non-zero.
```
> Note: needs data/corpus.txt from Human Step 3.

---

## Phase 3 — Training Infrastructure

### 🤖 Prompt 12 — Write lora.py
```
Read the "LoRA" section in agentmind_training_infra.md.
Implement LoRALinear and apply_lora() fully.

The apply_lora function must:
- Freeze the entire model first using model.freeze()
- Walk every module and replace target nn.Linear layers with LoRALinear
- Print the number of trainable params after

Test:
python -c "
from config import AgentMindConfig
from model.agent_lm import AgentMind
from lora import apply_lora
from init import init_agentmind

cfg = AgentMindConfig()
model = AgentMind(cfg)
model = init_agentmind(model, cfg)
model = apply_lora(model, rank=16, alpha=32.0)
"
Trainable params should be roughly 1-5% of total params.
```

---

### 🤖 Prompt 13 — Write scheduler.py
```
Read the "LR Scheduler" section in agentmind_training_infra.md.
Implement CosineWarmupScheduler exactly as specified.

Test:
python -c "
import mlx.optimizers as optim
from scheduler import CosineWarmupScheduler
opt = optim.AdamW(learning_rate=2e-4)
sched = CosineWarmupScheduler(opt, base_lr=2e-4, warmup_steps=10, total_steps=100)
lrs = [sched.step() for _ in range(100)]
print('Step 0 LR (should be ~0):', lrs[0])
print('Step 10 LR (should be 2e-4):', lrs[9])
print('Step 100 LR (should be ~2e-5):', lrs[-1])
"
```

---

### 🤖 Prompt 14 — Write model/latent.py
```
Read the "Latent Reasoning Training" section in agentmind_training_infra.md.
Implement:
1. inject_latent_tokens(sample, tokenizer, stage) — data preprocessing
2. LatentReasoningWrapper(mamba_block, cfg, n_steps) — model wrapper
3. latent_loss_mask(input_ids, labels, think_start_id, think_end_id)

Do NOT integrate LatentReasoningWrapper into the model yet.
We train stage 1 first (no latent). Wrapper gets added in Phase 5.

Test inject_latent_tokens:
python -c "
from model.latent import inject_latent_tokens

sample = {
    'type': 'agent_multi',
    'messages': [
        {'role': 'user', 'content': 'Do something'},
        {'role': 'assistant', 'content': '<|scratch|>Let me think<|tool_call|>{\"name\": \"search\"}'}
    ]
}

class DummyTok: pass

out_s1 = inject_latent_tokens(sample.copy(), DummyTok(), stage=1)
out_s2 = inject_latent_tokens(sample.copy(), DummyTok(), stage=2)
print('Stage 1:', out_s2['messages'][1]['content'][:80])
print('Stage 2:', out_s2['messages'][1]['content'][:80])
"
Stage 2 should have <|think_start|> added.
```

---

### 🤖 Prompt 15 — Write complete train.py
```
Read the "Complete Training Loop" section in agentmind_training_infra.md.
Implement train.py fully. It must include:
- Model init with init_agentmind()
- LoRA wrapping with apply_lora()
- CosineWarmupScheduler
- Gradient accumulation (every grad_accum steps)
- Gradient clipping
- Eval call every eval_every steps
- Checkpoint save every save_every steps (weights + log.json)
- Clean console output: step | loss | lr | grad_norm | tok/s

Then do a 5-step smoke test to verify it runs without error:
python -c "
import mlx.core as mx
from config import AgentMindConfig
from model.agent_lm import AgentMind
from lora import apply_lora
from init import init_agentmind
from scheduler import CosineWarmupScheduler
import mlx.optimizers as optim

cfg = AgentMindConfig()
model = AgentMind(cfg)
model = init_agentmind(model, cfg)
model = apply_lora(model, rank=16)

opt = optim.AdamW(learning_rate=2e-4)

# Fake batch
ids = mx.ones((1, 64), dtype=mx.int32)
targets = mx.ones((1, 64), dtype=mx.int32) * -100
targets = targets.at[:, 1:].set(ids[:, 1:])

import mlx.nn as nn
def loss_fn(params):
    model.update(params)
    logits, _ = model(ids)
    B, L, V = logits.shape
    flat = logits.reshape(-1, V)
    t = targets.reshape(-1)
    mask = (t != -100).astype(mx.float32)
    loss = nn.losses.cross_entropy(flat, mx.maximum(t, 0), reduction='none')
    return (loss * mask).sum() / (mask.sum() + 1e-8)

loss, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
mx.eval(loss)
print('Smoke test loss:', loss.item(), '— training loop works')
"
```

---

### 🤖 Prompt 16 — Write eval.py
```
Read the "Evaluation" section in agentmind_training_infra.md.
Implement eval.py with:
1. compute_perplexity()
2. tool_call_accuracy()
3. format_adherence()
4. evaluate() — combined function called from train.py

Make sure evaluate() signature matches what train.py expects:
  evaluate(model, val_dataset, tok, cfg) -> (val_loss_float, tool_acc_float)
```

---

## Phase 4 — Generate Training Data + First Training Run

### 🤖 Prompt 18 — Generate full training data
```
Using the updated data pipeline, generate and prepare all training data:

1. Build corpus from open datasets:
   python build_corpus.py
   (Downloads FineWeb, The Stack, UltraChat, AgentInstruct, ToolBench, WebArena)

2. Generate scaled synthetic data (11.5K samples):
   python generate_scaled_synthetic.py

3. Verify data:
   python -c "
   import json, os
   for f in ['data/scaled_synthetic.jsonl', 'data/test_synthetic.jsonl']:
       if os.path.exists(f):
           count = sum(1 for _ in open(f))
           print(f'{f}: {count} samples, {os.path.getsize(f)/1e6:.1f}MB')
   "
```

---

### 👤 Human Step 5 — Start training run (Stage 1)
```bash
cd agentmind
python train.py
```
> Data is already prepared: `data/corpus.txt` (~250MB) + `data/scaled_synthetic.jsonl` (11.5K samples).
> Watch the first 20 steps. Loss should start around 8-10 and drop within 50 steps.
> If loss is NaN from step 1 → weight init issue, stop and report.
> If loss is stuck above 8 after 200 steps → LR too high or data formatting broken.
> Training ~3000 steps takes 3-6 hours on MacBook Air M-series. Let it run.

---

### 👤 Human Step 6 — Monitor training
Watch for:
- Loss < 3.0 by step 500 → healthy
- Loss < 2.0 by step 1500 → on track
- Tool call accuracy > 50% by step 1000 → agent behavior emerging
- grad_norm spikes > 5 frequently → reduce lr to 1e-4 in TRAIN_CFG

---

## Phase 5 — Latent Reasoning (After Stage 1 Converges)

### 🤖 Prompt 19 — Integrate latent reasoning into training
```
Training stage 1 is complete (loss < 2.0, tool_acc > 60%).
Now integrate latent reasoning for stage 2.

1. In train.py, update TRAIN_CFG:
   "latent_stage": 2
   "max_steps": 1500  (additional steps on top of stage 1)

2. Update make_dataloader in data/pipeline.py to call inject_latent_tokens()
   on each sample before tokenizing, using the current latent_stage from TRAIN_CFG.

3. In the training loop, update the loss computation to use latent_loss_mask()
   from model/latent.py — zero out loss between <|think_start|> and <|think_end|>

4. Load the best checkpoint from stage 1:
   Add --resume flag to train.py that loads weights from a checkpoint path

Test that latent masking works:
python -c "
import mlx.core as mx
from model.latent import latent_loss_mask
from config import AgentMindConfig
cfg = AgentMindConfig()
ids = mx.array([1, cfg.think_start_id, 100, 200, cfg.think_end_id, 50, 2])
labels = mx.array([1, cfg.think_start_id, 100, 200, cfg.think_end_id, 50, 2])
masked = latent_loss_mask(ids, labels, cfg.think_start_id, cfg.think_end_id)
print('Masked labels:', masked.tolist())
"
Tokens between think_start and think_end should be -100.
```

---

## Phase 6 — Export

### 🤖 Prompt 20 — Write export.py and run export
```
Read the "GGUF Export" section in agentmind_training_infra.md.
Implement export.py fully with:
- save_hf_format(model, cfg, out_dir)
- quantize_and_export(hf_dir, out_dir, bits=4)
- CLI with --checkpoint and --out flags

Then run export on the best checkpoint:
python export.py \
  --checkpoint ./checkpoints/step_03000 \
  --out ./agentmind-4bit \
  --bits 4

Verify the output:
python -c "
from mlx_lm import load, generate
model, tokenizer = load('./agentmind-4bit')
response = generate(model, tokenizer, prompt='<|user|>Hello<|assistant|>', max_tokens=50)
print(response)
"
```

---

### 🤖 Prompt 21 — Write and test agent.py
```
Read the "agent.py" section in agentmind_architecture.md.
Implement the full AgentLoop class.

Then wire up 3 real tools:
1. web_search(query: str) → use Python requests + DuckDuckGo lite API
2. run_python(code: str) → use subprocess with timeout=10s
3. read_file(path: str) → open and read, max 10KB

Register them and run a test:
python agent.py \
  --model ./agentmind-4bit \
  --query "Write and run a Python script that computes the first 10 prime numbers"

The agent should:
1. Plan the steps
2. Call run_python with the script
3. Observe the output
4. Return the answer
```

---

## Phase 7 — Parallel Scan Upgrade (Optional, After Everything Works)

### 🤖 Prompt 22 — Swap in parallel scan
```
Read the "Parallel Scan" section in agentmind_training_infra.md.
Implement model/parallel_scan.py.

Then in model/mamba_block.py, replace the sequential for-loop in _ssm()
with the parallel_scan_log() function.

Benchmark before and after:
python -c "
import mlx.core as mx
import time
from config import AgentMindConfig
from model.mamba_block import MambaBlock

cfg = AgentMindConfig()
block = MambaBlock(cfg)
x = mx.ones((1, 512, cfg.d_model))

# Warmup
block(x)
mx.eval()

t0 = time.time()
for _ in range(10):
    out, h = block(x)
    mx.eval(out)
print(f'Avg time per forward: {(time.time()-t0)/10*1000:.1f}ms')
"
Run this before and after the swap. Parallel should be 2-4x faster at seq_len=512.
```

---

## Summary — What You Do vs Claude Code

| Step | Who |
|---|---|
| Install dependencies | 👤 You |
| Verify MLX on Apple Silicon | 👤 You |
| Verify HF token access (optional) | 👤 You |
| Start training and watch first 20 steps | 👤 You |
| Monitor loss / decide when to move stages | 👤 You |
| All code writing | 🤖 Claude Code |
| All smoke tests | 🤖 Claude Code |
| Debugging errors | 🤖 Claude Code |
| Tokenizer training | 🤖 Claude Code |
| Corpus building (6 open datasets) | 🤖 Claude Code |
| Synthetic data generation (11.5K samples) | 🤖 Claude Code |
| Checkpoint saving/loading | 🤖 Claude Code |
| Export to 4-bit GGUF | 🤖 Claude Code |
| Wiring real tools into agent.py | 🤖 Claude Code |
| Parallel scan upgrade | 🤖 Claude Code |

---

## If Something Breaks — Debug Prompts

### 🤖 NaN loss from step 1
```
My AgentMind training is producing NaN loss from step 1.
Here is my init.py: [paste]
Here is my mamba_block.py _ssm() method: [paste]
Check:
1. Is A_log initialization producing -inf anywhere?
2. Is the ZOH discretization numerically stable?
3. Is dt_proj bias initialized correctly with inverse softplus?
Fix all numerical stability issues.
```

### 🤖 Tool call accuracy stuck at 0%
```
My model trains fine (loss < 2.5) but tool_call_accuracy() always returns 0%.
Here is my eval.py tool_call_accuracy function: [paste]
Here is a sample model output: [paste raw decoded string]
Debug why the JSON parsing is failing and fix the regex/parsing logic.
Also check if <|tool_call|> token ID is correctly assigned in the tokenizer.
```

### 🤖 OOM on MacBook Air
```
I'm hitting out-of-memory during training on 16GB MacBook Air.
Current settings: batch_size=1, seq_len=2048, d_model=2048, n_layers=24
Options to try in order:
1. Reduce seq_len to 1024
2. Reduce d_model to 1536
3. Reduce n_layers to 16
4. Add gradient checkpointing to MambaBlock
Implement gradient checkpointing for MambaBlock first and show me the memory
usage before and after using: python -c "import mlx.core as mx; print(mx.metal.get_active_memory() / 1e9, 'GB')"
```