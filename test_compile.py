import mlx.core as mx
import mlx.nn as nn
import time
from config import AgentMindConfig
from model.agent_lm import AgentMind
from init import init_agentmind

cfg = AgentMindConfig()
model = AgentMind(cfg)
model = init_agentmind(model, cfg)

def loss_fn(model, x, y):
    logits, _ = model(x, return_mtp=False)
    # Simple CE loss
    logits_s = logits[:, :-1, :]
    targets_s = y[:, 1:]
    flat_l = logits_s.reshape(-1, logits_s.shape[-1])
    flat_t = targets_s.reshape(-1)
    return nn.losses.cross_entropy(flat_l, flat_t, reduction='mean')

# Uncompiled
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

def step_uncompiled(x, y):
    loss, grads = loss_and_grad_fn(model, x, y)
    return loss, grads

# Compiled
@mx.compile
def step_compiled(x, y):
    loss, grads = loss_and_grad_fn(model, x, y)
    return loss, grads

x = mx.random.randint(0, cfg.vocab_size, shape=(1, 256))
y = mx.random.randint(0, cfg.vocab_size, shape=(1, 256))

# Warmup uncompiled
loss, grads = step_uncompiled(x, y)
mx.eval(loss, grads)

t0 = time.time()
loss, grads = step_uncompiled(x, y)
mx.eval(loss, grads)
print(f"Uncompiled step: {(time.time() - t0)*1000:.1f}ms")

# Warmup compiled
loss, grads = step_compiled(x, y)
mx.eval(loss, grads)

t0 = time.time()
loss, grads = step_compiled(x, y)
mx.eval(loss, grads)
print(f"Compiled step: {(time.time() - t0)*1000:.1f}ms")
