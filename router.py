import json, random, time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from pathlib import Path
from stats_logger import GLOBAL as log


class TaskRouter(nn.Module):
    def __init__(self, d_model=1024, hidden=64, n_domains=5, domain_names=None):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_domains),
        )
        self.domain_names = domain_names or []

    @staticmethod
    def _format_messages(messages):
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"<|system|>{content}"
            elif role == "user":
                text += f"<|user|>{content}"
            elif role == "assistant":
                text += f"<|assistant|>{content}<eos>"
        return text

    def __call__(self, hidden_state):
        if hidden_state.ndim == 3:
            last_hidden = hidden_state[:, -1, :]
        else:
            last_hidden = hidden_state
        return self.classifier(last_hidden)

    def select_expert(self, hidden_state, threshold=0.6):
        logits = self(hidden_state)
        probs = mx.softmax(logits, axis=-1)
        if mx.max(probs).item() < threshold:
            return "tool_caller"
        return self.domain_names[mx.argmax(logits, axis=-1).item()]

    def train(self, dataset, backbone, tokenizer=None, steps=200, lr=1e-3):
        backbone.eval()
        backbone.freeze()

        domain_to_id = {name: i for i, name in enumerate(self.domain_names)}

        # ── Cache hidden states: one 147M forward pass per sample ──
        cached = []
        for sample in dataset:
            domain = sample["domain"]
            messages = sample["messages"]
            label = domain_to_id[domain]

            if isinstance(messages, list) and messages and isinstance(messages[0], dict):
                if tokenizer is None:
                    raise ValueError(
                        "tokenizer required when messages are raw dicts; "
                        "pass a SentencePiece tokenizer with .encode(text, add_bos=True)"
                    )
                text = self._format_messages(messages)
                token_ids = tokenizer.encode(text, add_bos=True)
            else:
                token_ids = messages

            ids = mx.array([token_ids])
            backbone.forward_with_state(ids, {})
            # cache only last-token hidden state — (1, d_model), ~4KB per sample
            cached.append((backbone.last_hidden[:, -1, :], label, domain))

        print(f"  Cached {len(cached)} hidden states (one forward pass each)")

        # ── Train classifier on cached states ──
        optimizer = optim.Adam(learning_rate=lr)
        loss_fn = nn.losses.cross_entropy
        t_start = time.time()

        domain_names = self.domain_names
        per_domain = {d: {"correct": 0, "total": 0} for d in domain_names}

        for step in range(steps):
            random.shuffle(cached)
            total_loss = 0.0

            for hidden, label, domain in cached:
                label_arr = mx.array([label])

                def compute_loss(h):
                    logits = self.classifier(h)
                    return loss_fn(logits, label_arr, reduction="mean")

                loss, grads = nn.value_and_grad(self, compute_loss)(hidden)
                optimizer.update(self, grads)
                mx.eval(self.parameters(), optimizer.state)

                total_loss += loss.item()
                pred = mx.argmax(self.classifier(hidden), axis=-1).item()
                per_domain[domain]["correct"] += (pred == label)
                per_domain[domain]["total"] += 1

            if step % 50 == 0 or step == steps - 1:
                avg_loss = total_loss / len(cached)
                total_correct = sum(v["correct"] for v in per_domain.values())
                total_all = sum(v["total"] for v in per_domain.values())
                acc = total_correct / total_all * 100
                per_domain_acc = ", ".join(
                    f"{d}={v['correct']/v['total']*100:.0f}%"
                    for d, v in per_domain.items()
                )
                print(f"[router] step {step}/{steps}  loss={avg_loss:.4f}  acc={acc:.1f}%  [{per_domain_acc}]")
                log.step("router", step, steps, avg_loss, acc=acc,
                         per_domain_acc=per_domain_acc)

        elapsed = time.time() - t_start
        total_correct = sum(v["correct"] for v in per_domain.values())
        total_all = sum(v["total"] for v in per_domain.values())
        per_domain_acc = ", ".join(
            f"{d}={v['correct']/v['total']*100:.0f}%"
            for d, v in per_domain.items()
        )
        final_acc = total_correct / total_all * 100
        print(f"[router] Complete ({steps} steps, {elapsed:.0f}s) "
              f"acc={final_acc:.1f}%  [{per_domain_acc}]")
        log.summary("router", steps=steps, elapsed=elapsed, acc=final_acc,
                    per_domain_acc=per_domain_acc)
        return self

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path = path.with_suffix('.safetensors')

        params = dict(tree_flatten(self.parameters()))
        metadata = {
            "domain_names": json.dumps(self.domain_names),
        }
        mx.save_safetensors(str(path), params, metadata)
        total_kb = sum(v.nbytes for v in params.values()) // 1024
        print(f"[router] saved TaskRouter \u2192 {path} ({total_kb} KB)")

    @classmethod
    def load(cls, path):
        path = Path(str(path))
        if path.suffix != '.safetensors':
            path = path.with_suffix('.safetensors')
        if not path.exists():
            raise FileNotFoundError(f"TaskRouter not found: {path}")

        loaded = mx.load(str(path))
        metadata = loaded.get("metadata", {})
        if "metadata" in loaded:
            del loaded["metadata"]

        w0 = loaded["classifier.layers.0.weight"]
        hidden, d_model = w0.shape
        w2 = loaded["classifier.layers.2.weight"]
        n_domains = w2.shape[0]

        domain_names = json.loads(metadata.get("domain_names", "[]"))
        if not domain_names:
            domain_names = [str(i) for i in range(n_domains)]

        router = cls(
            d_model=d_model,
            hidden=hidden,
            n_domains=n_domains,
            domain_names=domain_names,
        )
        nested = tree_unflatten(dict(loaded))
        router.update(nested)
        print(f"[router] loaded TaskRouter from {path} "
              f"({n_domains} domains, d_model={d_model}, hidden={hidden})")
        return router
