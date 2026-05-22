import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


class TaskRouter(nn.Module):
    def __init__(self, d_model=1024, hidden=64, n_domains=5, domain_names=None):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_domains),
        )
        self.domain_names = domain_names or []

    def __call__(self, hidden_state):
        pooled = mx.mean(hidden_state, axis=1)
        return self.classifier(pooled)

    def select_expert(self, hidden_state, threshold=0.6):
        logits = self(hidden_state)
        probs = mx.softmax(logits, axis=-1)
        if mx.max(probs).item() < threshold:
            return "tool_caller"
        return self.domain_names[mx.argmax(logits, axis=-1).item()]

    def train(self, dataset, backbone, steps=200, lr=1e-3):
        backbone.eval()
        backbone.freeze()
        optimizer = optim.Adam(learning_rate=lr)
        loss_fn = nn.losses.cross_entropy

        domain_to_id = {name: i for i, name in enumerate(self.domain_names)}

        for step in range(steps):
            total_loss = 0.0
            for sample in dataset:
                domain = sample["domain"]
                token_ids = sample["messages"]
                label = mx.array([domain_to_id[domain]])

                ids = mx.array([token_ids])
                backbone.forward_with_state(ids, {})
                hidden = backbone.last_hidden

                def compute_loss(h):
                    logits = self(h)
                    return loss_fn(logits, label, reduction="mean")

                loss_and_grad = nn.value_and_grad(self, compute_loss)
                loss, grads = loss_and_grad(hidden)
                optimizer.update(self, grads)
                mx.eval(self.parameters(), optimizer.state)

                total_loss += loss.item()

            if step % 50 == 0 or step == steps - 1:
                avg_loss = total_loss / len(dataset)
                print(f"[router] step {step}/{steps}  loss={avg_loss:.4f}")

        return self
