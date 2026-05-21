import math

class CosineWarmupScheduler:
    """
    Linear warmup → cosine decay.
    Standard for instruction-tuned models.
    """

    def __init__(
        self,
        optimizer,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.opt = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = base_lr * min_lr_ratio
        self.step_count = 0

    def step(self):
        self.step_count += 1
        lr = self._get_lr()
        self.opt.learning_rate = lr
        return lr

    def _get_lr(self):
        s = self.step_count
        W = self.warmup_steps
        T = self.total_steps

        if s < W:
            # Linear warmup
            return self.base_lr * (s / max(1, W))
        else:
            # Cosine decay
            progress = (s - W) / max(1, T - W)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine
