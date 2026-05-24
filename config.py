import math
from dataclasses import dataclass, field

@dataclass
class AgentMindConfig:
    # Vocabulary
    vocab_size: int = 32_000

    # Model dimensions
    d_model: int = 1024
    n_layers: int = 12

    # Mamba SSM
    d_state: int = 64        # memory per channel (64 for structured output; Mamba-2 default)
    d_conv: int = 4          # causal conv kernel
    expand: int = 2          # d_inner = expand × d_model = 4096
    dt_rank: int = 64        # explicit rank; matches Mamba-2 recipe for recall tasks

    # Hybrid attention
    n_heads: int = 8
    attn_window: int = 512   # local attention window (doubled — covers user query at later positions)
    attn_every: int = 2      # attention layer every N blocks (6 Mamba + 6 Attn with n_layers=12)

    # FFN (SwiGLU)
    ffn_mult: float = 8 / 3  # standard SwiGLU multiplier

    # Runtime
    max_seq_len: int = 8192
    tie_embeddings: bool = True

    # Special token IDs — set via hydrate_config() after tokenizer init
    pad_id: int = -1
    bos_id: int = -1
    eos_id: int = -1
    tool_call_id: int = -1
    plan_id: int = -1
    memory_id: int = -1
    scratch_id: int = -1
    observe_id: int = -1
    think_start_id: int = -1
    think_end_id: int = -1
    system_id: int = -1
    user_id: int = -1
    assistant_id: int = -1

    def __post_init__(self):
        if self.dt_rank == -1:
            self.dt_rank = math.ceil(self.d_model / 16)

    @property
    def d_inner(self) -> int:
        return int(self.expand * self.d_model)

    @property
    def dt_rank_val(self) -> int:
        return self.dt_rank

    @property
    def ffn_hidden(self) -> int:
        raw = int(self.d_model * self.ffn_mult)
        return (raw // 256) * 256  # align to 256 for hardware efficiency

    def is_attn_layer(self, i: int) -> bool:
        return (i + 1) % self.attn_every == 0

    @property
    def param_count_estimate(self) -> int:
        V = self.vocab_size
        d = self.d_model
        di = self.d_inner
        ds = self.d_state
        dr = self.dt_rank
        dh = self.ffn_hidden
        H = self.n_heads

        embed = V * d
        lm_head = V * d if not self.tie_embeddings else 0

        n_mamba = sum(1 for i in range(self.n_layers) if not self.is_attn_layer(i))
        n_attn = sum(1 for i in range(self.n_layers) if self.is_attn_layer(i))

        mamba_per_layer = (
            d * di * 2            # in_proj
            + di                  # conv bias
            + di * self.d_conv    # conv kernel (depthwise)
            + di * (dr + ds * 2)  # x_proj
            + dr * di             # dt_proj
            + di * d              # out_proj
        )

        attn_per_layer = (
            d * d * 3           # q, k, v proj
            + d * d             # o proj
            + d * dh * 2        # gate, up proj
            + dh * d            # down proj
        )

        norm = self.n_layers * d * 2  # RMSNorm scales

        total = embed + lm_head + n_mamba * mamba_per_layer + n_attn * attn_per_layer + norm
        return total


@dataclass
class TrainingConfig:
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    total_steps: int = 3000
    grad_clip: float = 1.0
    batch_size: int = 1
    grad_accum: int = 8
    seq_len: int = 512
    seq_len_schedule: dict = field(default_factory=lambda: {0: 128, 800: 256, 2000: 512})
    use_mtp: bool = True
    mtp_weight: float = 0.2
    mtp_start: int = 500
    lora_rank: int = 16
    lora_alpha: float = 32.0
    save_dir: str = "./checkpoints"
    eval_every: int = 500
    save_every: int = 200


@dataclass
class SpecialistConfig(TrainingConfig):
    syntax_aux_weight: float = 0.05
    boundary_weight: float = 1.5
    boundary_steps: int = 300
    lora_rank: int = 16
    lora_alpha: float = 32.0


@dataclass
class DistillConfig(TrainingConfig):
    beta: float = 0.5
    lr: float = 1e-5
    mtp_start: int = 20
    mtp_weight: float = 0.2


APPRENTICE_ROUNDS = [
    {
        "domain": "tool_caller",
        "file": "data/apprentice_tool_caller.jsonl",
        "specialist_steps": 2000,
        "seq_len": 256,
        "seq_len_schedule": {0: 384, 200: 512},
        "distill_steps": 200,
        "adversarial": 0.3,
        "latent_stage": 1,
    },
    {
        "domain": "planner",
        "file": "data/apprentice_planner.jsonl",
        "specialist_steps": 300,
        "seq_len": 512,
        "seq_len_schedule": None,
        "distill_steps": 150,
        "adversarial": 0.3,
        "latent_stage": 2,
    },
    {
        "domain": "recovery",
        "file": "data/apprentice_recovery.jsonl",
        "specialist_steps": 300,
        "seq_len": 256,
        "seq_len_schedule": {0: 128, 150: 256},
        "distill_steps": 150,
        "adversarial": 0.4,
        "latent_stage": 2,
    },
    {
        "domain": "code",
        "file": "data/apprentice_code.jsonl",
        "specialist_steps": 300,
        "seq_len": 512,
        "seq_len_schedule": None,
        "distill_steps": 150,
        "adversarial": 0.3,
        "latent_stage": 4,
    },
    {
        "domain": "research",
        "file": "data/apprentice_research.jsonl",
        "specialist_steps": 300,
        "seq_len": 1024,
        "seq_len_schedule": None,
        "distill_steps": 150,
        "adversarial": 0.3,
        "latent_stage": 4,
    },
]
