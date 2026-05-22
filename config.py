import math
from dataclasses import dataclass

@dataclass
class AgentMindConfig:
    # Vocabulary
    vocab_size: int = 32_000

    # Model dimensions
    d_model: int = 1024
    n_layers: int = 16

    # Mamba SSM
    d_state: int = 16        # memory per channel
    d_conv: int = 4          # causal conv kernel
    expand: int = 2          # d_inner = expand × d_model = 4096
    dt_rank: int = -1        # -1 = auto: ceil(d_model / 16) = 64

    # Hybrid attention
    n_heads: int = 8
    attn_window: int = 256   # local attention window
    attn_every: int = 4      # attention layer every N blocks

    # FFN (SwiGLU)
    ffn_mult: float = 8 / 3  # standard SwiGLU multiplier

    # Runtime
    max_seq_len: int = 8192
    tie_embeddings: bool = True

    # Special token IDs (assigned after tokenizer init)
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 5
    tool_call_id: int = 6
    plan_id: int = 7
    memory_id: int = 8
    scratch_id: int = 9
    observe_id: int = 10
    think_start_id: int = 11
    think_end_id: int = 12
    system_id: int = 13
    user_id: int = 14
    assistant_id: int = 15

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
