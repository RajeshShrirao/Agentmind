import mlx.core as mx
import mlx.nn as nn
from .mamba_block import MambaBlock
from .attention_block import LocalAttentionBlock
from .mtp_head import MTPHead

class AgentMind(nn.Module):
    """
    Hybrid SSM + Local Attention Language Model.
    
    Layer pattern (n_layers=16, attn_every=4):
    [M M M A | M M M A | M M M A | M M M A]
     12 Mamba blocks + 4 Attention blocks = 16 total ≈ 147M params
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.blocks = [
            LocalAttentionBlock(cfg) if cfg.is_attn_layer(i)
            else MambaBlock(cfg)
            for i in range(cfg.n_layers)
        ]

        self.norm = nn.RMSNorm(cfg.d_model)

        # Tied LM head
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # MTP auxiliary head
        self.mtp = MTPHead(cfg, K=4)

    def __call__(self, input_ids, return_mtp=False):
        # input_ids: [B, L]
        x = self.embed(input_ids)
        h_states = {}

        for i, block in enumerate(self.blocks):
            if isinstance(block, MambaBlock):
                x, h = block(x)
                h_states[i] = h
            else:
                x = block(x)

        # Store last hidden state for inference
        self.last_hidden = x

        # MTP auxiliary predictions (only when requested)
        if return_mtp:
            self.last_mtp_logits = self.mtp(x)

        x = self.norm(x)
        logits = self.lm_head(x)  # [B, L, vocab_size]
        return logits, h_states

    def load_lora(self, adapter_weights: dict):
        for name, param in adapter_weights.items():
            parts = name.split('.')
            module = self
            for part in parts[:-1]:
                module = module[int(part)] if part.isdigit() else getattr(module, part)
            setattr(module, parts[-1], param)

    def forward_with_state(self, input_ids, past_h_states=None):
        """Used during agentic inference to preserve SSM state across calls."""
        x = self.embed(input_ids)
        new_h_states = {}

        for i, block in enumerate(self.blocks):
            if isinstance(block, MambaBlock):
                h_in = past_h_states.get(i) if past_h_states else None
                x, h = block(x, h_in)
                new_h_states[i] = h
            else:
                x = block(x)

        # Store last hidden state for MTP
        self.last_hidden = x

        # MTP auxiliary predictions
        self.last_mtp_logits = self.mtp(x)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_h_states
