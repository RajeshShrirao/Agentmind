from dataclasses import dataclass, field


@dataclass
class AgentMindConfig:
    backbone_id: str = "./checkpoints/Qwen2.5-0.5B"
    d_model: int = 896
    vocab_size: int = 151_936
    max_seq_len: int = 8192

    # Special token IDs — set at runtime via tokenizer
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
    lora_rank: int = 16
    lora_alpha: float = 32.0
    save_dir: str = "./checkpoints"
    eval_every: int = 500
    save_every: int = 200


@dataclass
class DistillConfig(TrainingConfig):
    beta: float = 0.5
    lr: float = 1e-5


APPRENTICE_ROUNDS = [
    {
        "domain": "tool_caller",
        "file": "data/apprentice_tool_caller.jsonl",
        "specialist_steps": 2000,
        "seq_len": 256,
        "seq_len_schedule": {0: 384, 200: 512},
        "distill_steps": 200,
        "adversarial": 0.3,
    },
    {
        "domain": "planner",
        "file": "data/apprentice_planner.jsonl",
        "specialist_steps": 300,
        "seq_len": 512,
        "seq_len_schedule": None,
        "distill_steps": 150,
        "adversarial": 0.3,
    },
    {
        "domain": "recovery",
        "file": "data/apprentice_recovery.jsonl",
        "specialist_steps": 300,
        "seq_len": 256,
        "seq_len_schedule": {0: 128, 150: 256},
        "distill_steps": 150,
        "adversarial": 0.4,
    },
    {
        "domain": "code",
        "file": "data/apprentice_code.jsonl",
        "specialist_steps": 300,
        "seq_len": 512,
        "seq_len_schedule": None,
        "distill_steps": 150,
        "adversarial": 0.3,
    },
    {
        "domain": "research",
        "file": "data/apprentice_research.jsonl",
        "specialist_steps": 300,
        "seq_len": 1024,
        "seq_len_schedule": None,
        "distill_steps": 150,
        "adversarial": 0.3,
    },
]
