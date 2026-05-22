import sentencepiece as spm
from pathlib import Path
from dataclasses import dataclass

SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>",
    # Agentic control tokens
    "<|tool_call|>",     # model wants to invoke a tool
    "<|plan|>",          # structured multi-step plan
    "<|memory|>",        # write to persistent memory
    "<|scratch|>",       # internal scratchpad (visible)
    "<|observe|>",       # tool result injection
    "<|think_start|>",   # begin latent reasoning window
    "<|think_end|>",     # surface output after latent steps
    # Role tokens
    "<|system|>", "<|user|>", "<|assistant|>",
]

@dataclass(frozen=True)
class SpecialTokenIDs:
    pad_id: int
    bos_id: int
    eos_id: int
    unk_id: int
    tool_call_id: int
    plan_id: int
    memory_id: int
    scratch_id: int
    observe_id: int
    think_start_id: int
    think_end_id: int
    system_id: int
    user_id: int
    assistant_id: int

def get_token_ids(tokenizer) -> SpecialTokenIDs:
    return SpecialTokenIDs(
        pad_id=tokenizer.pad_id(),
        bos_id=tokenizer.bos_id(),
        eos_id=tokenizer.piece_to_id("<eos>"),
        unk_id=tokenizer.unk_id(),
        tool_call_id=tokenizer.piece_to_id("<|tool_call|>"),
        plan_id=tokenizer.piece_to_id("<|plan|>"),
        memory_id=tokenizer.piece_to_id("<|memory|>"),
        scratch_id=tokenizer.piece_to_id("<|scratch|>"),
        observe_id=tokenizer.piece_to_id("<|observe|>"),
        think_start_id=tokenizer.piece_to_id("<|think_start|>"),
        think_end_id=tokenizer.piece_to_id("<|think_end|>"),
        system_id=tokenizer.piece_to_id("<|system|>"),
        user_id=tokenizer.piece_to_id("<|user|>"),
        assistant_id=tokenizer.piece_to_id("<|assistant|>"),
    )

def assert_token_ids_real(tokenizer, ids: SpecialTokenIDs):
    print("=" * 60)
    print("Special Token ID Verification")
    print("=" * 60)
    fmt = "  {:25s} {}"
    print(fmt.format("pad_id", ids.pad_id))
    print(fmt.format("bos_id (<s>)", ids.bos_id))
    print(fmt.format("eos_id (<eos>)", ids.eos_id))
    print(fmt.format("spm_eos_id (</s>)", tokenizer.eos_id()))
    print(fmt.format("unk_id", ids.unk_id))
    print(fmt.format("tool_call_id", ids.tool_call_id))
    print(fmt.format("plan_id", ids.plan_id))
    print(fmt.format("memory_id", ids.memory_id))
    print(fmt.format("scratch_id", ids.scratch_id))
    print(fmt.format("observe_id", ids.observe_id))
    print(fmt.format("think_start_id", ids.think_start_id))
    print(fmt.format("think_end_id", ids.think_end_id))
    print(fmt.format("system_id", ids.system_id))
    print(fmt.format("user_id", ids.user_id))
    print(fmt.format("assistant_id", ids.assistant_id))

    all_ids = [ids.pad_id, ids.bos_id, ids.eos_id, ids.unk_id,
               ids.tool_call_id, ids.plan_id, ids.memory_id, ids.scratch_id,
               ids.observe_id, ids.think_start_id, ids.think_end_id,
               ids.system_id, ids.user_id, ids.assistant_id]

    assert len(set(all_ids)) == len(all_ids), \
        f"Duplicate token IDs detected! {len(all_ids)} tokens but {len(set(all_ids))} unique"

    assert ids.pad_id == tokenizer.pad_id(), \
        f"pad_id mismatch: cfg={ids.pad_id}, tokenizer={tokenizer.pad_id()}"
    assert ids.bos_id == tokenizer.bos_id(), \
        f"bos_id mismatch: cfg={ids.bos_id}, tokenizer={tokenizer.bos_id()}"
    assert ids.unk_id == tokenizer.unk_id(), \
        f"unk_id mismatch: cfg={ids.unk_id}, tokenizer={tokenizer.unk_id()}"
    assert all(t > 3 for t in all_ids[4:]), \
        "Agentic control tokens must have IDs > 3"

    print(f"✅ All {len(all_ids)} special token IDs are valid and unique.")
    print("=" * 60)

def hydrate_config(cfg, tokenizer):
    """Set all cfg.*_id attributes from tokenizer-derived IDs."""
    ids = get_token_ids(tokenizer)
    for attr in ("pad_id", "bos_id", "eos_id", "tool_call_id", "plan_id",
                 "memory_id", "scratch_id", "observe_id", "think_start_id",
                 "think_end_id", "system_id", "user_id", "assistant_id"):
        setattr(cfg, attr, getattr(ids, attr))
    # Verify every ID was set (none left at sentinel)
    for attr in ("pad_id", "bos_id", "eos_id", "tool_call_id", "plan_id",
                 "memory_id", "scratch_id", "observe_id", "think_start_id",
                 "think_end_id", "system_id", "user_id", "assistant_id"):
        assert getattr(cfg, attr) >= 0, \
            f"hydrate_config failed: {attr} is still {getattr(cfg, attr)}"
    # Regression guard: verify IDs match the tokenizer, not stale hardcoded values
    assert cfg.eos_id == tokenizer.piece_to_id("<eos>"), \
        f"eos_id {cfg.eos_id} is not the tokenizer's <eos>"
    assert cfg.tool_call_id == tokenizer.piece_to_id("<|tool_call|>"), \
        f"tool_call_id {cfg.tool_call_id} is not the tokenizer's <|tool_call|>"
    assert cfg.assistant_id == tokenizer.piece_to_id("<|assistant|>"), \
        f"assistant_id {cfg.assistant_id} is not the tokenizer's <|assistant|>"
    return cfg

def train_tokenizer(corpus_path: str, model_prefix: str = "agentmind_tok"):
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=32_000,
        character_coverage=0.9999,
        model_type="bpe",
        pad_id=0,
        pad_piece="<pad>",
        bos_id=1,
        bos_piece="<s>",
        eos_id=2,
        eos_piece="</s>",
        unk_id=3,
        unk_piece="<unk>",
        user_defined_symbols=SPECIAL_TOKENS[1:],  # custom tokens starting with <bos> and <eos>
        byte_fallback=True,             # handles any unicode
        add_dummy_prefix=False,
        split_digits=True,              # tokenize digits separately (better for tool args)
    )
    print(f"Tokenizer saved: {model_prefix}.model")

def load_tokenizer(model_path: str):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp
