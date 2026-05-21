import sentencepiece as spm
from pathlib import Path

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

def train_tokenizer(corpus_path: str, model_prefix: str = "agentmind_tok"):
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=32_000,
        character_coverage=0.9999,
        model_type="bpe",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        user_defined_symbols=SPECIAL_TOKENS[3:],  # custom tokens after <pad/bos/eos>
        byte_fallback=True,             # handles any unicode
        add_dummy_prefix=False,
        split_digits=True,              # tokenize digits separately (better for tool args)
    )
    print(f"Tokenizer saved: {model_prefix}.model")

def load_tokenizer(model_path: str):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp
