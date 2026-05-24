"""train_pytorch.py — PyTorch + PEFT LoRA training for Qwen2.5-0.5B AgentMind.

Usage:
    python train_pytorch.py --data data/apprentice_tool_caller.jsonl --domain tool_caller
"""

import time
import json
import math
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
import peft
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

from data.pipeline_pytorch import AgentDataset, make_dataloader, get_seq_len_from_schedule, collate_fn

SPECIAL_TOKENS = [
    "<|tool_call|>", "<|plan|>", "<|memory|>", "<|scratch|>", "<|observe|>",
    "<|think_start|>", "<|think_end|>", "<|system|>", "<|user|>", "<|assistant|>",
]


def load_model_and_tokenizer(backbone_id="Qwen/Qwen2.5-0.5B", use_4bit=True):
    tokenizer = AutoTokenizer.from_pretrained(backbone_id, trust_remote_code=True)

    num_added = tokenizer.add_tokens(SPECIAL_TOKENS)
    print(f"Added {num_added} special tokens (vocab size now {len(tokenizer)})")

    if use_4bit:
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            backbone_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            backbone_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def apply_lora_peft(model, rank=16, alpha=32):
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied | Trainable params: {trainable:,} ({trainable/1e6:.2f}M) / Total: {total:,} ({total/1e6:.2f}M)")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    return model


def train_specialist(model, tokenizer, dataset, domain_name,
                     steps=2000, lr=2e-4, seq_len=256,
                     seq_len_schedule=None, grad_accum=8,
                     grad_clip=1.0, output_dir="./checkpoints",
                     device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup_steps = min(200, max(1, steps // 10))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=steps
    )
    scaler = torch.cuda.amp.GradScaler()

    if seq_len_schedule is None:
        seq_len_schedule = {0: seq_len}

    n_total = len(dataset)
    n_val = max(1, n_total // 20)
    val_indices = set(random.sample(range(n_total), n_val))
    train_indices = [i for i in range(n_total) if i not in val_indices]

    opt_step = 0
    micro_step = 0
    accum_loss = 0.0
    t_start = time.time()
    last_logged_step = 0
    current_seq_len = 0
    n_retrace = 0
    nan_count = 0

    while opt_step < steps:
        target_seq_len = get_seq_len_from_schedule(opt_step, seq_len_schedule)
        if target_seq_len != current_seq_len:
            print(f"  [seq_len] {current_seq_len} -> {target_seq_len}")
            current_seq_len = target_seq_len
            n_retrace += 1

        n_epoch = min(5000, len(train_indices))
        epoch_indices = random.sample(train_indices, n_epoch)

        dataset.max_len = current_seq_len
        loader = make_dataloader(dataset, batch_size=2, shuffle=True, indices=epoch_indices)

        for batch in loader:
            if opt_step >= steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )
                loss = outputs.loss / grad_accum

            loss_finite = torch.isfinite(loss)
            if not loss_finite:
                nan_count += 1
                optimizer.zero_grad()
                continue

            accum_loss += outputs.loss.item()
            scaler.scale(loss).backward()
            micro_step += 1

            if micro_step % grad_accum == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                avg_loss = accum_loss / grad_accum
                accum_loss = 0.0

                if opt_step % 100 == 0 or opt_step == steps - 1:
                    elapsed = time.time() - t_start
                    steps_since = opt_step - last_logged_step + 1
                    tok_per_sec = (input_ids.shape[1] * grad_accum * steps_since) / (elapsed + 1e-8)
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"[{domain_name}] step {opt_step:3d}/{steps} "
                          f"loss {avg_loss:.4f} lr {current_lr:.2e} "
                          f"grad_norm {grad_norm:.3f} {tok_per_sec:.0f} tok/s "
                          f"retrace={n_retrace}")
                    t_start = time.time()
                    last_logged_step = opt_step

                opt_step += 1

    elapsed = time.time() - t_start
    print(f"[{domain_name}] Training complete ({opt_step} steps, {elapsed:.0f}s) nan={nan_count}")
    return model


def main():
    parser = argparse.ArgumentParser(description="PyTorch LoRA training for AgentMind")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL data file")
    parser.add_argument("--domain", type=str, default="tool_caller", help="Domain name")
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-0.5B", help="Backbone model ID")
    parser.add_argument("--steps", type=int, default=2000, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(args.backbone, use_4bit=not args.no_4bit)
    model = apply_lora_peft(model, rank=args.lora_rank, alpha=args.lora_alpha)

    dataset = AgentDataset.from_raw(args.data, tokenizer, max_len=args.seq_len)
    print(f"Loaded {len(dataset)} samples from {args.data}")

    model = train_specialist(
        model, tokenizer, dataset,
        domain_name=args.domain,
        steps=args.steps,
        lr=args.lr,
        seq_len=args.seq_len,
        grad_accum=args.grad_accum,
        grad_clip=args.grad_clip,
        output_dir=args.output_dir,
        device=device,
    )

    output_dir = f"{args.output_dir}/adapters/{args.domain}"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved adapter and tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
