"""
train_gpt.py
Production-grade training script for custom Decoder-Only Causal Transformer (Mini-GPT).
Features:
- Subword BPE Tokenizer integration
- AdamW optimizer with decoupled weight decay
- Linear warmup + Cosine decay scheduler
- Gradient clipping
- Checkpointing best_gpt.pt and latest_gpt.pt
- Live dialogue sample generation during evaluation
- Loss curve visualization
"""

import os
import sys
import math
import time
import argparse
from typing import Tuple, Optional
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from tokenizers import Tokenizer

from gpt_model import MiniGPT, GPTConfig

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TOKENIZER_PATH = os.path.join("data", "bpe_tokenizer.json")
TRAIN_BIN = os.path.join("data", "train_bpe.pt")
VAL_BIN = os.path.join("data", "val_bpe.pt")
CHECKPOINT_DIR = "checkpoints"


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts random chunks of sequences for input x and target y."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


def get_cosine_lr(step: int, max_steps: int, warmup_steps: int, lr: float, min_lr: float) -> float:
    """Computes learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(model: nn.Module, train_data: torch.Tensor, val_data: torch.Tensor, batch_size: int, block_size: int, eval_iters: int, device: str) -> dict:
    """Computes mean loss over multiple batches for train and val splits."""
    model.eval()
    out = {}
    for split_name, dataset in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(dataset, batch_size, block_size, device)
            _, loss = model(x, targets=y)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()
    model.train()
    return out


def plot_loss_curve(steps: list, train_losses: list, val_losses: list, output_path: str):
    """Plots and saves loss curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(steps, train_losses, label="Train Loss", color="#2563eb", linewidth=2)
    plt.plot(steps, val_losses, label="Validation Loss", color="#ea580c", linewidth=2)
    plt.title("Mini-GPT Language Model: Training & Validation Loss", fontsize=14)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss (Cross-Entropy)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 70)
    print(f"🚀 Training Custom Mini-GPT Architecture from Scratch on {device.upper()}")
    print("=" * 70)

    # 1. Load BPE Tokenizer & Binary Tensors
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"{TOKENIZER_PATH} not found!")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()

    train_data = torch.load(TRAIN_BIN, weights_only=True)
    val_data = torch.load(VAL_BIN, weights_only=True)

    # 2. Configure Model
    block_size = args.block_size
    batch_size = args.batch_size
    max_steps = args.steps
    eval_interval = args.eval_interval
    lr = args.lr
    min_lr = args.min_lr
    warmup_steps = args.warmup_steps

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        device=device
    )

    model = MiniGPT(config).to(device)
    total_params = model.get_num_params()
    print(f"Model Configuration : {config.n_layer} Layers | {config.n_head} Heads | {config.n_embd} Dim")
    print(f"Total Parameters    : {total_params:,d}")
    print(f"Context Window      : {block_size} subword tokens")
    print(f"Max Steps           : {max_steps:,d} (Eval every {eval_interval} steps)")
    print(f"Initial LR          : {lr} (Warmup: {warmup_steps} steps)")
    print("=" * 70)

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    best_val_loss = float("inf")
    best_step = 0
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_gpt.pt")
    latest_ckpt_path = os.path.join(CHECKPOINT_DIR, "latest_gpt.pt")

    if args.resume and os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        old_cfg = ckpt.get("config", {})
        if (old_cfg.get("n_layer") == config.n_layer and 
            old_cfg.get("n_head") == config.n_head and 
            old_cfg.get("n_embd") == config.n_embd and
            old_cfg.get("vocab_size") == config.vocab_size):
            print(f"🔄 Resuming model weights from {best_ckpt_path}...")
            model.load_state_dict(ckpt["model_state_dict"])
            prev_loss = ckpt.get("val_loss", "N/A")
            print(f"   Loaded checkpoint with previous Val Loss: {prev_loss}")
            best_val_loss = float("inf")
        else:
            print("⚠️ Checkpoint configuration differs from current model architecture. Starting fresh training from scratch!")

    eval_steps = []
    train_loss_history = []
    val_loss_history = []

    start_time = time.time()
    pbar = tqdm(range(max_steps), desc="Training Mini-GPT", unit="step")

    for step in pbar:
        # Cosine learning rate scheduling
        cur_lr = get_cosine_lr(step, max_steps, warmup_steps, lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr

        # Forward & Backward pass
        x, y = get_batch(train_data, batch_size, block_size, device)
        _, loss = model(x, targets=y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Periodic Evaluation
        if step % eval_interval == 0 or step == max_steps - 1:
            eval_metrics = estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters=8, device=device)
            tr_loss = eval_metrics["train"]
            va_loss = eval_metrics["val"]

            eval_steps.append(step)
            train_loss_history.append(tr_loss)
            val_loss_history.append(va_loss)

            pbar.set_postfix({"tr": f"{tr_loss:.3f}", "val": f"{va_loss:.3f}", "lr": f"{cur_lr:.1e}"})
            tqdm.write(f"Step {step:4d}/{max_steps} | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | LR: {cur_lr:.2e}")
            sys.stdout.flush()

            # Save latest checkpoint
            checkpoint_data = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "val_loss": va_loss,
                "train_loss": tr_loss,
            }
            torch.save(checkpoint_data, latest_ckpt_path)

            # Save best checkpoint
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_step = step
                torch.save(checkpoint_data, best_ckpt_path)

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🎉 Mini-GPT Training Completed!")
    print(f"Total Time        : {elapsed_time:.1f}s ({elapsed_time / 60:.2f} mins)")
    print(f"Best Val Loss     : {best_val_loss:.4f} at Step {best_step}")
    print(f"Best Model Saved  : {best_ckpt_path}")
    print("=" * 70)

    # Save loss curve
    loss_curve_file = os.path.join(CHECKPOINT_DIR, "gpt_loss_curve.png")
    plot_loss_curve(eval_steps, train_loss_history, val_loss_history, loss_curve_file)
    print(f"Loss curve saved to: {loss_curve_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mini-GPT Model from Scratch")
    parser.add_argument("--steps", type=int, default=4500, help="Total training steps")
    parser.add_argument("--eval_interval", type=int, default=250, help="Evaluation interval")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--block_size", type=int, default=128, help="Context length")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=192, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=1.2e-3, help="Max learning rate")
    parser.add_argument("--min_lr", type=float, default=1.2e-4, help="Min learning rate")
    parser.add_argument("--warmup_steps", type=int, default=200, help="Warmup steps")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint if available")
    args = parser.parse_args()

    train(args)
