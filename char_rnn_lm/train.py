"""
Training Script for Character-Level GRU Language Model.
Includes:
- AdamW optimizer with weight decay
- Linear warmup + Cosine learning rate decay scheduler
- Gradient clipping (prevents exploding gradients)
- Multilingual validation tracking: separate loss for Hindi/Hinglish val set
- tqdm live progress bar (step, lr, train loss, val loss, val_hindi loss)
- Checkpoint persistence (best_model.pt and latest_model.pt)
- Overfitting detection (tracks step where val loss begins diverging)
- Loss curve plotting (saved as loss_curve.png)
- Argparse CLI options
"""

import os
import sys
import math
import argparse
import json
from typing import Optional
import time
import torch
import torch.nn as nn
from dataclasses import asdict
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from model import CharGRUModel
from data import load_dataset_splits, get_batch, CharTokenizer

# Ensure safe UTF-8 terminal output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_cosine_lr(
    step: int,
    max_steps: int,
    warmup_steps: int,
    learning_rate: float,
    min_lr: float
) -> float:
    """Computes learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return learning_rate * (step + 1) / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    val_multi_data: Optional[torch.Tensor],
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: str
) -> dict:
    """
    Evaluates average loss over several batches for:
    1. train: overall training split
    2. val: overall validation split
    3. val_hindi: dedicated validation split containing ONLY Hindi/Hinglish lines
    """
    out = {}
    model.eval()
    eval_splits = [("train", train_data), ("val", val_data)]
    if val_multi_data is not None and len(val_multi_data) > block_size + 2:
        eval_splits.append(("val_hindi", val_multi_data))

    for split_name, dataset in eval_splits:
        losses = torch.zeros(eval_iters)
        # Dynamically size batch if dataset is compact
        cur_batch_size = min(batch_size, max(1, (len(dataset) - block_size - 1) // 2))
        for k in range(eval_iters):
            x, y = get_batch(dataset, cur_batch_size, block_size, device)
            _, loss, _ = model(x, targets=y)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()
    model.train()
    return out


def plot_loss_curves(
    steps: list,
    train_losses: list,
    val_losses: list,
    val_hindi_losses: Optional[list] = None,
    output_path: str = "loss_curve.png"
):
    """Saves training, validation (all), and validation (Hindi/Hinglish) loss curves."""
    plt.figure(figsize=(9, 5))
    plt.plot(steps, train_losses, label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(steps, val_losses, label="Val Loss (All)", color="#ff7f0e", linewidth=2, linestyle="--")
    if val_hindi_losses and len(val_hindi_losses) == len(steps):
        plt.plot(steps, val_hindi_losses, label="Val Loss (Hindi/Hinglish)", color="#2ca02c", linewidth=2, linestyle="-.")
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.title("Character-Level GRU Training & Multilingual Validation Loss Curves", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"📈 Loss curves plot saved to: {output_path}")


def train(args):
    # 1. Device selection
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"⚡ Device: CUDA GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("💻 Device: CPU (Autocast mixed precision skipped - CPU float32 is optimal)")

    start_time = time.time()

    # 2. Load Configuration Preset & Overrides
    cfg = Config.get_preset(args.preset)
    if args.steps:
        cfg.max_steps = args.steps
    if args.eval_interval:
        cfg.eval_interval = args.eval_interval
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    if args.min_lr:
        cfg.min_lr = args.min_lr
    if args.hidden_dim:
        cfg.hidden_dim = args.hidden_dim
    if args.embedding_dim:
        cfg.embedding_dim = args.embedding_dim
    if args.num_layers:
        cfg.num_layers = args.num_layers
    if args.block_size:
        cfg.block_size = args.block_size
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
    cfg.device = device

    torch.manual_seed(cfg.seed)

    # 3. Load Data Splits & Tokenizer
    train_file = os.path.join("data", "train.txt")
    val_file = os.path.join("data", "val.txt")
    val_multi_file = os.path.join("data", "val_multilingual.txt")
    
    train_data, val_data, val_multi_data, tokenizer = load_dataset_splits(train_file, val_file, val_multi_file)
    cfg.vocab_size = tokenizer.vocab_size
    tokenizer.save_vocab(cfg.vocab_path)

    print("=" * 70)
    print(f"🏋️ Starting Training: Preset '{cfg.preset_name.upper()}' on {device.upper()}")
    print("=" * 70)
    print(f"Max Steps         : {cfg.max_steps}")
    print(f"Eval Interval     : {cfg.eval_interval}")
    print(f"Batch Size        : {cfg.batch_size}")
    print(f"Block Size        : {cfg.block_size}")
    print(f"Learning Rate     : {cfg.learning_rate} (Warmup: {cfg.warmup_steps})")
    print(f"Embedding/Hidden  : {cfg.embedding_dim} / {cfg.hidden_dim}")
    print(f"GRU Layers        : {cfg.num_layers}")
    print(f"Vocabulary Size   : {cfg.vocab_size}")

    # 4. Instantiate Model & Optimizer
    model = CharGRUModel(cfg).to(device)
    print(f"Total Parameters  : {model.get_num_params():,d}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    start_step = 0
    best_val_loss = float("inf")
    best_step = 0
    patience_counter = 0

    # 5. Checkpoints Setup
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
    latest_ckpt_path = os.path.join(cfg.checkpoint_dir, "latest_model.pt")

    if args.resume and os.path.exists(latest_ckpt_path):
        print(f"🔄 Resuming from checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"] + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        best_step = checkpoint.get("step", 0)
        print(f"   Resumed at step {start_step} (Best Val Loss: {best_val_loss:.4f} at step {best_step})")

    eval_steps_record = []
    train_losses_record = []
    val_losses_record = []
    val_hindi_losses_record = []

    # 6. Training Loop
    model.train()
    pbar = tqdm(range(start_step, cfg.max_steps), desc="Training", unit="step")
    
    for step in pbar:
        # Dynamic Learning Rate Schedule
        lr = get_cosine_lr(
            step,
            cfg.max_steps,
            cfg.warmup_steps,
            cfg.learning_rate,
            cfg.min_lr
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Fetch Batch
        x, y = get_batch(train_data, cfg.batch_size, cfg.block_size, device=device)

        # Forward & Loss
        logits, loss, _ = model(x, targets=y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient Clipping
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        # Validation Evaluation
        if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            eval_metrics = estimate_loss(
                model, train_data, val_data, val_multi_data,
                cfg.batch_size, cfg.block_size, cfg.eval_iters, device
            )
            tr_loss = eval_metrics["train"]
            va_loss = eval_metrics["val"]
            va_hn_loss = eval_metrics.get("val_hindi", float("nan"))

            eval_steps_record.append(step)
            train_losses_record.append(tr_loss)
            val_losses_record.append(va_loss)
            if not math.isnan(va_hn_loss):
                val_hindi_losses_record.append(va_hn_loss)

            pbar.set_postfix({
                "tr": f"{tr_loss:.3f}",
                "val": f"{va_loss:.3f}",
                "val_hn": f"{va_hn_loss:.3f}",
                "lr": f"{lr:.2e}"
            })
            tqdm.write(
                f"Step {step:4d}/{cfg.max_steps} | "
                f"Train Loss: {tr_loss:.4f} | "
                f"Val Loss: {va_loss:.4f} | "
                f"Val Hindi Loss: {va_hn_loss:.4f} | "
                f"LR: {lr:.2e}"
            )
            sys.stdout.flush()

            # Checkpoint Payload
            latest_payload = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "val_hindi_loss": va_hn_loss,
                "config": asdict(cfg),
                "vocab": tokenizer.stoi
            }
            torch.save(latest_payload, latest_ckpt_path)

            # Best Model Checkpoint
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_step = step
                patience_counter = 0
                torch.save(latest_payload, best_ckpt_path)
            else:
                patience_counter += 1
                if args.early_stopping and patience_counter >= args.patience:
                    print(f"\n🛑 Early stopping triggered at step {step} (Patience: {args.patience})")
                    break

    # 7. Overfitting Analysis & Summary
    total_time_sec = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🎉 Training Finished! (Total Steps: {cfg.max_steps})")
    print(f"   ⏱️ Total Training Time  : {total_time_sec:.2f} seconds ({total_time_sec / 60:.2f} minutes)")
    print(f"   🧠 Total Parameters     : {model.get_num_params():,d}")
    print(f"   Lowest Validation Loss : {best_val_loss:.4f} achieved at Step {best_step}")
    print(f"   Best Model Saved to    : {best_ckpt_path} (from Step {best_step})")
    print(f"   Latest Model Saved to  : {latest_ckpt_path}")

    # Detect Overfitting Step
    if best_step < cfg.max_steps - cfg.eval_interval:
        print(f"\n🔍 Overfitting Analysis:")
        print(f"   Step {best_step} ke baad validation loss plateau ya badhne laga jabki train loss girta raha.")
        print(f"   isliye 'best_model.pt' strictly Step {best_step} ke weights par preserved hai!")
    else:
        print("\n🔍 Overfitting Analysis:")
        print("   Model ne end tak healthy learning ki (validation loss aakhiri steps tak improve hua).")

    if eval_steps_record:
        curve_path = os.path.join(cfg.checkpoint_dir, "loss_curve.png")
        plot_loss_curves(eval_steps_record, train_losses_record, val_losses_record, val_hindi_losses_record, curve_path)
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Character-level GRU Language Model")
    parser.add_argument("--preset", type=str, default="small", choices=["tiny", "small", "medium"])
    parser.add_argument("--steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--eval_interval", type=int, default=None, help="Override validation evaluation interval")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--min_lr", type=float, default=None, help="Override minimum learning rate")
    parser.add_argument("--hidden_dim", type=int, default=None, help="Override hidden dimension")
    parser.add_argument("--embedding_dim", type=int, default=None, help="Override embedding dimension")
    parser.add_argument("--num_layers", type=int, default=None, help="Override number of GRU layers")
    parser.add_argument("--block_size", type=int, default=None, help="Override context block size")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping evaluation patience")
    args = parser.parse_args()

    train(args)
