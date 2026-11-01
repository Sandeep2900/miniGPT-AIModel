"""
finetune_tool.py
Fine-tunes the 3.08M MiniGPT model to learn [TOOL: search("<query>")] call patterns
while preserving natural conversational ability.
Leaves checkpoints/best_gpt.pt completely untouched.
Saves fine-tuned weights to checkpoints/tool_gpt.pt.
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from gpt_model import MiniGPT, GPTConfig

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_CHECKPOINT = os.path.join("checkpoints", "best_gpt.pt")
TOOL_CHECKPOINT = os.path.join("checkpoints", "tool_gpt.pt")
TOKENIZER_PATH = os.path.join("data", "bpe_tokenizer.json")
TRAIN_TXT = os.path.join("data", "tool_train.txt")
VAL_TXT = os.path.join("data", "tool_val.txt")


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, iters, device):
    model.eval()
    losses = {}
    for split, dataset in [("train", train_data), ("val", val_data)]:
        loss_vals = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(dataset, batch_size, block_size, device)
            _, loss = model(x, targets=y)
            loss_vals[k] = loss.item()
        losses[split] = loss_vals.mean().item()
    model.train()
    return losses


def finetune():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 68)
    print(f"🔥 Fine-Tuning MiniGPT for Tool Calls on {device.upper()}")
    print("=" * 68)

    # 1. Load Tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    # 2. Prepare Data Tensors
    with open(TRAIN_TXT, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(VAL_TXT, "r", encoding="utf-8") as f:
        val_text = f.read()

    train_ids = tokenizer.encode(train_text).ids
    val_ids = tokenizer.encode(val_text).ids
    train_data = torch.tensor(train_ids, dtype=torch.long)
    val_data = torch.tensor(val_ids, dtype=torch.long)

    print(f"Dataset: Train Tokens = {len(train_data):,d} | Val Tokens = {len(val_data):,d}")

    # 3. Load Base Checkpoint
    if not os.path.exists(BASE_CHECKPOINT):
        raise FileNotFoundError(f"Base checkpoint {BASE_CHECKPOINT} not found!")

    ckpt = torch.load(BASE_CHECKPOINT, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = GPTConfig(**cfg_dict)
    cfg.device = device

    model = MiniGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    base_val_loss = ckpt.get("val_loss", 0.0)
    print(f"Loaded base model: {model.get_num_params():,d} params (Base Val Loss: {base_val_loss:.4f})")

    # 4. Hyperparameters
    max_steps = 180
    batch_size = 8
    block_size = cfg.block_size
    lr = 4e-5  # Low fine-tuning learning rate to preserve conversational ability
    eval_interval = 30
    eval_iters = 10

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print("\nStarting fine-tuning...")
    best_val_loss = float("inf")
    start_time = time.time()

    for step in range(1, max_steps + 1):
        x, y = get_batch(train_data, batch_size, block_size, device)
        logits, loss = model(x, targets=y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == max_steps:
            losses = estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device)
            print(f"Step {step:03d}/{max_steps} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "val_loss": best_val_loss,
                    "base_checkpoint": BASE_CHECKPOINT,
                }, TOOL_CHECKPOINT)
                print(f"  ⭐ Saved best tool model to {TOOL_CHECKPOINT} (Val Loss: {best_val_loss:.4f})")

    elapsed = time.time() - start_time
    print(f"\n✅ Fine-tuning complete in {elapsed:.1f}s! Best Val Loss: {best_val_loss:.4f}")
    print(f"Original checkpoint preserved: {BASE_CHECKPOINT}")
    print(f"Fine-tuned tool checkpoint: {TOOL_CHECKPOINT}")


if __name__ == "__main__":
    finetune()
