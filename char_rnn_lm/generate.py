"""
Text Generation / Inference Script for Character-Level GRU Language Model.
Loads trained checkpoint (checkpoints/best_model.pt) and vocabulary.
Supports:
- Temperature scaling (entropy / randomness control)
- Top-K sampling (truncating unlikely tail tokens)
- Top-P nucleus sampling (dynamic probability mass threshold)
- Seed control for reproducibility
"""

import os
import sys
import argparse
import torch
from config import Config
from model import CharGRUModel
from data import CharTokenizer

# Ensure safe UTF-8 terminal output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_model_and_tokenizer(checkpoint_path: str, device: str = "cpu"):
    """Loads checkpoint state dictionary, configuration, and tokenizer."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' not found! Run 'python train.py' first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg_dict = checkpoint["config"]
    cfg = Config(**cfg_dict)
    cfg.device = device

    # Load tokenizer from checkpoint vocabulary mapping
    tokenizer = CharTokenizer()
    tokenizer.stoi = checkpoint["vocab"]
    tokenizer.itos = {int(v): k for k, v in checkpoint["vocab"].items()}
    tokenizer.chars = sorted(list(tokenizer.stoi.keys()))
    tokenizer.vocab_size = len(tokenizer.chars)
    cfg.vocab_size = tokenizer.vocab_size

    # Instantiate model and load trained weights
    model = CharGRUModel(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_loss = checkpoint.get("val_loss", None)
    step = checkpoint.get("step", None)
    return model, tokenizer, cfg, step, val_loss


def generate_text(
    model: CharGRUModel,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    device: str = "cpu"
) -> str:
    """Encodes prompt, runs autoregressive generation, and decodes back to text."""
    # Encode prompt (if empty, start with newline or first character)
    if not prompt:
        prompt = "\n"
    
    encoded = tokenizer.encode(prompt)
    if not encoded:
        encoded = [0]
        
    idx = torch.tensor([encoded], dtype=torch.long, device=device)
    generated_idx = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    return tokenizer.decode(generated_idx[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from trained Character-level GRU model")
    parser.add_argument("--prompt", type=str, default="आज सुबह ", help="Starting prompt string")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Number of new characters to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (lower=conservative, higher=creative)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K token filter (0 to disable)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus probability threshold (1.0 to disable)")
    parser.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints", "best_model.pt"))
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model, tokenizer, cfg, step, val_loss = load_model_and_tokenizer(args.checkpoint, device=device)
        print("=" * 65)
        print(f"[GENERATE] Checkpoint: {args.checkpoint}")
        if step is not None:
            print(f"   Trained Steps : {step} | Best Val Loss: {val_loss:.4f}")
        print(f"   Settings      : Temp={args.temperature}, Top-K={args.top_k}, Top-P={args.top_p}, Tokens={args.max_new_tokens}, Seed={args.seed}")
        print("=" * 65)
        print(f"Prompt: {args.prompt!r}\n")
        print("Generated Output:")
        print("-" * 65)

        # Seed right before generation for clean, reproducible autoregressive sampling
        if args.seed is not None:
            torch.manual_seed(args.seed)

        output = generate_text(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device
        )
        print(output)
        print("-" * 65)
    except FileNotFoundError as e:
        print(f"[!] {e}")
