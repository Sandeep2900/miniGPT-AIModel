"""
chat_gpt.py
Interactive Command-Line Chat Interface for trained custom Mini-GPT model.
Uses BPE tokenizer (data/bpe_tokenizer.json) and trained weights (checkpoints/best_gpt.pt).
"""

import os
import sys
import torch
from tokenizers import Tokenizer
from gpt_model import MiniGPT, GPTConfig

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CHECKPOINT_PATH = os.path.join("checkpoints", "best_gpt.pt")
TOKENIZER_PATH = os.path.join("data", "bpe_tokenizer.json")


def load_model_and_tokenizer(checkpoint_path: str = CHECKPOINT_PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found! Train the model first.")
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer '{TOKENIZER_PATH}' not found!")

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg_dict = ckpt["config"]
    cfg = GPTConfig(**cfg_dict)
    cfg.device = device

    model = MiniGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_loss = ckpt.get("val_loss", None)
    step = ckpt.get("step", None)
    return model, tokenizer, cfg, step, val_loss, device


def chat_turn(model, tokenizer, user_text: str, device: str, max_new_tokens: int = 100, temperature: float = 0.35):
    """Formats turn, encodes with BPE, generates autoregressively, and decodes."""
    prompt = f"User: {user_text.strip()}\nAssistant:"
    encoded = tokenizer.encode(prompt).ids
    idx = torch.tensor([encoded], dtype=torch.long, device=device)

    generated_ids = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.12
    )[0].tolist()

    full_output = tokenizer.decode(generated_ids)
    # Extract only assistant's response cleanly
    if "Assistant:" in full_output:
        reply = full_output.split("Assistant:", 1)[1].strip()
    else:
        reply = full_output[len(prompt):].strip()

    # Remove any trailing user/assistant prompt turns or multiple newlines
    for stop_str in ["\nUser:", "\nAssistant:", "\n\n"]:
        if stop_str in reply:
            reply = reply.split(stop_str)[0]
    
    reply = reply.strip()
    if reply.startswith(":"):
        reply = reply.lstrip(":").strip()
    return reply


def interactive_chat():
    try:
        model, tokenizer, cfg, step, val_loss, device = load_model_and_tokenizer()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    print("=" * 68)
    print("🤖 Custom Mini-GPT (Decoder-Only Transformer) Interactive Chat")
    print("=" * 68)
    print(f"Architecture : {cfg.n_layer} Transformer Layers | {cfg.n_head} Heads | {cfg.n_embd} Dim")
    print(f"Parameters   : {model.get_num_params():,d} (100% custom trained from scratch)")
    print(f"Checkpoint   : Step {step} (Val Loss: {val_loss:.4f})")
    print("Instructions : Type in Hindi, Hinglish, or English. Type 'exit' to quit.")
    print("=" * 68)

    while True:
        try:
            user_input = input("\n📝 You > ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                print("\nAlvida! Chat session ended.")
                break
            if not user_input:
                continue

            response = chat_turn(model, tokenizer, user_input, device=device)
            print("\n🤖 Assistant:")
            print("-" * 50)
            print(response if response else "(empty response)")
            print("-" * 50)
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


if __name__ == "__main__":
    interactive_chat()
