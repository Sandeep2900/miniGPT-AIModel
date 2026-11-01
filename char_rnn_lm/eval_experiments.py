"""
Evaluation script to compute KL divergence between Hinglish and English prompt distributions
and generate sample text across seeds.
"""

import os
import sys
import torch
import torch.nn.functional as F
from config import Config
from model import CharGRUModel
from data import CharTokenizer
from generate import load_model_and_tokenizer

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def compute_kl_divergence(model, tokenizer, prompt_1="Aaj subah ", prompt_2="The morning ", device="cpu"):
    """
    Computes KL Divergence between next-character probability distributions:
    D_KL(p || q) and D_KL(q || p), along with L1 norm of logit difference.
    """
    model.eval()
    id1 = torch.tensor([tokenizer.encode(prompt_1)], dtype=torch.long, device=device)
    id2 = torch.tensor([tokenizer.encode(prompt_2)], dtype=torch.long, device=device)

    # Process prompts through GRU
    out1, _ = model.gru(model.token_embedding(id1), None)
    out2, _ = model.gru(model.token_embedding(id2), None)

    logits1 = model.lm_head(out1[:, -1, :])  # (1, vocab_size)
    logits2 = model.lm_head(out2[:, -1, :])  # (1, vocab_size)

    # Convert to probability distributions with smoothing
    p = F.softmax(logits1, dim=-1)
    q = F.softmax(logits2, dim=-1)

    eps = 1e-10
    p_safe = torch.clamp(p, min=eps)
    q_safe = torch.clamp(q, min=eps)

    kl_p_q = torch.sum(p_safe * torch.log(p_safe / q_safe)).item()
    kl_q_p = torch.sum(q_safe * torch.log(q_safe / p_safe)).item()
    js_div = 0.5 * (kl_p_q + kl_q_p)
    logit_diff_norm = torch.norm(logits1 - logits2).item()

    return {
        "kl_p_q": kl_p_q,
        "kl_q_p": kl_q_p,
        "js_div": js_div,
        "logit_diff_norm": logit_diff_norm,
        "top5_p1": [(tokenizer.itos[i.item()], round(p[0, i].item() * 100, 2)) for i in torch.topk(p[0], 5).indices],
        "top5_p2": [(tokenizer.itos[i.item()], round(q[0, i].item() * 100, 2)) for i in torch.topk(q[0], 5).indices],
    }


def generate_with_seed(model, tokenizer, prompt, seed, max_new_tokens=150, temperature=0.7, top_k=40, top_p=0.9, device="cpu"):
    torch.manual_seed(seed)
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor([encoded], dtype=torch.long, device=device)
    out_idx = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    return tokenizer.decode(out_idx[0].tolist())
