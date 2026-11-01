import torch
import torch.nn.functional as F
from eval_experiments import compute_kl_divergence, generate_with_seed
from generate import load_model_and_tokenizer

ckpts = {
    'Exp A (Best @ Step 4000)': 'checkpoints/exp_a/best_model.pt',
    'Exp B (Best @ Step 1750)': 'checkpoints/exp_b/best_model.pt',
    'Exp C (Best @ Step 1500)': 'checkpoints/exp_c/best_model.pt',
    'Old 1500-step baseline' : 'checkpoints/best_model.pt',
}

print('=' * 75)
print('KL DIVERGENCE TEST: Hinglish ("Aaj subah ") vs English ("The morning ")')
print('=' * 75)
for name, ckpt_path in ckpts.items():
    model, tok, cfg, step, val_loss = load_model_and_tokenizer(ckpt_path)
    res = compute_kl_divergence(model, tok, 'Aaj subah ', 'The morning ')
    print(f'\n--- {name} (Val Loss: {val_loss:.4f}) ---')
    print(f'Logits Diff L1 Norm   : {res["logit_diff_norm"]:.4f}')
    print(f'KL Divergence D(p||q) : {res["kl_p_q"]:.6f}')
    print(f'KL Divergence D(q||p) : {res["kl_q_p"]:.6f}')
    print(f'Jensen-Shannon Div    : {res["js_div"]:.6f}')
    print(f'Top-5 for "Aaj subah "  : {res["top5_p1"]}')
    print(f'Top-5 for "The morning ": {res["top5_p2"]}')
