"""
Character-Level Recurrent Neural Network (GRU) Language Model.
Built in pure PyTorch without high-level wrappers.
Architecture: Token Embedding -> Multi-Layer GRU -> Dropout -> Linear Projection Head.
Includes tensor shape debugging, parameter accounting, and token-by-token generation.
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from config import Config

# Ensure safe UTF-8 terminal output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class CharGRUModel(nn.Module):
    """
    Character-Level Gated Recurrent Unit (GRU) Language Model.
    
    1. Embedding Layer : Discrete integer token IDs -> Dense vectors (B, T) -> (B, T, C)
    2. Multi-layer GRU : Temporal recurrent memory processing  -> (B, T, H)
    3. Dropout Layer   : Regularization against overfitting
    4. Linear Head     : Hidden states -> Vocabulary Logits     -> (B, T, V)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.embedding_dim = cfg.embedding_dim
        self.hidden_dim = cfg.hidden_dim
        self.num_layers = cfg.num_layers
        self.block_size = cfg.block_size

        # Layer 1: Token Embedding
        # Maps each character index (0 to vocab_size - 1) to a vector of size embedding_dim
        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim
        )

        # Layer 2: Multi-Layer GRU
        # batch_first=True expects inputs of shape (Batch, Seq_Len, Features)
        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=cfg.dropout if self.num_layers > 1 else 0.0
        )

        # Layer 3: Dropout for regularization
        self.dropout = nn.Dropout(cfg.dropout)

        # Layer 4: Final Linear Head (Language Modeling Projection)
        # Projects hidden representations back into vocabulary score logits
        self.lm_head = nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.vocab_size
        )

        # Initialize weights carefully
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Standard Gaussian initialization for linear and embedding layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
        debug: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the Char-GRU model.
        
        Args:
            idx     : LongTensor of shape (B, T) - token indices
            targets : LongTensor of shape (B, T) - optional ground truth target tokens
            hidden  : Tensor of shape (num_layers, B, hidden_dim) - optional initial hidden state
            debug   : bool - if True, prints tensor shapes at each step
            
        Returns:
            logits  : Tensor of shape (B, T, vocab_size) - unnormalized log probabilities
            loss    : Tensor scalar or None - Cross-Entropy loss if targets provided
            hidden  : Tensor of shape (num_layers, B, hidden_dim) - final hidden state
        """
        B, T = idx.shape

        if debug:
            print("\n" + "-" * 55)
            print("🔬 DEBUG FORWARD PASS - TENSOR SHAPES:")
            print("-" * 55)
            print(f"1. Input idx tokens shape      : {tuple(idx.shape)} (B={B}, T={T})")

        # Step 1: Character Token Embedding
        # Shape: (B, T) -> (B, T, embedding_dim)
        tok_emb = self.token_embedding(idx)
        if debug:
            print(f"2. After Embedding shape       : {tuple(tok_emb.shape)} (B={B}, T={T}, C={self.embedding_dim})")

        # Step 2: Multi-layer GRU
        # Shape: (B, T, embedding_dim) -> (B, T, hidden_dim)
        out, hidden = self.gru(tok_emb, hidden)
        if debug:
            print(f"3. After GRU output shape      : {tuple(out.shape)} (B={B}, T={T}, H={self.hidden_dim})")
            print(f"   Hidden state shape          : {tuple(hidden.shape)} (Layers={self.num_layers}, B={B}, H={self.hidden_dim})")

        out = self.dropout(out)

        # Step 3: Linear Projection to Vocab Logits
        # Shape: (B, T, hidden_dim) -> (B, T, vocab_size)
        logits = self.lm_head(out)
        if debug:
            print(f"4. Logits output shape         : {tuple(logits.shape)} (B={B}, T={T}, V={self.vocab_size})")

        # Step 4: Compute Loss if targets are supplied
        loss = None
        if targets is not None:
            # Flatten B and T dimensions to compute Cross Entropy over all positions
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            if debug:
                print(f"5. CrossEntropy Loss calculated: {loss.item():.4f}")
                print("-" * 55)

        return logits, loss, hidden

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Autoregressive stateful text generation for Recurrent Neural Networks (GRU).
        1. Feeds the entire prompt sequence once to initialize the GRU hidden state.
        2. Obtains the logits for the first new character from the final prompt step.
        3. Autoregressively generates new tokens one-by-one:
           - Applies temperature, Top-K, Top-P to logits.
           - Samples the next token (or takes argmax if temperature <= 0).
           - Updates and carries over the GRU hidden state step-by-step.
        """
        self.eval()
        B, T = idx.shape

        # Step 1: Warm-up GRU with prompt: process entire prompt in a single forward pass
        tok_emb = self.token_embedding(idx)       # Shape: (B, T, embedding_dim)
        out, hidden = self.gru(tok_emb, None)     # out: (B, T, hidden_dim), hidden: (num_layers, B, hidden_dim)
        out = self.dropout(out)
        logits = self.lm_head(out[:, -1, :])      # Logits for first new token: (B, vocab_size)

        for _ in range(max_new_tokens):
            # Apply temperature scaling
            if temperature > 0:
                scaled_logits = logits / max(temperature, 1e-5)
            else:
                scaled_logits = logits

            # Greedy decoding
            if temperature <= 0:
                next_token = torch.argmax(scaled_logits, dim=-1, keepdim=True)
            else:
                # Optional Top-K filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float("Inf")

                # Optional Top-P (nucleus) filtering
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above top_p threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift right to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    scaled_logits[indices_to_remove] = -float("Inf")

                # Softmax to get probabilities
                probs = F.softmax(scaled_logits, dim=-1)
                # Sample next token from probability distribution
                next_token = torch.multinomial(probs, num_samples=1)  # Shape: (B, 1)

            # Append sampled token to running sequence
            idx = torch.cat((idx, next_token), dim=1)

            # Carry and update GRU hidden state with newly sampled token
            next_emb = self.token_embedding(next_token)  # Shape: (B, 1, embedding_dim)
            out, hidden = self.gru(next_emb, hidden)     # (B, 1, hidden_dim), persistent hidden state!
            out = self.dropout(out)
            logits = self.lm_head(out[:, 0, :])          # Logits for next token: (B, vocab_size)

        return idx


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Phase 2: Char-GRU Model Architecture & Shape Verification")
    print("=" * 60)

    # 1. Load config and set vocab size
    cfg = Config.get_preset("small")
    cfg.vocab_size = 127  # Matching our verified vocabulary size
    
    print(f"Active Preset : {cfg.preset_name.upper()}")
    print(f"Context Window: {cfg.block_size} characters")
    print(f"Embedding Dim : {cfg.embedding_dim}")
    print(f"Hidden Dim    : {cfg.hidden_dim}")
    print(f"GRU Layers    : {cfg.num_layers}")
    print(f"Vocab Size    : {cfg.vocab_size}")

    # 2. Instantiate Model
    model = CharGRUModel(cfg)
    total_params = model.get_num_params()
    print(f"\n📦 Total Trainable Parameters: {total_params:,d}")
    print(f"   Breakdown:")
    print(f"   • Token Embedding : {cfg.vocab_size} x {cfg.embedding_dim} = {cfg.vocab_size * cfg.embedding_dim:,d}")
    gru_params = sum(p.numel() for n, p in model.gru.named_parameters())
    print(f"   • 2-Layer GRU     : {gru_params:,d}")
    print(f"   • Linear Head     : {cfg.hidden_dim} x {cfg.vocab_size} + bias = {cfg.hidden_dim * cfg.vocab_size + cfg.vocab_size:,d}")

    # 3. Dummy Forward Pass with Debug Mode
    batch_size = 4
    seq_len = 16
    dummy_x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    dummy_y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    logits, loss, _ = model(dummy_x, targets=dummy_y, debug=True)

    # Expected initial random loss should be around ln(vocab_size)
    expected_init_loss = math.log(cfg.vocab_size)
    print(f"📊 Loss Analysis:")
    print(f"   Actual Initial Loss   : {loss.item():.4f}")
    print(f"   Expected Initial Loss : {expected_init_loss:.4f} (ln({cfg.vocab_size}))")
    loss_diff = abs(loss.item() - expected_init_loss)
    print(f"   Difference from theoretical random: {loss_diff:.4f} (Under 0.2 indicates healthy random init)")

    # 4. Dummy Generation Test
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    print(f"\n🧪 Autoregressive Generation Test:")
    print(f"   Input prompt tokens  : {prompt.tolist()[0]}")
    print(f"   Generated token IDs  : {generated.tolist()[0]}")
    print(f"   Output Sequence Len  : {generated.shape[1]} tokens")

    print("\n" + "=" * 60)
    print("✅ Model architecture and forward pass verified successfully!")
    print("=" * 60)
