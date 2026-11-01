"""
Configuration Dataclass & Hardware-tuned Presets.
Centralizes all hyperparameters for data, model architecture, and training.
Includes rebalancing options: max_kb_per_file and upsample_multilingual_train.
"""

import os
import json
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # 1. Data Pipeline & Paths
    raw_dir: str = os.path.join("data", "raw")
    data_path: str = os.path.join("data", "data.txt")
    train_path: str = os.path.join("data", "train.txt")
    val_path: str = os.path.join("data", "val.txt")
    vocab_path: str = os.path.join("data", "vocab.json")
    checkpoint_dir: str = "checkpoints"
    train_val_split: float = 0.90
    seed: int = 42

    # 2. Rebalancing & Capping Options
    max_kb_per_file: int = 50           # Max KB per raw file (continuous slice from beginning)
    upsample_multilingual_train: int = 1 # Repeat factor for multilingual files in TRAIN split only (1 = no upsampling)

    # 3. Sequence & Batch Dimensions
    block_size: int = 64       # Context window length (characters)
    batch_size: int = 32       # Number of parallel sequences processed per step
    
    # 4. Model Architecture (Char-GRU)
    vocab_size: int = 130      # Will be updated dynamically by data pipeline
    embedding_dim: int = 64    # Size of character embedding vectors
    hidden_dim: int = 128      # Hidden state size in each GRU cell
    num_layers: int = 2        # Number of stacked GRU layers
    dropout: float = 0.1       # Dropout probability for regularization
    
    # 5. Training Optimization
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 1e-2
    max_steps: int = 1500      # Total training iterations
    warmup_steps: int = 50     # Linear learning rate warmup steps
    eval_interval: int = 100   # Compute validation loss every N steps
    eval_iters: int = 20       # Number of validation batches to average
    grad_clip: float = 1.0     # Max gradient norm to prevent exploding gradients
    
    # 6. Device & Preset
    device: str = "cpu"
    preset_name: str = "small"

    @classmethod
    def get_preset(cls, preset_name: str = "small") -> "Config":
        """
        Returns a configured Config object based on preset name:
        - 'tiny'   : ~10K params | ~150 MB RAM | ~2-3 mins on CPU (Sanity check)
        - 'small'  : ~130K params | ~250 MB RAM | ~10-15 mins on CPU (Balanced & Recommended)
        - 'medium' : ~500K params | ~450 MB RAM | ~25-35 mins on CPU (High quality within 30 min budget)
        """
        preset_name = preset_name.lower().strip()
        if preset_name == "tiny":
            return cls(
                preset_name="tiny",
                block_size=32,
                batch_size=16,
                embedding_dim=32,
                hidden_dim=64,
                num_layers=1,
                dropout=0.0,
                max_steps=500,
                learning_rate=2e-3,
                eval_interval=50
            )
        elif preset_name == "small":
            return cls(
                preset_name="small",
                block_size=64,
                batch_size=32,
                embedding_dim=64,
                hidden_dim=128,
                num_layers=2,
                dropout=0.1,
                max_steps=1500,
                learning_rate=1e-3,
                eval_interval=100
            )
        elif preset_name == "medium":
            return cls(
                preset_name="medium",
                block_size=128,
                batch_size=32,
                embedding_dim=128,
                hidden_dim=256,
                num_layers=2,
                dropout=0.15,
                max_steps=3000,
                learning_rate=8e-4,
                eval_interval=150
            )
        else:
            raise ValueError(f"Unknown preset '{preset_name}'. Choose from: 'tiny', 'small', 'medium'")

    def to_json(self, filepath: str) -> None:
        """Save configuration dictionary to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> "Config":
        """Load configuration from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


if __name__ == "__main__":
    print("=" * 60)
    print("[*] Configuration Module & Presets Summary")
    print("=" * 60)
    cfg = Config.get_preset("small")
    print(f"Active Preset: {cfg.preset_name.upper()}")
    print(f"  Max KB Per File:            {cfg.max_kb_per_file} KB")
    print(f"  Upsample Multilingual Train: {cfg.upsample_multilingual_train}x")
    print(f"  Context Window (block_size): {cfg.block_size}")
    print(f"  Batch Size:                 {cfg.batch_size}")
    print(f"  Embedding / Hidden Dim:     {cfg.embedding_dim} / {cfg.hidden_dim}")
    print(f"  GRU Layers:                 {cfg.num_layers}")
    print(f"  Max Steps:                  {cfg.max_steps}")
    print("=" * 60)
