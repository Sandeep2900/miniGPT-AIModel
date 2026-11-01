# Fine-tune only the final transformer block to reduce forgetting.

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
APPROVED_DIR = Path(__file__).resolve().parent
DATA_FILE = APPROVED_DIR / "ai_reviewed_pairs.jsonl"
MANIFEST_FILE = APPROVED_DIR / "ai_review_manifest.json"
BASE_CHECKPOINT = ROOT / "checkpoints" / "best_gpt.pt"
TOKENIZER_FILE = ROOT / "data" / "bpe_tokenizer.json"
OUTPUT_CHECKPOINT = ROOT / "checkpoints" / "v2" / "best_conservative.pt"
SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 3e-5
MAX_STEPS = 120
EVAL_INTERVAL = 10
PATIENCE_LIMIT = 5
GRAD_CLIP = 1.0


def load_data():
    """Validate AI-reviewed data and verify the manifest hash."""
    from validate_ai_reviewed import validate

    issues, total, valid, _, _ = validate()
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    digest = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()
    if issues or valid != total or manifest.get("sha256") != digest:
        raise ValueError("Dataset validation or manifest hash check failed.")
    if manifest.get("human_verified") is not False:
        raise ValueError("Expected an AI-reviewed, not human-verified, manifest.")
    return [json.loads(line) for line in DATA_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_records(records):
    """Split whole conversation IDs into train and validation sets."""
    groups = {}
    for record in records:
        groups.setdefault(record["conversation_id"], []).append(record)
    ids = list(groups)
    random.Random(SEED).shuffle(ids)
    validation_ids = set(ids[: max(1, round(len(ids) * 0.10))])
    train = [record for key, group in groups.items() if key not in validation_ids for record in group]
    validation = [record for key, group in groups.items() if key in validation_ids for record in group]
    return train, validation


def encode_pair(record, tokenizer, block_size):
    """Encode a turn and mask all prompt-token targets."""
    prefix = f"User: {record['user']}\nAssistant:"
    prefix_ids = tokenizer.encode(prefix).ids
    answer_ids = tokenizer.encode(f" {record['assistant']}\n").ids
    answer_ids = answer_ids[:block_size]
    prefix_ids = prefix_ids[-max(1, block_size + 1 - len(answer_ids)) :]
    full_ids = prefix_ids + answer_ids
    inputs = full_ids[:-1]
    targets = full_ids[1:].copy()
    for index in range(max(0, len(prefix_ids) - 1)):
        targets[index] = -1
    return inputs, targets


def make_batch(records, tokenizer, pad_id, block_size):
    """Right-pad inputs and ignore padded target positions."""
    pairs = [encode_pair(record, tokenizer, block_size) for record in records]
    width = max(len(inputs) for inputs, _ in pairs)
    inputs = torch.full((len(pairs), width), pad_id, dtype=torch.long)
    targets = torch.full((len(pairs), width), -1, dtype=torch.long)
    for row, (x, y) in enumerate(pairs):
        inputs[row, : len(x)] = torch.tensor(x)
        targets[row, : len(y)] = torch.tensor(y)
    return inputs, targets


@torch.no_grad()
def evaluate(model, records, tokenizer, pad_id, block_size):
    """Compute mean assistant-only loss across a fixed split."""
    model.eval()
    values = []
    for start in range(0, len(records), BATCH_SIZE):
        x, y = make_batch(records[start : start + BATCH_SIZE], tokenizer, pad_id, block_size)
        _, loss = model(x, targets=y)
        values.append(float(loss.item()))
    model.train()
    return sum(values) / max(1, len(values))


def save_best(model, config, step, loss, data_hash, manifest_hash, base_hash):
    """Save to a new conservative checkpoint without replacing best.pt."""
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "val_loss": loss,
            "base_checkpoint": "checkpoints/best_gpt.pt",
            "base_checkpoint_sha256": base_hash,
            "training_data_sha256": data_hash,
            "ai_review_manifest_sha256": manifest_hash,
            "ai_reviewed_only": True,
            "human_verified": False,
            "trainable_scope": "final_transformer_block_and_final_layer_norm",
        },
        OUTPUT_CHECKPOINT,
    )


def main():
    if OUTPUT_CHECKPOINT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT_CHECKPOINT}")

    sys.path.insert(0, str(ROOT))
    from gpt_model import GPTConfig, MiniGPT
    from tokenizers import Tokenizer

    random.seed(SEED)
    torch.manual_seed(SEED)
    records = load_data()
    train_records, val_records = split_records(records)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    base = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)
    config = GPTConfig(**base["config"])
    config.device = "cpu"
    config.dropout = 0.1
    model = MiniGPT(config).to("cpu")
    model.load_state_dict(base["model_state_dict"])

    # Freeze the embeddings and earlier blocks to preserve base language patterns.
    for parameter in model.parameters():
        parameter.requires_grad = False
    final_block = model.transformer.h[-1]
    for parameter in final_block.parameters():
        parameter.requires_grad = True
    for parameter in model.transformer.ln_f.parameters():
        parameter.requires_grad = True
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise ValueError("Tokenizer is missing <pad>.")

    OUTPUT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    data_hash = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(MANIFEST_FILE.read_bytes()).hexdigest()
    base_hash = hashlib.sha256(BASE_CHECKPOINT.read_bytes()).hexdigest()
    best_loss = evaluate(model, val_records, tokenizer, pad_id, config.block_size)
    best_step = 0
    save_best(model, config, best_step, best_loss, data_hash, manifest_hash, base_hash)
    print(f"Train pairs={len(train_records)} validation pairs={len(val_records)}")
    print(f"Trainable parameters={sum(parameter.numel() for parameter in trainable):,}")
    print(f"Initial validation loss={best_loss:.4f}")

    sampler = random.Random(SEED + 1)
    patience = 0
    start_time = time.time()
    for step in range(1, MAX_STEPS + 1):
        model.train()
        batch = sampler.choices(train_records, k=BATCH_SIZE)
        x, y = make_batch(batch, tokenizer, pad_id, config.block_size)
        _, loss = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            val_loss = evaluate(model, val_records, tokenizer, pad_id, config.block_size)
            print(f"step={step} train_batch_loss={float(loss.item()):.4f} val_loss={val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                best_step = step
                patience = 0
                save_best(model, config, step, val_loss, data_hash, manifest_hash, base_hash)
                print(f"  saved best conservative checkpoint at step={best_step}")
            else:
                patience += 1
                if patience >= PATIENCE_LIMIT:
                    print(f"Early stopping at step={step}; best_step={best_step}")
                    break

    print(f"Training seconds={time.time() - start_time:.1f}")
    print(f"Best validation loss={best_loss:.4f} at step={best_step}")
    print(f"Checkpoint={OUTPUT_CHECKPOINT}")


if __name__ == "__main__":
    main()
