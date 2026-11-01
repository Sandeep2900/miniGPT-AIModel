# Fine-tune a new checkpoint on AI-reviewed records with assistant-only loss.

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
TOKENIZER_FILE = ROOT / "data" / "bpe_tokenizer.json"
BASE_CHECKPOINT = ROOT / "checkpoints" / "best_gpt.pt"
V2_DIR = ROOT / "checkpoints" / "v2"
OUTPUT_CHECKPOINT = V2_DIR / "best.pt"

SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
MAX_STEPS = 600
EVAL_INTERVAL = 25
EARLY_STOP_PATIENCE = 6


def load_and_check_data():
    """Validate the data and verify that its hash matches the AI manifest."""
    from validate_ai_reviewed import validate

    if not DATA_FILE.is_file() or not MANIFEST_FILE.is_file():
        raise FileNotFoundError("AI-reviewed dataset or manifest is missing.")
    issues, total, valid, _, _ = validate()
    if issues or valid != total:
        raise ValueError(f"AI-reviewed data failed validation: {dict(issues)}")

    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    actual_hash = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()
    if manifest.get("sha256") != actual_hash:
        raise ValueError("Dataset hash does not match the AI-review manifest.")
    if manifest.get("human_verified") is not False:
        raise ValueError("Manifest must explicitly identify this data as not human-verified.")

    records = [
        json.loads(line)
        for line in DATA_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records, manifest


def split_by_conversation(records):
    """Split whole conversation IDs into train and validation partitions."""
    conversations: dict[str, list[dict[str, object]]] = {}
    for record in records:
        conversations.setdefault(str(record["conversation_id"]), []).append(record)

    conversation_ids = list(conversations)
    random.Random(SEED).shuffle(conversation_ids)
    validation_count = max(1, round(len(conversation_ids) * 0.10))
    validation_ids = set(conversation_ids[:validation_count])
    train_records = [record for key, group in conversations.items() if key not in validation_ids for record in group]
    validation_records = [record for key, group in conversations.items() if key in validation_ids for record in group]
    if not train_records or not validation_records:
        raise ValueError("Conversation split must have non-empty train and validation partitions.")
    return train_records, validation_records


def encode_pair(record, tokenizer, block_size: int):
    """Encode one dialogue pair and mask prompt targets from the loss."""
    prefix = f"User: {record['user']}\nAssistant:"
    prefix_ids = tokenizer.encode(prefix).ids
    answer_ids = tokenizer.encode(f" {record['assistant']}\n").ids

    # Keep the full answer when possible and trim only the oldest prompt tokens.
    if len(answer_ids) > block_size:
        answer_ids = answer_ids[:block_size]
    max_prefix_ids = max(1, block_size + 1 - len(answer_ids))
    if len(prefix_ids) > max_prefix_ids:
        prefix_ids = prefix_ids[-max_prefix_ids:]

    full_ids = prefix_ids + answer_ids
    inputs = full_ids[:-1]
    targets = full_ids[1:].copy()
    first_answer_target = max(0, len(prefix_ids) - 1)
    for index in range(first_answer_target):
        targets[index] = -1
    return inputs, targets


def make_batch(records, tokenizer, pad_id: int, block_size: int, device: str):
    """Pad a batch on the right; padded targets use the ignored label -1."""
    encoded = [encode_pair(record, tokenizer, block_size) for record in records]
    max_length = max(len(item[0]) for item in encoded)
    input_batch = torch.full((len(encoded), max_length), pad_id, dtype=torch.long)
    target_batch = torch.full((len(encoded), max_length), -1, dtype=torch.long)
    for row, (inputs, targets) in enumerate(encoded):
        input_batch[row, : len(inputs)] = torch.tensor(inputs, dtype=torch.long)
        target_batch[row, : len(targets)] = torch.tensor(targets, dtype=torch.long)
    return input_batch.to(device), target_batch.to(device)


@torch.no_grad()
def estimate_loss(model, records, tokenizer, pad_id, block_size, device):
    """Average assistant-only loss over deterministic validation batches."""
    model.eval()
    losses = []
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        inputs, targets = make_batch(batch, tokenizer, pad_id, block_size, device)
        _, loss = model(inputs, targets=targets)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def save_checkpoint(model, config, step, val_loss, train_loss, manifest, data_hash):
    """Save only the new v2 checkpoint path."""
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "base_checkpoint": "checkpoints/best_gpt.pt",
            "base_checkpoint_sha256": hashlib.sha256(BASE_CHECKPOINT.read_bytes()).hexdigest(),
            "ai_review_manifest_sha256": hashlib.sha256(MANIFEST_FILE.read_bytes()).hexdigest(),
            "training_data_sha256": data_hash,
            "ai_reviewed_only": True,
            "human_verified": False,
        },
        OUTPUT_CHECKPOINT,
    )


def main():
    if OUTPUT_CHECKPOINT.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {OUTPUT_CHECKPOINT}")

    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from gpt_model import GPTConfig, MiniGPT
    from tokenizers import Tokenizer

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    records, manifest = load_and_check_data()
    train_records, validation_records = split_by_conversation(records)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    base = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)
    config = GPTConfig(**base["config"])
    config.device = "cpu"
    config.dropout = 0.1
    if config.vocab_size != tokenizer.get_vocab_size():
        raise ValueError("Checkpoint and tokenizer vocabulary sizes do not match.")

    model = MiniGPT(config).to("cpu")
    model.load_state_dict(base["model_state_dict"])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise ValueError("Tokenizer has no <pad> token.")

    V2_DIR.mkdir(parents=True, exist_ok=True)
    data_hash = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()
    best_val_loss = estimate_loss(model, validation_records, tokenizer, pad_id, config.block_size, "cpu")
    best_step = 0
    save_checkpoint(model, config, 0, best_val_loss, None, manifest, data_hash)
    print(f"Base checkpoint loaded: step={base.get('step')} val_loss={base.get('val_loss')}")
    print(f"AI-reviewed data: records={len(records)} train={len(train_records)} validation={len(validation_records)}")
    print(f"Assistant-only loss enabled; language model parameters={model.get_num_params():,}")
    print(f"Initial validation loss={best_val_loss:.4f} saved to {OUTPUT_CHECKPOINT}")

    sampler = random.Random(SEED + 1)
    patience = 0
    start_time = time.time()
    for step in range(1, MAX_STEPS + 1):
        batch_records = sampler.choices(train_records, k=BATCH_SIZE)
        inputs, targets = make_batch(batch_records, tokenizer, pad_id, config.block_size, "cpu")
        _, loss = model(inputs, targets=targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            train_loss = estimate_loss(model, train_records, tokenizer, pad_id, config.block_size, "cpu")
            val_loss = estimate_loss(model, validation_records, tokenizer, pad_id, config.block_size, "cpu")
            print(f"step={step} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                patience = 0
                save_checkpoint(model, config, step, val_loss, train_loss, manifest, data_hash)
                print(f"  saved new best checkpoint at step={best_step}")
            else:
                patience += 1
                if patience >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping at step={step}; best_step={best_step}")
                    break

    print(f"Training wall time seconds={time.time() - start_time:.1f}")
    print(f"Best validation loss={best_val_loss:.4f} at step={best_step}")
    print(f"New checkpoint: {OUTPUT_CHECKPOINT}")
    print("Original checkpoints/best_gpt.pt was loaded as initialization and not written.")


if __name__ == "__main__":
    main()
