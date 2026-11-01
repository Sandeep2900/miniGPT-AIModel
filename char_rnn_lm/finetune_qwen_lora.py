"""
finetune_qwen_lora.py
Fine-tunes Qwen2.5-0.5B-Instruct using LoRA (PEFT) on custom conversational JSONL pairs.
Loss is computed strictly on assistant response tokens (prompt loss masking).
Saves trained LoRA adapters to checkpoints/qwen_lora/ (never overwrites base checkpoints).
"""

import os
import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = os.path.join("scratch", "finetune_dataset.jsonl")
OUTPUT_DIR = os.path.join("checkpoints", "qwen_lora")


class ConversationDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Dataset '{jsonl_path}' not found!")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                user_msg = data.get("user", "").strip()
                asst_msg = data.get("assistant", "").strip()
                if user_msg and asst_msg:
                    self.samples.append((user_msg, asst_msg))

        print(f"Loaded {len(self.samples)} training pairs from {jsonl_path}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_msg, asst_msg = self.samples[idx]
        
        # Build prompt and full conversation
        prompt_msgs = [
            {"role": "system", "content": "You are a helpful and polite conversational assistant."},
            {"role": "user", "content": user_msg}
        ]
        prompt_text = self.tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_msgs = prompt_msgs + [{"role": "assistant", "content": asst_msg}]
        full_text = self.tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids

        # Truncate if necessary
        if len(full_ids) > self.max_len:
            full_ids = full_ids[:self.max_len]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        
        # Mask prompt tokens with -100 so cross-entropy loss is computed ONLY on assistant reply
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels[:prompt_len] = -100

        return input_ids, labels


def collate_fn(batch):
    input_ids_list, labels_list = zip(*batch)
    max_len = max(len(ids) for ids in input_ids_list)
    pad_token_id = 151643  # Qwen endoftext / pad token

    padded_inputs = []
    padded_labels = []

    for inp, lbl in zip(input_ids_list, labels_list):
        pad_len = max_len - len(inp)
        if pad_len > 0:
            inp = torch.cat([inp, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
        padded_inputs.append(inp)
        padded_labels.append(lbl)

    return torch.stack(padded_inputs), torch.stack(padded_labels)


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cpu"
    print("=" * 70)
    print("INITIALIZING QWEN2.5-0.5B LoRA FINE-TUNING PIPELINE")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        device_map=device,
        low_cpu_mem_usage=True
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = ConversationDataset(DATA_PATH, tokenizer, max_len=192)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    num_epochs = 2
    step = 0

    print(f"\nStarting training for {num_epochs} epochs on CPU...")
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (input_ids, labels) in enumerate(loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            epoch_loss += loss.item()

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(loader):
                print(f"Epoch [{epoch+1}/{num_epochs}] Step {step} | Batch [{batch_idx+1}/{len(loader)}] | Loss: {loss.item():.4f}")

    # Save LoRA adapter weights
    print(f"\nSaving fine-tuned LoRA weights to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✓ LoRA fine-tuning complete and saved successfully!")

if __name__ == "__main__":
    train()
