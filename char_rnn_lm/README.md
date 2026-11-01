# Multilingual Character-Level GRU Language Model (From Scratch)

A 100% local, pure PyTorch recurrent neural network language model trained from scratch with random weights on mixed **Hindi (Devanagari)**, **Hinglish**, and **English** text. Zero high-level wrappers, zero pretrained weights.

---

## Quickstart in 5 Commands

```powershell
# 1. Navigate to project folder
cd d:\Projects\Python\char_rnn_lm

# 2. Check hardware and dependencies
python check_setup.py

# 3. Clean raw files, rebalance, deduplicate, and create continuous per-file splits
python prepare_data.py

# 4. Verify data pipeline, script distribution, and tokenization
python data.py

# 5. Run sanity check (verify single-batch overfit test)
python sanity_check.py
```

### Training & Generation

```powershell
# Train the model (e.g. 1,000 steps with Cosine LR schedule)
python train.py --steps 1000 --eval_interval 100

# Generate text from a prompt
python generate.py --prompt "आज सुबह " --temperature 0.75

# Interactive chat completion loop
python chat.py

# Evaluate perplexity & compare multi-temperature samples
python evaluate.py
```

---

## Project Structure

```text
char_rnn_lm/
│
├── data/
│   ├── raw/                 # Raw input files (auto-detected: my_data.txt, shakespeare.txt, etc.)
│   ├── train.txt            # Continuous 90% train split (rebalanced)
│   ├── val.txt              # Continuous 10% validation split
│   ├── data.txt             # Combined dataset
│   └── vocab.json           # Unified character vocabulary mapping
│
├── checkpoints/
│   ├── best_model.pt        # Model weights with lowest validation loss
│   └── latest_model.pt      # Most recent checkpoint
│
├── check_setup.py           # Environment, PyTorch CPU, and Windows console check
├── prepare_data.py          # Auto-detector, cross-file deduplication, Shakespeare capping
├── config.py                # Hyperparameters, paths, and hardware presets (tiny/small/medium)
├── data.py                  # Tokenizer, continuous batch generator, script distribution
├── model.py                 # Pure PyTorch Char-GRU neural network architecture
├── train.py                 # Training engine (AdamW, Cosine decay, validation, loss curves)
├── sanity_check.py          # Overfit single-batch test and initial loss verification
├── generate.py              # Text generator with Temperature, Top-K, and Top-P
├── chat.py                  # Interactive CLI prompt loop
├── evaluate.py              # Perplexity evaluation and multi-temperature generation
├── loss_curve.png           # Auto-generated loss plot
├── samples.txt              # Sample text generations at different temperatures
└── requirements.txt         # Pip dependencies
```

---

## Model Architecture

$$\text{Token IDs } (B, T) \longrightarrow \text{Embedding Layer } (B, T, 64) \longrightarrow \text{2-Layer GRU } (B, T, 128) \longrightarrow \text{Linear Head } (B, T, 127)$$

- **Total Parameters:** ~198,079
- **Initial Loss:** $\approx \ln(V) = \ln(127) \approx 4.84$
- **Target Loss:** $< 2.0$

---

## Sharing & Exporting Your Model

To share or deploy your trained model to any other computer:
1. Copy `checkpoints/best_model.pt` (contains model weights, optimizer state, config, and vocabulary).
2. Copy `data/vocab.json`.
3. Run `python generate.py --checkpoint best_model.pt --prompt "Your prompt"`. No internet or external APIs required!
