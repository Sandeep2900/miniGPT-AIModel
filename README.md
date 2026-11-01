# MiniGPT AI Assistant

An intelligent, 100% local, privacy-first conversational AI assistant featuring a **dual-engine neural architecture**, **real-time tool augmentation (Live Clock & DuckDuckGo Web Search)**, **multi-turn conversational memory**, and a **ChatGPT/Claude-grade modern Web UI**.

Runs completely offline on standard CPU/GPU without requiring any paid API keys (No OpenAI, Anthropic, or external subscriptions needed).

---

## 🌟 Key Features & Capabilities

### 1. What the AI Can Do Right Now

- **Multilingual Fluency**: Seamlessly understands and responds in **English**, **Hindi (Devanagari script)**, and **Hinglish (Romanized Hindi)**.
- **Coding & DSA Problem Solving**: Generates and explains complete, executable code (Python, C++, Java, JavaScript, etc.) for Data Structures, Algorithms (e.g. Two Sum, Binary Trees, Dynamic Programming), and debugging tasks.
- **Context-Aware Dialogue (Multi-Turn Memory)**: Maintains conversation history across turns. It accurately understands referential follow-up questions (e.g., *"where is the code?"*, *"explain that step in Hindi"*, *"what is its time complexity?"*).
- **Live Tool Augmentation**:
  - **DuckDuckGo Web Search**: Automatically triggers real-time web search for current events, news, weather, prices, or sports updates.
  - **System Clock**: Accurately reports real-time system timestamps and dates.
- **Modern Minimal Web Interface**:
  - Clean Dark and Light mode themes.
  - Markdown formatting with rich table and bullet rendering.
  - Syntax-highlighted code blocks with a one-click **Copy Code** button.
  - Word-by-word streaming reveal animation.
  - **Instant Stop Generation**: Network-level `AbortController` and token halt allowing you to cancel responses instantly via the Stop button or `Esc` key.
  - Circular token safety meter tracking input budgets.
  - Conversation session history saved locally in `localStorage`.
  - Architecture specs modal and transparent system audit log viewer.
- **Dual-Engine Architecture**:
  - **Engine 1 (Default)**: Pretrained `Qwen2.5-0.5B-Instruct` model optimized for fast CPU inference, reasoning, and multi-turn dialogue.
  - **Engine 2**: Custom locally trained 3.08M parameter MiniGPT Decoder Transformer with custom Byte Pair Encoding (BPE) tokenizer.

---

## 📋 System Prerequisites

Before installing, ensure your system meets the following requirements:

| Component | Minimum Requirement | Recommended |
|---|---|---|
| **Operating System** | Windows 10/11, macOS, or Linux | Windows 11 / Ubuntu 22.04+ |
| **Python** | Python 3.9 - 3.12 | Python 3.10 or 3.11 |
| **RAM** | 4 GB | 8 GB or higher |
| **Storage** | ~2 GB free disk space | SSD with 5 GB+ free space |
| **GPU (Optional)** | None (runs on standard CPU) | NVIDIA GPU with CUDA for faster inference |

---

## 🚀 Step-by-Step Installation Guide

Follow these steps to set up and run the project from scratch on any machine:

### Step 1: Clone the Repository

Open your terminal (PowerShell, Command Prompt, or Bash) and clone this repository:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd Python/char_rnn_lm
```

> **Note**: Ensure your terminal working directory is inside the `char_rnn_lm` folder before running scripts.

---

### Step 2: Create a Virtual Environment (Recommended)

Creating an isolated Python virtual environment prevents package conflicts.

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Required Dependencies

Install all necessary packages via `pip`:

```bash
pip install -r requirements.txt
```

> **PyTorch Note**:
> - The `requirements.txt` file defaults to CPU-optimized PyTorch wheels.
> - If you have an NVIDIA GPU and want GPU acceleration, install the CUDA-enabled PyTorch build:
>   ```bash
>   pip install torch --index-url https://download.pytorch.org/whl/cu121
>   ```

---

### Step 4: Verify the Installation

Run the setup validation script to confirm Python version, PyTorch installation, and directory structure:

```bash
python check_setup.py
```

You should see all green verification checks.

---

### Step 5: Launch the Web Assistant

Start the Flask web application server:

```bash
python app.py
```

#### What happens on first launch:
1. The script will automatically download the lightweight `Qwen2.5-0.5B-Instruct` weights (~900MB) from Hugging Face.
2. The weights are cached locally in your standard Hugging Face cache directory (`~/.cache/huggingface/`).
3. Once loaded, the server starts at `http://127.0.0.1:5000`.

---

### Step 6: Open the Web UI in Your Browser

Open any web browser (Chrome, Edge, Firefox, Brave) and navigate to:

```text
http://127.0.0.1:5000
```

You can now start asking questions in **English**, **Hinglish**, or **Hindi**!

---

## ⚙️ Advanced Configuration & CLI Modes

### Switching to the Custom MiniGPT Transformer Engine

By default, the assistant runs the high-capability `Qwen2.5-0.5B-Instruct` model. To switch to the locally trained 3.08M parameter MiniGPT model:

**On Windows (PowerShell):**
```powershell
$env:AGENT_ENGINE="minigpt"
python app.py
```

**On Linux / macOS:**
```bash
export AGENT_ENGINE="minigpt"
python app.py
```

---

### Running CLI Interactive Chat (Terminal Mode)

If you prefer interacting directly from the command line without launching the web server:

```bash
python chat.py
```

---

### Training Your Own MiniGPT Model from Scratch

To retrain the custom transformer on your own data:

```powershell
# 1. Clean and prepare datasets
python prepare_data.py

# 2. Train the custom Byte Pair Encoding (BPE) tokenizer
python train_bpe_tokenizer.py

# 3. Train the MiniGPT model
python train_gpt.py --steps 5000 --eval_interval 200
```

---

## 📁 Project Directory Structure

```text
char_rnn_lm/
│
├── app.py                      # Flask backend API server (/api/chat, /api/status, /api/logs)
├── agent.py                    # Autonomous agent logic (Tool routing, Qwen/MiniGPT engines, SessionMemory)
├── model_gpt.py                # PyTorch MiniGPT Decoder Transformer (Multi-head Self-Attention, GELU, RMSNorm)
├── bpe_tokenizer.py            # Custom HuggingFace Tokenizers wrapper
├── tools.py                    # Live tools: DuckDuckGo search and scraping
├── check_setup.py              # Environment and hardware diagnostics
│
├── templates/
│   └── index.html              # Modern, semantic chat interface HTML
│
├── static/
│   ├── css/
│   │   └── style.css           # Premium responsive stylesheet (Dark/Light themes, animations)
│   └── script.js               # Frontend controller (AbortController, Markdown parser, Streaming, Sessions)
│
├── data/
│   ├── raw/                    # Raw text corpora
│   ├── train.txt               # Prepared training dataset
│   ├── val.txt                 # Validation split
│   ├── bpe_vocab.json          # BPE tokenizer vocabulary
│   └── bpe_merges.txt          # BPE tokenizer merges
│
├── checkpoints/
│   ├── best_gpt.pt             # Trained MiniGPT model checkpoint
│   └── tool_gpt.pt             # Tool-augmented MiniGPT checkpoint
│
├── logs/
│   └── agent_log.txt           # Auditing logs of all queries, routes, and responses
│
└── requirements.txt            # Python dependencies
```

---

## ⚠️ Current Limitations & Constraints

While this AI assistant is fast, lightweight, and completely private, users should be aware of its current limitations:

1. **Parameter Scale (0.5B Model vs. 70B+ Frontier Models)**:
   - Designed specifically to run smoothly on standard laptops and CPUs without requiring heavy GPU clusters.
   - Ideal for straightforward coding problems, general Q&A, Hinglish dialogue, and factual summarization.
   - For deeply abstract reasoning, complex multi-file enterprise refactoring, or advanced academic mathematics, smaller models may occasionally produce simplified solutions.

2. **Context Window Safety Budget**:
   - To maintain fast token generation on CPU, the prompt context is optimized around 128 to 512 tokens.
   - Very long codebases or multi-page documents should be asked about in focused, modular sections.

3. **CPU Inference Latency**:
   - On a modern multi-core CPU, short answers generate in 2–5 seconds. Longer code implementations (300+ tokens) may take 15–25 seconds.
   - *Tip*: Running on an NVIDIA GPU with CUDA reduces response times to sub-second speeds.

4. **Web Search Dependent on Snippets**:
   - DuckDuckGo web search relies on live search snippets and clean textual web pages.
   - Sites behind strict paywalls, captchas, or heavy client-side JavaScript rendering cannot be fully scraped.

5. **Hallucination Risk**:
   - Like all autoregressive language models, the AI can occasionally generate plausible-sounding but incorrect code syntax or facts. Always verify generated code before deploying to production systems.

---

## ❓ Troubleshooting

- **Port 5000 Already in Use (`Address already in use`)**:
  If port 5000 is occupied by a previously running process, free it or specify a different port:
  ```powershell
  # Free port 5000 on Windows
  Get-NetTCPConnection -LocalPort 5000 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
  
  # Or run on a custom port
  $env:PORT="5050"
  python app.py
  ```

- **Hugging Face Model Download Hangs or Fails**:
  If your internet connection drops while downloading `Qwen2.5-0.5B-Instruct`, simply re-run `python app.py`. Hugging Face will resume downloading from the last completed chunk.

---

## 📄 License & Attribution

This project is open-source under the MIT License. Feel free to use, modify, and distribute it for personal and educational projects.
