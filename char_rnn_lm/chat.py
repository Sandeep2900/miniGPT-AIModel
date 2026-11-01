"""
chat.py
Interactive Command-Line Online Assistant Interface for custom Mini-GPT.
Features:
- Online tool execution with DuckDuckGo Search & compact web scraping
- Real-time System Clock awareness
- Strict 128-token context budget protection
- Tool call visualization (🔍 Searching web: <query>)
- Web Source Citations (🌐 Sources: <url>)
"""

import os
import sys

# Ensure UTF-8 terminal encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from agent import MiniGPTAgent


def main():
    try:
        agent = MiniGPTAgent()
    except Exception as e:
        print(f"[ERROR] Failed to initialize agent: {e}")
        return

    ckpt_name = os.path.basename(agent.checkpoint_path)
    print("=" * 72)
    print("🤖 Custom Mini-GPT (Decoder-Only Transformer) Online Assistant")
    print("=" * 72)
    print(f"Architecture : {agent.cfg.n_layer} Transformer Layers | {agent.cfg.n_head} Heads | {agent.cfg.n_embd} Dim")
    print(f"Parameters   : {agent.model.get_num_params():,d} (100% custom trained from scratch)")
    print(f"Checkpoint   : {ckpt_name} (Step {agent.step}, Val Loss: {agent.val_loss:.4f})")
    print("Capabilities : Online Web Search 🌐 | Compact Web Scraping 📄 | System Clock ⏰")
    print("Context Limit: 128 tokens strictly enforced")
    print("Instructions : Type in Hindi, Hinglish, or English. Type 'exit' to quit.")
    print("=" * 72)

    while True:
        try:
            user_input = input("\n📝 You > ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                print("\nAlvida! Chat session ended.")
                break
            if not user_input:
                continue

            # Process turn through agent orchestrator
            result = agent.respond(user_input)

            # Visual indicator if a web search was triggered
            if result.get("tool_call"):
                tool_str = result["tool_call"]
                # Extract query from [TOOL: search("query")]
                q_match = tool_str.replace('[TOOL: search("', '').rstrip('")]')
                print(f"🔍 Searching web: '{q_match}'...")

            print("\n🤖 Assistant:")
            print("-" * 52)
            print(result["response"] if result["response"] else "(no response)")
            print("-" * 52)

            # Print clickable source citations if search occurred
            sources = result.get("sources", [])
            if sources:
                print("🌐 Sources:")
                for s in sources:
                    print(f"   • {s}")

        except (KeyboardInterrupt, EOFError):
            print("\nAlvida! Chat session ended.")
            break


if __name__ == "__main__":
    main()
