"""
app.py
Flask Web Server for Custom Mini-GPT Online Tool-Using Assistant.
Serves a modern glassmorphic web UI with live search indicators,
source citations, system clock sync, and real-time audit logs.
"""

import os
import sys
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from agent import MiniGPTAgent
from tools.clock import get_current_time

app = Flask(__name__)

# Initialize MiniGPT Agent once on startup
print("Initializing MiniGPT Agent for Web UI...")
agent = MiniGPTAgent()
print(f"Agent Ready! Checkpoint: {agent.checkpoint_path} (Val Loss: {agent.val_loss:.4f})")


@app.route("/")
def index():
    """Serves the main web UI."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns model specifications, device, and system clock."""
    if hasattr(agent, "get_status_dict"):
        status_info = agent.get_status_dict()
        status_info["current_time"] = get_current_time()
        return jsonify(status_info)
    return jsonify({
        "model_name": "Custom Mini-GPT (Decoder-Only Transformer)",
        "params": agent.model.get_num_params(),
        "n_layer": agent.cfg.n_layer,
        "n_head": agent.cfg.n_head,
        "n_embd": agent.cfg.n_embd,
        "block_size": agent.cfg.block_size,
        "step": agent.step,
        "val_loss": round(agent.val_loss, 4),
        "device": agent.device,
        "checkpoint": os.path.basename(agent.checkpoint_path) if agent.checkpoint_path else "none",
        "current_time": get_current_time(),
        "status": "online"
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """Processes a user message through the online orchestrator."""
    data = request.get_json(force=True, silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    start_ts = datetime.now()
    try:
        result = agent.respond(user_message)
        latency_ms = int((datetime.now() - start_ts).total_seconds() * 1000)

        return jsonify({
            "response": result["response"],
            "route": result["route"],
            "sources": result.get("sources", []),
            "tool_call": result.get("tool_call"),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "latency_ms": latency_ms,
            "tokens_limit": agent.cfg.block_size
        })
    except Exception as e:
        return jsonify({"error": f"Internal agent error: {str(e)}"}), 500


@app.route("/api/logs", methods=["GET"])
def get_logs():
    """Returns recent entries from logs/agent_log.txt."""
    log_file = os.path.join("logs", "agent_log.txt")
    if not os.path.exists(log_file):
        return jsonify({"logs": []})

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        raw_blocks = content.split("-" * 60)
        logs = [b.strip() for b in raw_blocks if b.strip()]
        # Return latest 25 logs reversed
        return jsonify({"logs": logs[-25:][::-1]})
    except Exception as e:
        return jsonify({"error": str(e), "logs": []})


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting MiniGPT Online Assistant Web App on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
