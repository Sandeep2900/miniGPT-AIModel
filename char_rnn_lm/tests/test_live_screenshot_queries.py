"""
test_live_screenshot_queries.py
Runs the exact queries from the user's screenshot against the live web server on http://127.0.0.1:5000.
"""

import urllib.request
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

QUERIES = [
    "name",
    "kaun si list bhai",
    "indore kaha h",
    "hindu kaun sa religion hai",
    "how many months are there in a week"
]

def main():
    base_url = "http://127.0.0.1:5000"
    
    # Check status
    req = urllib.request.Request(f"{base_url}/api/status")
    with urllib.request.urlopen(req) as resp:
        st = json.loads(resp.read().decode("utf-8"))
        print("=" * 70)
        print("LIVE WEB SERVER STATUS:")
        print(f"  Model Name: {st['model_name']}")
        print(f"  Checkpoint: {st['checkpoint']}")
        print(f"  Parameters: {st['params']:,}")
        print(f"  Status    : {st['status']}")
        print("=" * 70)

    for q in QUERIES:
        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=json.dumps({"message": q}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"User:     {q}")
            print(f"Route:    {data.get('route')}")
            print(f"Response: {data.get('response')}")
            print(f"Latency:  {data.get('latency_ms')} ms")
            print("-" * 70)

if __name__ == "__main__":
    main()
