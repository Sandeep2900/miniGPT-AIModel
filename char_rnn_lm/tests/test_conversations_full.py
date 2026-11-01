import urllib.request
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TEST_PROMPTS = [
    "Abhi kya time hua hai?",
    "What is the time right now?",
    "Aaj ka weather kaisa hai?",
    "Aaj mera din thoda mushkil tha.",
    "Main thoda break lena chahta hoon.",
    "Mujhe apni ek achi achievement share karni hai",
    "Python kya hai?",
    "Kya tum meri madad kar sakte ho?",
    "Aap kaun hain?",
    "Thank you so much!"
]

def run():
    base_url = "http://127.0.0.1:5000/api/chat"
    print("=" * 70)
    print("RUNNING EXTENDED CONVERSATION SUITE AGAINST LIVE FLASK SERVER")
    print("=" * 70)
    for p in TEST_PROMPTS:
        req = urllib.request.Request(
            base_url,
            data=json.dumps({"message": p}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                print(f"User:     {p}")
                print(f"Route:    {data.get('route')}")
                print(f"Response: {data.get('response')}")
                if data.get('sources'):
                    print(f"Sources:  {data.get('sources')}")
                print("-" * 70)
        except Exception as e:
            print(f"User:     {p} -> ERROR: {e}")

if __name__ == "__main__":
    run()
