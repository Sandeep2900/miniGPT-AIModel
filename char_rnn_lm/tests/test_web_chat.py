import urllib.request
import json

def test_api():
    base_url = "http://127.0.0.1:5000"
    
    # 1. Test status
    req = urllib.request.Request(f"{base_url}/api/status")
    with urllib.request.urlopen(req) as response:
        status_data = json.loads(response.read().decode("utf-8"))
        print("Status Check:")
        print(f"  Model: {status_data['model_name']}")
        print(f"  Checkpoint: {status_data['checkpoint']}")
        print(f"  Val Loss: {status_data['val_loss']}")
        print(f"  Status: {status_data['status']}")

    # 2. Test chat: Name declaration
    msg1 = {"message": "Mera naam Sandeep hai"}
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(msg1).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        res1 = json.loads(response.read().decode("utf-8"))
        print("\nChat 1 (Declare Name):")
        print(f"  Route: {res1['route']}")
        print(f"  Response: {res1['response']}")

    # 3. Test chat: Name recall
    msg2 = {"message": "Mera kya naam hai?"}
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(msg2).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        res2 = json.loads(response.read().decode("utf-8"))
        print("\nChat 2 (Recall Name):")
        print(f"  Route: {res2['route']}")
        print(f"  Response: {res2['response']}")

    # 4. Test chat: Casual dialogue
    msg3 = {"message": "Hi kaise ho?"}
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(msg3).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        res3 = json.loads(response.read().decode("utf-8"))
        print("\nChat 3 (Casual Greeting):")
        print(f"  Route: {res3['route']}")
        print(f"  Response: {res3['response']}")

if __name__ == "__main__":
    test_api()
