"""
clean_dailydialog.py
Extracts only clean, generic conversations from dailydialog.txt.
Removes character-roleplay names and human specific roleplay sentences.
"""

import os
import re

NAMES = [
    'Jim', 'Mary', 'Mark', 'Kark', 'Bill', 'Sally', 'Lean', 'Alike', 'Nicole',
    'Emily', 'Francis', 'Charles', 'Rose', 'Jack', 'Tom', 'Martin', 'Jane',
    'Cody', 'Penny', 'Leonard', 'Steven', 'Lin', 'Cindy', 'Andy', 'David',
    'Jenny', 'Tony', 'Gina', 'Brian', 'Steve', 'Vicky', 'Julia', 'Mike',
    'Hans', 'Harry', 'Shirley', 'Ann', 'Bob', 'Alice', 'George', 'Paul',
    'John', 'Lucy', 'Kate', 'Sam', 'Dan', 'Ken', 'Susan', 'Betty', 'Nancy'
]
name_pat = re.compile(r'\b(' + '|'.join(NAMES) + r')\b', re.IGNORECASE)

HUMAN_ROLEPLAY = [
    'my wife', 'my husband', 'my boyfriend', 'my girlfriend', 'my son',
    'my daughter', 'my boss', 'my car', 'room service', 'check out of the hotel',
    'order food', 'menu, please', 'have a beer', 'going for beers', 'push-ups'
]
roleplay_pat = re.compile(r'\b(' + '|'.join(HUMAN_ROLEPLAY) + r')\b', re.IGNORECASE)

IN_PATH = os.path.join("data", "raw", "dailydialog.txt")
OUT_PATH = os.path.join("data", "raw", "clean_dailydialog.txt")


def filter_dailydialog():
    if not os.path.exists(IN_PATH):
        print(f"File {IN_PATH} does not exist.")
        return

    with open(IN_PATH, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    dialogs = text.split("\n\n")
    clean_dialogs = []

    for d in dialogs:
        d_clean = d.strip()
        if len(d_clean) < 30 or len(d_clean) > 600:
            continue
        if name_pat.search(d_clean) or roleplay_pat.search(d_clean):
            continue
        # Ensure it has both User and Assistant
        if "User:" in d_clean and "Assistant:" in d_clean:
            clean_dialogs.append(d_clean)

    # Take up to 2,500 pristine dialogues
    selected = clean_dialogs[:2500]
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n\n".join(selected) + "\n")

    size_kb = os.path.getsize(OUT_PATH) / 1024.0
    print(f"Retained {len(selected)} clean dialogues out of {len(dialogs)}.")
    print(f"Saved: {OUT_PATH} ({size_kb:.2f} KB)")


if __name__ == "__main__":
    filter_dailydialog()
