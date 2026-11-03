import json
import os

nb_path = r"C:\Users\aryan\Downloads\indian-sign-language-classification.ipynb"
if os.path.exists(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print("Found classification notebook. Cell count:", len(nb["cells"]))
    for i, cell in enumerate(nb["cells"]):
        text = "".join(cell.get("source", []))
        if any(w in text.lower() for w in ["split", "signer", "test", "val", "accuracy", "model", "classes"]):
            print(f"--- Cell {i} ({cell.get('cell_type')}) ---")
            print(text[:400])
