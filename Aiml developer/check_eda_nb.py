import json
import os

nb_path = r"C:\Users\aryan\Downloads\indian-sign-language-mediapipe-landmark-eda.ipynb"
if os.path.exists(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print("Found EDA notebook. Cell count:", len(nb["cells"]))
    for i, cell in enumerate(nb["cells"]):
        text = "".join(cell.get("source", []))
        print(f"--- Cell {i} ({cell.get('cell_type')}) ---")
        print(text[:400])
