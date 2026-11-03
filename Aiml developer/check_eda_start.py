import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
nb_path = r"C:\Users\aryan\Downloads\indian-sign-language-mediapipe-landmark-eda.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i in range(1, 10):
    cell = nb["cells"][i]
    print(f"=== Cell {i} ({cell.get('cell_type')}) ===")
    print("".join(cell.get("source", []))[:800])
