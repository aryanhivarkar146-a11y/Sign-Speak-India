import json

nb_path = r"C:\Users\aryan\Downloads\isl-mediapipe-holistic-landmarks-generation.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell_idx in [1, 3]:
    print(f"=== Cell {cell_idx} ===")
    print("".join(nb["cells"][cell_idx]["source"]))
