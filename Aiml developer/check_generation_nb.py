import json

nb_path = r"C:\Users\aryan\Downloads\isl-mediapipe-holistic-landmarks-generation.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if any(w in source for w in ["mp_holistic", "Holistic", "left_hand", "process", "landmark"]):
            print(f"--- Cell {i} ---")
            print(source[:1000])
