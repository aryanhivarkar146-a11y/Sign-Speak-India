with open(r"..\index.html", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(125, 140):
    if i < len(lines):
        print(f"{i+1:3d}: {lines[i].rstrip()}")
