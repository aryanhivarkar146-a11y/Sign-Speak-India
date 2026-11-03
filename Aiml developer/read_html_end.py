with open(r"..\index.html", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
for i in range(350, len(lines)):
    print(f"{i+1:3d}: {lines[i].rstrip()}")
