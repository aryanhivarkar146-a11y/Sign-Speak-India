import sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r"..\index.html", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines in index.html: {len(lines)}")
for i, line in enumerate(lines[:120]):
    print(f"{i+1:3d}: {line.rstrip()}")
