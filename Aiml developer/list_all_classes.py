import json
import os

with open(r"C:\Users\aryan\Downloads\isl_data\label_map.json") as f:
    label_map = json.load(f)

print(f"Total classes in dataset: {len(label_map)}")
print("Classes sorted alphabetically:")
classes = sorted(list(label_map.keys()))
for i in range(0, len(classes), 5):
    print("  " + ", ".join(f"{c} ({label_map[c]})" for c in classes[i:i+5]))
