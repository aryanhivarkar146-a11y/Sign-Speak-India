import os
import glob
import re
from collections import Counter
import pandas as pd

kp_root = r"C:\Users\aryan\Downloads\isl_data\keypoints"
files = glob.glob(os.path.join(kp_root, "*", "*", "*.parquet"))

video_names = [os.path.splitext(os.path.basename(f))[0] for f in files]
print(f"Total parquet files: {len(files)}")
print(f"Unique video names: {len(set(video_names))}")

# Check video name formats (e.g., MVI_4470, etc.)
prefixes = Counter([re.sub(r'\d+', '', v) for v in video_names])
print("Prefixes:", prefixes)

# Extract numbers
numbers = [int(re.search(r'\d+', v).group()) for v in video_names if re.search(r'\d+', v)]
if numbers:
    print(f"Video numbers range: min={min(numbers)}, max={max(numbers)}")
    # Group numbers by thousands or clusters
    ranges = pd.cut(numbers, bins=10).value_counts().sort_index()
    print("Video number clusters (signers/sessions):\n", ranges)
