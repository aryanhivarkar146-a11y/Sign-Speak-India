import os
import json
import glob
import pandas as pd
import numpy as np

LABEL_MAP_PATH = r"C:\Users\aryan\Downloads\filtered_label_map.json"
DATA_ROOT = r"C:\Users\aryan\Downloads\isl_data"

def inspect():
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    print(f"Loaded {len(label_map)} target classes from filtered_label_map.json:")
    for k, v in label_map.items():
        print(f"  {k} -> ID {v}")

    # Search for all parquet files matching each label
    found_files = {}
    for word in label_map.keys():
        # search recursively under DATA_ROOT for word directory
        pattern = os.path.join(DATA_ROOT, "**", word, "*.parquet")
        matches = glob.glob(pattern, recursive=True)
        found_files[word] = matches
        print(f"Class '{word}': found {len(matches)} sample parquet files.")

    total_samples = sum(len(v) for v in found_files.values())
    print(f"\nTotal sample files found across target classes: {total_samples}")

    # Inspect a few sample parquet files
    sample_file = None
    for word, files in found_files.items():
        if files:
            sample_file = files[0]
            break

    if sample_file:
        print(f"\nInspecting sample parquet file: {sample_file}")
        df = pd.read_parquet(sample_file)
        print(f"Shape: {df.shape} (rows/frames x columns/coordinates)")
        print(f"Columns count: {len(df.columns)}")
        print(f"Sample columns: {df.columns[:15].tolist()} ... {df.columns[-5:].tolist()}")
        print(f"Data types present: {df.dtypes.value_counts().to_dict()}")
        print(f"Null count in sample: {df.isnull().sum().sum()}")
        print("\nHead of sample:")
        print(df.iloc[:3, :8])

        # Analyze frame lengths across samples
        frame_lengths = []
        for word, files in found_files.items():
            for fpath in files[:5]: # sample up to 5 per class for speed
                try:
                    d = pd.read_parquet(fpath)
                    frame_lengths.append(len(d))
                except Exception as e:
                    print(f"Error reading {fpath}: {e}")

        if frame_lengths:
            print(f"\nSequence length stats across sample subset (N={len(frame_lengths)}):")
            print(f"  Min frames: {min(frame_lengths)}")
            print(f"  Max frames: {max(frame_lengths)}")
            print(f"  Mean frames: {np.mean(frame_lengths):.1f}")
            print(f"  Median frames: {np.median(frame_lengths):.1f}")

if __name__ == "__main__":
    inspect()
