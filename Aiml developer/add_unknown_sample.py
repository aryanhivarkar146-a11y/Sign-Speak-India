# add_unknown_sample.py
"""
Utility script to generate a single synthetic "unknown" sample and store it as a Parquet file
compatible with the training pipeline. The file will be placed under
`browser_samples/unknown/` inside the Aiml developer workspace.
"""

import os
import numpy as np
import pandas as pd

FIXED_FRAMES = 30
FEATURES_PER_FRAME = 225  # 75 landmarks * 3 coordinates

def generate_random_sequence():
    """Create a (30, 225) array of random floats in [0, 1]."""
    return np.random.rand(FIXED_FRAMES, FEATURES_PER_FRAME).astype("float32")

def build_dataframe(seq):
    rows = []
    for f in range(FIXED_FRAMES):
        # Pose landmarks (33 * 3 = 99 values)
        for i in range(33):
            rows.append([f, "pose", i,
                         seq[f, i * 3],
                         seq[f, i * 3 + 1],
                         seq[f, i * 3 + 2]])
        offset = 33 * 3
        # Left hand (21 * 3 = 63 values)
        for i in range(21):
            rows.append([f, "left_hand", i,
                         seq[f, offset + i * 3],
                         seq[f, offset + i * 3 + 1],
                         seq[f, offset + i * 3 + 2]])
        offset += 21 * 3
        # Right hand (21 * 3 = 63 values)
        for i in range(21):
            rows.append([f, "right_hand", i,
                         seq[f, offset + i * 3],
                         seq[f, offset + i * 3 + 1],
                         seq[f, offset + i * 3 + 2]])
    return pd.DataFrame(rows, columns=["frame", "type", "landmark_index", "x", "y", "z"])

def main():
    workspace_dir = os.path.dirname(__file__)
    out_dir = os.path.join(workspace_dir, "browser_samples", "unknown")
    os.makedirs(out_dir, exist_ok=True)
    seq = generate_random_sequence()
    df = build_dataframe(seq)
    out_path = os.path.join(out_dir, "synthetic_unknown.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Synthetic unknown sample written to {out_path}")

if __name__ == "__main__":
    main()
