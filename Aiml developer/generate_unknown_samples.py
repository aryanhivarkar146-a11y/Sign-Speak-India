# generate_unknown_samples.py
"""
Utility script to generate multiple synthetic "unknown" samples for the hand‑only training pipeline.
Each sample is saved under `browser_samples/unknown/` with a unique filename.
"""

import os
import numpy as np
import pandas as pd

FIXED_FRAMES = 30
FEATURES_PER_FRAME = 126  # 2 hands * 21 landmarks * 3 coordinates


def generate_random_sequence():
    """Create a (30, 126) array of random floats in [0, 1]."""
    return np.random.rand(FIXED_FRAMES, FEATURES_PER_FRAME).astype("float32")


def build_dataframe(seq):
    rows = []
    for f in range(FIXED_FRAMES):
        # Left hand (21 landmarks)
        for i in range(21):
            rows.append([
                f, "left_hand", i,
                seq[f, i * 3],
                seq[f, i * 3 + 1],
                seq[f, i * 3 + 2]
            ])
        offset = 21 * 3
        # Right hand (21 landmarks)
        for i in range(21):
            rows.append([
                f, "right_hand", i,
                seq[f, offset + i * 3],
                seq[f, offset + i * 3 + 1],
                seq[f, offset + i * 3 + 2]
            ])
    return pd.DataFrame(rows, columns=["frame", "type", "landmark_index", "x", "y", "z"])


def main(num_samples: int = 20):
    workspace_dir = os.path.dirname(__file__)
    out_dir = os.path.join(workspace_dir, "browser_samples", "unknown")
    os.makedirs(out_dir, exist_ok=True)
    for idx in range(num_samples):
        seq = generate_random_sequence()
        df = build_dataframe(seq)
        out_path = os.path.join(out_dir, f"synthetic_unknown_{idx}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"Synthetic unknown sample #{idx} written to {out_path}")


if __name__ == "__main__":
    main()
