"""Convert the downloaded ISL500 hand landmarks into fixed-length sequences.

The ISL500 MediaPipe HDF5 files contain two hands (2 x 21 x 3 = 126
features/frame), while the current SignSpeak model expects pose + hands
(225 features/frame). This converter intentionally keeps the source format's
126 hand features and does not fabricate pose values.

Run from the project root with the AI/ML environment:
    python backend/prepare_isl500_dataset.py
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_SOURCE = Path.home() / "Downloads" / "ISL500-SignSpeak" / "Landmarks" / "MediaPipe"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "isl500_hands_sequences.npz"
SEQUENCE_FRAMES = 30
EXPECTED_PER_FRAME = (2, 21, 3)


def resample_sequence(data: np.ndarray, target_frames: int) -> np.ndarray:
    """Linearly resample (time, hands, landmarks, xyz) to a fixed frame count."""
    if data.ndim != 4 or tuple(data.shape[1:]) != EXPECTED_PER_FRAME:
        raise ValueError(f"Expected shape (T, 2, 21, 3), got {data.shape}")
    if data.shape[0] == 0:
        raise ValueError("Sequence contains no frames")
    if not np.isfinite(data).all():
        raise ValueError("Sequence contains NaN or infinite values")

    flattened = data.reshape(data.shape[0], -1).astype(np.float32, copy=False)
    if len(flattened) == target_frames:
        return flattened.copy()

    sample_positions = np.linspace(0, len(flattened) - 1, target_frames)
    source_positions = np.arange(len(flattened))
    output = np.empty((target_frames, flattened.shape[1]), dtype=np.float32)
    for feature_index in range(flattened.shape[1]):
        output[:, feature_index] = np.interp(
            sample_positions,
            source_positions,
            flattened[:, feature_index],
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=SEQUENCE_FRAMES)
    args = parser.parse_args()

    if args.frames < 1:
        parser.error("--frames must be a positive integer")
    if not args.source_dir.is_dir():
        parser.error(f"Dataset directory not found: {args.source_dir}")

    paths = sorted(args.source_dir.rglob("*.h5"))
    if not paths:
        parser.error(f"No .h5 files found beneath {args.source_dir}")

    class_names = sorted({path.name.split("__", 1)[0].lower() for path in paths})
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    signers: list[str] = []
    filenames: list[str] = []
    errors: list[str] = []

    for path in paths:
        label = path.name.split("__", 1)[0].lower()
        try:
            with h5py.File(path, "r") as data_file:
                if "intermediate" not in data_file:
                    raise ValueError("missing 'intermediate' dataset")
                sequence = resample_sequence(data_file["intermediate"][:], args.frames)
            sequences.append(sequence)
            labels.append(class_to_idx[label])
            signers.append(path.parent.name)
            filenames.append(path.name)
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"{path}: {exc}")

    if not sequences:
        raise SystemExit("No valid sequences could be prepared.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    X = np.stack(sequences).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)
    signer_ids = np.asarray(signers, dtype=str)
    source_files = np.asarray(filenames, dtype=str)
    metadata = {
        "source": "ISL500/ISL-DATA Landmarks/MediaPipe",
        "sequence_frames": args.frames,
        "features_per_frame": 126,
        "feature_layout": "left_hand and right_hand; 21 landmarks x (x,y,z) each",
        "classes": class_names,
        "class_to_idx": class_to_idx,
        "sample_count": len(X),
        "signer_count": len(set(signers)),
        "skipped_count": len(errors),
    }
    np.savez_compressed(
        args.output,
        X=X,
        y=y,
        signers=signer_ids,
        source_files=source_files,
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    class_counts = Counter(class_names[label] for label in labels)
    print(f"Prepared {len(X)} sequences with shape {X.shape} -> {args.output}")
    print(f"Labels ({len(class_names)}): {dict(sorted(class_counts.items()))}")
    print(f"Signers: {len(set(signers))}; skipped invalid files: {len(errors)}")
    print("Feature warning: 126 hand values/frame; current model expects 225 pose+hand values/frame.")
    if errors:
        error_log = args.output.with_suffix(".errors.txt")
        error_log.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"Skipped-file details: {error_log}")


if __name__ == "__main__":
    main()
