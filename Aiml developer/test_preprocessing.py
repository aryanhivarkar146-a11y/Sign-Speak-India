import os
import glob
import json
import numpy as np
import pandas as pd


LABEL_MAP_PATH = r"C:\Users\aryan\Downloads\filtered_label_map.json"
DATA_ROOT = r"C:\Users\aryan\Downloads\isl_data"
FIXED_FRAMES = 30
FEATURES_PER_FRAME = 75 * 3  # 33 pose + 21 left_hand + 21 right_hand = 75 landmarks * 3 coords = 225

def load_and_preprocess_file(fpath):
    df = pd.read_parquet(fpath)
    
    # Filter for pose, left_hand, right_hand
    df = df[df['type'].isin(['pose', 'left_hand', 'right_hand'])].copy()
    
    # Fill NaN values with 0.0
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].fillna(0.0)
    
    # Sort deterministically by frame, type, landmark_index
    # We define a custom order for type
    type_order = {'pose': 0, 'left_hand': 1, 'right_hand': 2}
    df['type_rank'] = df['type'].map(type_order)
    df = df.sort_values(by=['frame', 'type_rank', 'landmark_index'])
    
    frames = sorted(df['frame'].unique())
    num_frames = len(frames)
    if num_frames == 0:
        return None
    
    # Extract coordinates per frame
    # Each frame should have 75 landmarks (33 + 21 + 21)
    frame_features = []
    for f in frames:
        f_df = df[df['frame'] == f]
        coords = f_df[['x', 'y', 'z']].values.flatten() # 225 floats
        if len(coords) == FEATURES_PER_FRAME:
            frame_features.append(coords)
        else:
            # Handle possible missing landmarks in frame
            return None
            
    frame_features = np.array(frame_features, dtype=np.float32) # (T, 225)
    
    # Temporal resampling to FIXED_FRAMES
    T = len(frame_features)
    if T != FIXED_FRAMES:
        indices = np.linspace(0, T - 1, FIXED_FRAMES)
        # Linear interpolation across time
        resampled = np.zeros((FIXED_FRAMES, FEATURES_PER_FRAME), dtype=np.float32)
        for col in range(FEATURES_PER_FRAME):
            resampled[:, col] = np.interp(indices, np.arange(T), frame_features[:, col])
        return resampled
    return frame_features

def test_load_all():
    with open(LABEL_MAP_PATH, "r") as f:
        raw_label_map = json.load(f)
    
    # Create contiguous 0-indexed class labels
    classes = sorted(list(raw_label_map.keys()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    print(f"Mapped {len(classes)} classes to contiguous indices 0..{len(classes)-1}")
    
    data_list = []
    labels_list = []
    file_records = []
    
    for word in classes:
        pattern = os.path.join(DATA_ROOT, "**", word, "*.parquet")
        files = glob.glob(pattern, recursive=True)
        print(f"Loading {len(files)} files for class '{word}' (id: {class_to_idx[word]})...")
        for fpath in files:
            feat = load_and_preprocess_file(fpath)
            if feat is not None:
                data_list.append(feat)
                labels_list.append(class_to_idx[word])
                file_records.append((fpath, word))
            else:
                print(f"Warning: skipped {fpath} due to incomplete landmarks.")
                
    X = np.array(data_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    print(f"\nSuccessfully loaded dataset:")
    print(f"X shape: {X.shape} (N_samples x Frames x Features)")
    print(f"y shape: {y.shape}")
    print(f"Class distribution: {dict(pd.Series(labels_list).value_counts().sort_index())}")
    return X, y, class_to_idx

if __name__ == "__main__":
    test_load_all()
