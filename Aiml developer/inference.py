"""
Inference Module for SignSpeak India
Role: Member 2 (AI/ML Developer)
Provides a clean interface for Member 3 (Backend Developer) to run real-time inference
on incoming MediaPipe keypoint sequences from WebSocket.
"""

import os
import json
import numpy as np
import pandas as pd
os.environ["KERAS_BACKEND"] = "torch"
import keras

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(CURRENT_DIR, "sign_speak_model_hand.keras")
H5_PATH = os.path.join(CURRENT_DIR, "sign_speak_model.h5")
META_PATH = os.path.join(CURRENT_DIR, "model_metadata.json")

class SignSpeakPredictor:
    def __init__(self):
        # Load metadata
        with open(META_PATH, "r") as f:
            self.metadata = json.load(f)
            
        self.classes = self.metadata["classes"]
        self.idx_to_class = {int(k): v for k, v in self.metadata["idx_to_class"].items()}
        self.seq_len = self.metadata["sequence_frames"]
        self.feat_dim = self.metadata["features_per_frame"]
        
        # Load model (prefer hand-only .keras, fallback to full .keras, then .h5)
        MODEL_PATH = os.path.join(CURRENT_DIR, "sign_speak_model.keras")
        target_model = HAND_MODEL_PATH if os.path.exists(HAND_MODEL_PATH) else (MODEL_PATH if os.path.exists(MODEL_PATH) else H5_PATH)
        print(f"Loading trained model from {target_model}...")
        self.model = keras.models.load_model(target_model)
        print("Model loaded successfully!")

    def predict(self, sequence):
        """
        Takes a sequence of keypoints of shape (T, 126) or (1, 30, 126).
        Returns:
            dict with {
                'predicted_sign': str,
                'confidence': float,
                'top3': list of (class_name, confidence)
            }
        """
        seq = np.array(sequence, dtype=np.float32)
        
        
        # If sequence has different number of frames, temporally resample
        if seq.ndim == 2:
            T, F = seq.shape
            if T != self.seq_len:
                indices = np.linspace(0, T - 1, self.seq_len)
                resampled = np.zeros((self.seq_len, F), dtype=np.float32)
                for c in range(F):
                    resampled[:, c] = np.interp(indices, np.arange(T), seq[:, c])
                seq = resampled
            seq = np.expand_dims(seq, axis=0) # (1, 30, 225) – now always 225 dims after padding
            
        probs = self.model.predict(seq, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        
        top3_indices = np.argsort(probs)[-3:][::-1]
        top3 = [(self.idx_to_class[int(i)], float(probs[i])) for i in top3_indices]
        
        return {
            "predicted_sign": self.idx_to_class[top_idx],
            "confidence": round(top_conf * 100, 2),
            "top3": top3
        }

if __name__ == "__main__":
    predictor = SignSpeakPredictor()
    # Try loading a synthetic unknown sample if available
    sample_path = os.path.join(CURRENT_DIR, "browser_samples", "unknown", "synthetic_unknown.parquet")
    if os.path.exists(sample_path):
        df = pd.read_parquet(sample_path)
        # Build 126‑dim sequence from left/right hand landmarks
        seq = np.zeros((30, 126), dtype=np.float32)
        for f in range(30):
            left = df[(df.frame == f) & (df.type == "left_hand")].sort_values('landmark_index')
            right = df[(df.frame == f) & (df.type == "right_hand")].sort_values('landmark_index')
            seq[f] = np.concatenate([left[['x','y','z']].values.ravel(), right[['x','y','z']].values.ravel()])
    else:
        seq = np.random.normal(0.5, 0.1, (30, 126)).astype(np.float32)
    res = predictor.predict(seq)
    print("\nSample Test Inference Output:")
    print(json.dumps(res, indent=2))
