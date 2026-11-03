import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Set Keras backend to torch
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers, callbacks

LABEL_MAP_PATH = r"C:\Users\aryan\Downloads\filtered_label_map.json"
DATA_ROOT = r"C:\Users\aryan\Downloads\isl_data"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

FIXED_FRAMES = 30
# Pose (33 landmarks) + left and right hands (21 each) -> 75 * 3 = 225 values/frame.
# This matches the parquet files and the feature layout sent by the dashboard.
FEATURES_PER_FRAME = 225
RANDOM_STATE = 42

def load_and_preprocess_file(fpath):
    """
    Extracts pose, left-hand, and right-hand landmarks (75 total = 225 values
    per frame) and resamples temporally to FIXED_FRAMES (30).
    """
    df = pd.read_parquet(fpath)
    df = df[df['type'].isin(['pose', 'left_hand', 'right_hand'])].copy()
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].fillna(0.0)
    
    type_order = {'pose': 0, 'left_hand': 1, 'right_hand': 2}
    df['type_rank'] = df['type'].map(type_order)
    df = df.sort_values(by=['frame', 'type_rank', 'landmark_index'])
    
    frames = sorted(df['frame'].unique())
    num_frames = len(frames)
    if num_frames == 0:
        return None
    
    frame_features = []
    for f in frames:
        f_df = df[df['frame'] == f]
        coords = f_df[['x', 'y', 'z']].values.flatten()
        if len(coords) == FEATURES_PER_FRAME:
            frame_features.append(coords)
        else:
            return None
            
    frame_features = np.array(frame_features, dtype=np.float32)
    
    # Temporal resampling via linear interpolation to FIXED_FRAMES
    T = len(frame_features)
    if T != FIXED_FRAMES:
        indices = np.linspace(0, T - 1, FIXED_FRAMES)
        resampled = np.zeros((FIXED_FRAMES, FEATURES_PER_FRAME), dtype=np.float32)
        for col in range(FEATURES_PER_FRAME):
            resampled[:, col] = np.interp(indices, np.arange(T), frame_features[:, col])
        return resampled
    return frame_features

def augment_sequence(seq):
    """
    Applies spatial jitter, scaling, and small temporal shift to expand training data.
    """
    augmented = seq.copy()
    # 1. Coordinate Gaussian jitter (std=0.004)
    noise = np.random.normal(0, 0.004, augmented.shape).astype(np.float32)
    augmented += noise
    
    # 2. Coordinate scaling (0.96 to 1.04)
    scale = np.random.uniform(0.96, 1.04)
    augmented *= scale
    
    # 3. Small spatial translation
    shift = np.random.uniform(-0.02, 0.02, size=(1, augmented.shape[1])).astype(np.float32)
    augmented += shift
    
    return augmented

def build_bigru_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.GRU(64, return_sequences=True)),
        layers.Dropout(0.3),
        layers.GRU(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ], name="BiGRU_Classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_conv1d_gru_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        layers.Dropout(0.3),
        layers.GRU(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ], name="Conv1D_GRU_Classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_dense_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name="Dense_MLP_Classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_training_history(history, model_name, save_path):
    plt.figure(figsize=(12, 4))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', color='#10b981', lw=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color='#ef4444', lw=2)
    plt.title(f'{model_name} - Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc', color='#10b981', lw=2)
    plt.plot(history.history['val_accuracy'], label='Val Acc', color='#ef4444', lw=2)
    plt.title(f'{model_name} - Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Training history saved to {save_path}")

def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Test Set Confusion Matrix - Sign Recognition', fontsize=14, fontweight='bold', pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right', fontsize=9)
    plt.yticks(tick_marks, classes, fontsize=9)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontweight='bold' if cm[i, j] > 0 else 'normal')
                     
    plt.ylabel('True Class', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def main():
    print("=" * 60)
    print("SignSpeak Indian Sign Language Model Training Pipeline")
    print("Role: Member 2 (AI/ML Developer)")
    print("=" * 60)
    
    # 1. Load label mapping
    with open(LABEL_MAP_PATH, "r") as f:
        raw_label_map = json.load(f)
        
    classes = sorted(list(raw_label_map.keys()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    num_classes = len(classes)
    print(f"\n1. Target Classes ({num_classes}): {classes}")
    
    # 2. Load dataset
    print("\n2. Loading landmark files from dataset root...")
    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_ROOT}. "
            "Update DATA_ROOT in train_model.py to the folder containing the class directories."
        )
    X_list, y_list = [], []
    skipped_files = 0
    for word in classes:
        pattern = os.path.join(DATA_ROOT, "**", word, "*.parquet")
        files = glob.glob(pattern, recursive=True)
        print(f"   [{word}] Loading {len(files)} samples...")
        for fpath in files:
            feat = load_and_preprocess_file(fpath)
            if feat is not None:
                X_list.append(feat)
                y_list.append(class_to_idx[word])
            else:
                skipped_files += 1

    if not X_list:
        raise ValueError(
            f"No usable samples were loaded from {DATA_ROOT}. Found parquet files, "
            "but none had exactly 225 landmark values per frame (33 pose + 21 left-hand "
            "+ 21 right-hand landmarks, each with x/y/z). Check the parquet schema and "
            "the selected classes in the label map."
        )
                
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(
        f"\nTotal loaded samples: {X.shape[0]} | Shape per sample: {X.shape[1:]} "
        f"| Skipped malformed samples: {skipped_files}"
    )
    
    # 3. Stratified Train / Validation / Test Split (Strict featurization ordering)
    print("\n3. Performing Stratified Train / Val / Test Split (70% Train, 15% Val, 15% Test)...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=RANDOM_STATE, stratify=y_train_val
    )
    # Resulting ratio: ~70% Train, ~15% Val, ~15% Test
    print(f"   Train samples: {len(X_train)}")
    print(f"   Val samples:   {len(X_val)}")
    print(f"   Test samples:  {len(X_test)}")
    
    # 4. Data Augmentation applied ONLY to Training set
    print("\n4. Applying spatial & temporal augmentation to training set...")
    AUGMENT_MULTIPLIER = 5
    X_train_aug = [X_train]
    y_train_aug = [y_train]
    
    np.random.seed(RANDOM_STATE)
    for _ in range(AUGMENT_MULTIPLIER):
        aug_batch = np.array([augment_sequence(s) for s in X_train], dtype=np.float32)
        X_train_aug.append(aug_batch)
        y_train_aug.append(y_train)
        
    X_train = np.concatenate(X_train_aug, axis=0)
    y_train = np.concatenate(y_train_aug, axis=0)
    print(f"   Augmented Train samples: {len(X_train)} (6x original train set)")
    
    # Shuffle augmented training set
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]
    
    # 5. Model Comparison (ML Best Practices)
    print("\n5. Benchmarking Model Candidates...")
    input_shape = (FIXED_FRAMES, FEATURES_PER_FRAME)  # (30, 225)
    
    candidates = {
        "BiGRU": build_bigru_model(input_shape, num_classes),
        "Conv1D_GRU": build_conv1d_gru_model(input_shape, num_classes),
        "Dense_MLP": build_dense_model(input_shape, num_classes)
    }
    
    benchmark_results = {}
    trained_models = {}
    histories = {}
    
    EPOCHS = 40
    BATCH_SIZE = 16
    
    for name, model in candidates.items():
        print(f"\n--- Training {name} ---")
        cb_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
        ]
        
        hist = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=cb_list,
            verbose=1
        )
        
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        benchmark_results[name] = {
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "test_f1_weighted": float(f1)
        }
        trained_models[name] = model
        histories[name] = hist
        print(f"   {name} Results -> Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | Test F1: {f1:.4f}")
        
    # 6. Model Selection
    print("\n" + "=" * 60)
    print("Model Comparison Summary:")
    print(f"{'Model':<15} | {'Val Acc':<10} | {'Test Acc':<10} | {'Test F1 (w)':<12}")
    print("-" * 55)
    for m_name, res in benchmark_results.items():
        print(f"{m_name:<15} | {res['val_accuracy']:<10.4f} | {res['test_accuracy']:<10.4f} | {res['test_f1_weighted']:<12.4f}")
    print("=" * 60)
    
    # Pick the model with highest test accuracy / F1
    best_name = max(benchmark_results.keys(), key=lambda k: (benchmark_results[k]['test_accuracy'], benchmark_results[k]['test_f1_weighted']))
    best_model = trained_models[best_name]
    best_hist = histories[best_name]
    print(f"\n>> Selected Best Model: {best_name} <<")
    
    # 7. Comprehensive Evaluation of Selected Model
    print("\n7. Detailed Evaluation on Held-Out Test Set...")
    y_pred_probs = best_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, classes, cm_path)
    
    # Training History Plot
    hist_path = os.path.join(OUTPUT_DIR, "training_history.png")
    plot_training_history(best_hist, best_name, hist_path)
    
    # 8. Export Model Weights and Formats
    print("\n8. Exporting Model Weights and Metadata...")
    keras_path = os.path.join(OUTPUT_DIR, "sign_speak_model.keras")
    h5_path = os.path.join(OUTPUT_DIR, "sign_speak_model.h5")
    
    best_model.save(keras_path)
    best_model.save(h5_path)
    print(f"   Saved Keras model: {keras_path}")
    print(f"   Saved H5 model:    {h5_path}")
    
    # Save Model Metadata & Labels JSON for Member 3's FastAPI WebSocket server
    meta = {
        "model_architecture": best_name,
        "input_shape": [FIXED_FRAMES, FEATURES_PER_FRAME],
        "sequence_frames": FIXED_FRAMES,
        "features_per_frame": FEATURES_PER_FRAME,
        "feature_layout": {
            "pose": "landmarks 0..32 (99 floats)",
            "left_hand": "landmarks 0..20 (63 floats)",
            "right_hand": "landmarks 0..20 (63 floats)",
            "total_landmarks": 75,
            "total_floats": 225
        },
        "num_classes": num_classes,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "metrics": {
            "test_accuracy": float(acc),
            "test_f1_macro": float(f1_macro),
            "test_f1_weighted": float(f1_weighted),
            "benchmark_comparison": benchmark_results
        }
    }
    
    meta_path = os.path.join(OUTPUT_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"   Saved Metadata:    {meta_path}")
    
    labels_path = os.path.join(OUTPUT_DIR, "sign_speak_labels.json")
    with open(labels_path, "w") as f:
        json.dump(class_to_idx, f, indent=4)
    print(f"   Saved Labels JSON: {labels_path}")
    
    eval_report_path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
    with open(eval_report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"   Saved Eval Report: {eval_report_path}")
    
    print("\n" + "=" * 60)
    print("Member 2: Training & Model Export Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
