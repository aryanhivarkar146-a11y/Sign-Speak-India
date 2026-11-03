# train_hand_model.py
"""
Hand‑only training pipeline for SignSpeak.
Loads only left_hand and right_hand landmarks (126 features per frame) and trains the same model candidates as the original script.
"""
import os, glob, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import keras
from keras import layers, callbacks

# Paths – adjust if needed
LABEL_MAP_PATH = r"C:\\Users\\aryan\\Downloads\\filtered_label_map.json"
DATA_ROOT = r"C:\\Users\\aryan\\Downloads\\isl_data"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

FIXED_FRAMES = 30
FEATURES_PER_FRAME = 126  # 2 hands * 21 landmarks * 3 coordinates
RANDOM_STATE = 42


def load_hand_only(fpath):
    """Load only left_hand and right_hand landmarks, resample to FIXED_FRAMES.
    Returns a (30, 126) array or None if malformed.
    """
    df = pd.read_parquet(fpath)
    df = df[df['type'].isin(['left_hand', 'right_hand'])].copy()
    df[['x','y','z']] = df[['x','y','z']].fillna(0.0)
    type_order = {'left_hand': 0, 'right_hand': 1}
    df['type_rank'] = df['type'].map(type_order)
    df = df.sort_values(by=['frame','type_rank','landmark_index'])
    frames = sorted(df['frame'].unique())
    if not frames:
        return None
    frame_features = []
    for f in frames:
        f_df = df[df['frame']==f]
        coords = f_df[['x','y','z']].values.flatten()
        if len(coords) == FEATURES_PER_FRAME:
            frame_features.append(coords)
        else:
            return None
    arr = np.array(frame_features, dtype=np.float32)
    T = arr.shape[0]
    if T != FIXED_FRAMES:
        idx = np.linspace(0, T-1, FIXED_FRAMES)
        resampled = np.zeros((FIXED_FRAMES, FEATURES_PER_FRAME), dtype=np.float32)
        for c in range(FEATURES_PER_FRAME):
            resampled[:,c] = np.interp(idx, np.arange(T), arr[:,c])
        return resampled
    return arr


def augment_sequence(seq):
    aug = seq.copy()
    aug += np.random.normal(0, 0.004, aug.shape).astype(np.float32)
    aug *= np.random.uniform(0.96, 1.04)
    aug += np.random.uniform(-0.02, 0.02, size=(1, aug.shape[1])).astype(np.float32)
    return aug

def build_conv1d_gru(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64,3,padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(64,3,padding='same',activation='relu'),
        layers.Dropout(0.3),
        layers.GRU(64),
        layers.Dropout(0.3),
        layers.Dense(64,activation='relu'),
        layers.Dense(num_classes,activation='softmax')
    ], name='Conv1D_GRU_Hand')
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(hist, name, path):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='val')
    plt.title(f'{name} loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(hist.history['accuracy'], label='train')
    plt.plot(hist.history['val_accuracy'], label='val')
    plt.title(f'{name} acc')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    with open(LABEL_MAP_PATH) as f:
        raw_map = json.load(f)
    classes = sorted(list(raw_map.keys()))
    class_to_idx = {c:i for i,c in enumerate(classes)}
    idx_to_class = {i:c for i,c in enumerate(classes)}
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")
    X, y = [], []
    for word in classes:
        pattern = os.path.join(DATA_ROOT, '**', word, '*.parquet')
        for fp in glob.glob(pattern, recursive=True):
            arr = load_hand_only(fp)
            if arr is not None:
                X.append(arr)
                y.append(class_to_idx[word])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"Loaded {X.shape[0]} samples, shape per sample {X.shape[1:]}.")
    X_trv, X_test, y_trv, y_test = train_test_split(X,y,test_size=0.15,random_state=RANDOM_STATE,stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_trv,y_trv,test_size=0.176,random_state=RANDOM_STATE,stratify=y_trv)
    aug_mult = 5
    X_aug = [X_train]
    y_aug = [y_train]
    for _ in range(aug_mult):
        X_aug.append(np.array([augment_sequence(s) for s in X_train], dtype=np.float32))
        y_aug.append(y_train)
    X_train = np.concatenate(X_aug)
    y_train = np.concatenate(y_aug)
    perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]
    input_shape = (FIXED_FRAMES, FEATURES_PER_FRAME)
    candidates = {'Conv1D_GRU_Hand': build_conv1d_gru(input_shape, num_classes)}
    results = {}
    for name, model in candidates.items():
        print(f"Training {name}")
        cb = [callbacks.EarlyStopping('val_loss',patience=8,restore_best_weights=True),
              callbacks.ReduceLROnPlateau('val_loss',factor=0.5,patience=4)]
        hist = model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=40,batch_size=16,callbacks=cb,verbose=1)
        val_acc = model.evaluate(X_val,y_val,verbose=0)[1]
        test_acc = model.evaluate(X_test,y_test,verbose=0)[1]
        y_pred = np.argmax(model.predict(X_test,verbose=0),axis=1)
        f1 = f1_score(y_test,y_pred,average='weighted')
        results[name] = {'val_acc':float(val_acc),'test_acc':float(test_acc),'f1':float(f1)}
        plot_history(hist, name, os.path.join(OUTPUT_DIR, f'{name}_history.png'))
    best_name = max(results, key=lambda k: results[k]['test_acc'])
    best = candidates[best_name]
    best.save(os.path.join(OUTPUT_DIR, 'sign_speak_model_hand.keras'))
    best.save(os.path.join(OUTPUT_DIR, 'sign_speak_model_hand.h5'))
    meta = {
        'model_architecture': best_name,
        'input_shape':[FIXED_FRAMES, FEATURES_PER_FRAME],
        'sequence_frames':FIXED_FRAMES,
        'features_per_frame':FEATURES_PER_FRAME,
        'feature_layout':{'left_hand':'landmarks 0..20 (63 floats)','right_hand':'landmarks 0..20 (63 floats)','total_landmarks':42,'total_floats':126},
        'num_classes':num_classes,
        'classes':classes,
        'class_to_idx':class_to_idx,
        'idx_to_class':idx_to_class,
        'metrics':{'test_accuracy':float(test_acc),'test_f1_weighted':float(f1)}
    }
    with open(os.path.join(OUTPUT_DIR,'model_metadata.json'),'w') as f:
        json.dump(meta, f, indent=4)
    print('Hand-only model exported.')

if __name__ == '__main__':
    main()
