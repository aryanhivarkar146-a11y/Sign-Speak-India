# SignSpeak backend

This FastAPI service receives MediaPipe landmark frames and streams predictions from the trained model.

## Run locally

The model and `inference.py` currently live in the sibling `Aiml developer` directory. From the dashboard project root, install the backend dependencies into the Python environment that has access to those model files, then run:

```powershell
python -m pip install -r "backend/requirements.txt"
python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```

The service loads `Aiml developer/inference.py` and its model on startup. To use a different model directory, set `SIGNSPEAK_MODEL_DIR` to the directory containing `inference.py`, `model_metadata.json`, and the `.keras` or `.h5` model. The model metadata must describe 30 frames and either 225 pose+hand values or 126 hand-only values per frame. The backend reads the model feature count from `inference.py` metadata and checks that metadata, the actual model input/output shapes, and class count agree at startup; inconsistent artifacts leave the service degraded with a diagnostic at `/health`.

The browser currently sends 225 values per frame in this order: pose `[0:99]`, first hand slot `[99:162]`, second hand slot `[162:225]`. If the loaded model expects 126 hand-only values, the backend removes the first 99 pose values and sends the remaining 126 to the model. A 225-feature model receives the original frame unchanged. `GET /health` reports both `features_per_frame` (model input) and `browser_features_per_frame` (browser input).

Model dimensions are taken from the loaded model metadata. If `SIGNSPEAK_FEATURES_PER_FRAME` is set in the environment to the old value `225`, update it to `126` or remove it before starting with a 126-feature model. `SIGNSPEAK_MODEL_FEATURES_PER_FRAME` can be used as the explicit model-dimension override.

The downloaded ISL500 subset can be prepared with:

```powershell
python backend/prepare_isl500_dataset.py
```

This creates `backend/data/isl500_hands_sequences.npz` with shape `(samples, 30, 126)`. The AI/ML training pipeline must train with that 126-value layout and preserve the feature order expected by the browser hand slots before replacing the current model. The current 225-feature model remains supported until then.

Check `http://localhost:8000/health` for model readiness. A missing model leaves the HTTP service running in a degraded state and returns a descriptive error to WebSocket clients.

## WebSocket contract

Connect to `ws://localhost:8000/ws`. Send each browser frame as a JSON array of 225 finite numbers, or as `{"keypoints": [...]}`. For a 126-feature hand-only model, the backend adapts browser frames automatically; clients may also send already-extracted 126-value hand frames. A batch can be sent as `{"frames": [[...], [...]]}`. The backend also accepts a raw two-dimensional array as a batch.

Until 30 frames have arrived, the server sends a `status` message with `frames_buffered` and `frames_required`. It then keeps a rolling 30-frame buffer and sends a `prediction` after each new frame:

```json
{
  "type": "prediction",
  "predicted_sign": "doctor",
  "confidence": 91.2,
  "top3": [
    {"sign": "doctor", "confidence": 91.2},
    {"sign": "adult", "confidence": 5.3},
    {"sign": "actor", "confidence": 2.1}
  ],
  "frames_buffered": 30
}
```

Malformed frames produce an `error` message and do not close the connection. The first prediction is sent after frame 30, then once per subsequent frame. The dashboard page now runs MediaPipe pose and two-hand landmark extraction in the browser, draws the landmarks over the camera, and sends the 225-value frames to this endpoint. Serve the dashboard from a local web server (such as VS Code Live Server on port 5500) and start this API on port 8000.

Set `SIGNSPEAK_ALLOWED_ORIGINS` to a comma-separated list of frontend origins when serving the page outside the default local development ports (`5500`, `5173`).
