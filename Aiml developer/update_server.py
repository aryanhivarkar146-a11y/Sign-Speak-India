import os

server_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "server.py"))

with open(server_py_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update CORS methods
content = content.replace('allow_methods=["GET"]', 'allow_methods=["*"]')

# 2. Add endpoints if not already added
if "/api/save_sample" not in content:
    endpoints_code = """

from pydantic import BaseModel, Field
import json
import time

SAMPLES_DIR = MODEL_DIR / "browser_samples"

class SamplePayload(BaseModel):
    label: str = Field(..., min_length=1)
    signer: str = Field(default="user")
    frames: list[list[float]]

@app.post("/api/save_sample")
async def save_sample(payload: SamplePayload) -> dict[str, Any]:
    \"\"\"Save a recorded landmark sequence from the browser data collection studio.\"\"\"
    try:
        frames_array = np.asarray(payload.frames, dtype=np.float32)
    except Exception as exc:
        return {"status": "error", "message": f"Invalid numeric data: {exc}"}
        
    if frames_array.ndim != 2 or frames_array.shape[1] != FRAME_SIZE:
        return {
            "status": "error",
            "message": f"Expected frames of shape (T, {FRAME_SIZE}), got {frames_array.shape}"
        }
        
    T = frames_array.shape[0]
    if T < 10:
        return {"status": "error", "message": f"Too few frames ({T}); expected ~{SEQUENCE_LENGTH} frames."}

    if T != SEQUENCE_LENGTH:
        indices = np.linspace(0, T - 1, SEQUENCE_LENGTH)
        resampled = np.zeros((SEQUENCE_LENGTH, FRAME_SIZE), dtype=np.float32)
        for col in range(FRAME_SIZE):
            resampled[:, col] = np.interp(indices, np.arange(T), frames_array[:, col])
        frames_array = resampled

    clean_label = payload.label.strip().lower().replace(" ", "_")
    target_dir = SAMPLES_DIR / clean_label
    target_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_ms = int(time.time() * 1000)
    file_path = target_dir / f"{payload.signer}_{timestamp_ms}.json"
    
    data = {
        "label": clean_label,
        "signer": payload.signer,
        "timestamp_ms": timestamp_ms,
        "frames_count": SEQUENCE_LENGTH,
        "features_per_frame": FRAME_SIZE,
        "landmarks": frames_array.tolist()
    }
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
        
    total_class_samples = len(list(target_dir.glob("*.json")))
    return {
        "status": "ok",
        "label": clean_label,
        "saved_file": file_path.name,
        "total_for_class": total_class_samples
    }

@app.get("/api/samples_summary")
async def samples_summary() -> dict[str, Any]:
    \"\"\"Return summary of all collected browser samples.\"\"\"
    summary = {}
    if SAMPLES_DIR.exists():
        for d in SAMPLES_DIR.iterdir():
            if d.is_dir():
                count = len(list(d.glob("*.json")))
                if count > 0:
                    summary[d.name] = count
    total = sum(summary.values())
    return {
        "status": "ok",
        "total_samples": total,
        "samples_by_class": summary
    }
"""
    content += endpoints_code

with open(server_py_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Updated server.py at {server_py_path} successfully!")
