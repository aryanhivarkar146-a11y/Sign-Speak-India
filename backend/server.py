"""FastAPI service for streaming MediaPipe landmarks to the SignSpeak model."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool


logger = logging.getLogger("signspeak")
BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

DEFAULT_MODEL_DIR = BACKEND_DIR.parent / "Aiml developer"
MODEL_DIR = Path(os.environ.get("SIGNSPEAK_MODEL_DIR", DEFAULT_MODEL_DIR)).expanduser().resolve()
PREDICTOR_MODULE = MODEL_DIR / "inference.py"
FRONTEND_FRAME_SIZE = 225  # pose (33 x 3), then two hands (21 x 3 each)
POSE_FEATURE_SIZE = 33 * 3
HAND_FEATURE_SIZE = 2 * 21 * 3
# Keep the old setting as an explicit model-dimension override, but let model
# metadata choose 126 or 225 automatically when no override is configured.
MODEL_FEATURE_SIZE_OVERRIDE = os.environ.get(
    "SIGNSPEAK_MODEL_FEATURES_PER_FRAME",
    os.environ.get("SIGNSPEAK_FEATURES_PER_FRAME"),
)
SEQUENCE_LENGTH = int(os.environ.get("SIGNSPEAK_SEQUENCE_FRAMES", "30"))


def load_predictor() -> Any:
    """Load the sibling AI/ML inference module without requiring a package name with spaces."""
    if not PREDICTOR_MODULE.is_file():
        raise FileNotFoundError(
            f"Predictor not found at {PREDICTOR_MODULE}. Set SIGNSPEAK_MODEL_DIR to its directory."
        )
    spec = importlib.util.spec_from_file_location("signspeak_inference", PREDICTOR_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load predictor module at {PREDICTOR_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predictor = module.SignSpeakPredictor()
    classes = list(predictor.classes)
    if predictor.metadata.get("num_classes") != len(classes):
        raise ValueError(
            "Model metadata class count is inconsistent: "
            f"num_classes={predictor.metadata.get('num_classes')}, classes={len(classes)}."
        )
    expected_class_to_idx = {name: index for index, name in enumerate(classes)}
    if predictor.metadata.get("class_to_idx") != expected_class_to_idx:
        raise ValueError("Model metadata class_to_idx does not match the ordered classes list.")
    expected_idx_to_class = {index: name for index, name in enumerate(classes)}
    actual_idx_to_class = {
        int(index): name
        for index, name in predictor.metadata.get("idx_to_class", {}).items()
    }
    if actual_idx_to_class != expected_idx_to_class:
        logger.warning("idx_to_class is incomplete or inconsistent; deriving it from classes/class_to_idx.")
    # The predictor's index map is used when turning model output into a label.
    # Deriving it here also handles a missing final entry such as "unknown".
    predictor.idx_to_class = expected_idx_to_class

    if predictor.feat_dim not in (HAND_FEATURE_SIZE, FRONTEND_FRAME_SIZE):
        raise ValueError(
            "Unsupported model feature layout: expected 126 hand-only or 225 pose+hand values, "
            f"got {predictor.feat_dim}."
        )
    declared_shape = predictor.metadata.get("input_shape")
    expected_shape = [predictor.seq_len, predictor.feat_dim]
    if declared_shape != expected_shape:
        raise ValueError(
            "Model metadata is inconsistent: input_shape must match sequence_frames and "
            f"features_per_frame; got input_shape={declared_shape}, expected={expected_shape}."
        )

    model_input_shape = predictor.model.input_shape
    if isinstance(model_input_shape, list):
        if len(model_input_shape) != 1:
            raise ValueError("SignSpeak model must have exactly one sequence input.")
        model_input_shape = model_input_shape[0]
    actual_shape = tuple(model_input_shape)
    if len(actual_shape) != 3 or actual_shape[1:] != (predictor.seq_len, predictor.feat_dim):
        raise ValueError(
            "Model file input shape does not match model metadata: "
            f"model_file={actual_shape}, metadata=({predictor.seq_len}, {predictor.feat_dim})."
        )

    model_output_shape = predictor.model.output_shape
    if isinstance(model_output_shape, list):
        if len(model_output_shape) != 1:
            raise ValueError("SignSpeak model must have exactly one classification output.")
        model_output_shape = model_output_shape[0]
    if len(model_output_shape) != 2 or model_output_shape[-1] != len(predictor.classes):
        raise ValueError(
            "Model output class count does not match metadata: "
            f"model_output={model_output_shape}, metadata_classes={len(predictor.classes)}."
        )

    if MODEL_FEATURE_SIZE_OVERRIDE is not None and predictor.feat_dim != int(MODEL_FEATURE_SIZE_OVERRIDE):
        raise ValueError(
            "SIGNSPEAK_MODEL_FEATURES_PER_FRAME/SIGNSPEAK_FEATURES_PER_FRAME does not match "
            f"model metadata: configured={MODEL_FEATURE_SIZE_OVERRIDE}, model={predictor.feat_dim}."
        )
    if predictor.seq_len != SEQUENCE_LENGTH:
        raise ValueError(
            "Backend input configuration does not match model metadata: "
            f"backend_sequence_frames={SEQUENCE_LENGTH}, "
            f"model=({predictor.seq_len}, {predictor.feat_dim})."
        )
    return predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model loading is synchronous and potentially slow; do not block the event loop.
    try:
        app.state.predictor = await run_in_threadpool(load_predictor)
        app.state.predictor_error = None
        logger.info("SignSpeak predictor ready from %s", MODEL_DIR)
    except Exception as exc:
        app.state.predictor = None
        app.state.predictor_error = str(exc)
        logger.exception("Could not load SignSpeak predictor")
    yield


app = FastAPI(title="SignSpeak API", version="1.0.0", lifespan=lifespan)

configured_origins = [
    origin.strip()
    for origin in os.environ.get("SIGNSPEAK_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins or [
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Report service and model readiness for local setup and deployment checks."""
    ready = app.state.predictor is not None
    model_feature_size = (
        app.state.predictor.feat_dim
        if ready
        else int(MODEL_FEATURE_SIZE_OVERRIDE or FRONTEND_FRAME_SIZE)
    )
    return {
        "status": "ok" if ready else "degraded",
        "model_ready": ready,
        "sequence_frames": SEQUENCE_LENGTH,
        "features_per_frame": model_feature_size,
        "browser_features_per_frame": FRONTEND_FRAME_SIZE,
        "input_adapter": "pose_and_hands_to_hands" if model_feature_size == HAND_FEATURE_SIZE else "none",
        **({"model_error": app.state.predictor_error} if not ready else {}),
    }


def _frames_from_message(payload: Any, model_feature_size: int) -> list[np.ndarray]:
    """Validate incoming frames and adapt browser pose+hands to hand-only models."""
    if isinstance(payload, dict):
        if "frames" in payload:
            raw_frames = payload["frames"]
        elif "keypoints" in payload:
            raw_frames = payload["keypoints"]
        else:
            raise ValueError("Expected a 'keypoints' frame or a 'frames' batch.")
    else:
        raw_frames = payload

    try:
        array = np.asarray(raw_frames, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Landmarks must be numeric arrays.") from exc

    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("Frames must be a one-dimensional frame or a two-dimensional frame batch.")
    if model_feature_size == HAND_FEATURE_SIZE and array.shape[1] == FRONTEND_FRAME_SIZE:
        # Browser contract: pose [0:99], first hand slot [99:162], second [162:225].
        array = array[:, POSE_FEATURE_SIZE:]
    elif array.shape[1] != model_feature_size:
        expected = (
            f"{model_feature_size} model values or {FRONTEND_FRAME_SIZE} browser values"
            if model_feature_size == HAND_FEATURE_SIZE
            else f"{model_feature_size} values"
        )
        raise ValueError(f"Each frame must contain {expected}; got {array.shape[1]}.")
    if array.shape[0] == 0:
        raise ValueError("At least one frame is required.")
    if not np.isfinite(array).all():
        raise ValueError("Landmarks must contain only finite values.")
    return [np.ascontiguousarray(frame, dtype=np.float32) for frame in array]


@app.websocket("/ws")
async def stream_predictions(websocket: WebSocket) -> None:
    await websocket.accept()
    predictor = app.state.predictor
    if predictor is None:
        await websocket.send_json({
            "type": "error",
            "error": "model_unavailable",
            "message": app.state.predictor_error or "The prediction model is unavailable.",
        })
        await websocket.close(code=1011, reason="Prediction model unavailable")
        return

    buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    try:
        while True:
            payload = await websocket.receive_json()
            try:
                frames = _frames_from_message(payload, predictor.feat_dim)
            except ValueError as exc:
                await websocket.send_json({"type": "error", "error": "invalid_frame", "message": str(exc)})
                continue

            for frame in frames:
                buffer.append(frame)
                if len(buffer) < SEQUENCE_LENGTH:
                    await websocket.send_json({
                        "type": "status",
                        "status": "buffering",
                        "frames_buffered": len(buffer),
                        "frames_required": SEQUENCE_LENGTH,
                    })
                    continue

                sequence = np.stack(buffer)
                try:
                    result = await run_in_threadpool(predictor.predict, sequence)
                except Exception:
                    logger.exception("SignSpeak inference failed")
                    await websocket.send_json({
                        "type": "error",
                        "error": "inference_failed",
                        "message": "Prediction failed for this sequence.",
                    })
                    continue

                await websocket.send_json({
                    "type": "prediction",
                    "predicted_sign": result["predicted_sign"],
                    "confidence": result["confidence"],
                    "top3": [
                        {"sign": sign, "confidence": round(float(confidence) * 100, 2)}
                        for sign, confidence in result["top3"]
                    ],
                    "frames_buffered": len(buffer),
                })
    except WebSocketDisconnect:
        logger.debug("SignSpeak client disconnected")
    except Exception:
        logger.exception("Unexpected WebSocket error")
        try:
            await websocket.close(code=1011, reason="Unexpected server error")
        except RuntimeError:
            pass


from pydantic import BaseModel, Field
import json
import time

SAMPLES_DIR = MODEL_DIR / "browser_samples"

class SamplePayload(BaseModel):
    label: str = Field(..., min_length=1)
    signer: str = Field(default="user")
    frames: list[list[float]]


class SignExplanationPayload(BaseModel):
    sign: str = Field(..., min_length=1, max_length=80)
    confidence: float = Field(..., ge=0, le=100)


@app.post("/api/explain_sign")
async def explain_sign(payload: SignExplanationPayload) -> dict[str, str]:
    """Explain a recognized label without sending camera frames to OpenAI."""
    sign = payload.sign.strip()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI is not configured. Add OPENAI_API_KEY to backend/.env and restart the server.",
        )
    if not sign:
        raise HTTPException(status_code=422, detail="A sign label is required.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=30.0, max_retries=1)
    try:
        response = await run_in_threadpool(
            client.responses.create,
            model=os.environ.get("OPENAI_MODEL", "gpt-6-astra"),
            instructions=(
                "You help users understand a sign label predicted by an Indian Sign Language demo. "
                "Explain the likely English word or idea in plain language in 2 to 4 short sentences. "
                "The input is only a class label and confidence score, not video or handshape data. "
                "Do not claim to verify the exact ISL sign, its handshape, or regional usage. "
                "Mention briefly that the model prediction may be wrong and signs can vary by region."
            ),
            input=f"Predicted sign label: {sign}\nModel confidence: {payload.confidence:.1f}%",
            max_output_tokens=160,
            store=False,
        )
    except Exception as exc:
        logger.exception("OpenAI sign explanation request failed")
        raise HTTPException(
            status_code=502,
            detail="OpenAI could not generate an explanation. Check the API key, model access, and backend logs.",
        ) from exc

    explanation = response.output_text.strip()
    if not explanation:
        raise HTTPException(status_code=502, detail="OpenAI returned an empty explanation.")
    return {"sign": sign, "explanation": explanation}

@app.post("/api/save_sample")
async def save_sample(payload: SamplePayload) -> dict[str, Any]:
    """Save a recorded landmark sequence from the browser data collection studio."""
    try:
        frames_array = np.asarray(payload.frames, dtype=np.float32)
    except Exception as exc:
        return {"status": "error", "message": f"Invalid numeric data: {exc}"}
        
    if frames_array.ndim != 2 or frames_array.shape[1] != FRONTEND_FRAME_SIZE:
        return {
            "status": "error",
            "message": f"Expected browser frames of shape (T, {FRONTEND_FRAME_SIZE}), got {frames_array.shape}"
        }
        
    T = frames_array.shape[0]
    if T < 10:
        return {"status": "error", "message": f"Too few frames ({T}); expected ~{SEQUENCE_LENGTH} frames."}

    if T != SEQUENCE_LENGTH:
        indices = np.linspace(0, T - 1, SEQUENCE_LENGTH)
        resampled = np.zeros((SEQUENCE_LENGTH, FRONTEND_FRAME_SIZE), dtype=np.float32)
        for col in range(FRONTEND_FRAME_SIZE):
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
        "features_per_frame": FRONTEND_FRAME_SIZE,
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
    """Return summary of all collected browser samples."""
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
