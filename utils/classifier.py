"""Modality C - bowler run-up CNN (Keras .h5).

The shipped model (cricket_runup_final.h5) is a full Keras-3 Functional model:
MobileNetV2 -> GlobalAveragePooling2D -> Dense(128, relu) -> Dropout -> Dense(1, sigmoid),
input 224x224x3. We load it directly; if that fails we rebuild exactly that
architecture and load the weights, and *verify* the weights really changed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .ocr import grab_frame


class ModelLoadError(RuntimeError):
    pass


@dataclass
class ModelInfo:
    input_size: Tuple[int, int] = (224, 224)     # (H, W)
    mode: str = "full model"
    warnings: List[str] = field(default_factory=list)


def _input_hw(model) -> Tuple[int, int]:
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        h, w = shape[1], shape[2]
        if h and w:
            return int(h), int(w)
    except Exception:
        pass
    return 224, 224


def _rebuild_and_load(tf, path: str, warnings: List[str]):
    L = tf.keras.layers
    base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    x = L.GlobalAveragePooling2D()(base.output)
    x = L.Dense(128, activation="relu")(x)
    x = L.Dropout(0.3)(x)
    out = L.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(base.input, out)

    before = [w.copy() for w in model.get_weights()]
    try:
        model.load_weights(path)
    except Exception as e1:
        warnings.append(f"Strict weight load failed ({type(e1).__name__}); trying by-name load.")
        try:
            model.load_weights(path, by_name=True, skip_mismatch=True)
        except Exception as e2:
            raise ModelLoadError(f"Could not load weights: {e2}") from e2
    after = model.get_weights()
    changed = sum(not np.array_equal(a, b) for a, b in zip(before, after))
    if changed / max(1, len(after)) < 0.8:
        raise ModelLoadError(
            f"Only {changed}/{len(after)} weight tensors were loaded - the layer layout in the "
            "file doesn't match the rebuilt architecture, so predictions would be random.")
    return model


def load_runup_model(path: str):
    """Return (model, ModelInfo). Raises ModelLoadError on failure."""
    if not path or not os.path.exists(path):
        raise ModelLoadError(f"Model file not found: {path}")
    try:
        import tensorflow as tf
    except Exception as e:
        raise ModelLoadError(f"TensorFlow is not installed/importable: {e}") from e

    info = ModelInfo()
    try:
        model = tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        info.warnings.append(
            f"load_model() failed ({type(e).__name__}: {str(e)[:160]}). "
            "This file was saved with Keras 3 - use tensorflow>=2.16. Rebuilding architecture instead.")
        model = _rebuild_and_load(tf, path, info.warnings)
        info.mode = "rebuilt architecture + weights"

    info.input_size = _input_hw(model)
    try:  # smoke test: catches shape/layer mismatches now rather than mid-run
        h, w = info.input_size
        p = model.predict(np.zeros((1, h, w, 3), np.float32), verbose=0)
        if np.asarray(p).size < 1:
            raise ValueError("empty prediction")
    except Exception as e:
        raise ModelLoadError(f"Model loaded but prediction failed: {e}") from e
    return model, info


# --------------------------------------------------------------------------- #
def preprocess(frame_bgr, size: Tuple[int, int], norm: str = "0to1") -> np.ndarray:
    h, w = size
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h)).astype(np.float32)
    if norm == "0to1":
        img /= 255.0
    elif norm == "-1to1":
        img = img / 127.5 - 1.0
    return img


def predict_frames(model, frames: Sequence[np.ndarray], size, norm="0to1", batch=16) -> np.ndarray:
    if not frames:
        return np.zeros(0, np.float32)
    x = np.stack([preprocess(f, size, norm) for f in frames])
    out = []
    for i in range(0, len(x), batch):
        p = np.asarray(model.predict(x[i:i + batch], verbose=0))
        out.append(p.reshape(len(p), -1)[:, 0])
    return np.concatenate(out).astype(np.float32)


def score_window(cap: cv2.VideoCapture, model, size, norm: str, t_end: float,
                 lookback: float = 10.0, fps: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Run-up probability at ``fps`` samples/second over [t_end-lookback, t_end]."""
    times = np.arange(max(0.0, t_end - lookback), t_end + 1e-6, 1.0 / fps)
    frames, kept = [], []
    for t in times:
        f = grab_frame(cap, float(t))
        if f is not None:
            frames.append(f)
            kept.append(t)
    return np.asarray(kept), predict_frames(model, frames, size, norm)


def locate_runup_start(times: np.ndarray, probs: np.ndarray, threshold: float = 0.5,
                       smooth_win: int = 3) -> Tuple[Optional[float], float]:
    """Sliding-window (moving-average) smoothing, then the first frame of the
    contiguous above-threshold region around the peak. Returns (start, confidence);
    start is None when nothing exceeds the threshold."""
    if len(probs) == 0:
        return None, 0.0
    w = max(1, min(smooth_win, len(probs)))
    sm = np.convolve(probs, np.ones(w) / w, mode="same")
    peak = int(np.argmax(sm))
    conf = float(sm[peak])
    if conf < threshold:
        return None, conf
    j = peak
    while j > 0 and sm[j - 1] >= threshold:
        j -= 1
    return float(times[j]), conf
