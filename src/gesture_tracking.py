"""
gesture_tracking.py

Tracks hand position over a video using MediaPipe's HandLandmarker (the
current Tasks API), computes vertical velocity as a proxy for "beat"
gesture movement, and detects local peaks as candidate gesture points.
"""

from pathlib import Path
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from scipy.signal import find_peaks

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
_MODEL_PATH = Path.home() / ".cache" / "mediapipe_models" / "hand_landmarker.task"


def _ensure_hand_landmarker_model() -> Path:
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _MODEL_PATH.exists():
        print(f"Downloading hand landmarker model to {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


def _make_landmarker(max_hands: int = 2) -> mp_vision.HandLandmarker:
    model_path = _ensure_hand_landmarker_model()
    base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def extract_hand_heights(video_path: Path, max_hands: int = 2, sample_every_n_frames: int = 1):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    times, heights = [], []
    frame_idx = 0

    landmarker = _make_landmarker(max_hands=max_hands)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n_frames == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                t = frame_idx / fps
                timestamp_ms = int(t * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.hand_landmarks:
                    ys = []
                    for hand_landmarks in result.hand_landmarks:
                        ys.append(hand_landmarks[0].y)
                    avg_y = float(np.mean(ys))
                    heights.append(1.0 - avg_y)  # invert: higher hand -> larger value
                else:
                    heights.append(np.nan)
                times.append(t)

            frame_idx += 1
    finally:
        landmarker.close()
        cap.release()

    return np.array(times), np.array(heights)


def compute_velocity(times: np.ndarray, heights: np.ndarray) -> np.ndarray:
    velocity = np.full_like(heights, np.nan, dtype=float)
    for i in range(1, len(heights)):
        if not (np.isnan(heights[i]) or np.isnan(heights[i - 1])):
            dt = times[i] - times[i - 1]
            if dt > 0:
                velocity[i] = (heights[i] - heights[i - 1]) / dt
    return velocity


def detect_gesture_peaks(times: np.ndarray, velocity: np.ndarray, min_prominence: float = 0.5):
    abs_vel = np.nan_to_num(np.abs(velocity), nan=0.0)
    peak_idx, _ = find_peaks(abs_vel, prominence=min_prominence)
    peak_times = times[peak_idx]

    merged = []
    for t in sorted(peak_times):
        if not merged or (t - merged[-1]) > 0.15:
            merged.append(t)
    return np.array(merged)


def get_gesture_peaks_for_video(video_path: Path):
    times, heights = extract_hand_heights(video_path)
    velocity = compute_velocity(times, heights)
    peaks = detect_gesture_peaks(times, velocity)
    return peaks, times, heights, velocity
