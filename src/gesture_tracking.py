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


def extract_hand_track(video_path: Path, max_hands: int = 2, min_conf: float = 0.3):
    """Per-frame mean wrist (x, y) in normalized image coords; NaN when no hand.
    Lower confidence than the original (0.5) so blurry beat gestures are not dropped."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    base = mp_tasks.BaseOptions(model_asset_path=str(_ensure_hand_landmarker_model()))
    lm = mp_vision.HandLandmarker.create_from_options(mp_vision.HandLandmarkerOptions(
        base_options=base, running_mode=mp_vision.RunningMode.VIDEO, num_hands=max_hands,
        min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf,
        min_tracking_confidence=min_conf))
    t, xs, ys, i = [], [], [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), int(i / fps * 1000))
            if res.hand_landmarks:
                xs.append(float(np.mean([h[0].x for h in res.hand_landmarks])))
                ys.append(float(np.mean([h[0].y for h in res.hand_landmarks])))
            else:
                xs.append(np.nan); ys.append(np.nan)
            t.append(i / fps); i += 1
    finally:
        lm.close(); cap.release()
    return np.array(t), np.array(xs), np.array(ys)


def detect_gesture_peaks_v2(times, x, y, max_gap_s: float = 0.2, k_mad: float = 2.0, min_spacing_s: float = 0.3):
    """Speaker-relative peak detection on 2D wrist speed.
    Gaps <= max_gap_s are interpolated, speed is smoothed, and a peak must exceed
    median + k_mad * (robust sigma) of that speaker's own speed distribution."""
    from scipy.signal import savgol_filter
    fps = 1.0 / np.median(np.diff(times))
    xi, yi = x.copy(), y.copy()
    isn = np.isnan(x); idx = np.arange(len(x))
    good = ~isn
    if good.sum() < 10:
        return np.array([])
    # interpolate only across short gaps
    xf = np.interp(idx, idx[good], x[good]); yf = np.interp(idx, idx[good], y[good])
    run = 0
    for i in range(len(x)):
        run = run + 1 if isn[i] else 0
        if isn[i]:
            j = i
            while j + 1 < len(x) and isn[j + 1]:
                j += 1
            if (j - i + 1) / fps <= max_gap_s and i > 0 and j < len(x) - 1:
                xi[i:j + 1], yi[i:j + 1] = xf[i:j + 1], yf[i:j + 1]
    valid = ~np.isnan(xi)
    speed = np.full(len(x), np.nan)
    sx = np.where(valid, xi, 0.0); sy = np.where(valid, yi, 0.0)
    if len(x) > 7:
        sx, sy = savgol_filter(sx, 5, 2), savgol_filter(sy, 5, 2)
    v = np.hypot(np.gradient(sx), np.gradient(sy)) * fps
    v[~valid] = np.nan
    ref = v[valid]
    med = np.median(ref); sig = 1.4826 * np.median(np.abs(ref - med))
    thr = med + k_mad * sig
    peaks, _ = __import__("scipy.signal", fromlist=["find_peaks"]).find_peaks(
        np.nan_to_num(v, nan=0.0), height=thr, distance=max(1, int(min_spacing_s * fps)))
    return times[peaks]
