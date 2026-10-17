"""
visibility.py

Finds when the speaker is actually on screen (vs. slides / cutaways) using
MediaPipe's BlazeFace detector sampled ~5x per second. Time is binned
into 1-second bins; a bin is 'visible' if a face shows up in >= min_frac of its
samples, then short gaps (< close_gap_s) are bridged so a head turn is not
treated as a slide.
"""
from pathlib import Path
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
_PATH = Path.home() / ".cache" / "mediapipe_models" / "blaze_face_short_range.tflite"


def speaker_visible_mask(video_path: Path, sample_hz: float = 5.0, min_frac: float = 0.2, close_gap_s: int = 2):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_hz)))
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _PATH.exists():
        urllib.request.urlretrieve(_URL, _PATH)
    det = mp_vision.FaceDetector.create_from_options(mp_vision.FaceDetectorOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(_PATH)), min_detection_confidence=0.5))
    hits, idx = {}, 0
    while cap.grab():
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                hits.setdefault(int(idx / fps), []).append(len(res.detections) > 0)
        idx += 1
    cap.release(); det.close()
    n = int(idx / fps) + 1
    mask = np.array([np.mean(hits[s]) >= min_frac if s in hits else False for s in range(n)])
    # bridge short gaps
    i = 0
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            if 0 < i and j < n and (j - i) <= close_gap_s:
                mask[i:j] = True
            i = j
        else:
            i += 1
    return mask  # mask[s] True => second s is speaker-visible
