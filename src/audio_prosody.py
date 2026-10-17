"""
audio_prosody.py

Extracts pitch (F0) and intensity contours from a video's audio track using
Parselmouth (Praat), and detects local peaks as candidate "stress" points.
"""

from pathlib import Path
import subprocess
import numpy as np
import parselmouth
from scipy.signal import find_peaks


def extract_audio(video_path: Path, out_wav: Path, sr: int = 16000) -> Path:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ac", "1", "-ar", str(sr),
            str(out_wav),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_wav


def get_prosodic_contours(wav_path: Path, time_step: float = 0.01):
    snd = parselmouth.Sound(str(wav_path))

    pitch = snd.to_pitch(time_step=time_step)
    intensity = snd.to_intensity(time_step=time_step)

    times = pitch.xs()
    f0 = np.array([pitch.get_value_at_time(t) or np.nan for t in times])
    inten = np.array([intensity.get_value(t) or np.nan for t in times])

    return times, f0, inten


def _voiced_runs(f0: np.ndarray, min_run_samples: int = 5):
    runs = []
    n = len(f0)
    i = 0
    while i < n:
        if not np.isnan(f0[i]):
            j = i
            while j + 1 < n and not np.isnan(f0[j + 1]):
                j += 1
            if (j - i + 1) >= min_run_samples:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def detect_stress_peaks(times, f0, intensity, min_prominence_f0=15.0, min_prominence_db=3.0):
    f0_peak_idx = []
    for start, end in _voiced_runs(f0):
        segment = f0[start:end + 1]
        local_idx, _ = find_peaks(segment, prominence=min_prominence_f0)
        f0_peak_idx.extend([start + i for i in local_idx])
    f0_peak_idx = np.array(sorted(f0_peak_idx), dtype=int)

    inten_filled = np.nan_to_num(intensity, nan=np.nanmin(intensity) if np.any(~np.isnan(intensity)) else 0.0)
    inten_peak_idx, _ = find_peaks(inten_filled, prominence=min_prominence_db)

    f0_times = times[f0_peak_idx] if len(f0_peak_idx) else np.array([])
    peak_times = np.concatenate([f0_times, times[inten_peak_idx]])
    peak_times.sort()

    merged = []
    for t in peak_times:
        if not merged or (t - merged[-1]) > 0.1:
            merged.append(t)
    return np.array(merged)


def detect_salient_stress_peaks(
    times, f0, intensity,
    min_prominence_st=1.0,
    min_prominence_db=3.0,
    coincidence_s=0.15,
    min_spacing_s=0.4,
):
    """Only perceptually salient stress: an F0 peak (semitones, so speaker-
    independent) that coincides with an intensity peak, spaced >= min_spacing_s.
    Mirrors the gesture side, which only fires on visibly large movements."""
    f0_st = 12.0 * np.log2(np.where(np.isnan(f0), np.nan, f0 / 100.0))
    f0_idx = []
    for start, end in _voiced_runs(f0):
        local, _ = find_peaks(f0_st[start:end + 1], prominence=min_prominence_st)
        f0_idx.extend(start + local)

    floor = np.nanmin(intensity) if np.any(~np.isnan(intensity)) else 0.0
    inten_idx, _ = find_peaks(np.nan_to_num(intensity, nan=floor), prominence=min_prominence_db)
    inten_t = times[inten_idx]

    peaks = []
    for i in sorted(f0_idx):
        t = times[i]
        if len(inten_t) and np.min(np.abs(inten_t - t)) <= coincidence_s:
            if not peaks or t - peaks[-1] >= min_spacing_s:
                peaks.append(t)
    return np.array(peaks)


def get_stress_peaks_for_video(video_path: Path, tmp_dir: Path = Path("data/tmp")):
    wav_path = tmp_dir / f"{video_path.stem}.wav"
    extract_audio(video_path, wav_path)
    times, f0, inten = get_prosodic_contours(wav_path)
    peaks = detect_stress_peaks(times, f0, inten)
    return peaks, times, f0, inten
