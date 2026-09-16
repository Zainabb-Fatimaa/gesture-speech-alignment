"""
alignment.py

Aligns gesture peaks with stress peaks and computes timing offsets.
Offset convention: offset = gesture_time - nearest_stress_time
  - negative offset -> gesture preceded the stress peak
  - positive offset -> gesture followed the stress peak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def match_peaks(gesture_peaks: np.ndarray, stress_peaks: np.ndarray, window: float = 0.5):
    rows = []
    for g in gesture_peaks:
        if len(stress_peaks) == 0:
            rows.append({"gesture_time": g, "matched_stress_time": np.nan, "offset_s": np.nan, "matched": False})
            continue
        diffs = stress_peaks - g
        nearest_idx = np.argmin(np.abs(diffs))
        nearest_diff = diffs[nearest_idx]
        if abs(nearest_diff) <= window:
            rows.append({
                "gesture_time": g,
                "matched_stress_time": stress_peaks[nearest_idx],
                "offset_s": -nearest_diff,  # gesture - stress
                "matched": True,
            })
        else:
            rows.append({"gesture_time": g, "matched_stress_time": np.nan, "offset_s": np.nan, "matched": False})
    return pd.DataFrame(rows)


def permutation_test_match_rate(
    gesture_peaks: np.ndarray,
    stress_peaks: np.ndarray,
    clip_duration: float,
    window: float = 0.5,
    n_permutations: int = 500,
    rng: np.random.Generator | None = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng()

    observed_df = match_peaks(gesture_peaks, stress_peaks, window=window)
    observed_rate = observed_df["matched"].mean() if len(observed_df) else np.nan

    if len(gesture_peaks) == 0 or len(stress_peaks) == 0:
        return {
            "observed_match_rate": observed_rate,
            "null_mean_match_rate": np.nan,
            "null_std_match_rate": np.nan,
            "p_value": np.nan,
            "n_permutations": n_permutations,
        }

    null_rates = np.empty(n_permutations)
    for k in range(n_permutations):
        shift = rng.uniform(0, clip_duration)
        shifted_stress = np.mod(stress_peaks + shift, clip_duration)
        shifted_df = match_peaks(gesture_peaks, shifted_stress, window=window)
        null_rates[k] = shifted_df["matched"].mean()

    p_value = float(np.mean(null_rates >= observed_rate))

    return {
        "observed_match_rate": float(observed_rate),
        "null_mean_match_rate": float(null_rates.mean()),
        "null_std_match_rate": float(null_rates.std()),
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


def summarize_offsets(df: pd.DataFrame) -> dict:
    matched = df[df["matched"]]
    if len(matched) == 0:
        return {
            "n_gestures": len(df),
            "n_matched": 0,
            "match_rate": 0.0,
            "median_offset_ms": np.nan,
            "mean_offset_ms": np.nan,
            "pct_gesture_precedes": np.nan,
        }
    offsets_ms = matched["offset_s"] * 1000
    return {
        "n_gestures": len(df),
        "n_matched": len(matched),
        "match_rate": len(matched) / len(df),
        "median_offset_ms": float(offsets_ms.median()),
        "mean_offset_ms": float(offsets_ms.mean()),
        "pct_gesture_precedes": float((offsets_ms < 0).mean() * 100),
    }


def plot_offset_histogram(all_offsets_ms: np.ndarray, out_path: str = "results/offset_histogram.png"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(all_offsets_ms, bins=20, color="#4C72B0", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Offset (ms): negative = gesture precedes stress")
    ax.set_ylabel("Count")
    ax.set_title("Gesture-Stress Timing Offset Distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_clip_timeline(times_label: str, gesture_peaks: np.ndarray, stress_peaks: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.eventplot(gesture_peaks, lineoffsets=1, colors="#4C72B0", label="Gesture peaks")
    ax.eventplot(stress_peaks, lineoffsets=0, colors="#DD8452", label="Stress peaks")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Stress", "Gesture"])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Timeline: {times_label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
