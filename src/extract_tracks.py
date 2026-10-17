"""One slow pass: cache per-frame hand tracks for every clip."""
from pathlib import Path
import numpy as np
from gesture_tracking import extract_hand_track
ROOT = Path(__file__).resolve().parent.parent
for c in sorted((ROOT / "data" / "clips").glob("*.mp4")):
    out = ROOT / "data" / "tmp" / f"{c.stem}_track.npz"
    if out.exists():
        continue
    t, x, y = extract_hand_track(c)
    np.savez(out, t=t, x=x, y=y)
    print(c.stem[:30], f"frames={len(t)} hand_detected={np.mean(~np.isnan(x))*100:.0f}%", flush=True)
