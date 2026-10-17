"""Full run: salient stress peaks vs gesture peaks, window-free test, pooled + mismatched-clip control."""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from audio_prosody import get_prosodic_contours, detect_salient_stress_peaks
from gesture_tracking import get_gesture_peaks_for_video
from alignment import nn_distance_test

ROOT = Path(__file__).resolve().parent.parent
cache = ROOT / "data" / "tmp"
clips = sorted((ROOT / "data" / "clips").glob("*.mp4"))
S, G, DUR = {}, {}, {}
for c in clips:
    t, f0, inten = get_prosodic_contours(cache / f"{c.stem}.wav")
    S[c.stem], DUR[c.stem] = detect_salient_stress_peaks(t, f0, inten), t[-1]
    gp = cache / f"{c.stem}_gesture_peaks.npy"
    if gp.exists():
        G[c.stem] = np.load(gp)
    else:
        G[c.stem] = get_gesture_peaks_for_video(c)[0]; np.save(gp, G[c.stem])
    print(c.stem[:30], "gestures", len(G[c.stem]), "stress", len(S[c.stem]), flush=True)

rng = np.random.default_rng(0)
rows = []
for k in S:
    r = nn_distance_test(G[k], S[k], DUR[k], rng=rng); r["clip"] = k; rows.append(r)
df = pd.DataFrame(rows)
print(df[["clip", "obs_mean_dist", "null_mean_dist", "z", "p_value"]].round(3).to_string())
z = df["z"].dropna(); n = len(z)
print(f"\nPOOLED Stouffer z = {z.sum()/np.sqrt(n):.2f} over {n} clips (positive = gestures closer to stress than chance)")

# control: gestures of clip i vs stress of clip j (should be ~0)
keys = list(S); cz = []
for i, a in enumerate(keys):
    for b in keys:
        if a != b:
            d = min(DUR[a], DUR[b]); g = G[a][G[a] < d]; s = S[b][S[b] < d]
            cz.append(nn_distance_test(g, s, d, n_permutations=500, rng=rng)["z"])
cz = np.array(cz); cz = cz[~np.isnan(cz)]
print(f"CONTROL (mismatched clips): mean z = {cz.mean():.2f}, sd = {cz.std():.2f}, n={len(cz)}")
df.to_csv(ROOT / "results" / "salient_per_clip.csv", index=False)
