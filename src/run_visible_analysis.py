"""Alignment test restricted to speaker-visible time (slides/cutaways removed)."""
from pathlib import Path
import numpy as np, pandas as pd
from audio_prosody import get_prosodic_contours, detect_salient_stress_peaks
from visibility import speaker_visible_mask
from alignment import nn_distance_test

ROOT = Path(__file__).resolve().parent.parent
cache = ROOT / "data" / "tmp"
rng = np.random.default_rng(0)


def compress(times, mask):
    """Map real time -> time on the concatenated visible axis; drop hidden times."""
    times = np.asarray(times); keep = np.array([t < len(mask) and mask[int(t)] for t in times], bool)
    csum = np.concatenate([[0], np.cumsum(mask)])  # visible seconds before second s
    t = times[keep]
    return csum[t.astype(int)] + (t - np.floor(t)), keep


rows, cz_inputs = [], {}
for c in sorted((ROOT / "data" / "clips").glob("*.mp4")):
    mp = cache / f"{c.stem}_visible.npy"
    mask = np.load(mp) if mp.exists() else speaker_visible_mask(c)
    np.save(mp, mask)
    t, f0, inten = get_prosodic_contours(cache / f"{c.stem}.wav")
    S = detect_salient_stress_peaks(t, f0, inten)
    G = np.load(cache / f"{c.stem}_gesture_peaks.npy")
    vis_dur = float(mask.sum())
    Sv, _ = compress(S, mask); Gv, _ = compress(G, mask)
    r = nn_distance_test(Gv, Sv, vis_dur, rng=rng) if vis_dur > 0 else {}
    r.update(clip=c.stem, visible_s=vis_dur, visible_pct=100 * vis_dur / len(mask),
             n_gestures=len(Gv), n_gestures_dropped=len(G) - len(Gv),
             n_stress=len(Sv), n_stress_dropped=len(S) - len(Sv))
    rows.append(r); cz_inputs[c.stem] = (Gv, Sv, vis_dur)
    print(f"{c.stem[:28]:28s} visible {r['visible_pct']:.0f}%  gestures {len(Gv)}  stress {len(Sv)} (dropped {r['n_stress_dropped']})", flush=True)

df = pd.DataFrame(rows)
cols = ["clip", "visible_pct", "n_gestures", "n_stress", "obs_mean_dist", "null_mean_dist", "z", "p_value"]
print(df[cols].round(3).to_string())
z = df["z"].dropna()
print(f"\nPOOLED Stouffer z = {z.sum()/np.sqrt(len(z)):.2f} over {len(z)} clips")
cz = []
for a, (Ga, _, da) in cz_inputs.items():
    for b, (_, Sb, db) in cz_inputs.items():
        if a != b and min(da, db) > 0:
            d = min(da, db); cz.append(nn_distance_test(Ga[Ga < d], Sb[Sb < d], d, n_permutations=500, rng=rng)["z"])
cz = np.array(cz); cz = cz[~np.isnan(cz)]
print(f"CONTROL (mismatched clips): mean z = {cz.mean():.2f}, sd = {cz.std():.2f}, n={len(cz)}")
df.to_csv(ROOT / "results" / "visible_per_clip.csv", index=False)
