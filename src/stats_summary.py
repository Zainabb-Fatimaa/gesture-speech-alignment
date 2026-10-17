"""
stats_summary.py -- group-level statistics from results/v2_per_clip.csv.
Run: python src/stats_summary.py
Convention: z > 0 means gestures are CLOSER to stress than chance (one-sided p = lower tail).
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "results" / "v2_per_clip.csv")
df["name"] = df["clip"].str.replace(r"_clip_30s-90s", "", regex=True).str[:28]
z, p, n = df["z"].values, df["p_value"].values, df["n_gestures"].values
k = len(df)
out = []
def P(s=""): print(s); out.append(s)

P(f"CLIPS: {k}   total gestures: {n.sum()}   total stress peaks: {df.n_stress.sum()}")

# 1. per-clip + multiplicity
order = np.argsort(p); holm = np.empty(k); bh = np.empty(k)
run = 0
for r, i in enumerate(order):
    run = max(run, (k - r) * p[i]); holm[i] = min(1, run)
prev = 1
for r in range(k - 1, -1, -1):
    i = order[r]; prev = min(prev, p[i] * k / (r + 1)); bh[i] = prev
P("\n1) PER CLIP (one-sided p, then corrected for 8 tests)")
t = pd.DataFrame({"clip": df.name, "n_g": n, "z": z.round(2), "p": p.round(3),
                  "Holm": holm.round(3), "BH_FDR": bh.round(3)})
P(t.to_string(index=False))
P(f"   clips with p<0.05 uncorrected: {(p<0.05).sum()}  (expected by luck: {0.05*k:.1f});  after Holm: {(holm<0.05).sum()}")

# 2. combined tests
P("\n2) COMBINING CLIPS (H0: no gesture-stress coupling)")
zs = z.sum() / np.sqrt(k); w = np.sqrt(n); zw = (w * z).sum() / np.sqrt((w**2).sum())
fisher = -2 * np.log(np.clip(p, 1e-12, 1)).sum()
P(f"   Stouffer (equal weights):   z={zs:.2f}  one-sided p={stats.norm.sf(zs):.3f}  two-sided p={2*stats.norm.sf(abs(zs)):.3f}")
P(f"   Stouffer (weight sqrt(n)):  z={zw:.2f}  one-sided p={stats.norm.sf(zw):.3f}")
P(f"   Fisher (detects ANY clip effect): chi2={fisher:.1f} df={2*k}  p={stats.chi2.sf(fisher, 2*k):.3f}")
P(f"   One-sample t-test on z (mean {z.mean():.2f}, sd {z.std(ddof=1):.2f}): t={stats.ttest_1samp(z,0).statistic:.2f}, p(two-sided)={stats.ttest_1samp(z,0).pvalue:.3f}")
P(f"   Wilcoxon signed-rank on z: p(two-sided)={stats.wilcoxon(z).pvalue:.3f}")
npos = (z > 0).sum(); sign = stats.binomtest(npos, k, 0.5)
P(f"   Sign test: {npos}/{k} clips positive, p(two-sided)={sign.pvalue:.3f}")

# 3. effect size
rel = (df.null_mean_dist - df.obs_mean_dist) / df.null_mean_dist   # >0 = closer than chance
rng = np.random.default_rng(0)
boot = np.array([rng.choice(rel, k).mean() for _ in range(20000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
P("\n3) EFFECT SIZE: how much closer to stress than chance (fraction of the chance distance)")
P(f"   mean = {rel.mean()*100:+.1f}%   95% bootstrap CI over clips = [{lo*100:+.1f}%, {hi*100:+.1f}%]")
P(f"   (0% = no coupling; positive = gestures nearer stress). Mean obs {df.obs_mean_dist.mean():.3f}s vs chance {df.null_mean_dist.mean():.3f}s")
ci_z = (z.mean() - stats.t.ppf(.975, k-1) * z.std(ddof=1)/np.sqrt(k), z.mean() + stats.t.ppf(.975, k-1) * z.std(ddof=1)/np.sqrt(k))
P(f"   mean z = {z.mean():.2f}, 95% t-CI = [{ci_z[0]:.2f}, {ci_z[1]:.2f}]")

# 4. heterogeneity
Q = ((z - z.mean())**2).sum(); I2 = max(0, (Q - (k-1)) / Q) * 100 if Q > 0 else 0
P("\n4) HETEROGENEITY across clips (are clips consistent with one common effect?)")
P(f"   Cochran Q={Q:.1f} df={k-1} p={stats.chi2.sf(Q, k-1):.3f}   I^2={I2:.0f}%")
P("   (z-scores of a true null should have variance 1; observed var = %.2f)" % z.var(ddof=1))

# 5. robustness
P("\n5) LEAVE-ONE-OUT pooled Stouffer z (does one clip drive it?)")
loo = [(df.name[i], (z.sum() - z[i]) / np.sqrt(k - 1)) for i in range(k)]
P("   " + "; ".join(f"-{a[:10]}: {b:.2f}" for a, b in loo))
P(f"   range: {min(b for _,b in loo):.2f} to {max(b for _,b in loo):.2f}")
# gestures-per-clip vs z (is the signal stronger where there is more data?)
rho = stats.spearmanr(n, z)
P(f"   Spearman(n_gestures, z) = {rho.statistic:.2f} (p={rho.pvalue:.2f}); corr(visible%, z) = {stats.spearmanr(df.visible_pct, z).statistic:.2f}")

# 6. power
P("\n6) SENSITIVITY: what could this design have detected?")
mdz = (1.645 + 0.84) / np.sqrt(k)
P(f"   Pooled z has SD 1 under H0, so 80% power at one-sided 0.05 needs a pooled shift of {1.645+0.84:.2f},")
P(f"   i.e. a mean per-clip z of about {mdz:.2f}. Observed mean z = {z.mean():.2f}, 95% CI [{ci_z[0]:.2f}, {ci_z[1]:.2f}].")
P(f"   The CI upper end ({ci_z[1]:.2f}) is above that, so a moderate effect is NOT ruled out with only {k} clips;")
P("   the data exclude only a large, consistent coupling (mean z/clip well above ~1.1).")
Path(ROOT / "results" / "stats_summary.txt").write_text("\n".join(out), encoding="utf-8")
