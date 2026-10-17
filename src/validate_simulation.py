"""
validate_simulation.py

Positive/negative control for the alignment tests. Plants gestures with a
KNOWN fraction locked to stress peaks, then checks how often each test
declares significance (power) and, at lock=0, how often it falsely does
(false-positive rate; should be ~5%).

Run: python src/validate_simulation.py
"""

import numpy as np
from alignment import match_peaks, permutation_test_match_rate, nn_distance_test

DUR = 60.0


def make_stress(rate, rng):
    t, out = 0.0, []
    while True:
        t += max(0.1, rng.exponential(1.0 / rate))
        if t >= DUR:
            break
        out.append(t)
    return np.array(out)


def make_gestures(stress, n, lock, lead_s, jitter_s, rng):
    g = []
    for _ in range(n):
        if rng.random() < lock:
            g.append(np.clip(rng.choice(stress) - lead_s + rng.normal(0, jitter_s), 0, DUR))
        else:
            g.append(rng.uniform(0, DUR))
    return np.sort(np.array(g))


def power(stress_rate, lock, n_gest=20, trials=100, window=0.5, seed=0):
    rng = np.random.default_rng(seed)
    old_hits = new_hits = 0
    for _ in range(trials):
        s = make_stress(stress_rate, rng)
        g = make_gestures(s, n_gest, lock, lead_s=0.05, jitter_s=0.08, rng=rng)
        old = permutation_test_match_rate(g, s, DUR, window=window, n_permutations=200, rng=rng)
        new = nn_distance_test(g, s, DUR, n_permutations=500, rng=rng)
        old_hits += old["p_value"] < 0.05
        new_hits += new["p_value"] < 0.05
    return old_hits / trials, new_hits / trials


if __name__ == "__main__":
    print(f"{'stress/s':>8} {'lock':>5} | {'old test':>8} {'new test':>8}   (fraction significant at p<0.05)")
    for rate in (7.0, 2.5):
        for lock in (0.0, 0.3, 0.6):
            o, n = power(rate, lock)
            tag = "  <- false-positive rate" if lock == 0 else ""
            print(f"{rate:>8} {lock:>5} | {o:>8.2f} {n:>8.2f}{tag}")
