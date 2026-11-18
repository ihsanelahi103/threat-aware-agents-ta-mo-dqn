#!/usr/bin/env python3
"""Offline design check for detector v2 (self-referenced) using DEVELOPMENT runs only
(seed 7, never the evaluation seeds 42/43/44).

Usage:  python3 design_detector.py results/<dev_MNIST>.json results/<dev_Fashion>.json

Each logged report carries (key, score, replay, conf_delta, pcf, dis_self, z_pop, dis_pop).
The score is re-computed offline for a grid of tolerances, the honest-only score range is
reported, and a drift stress test adds Gaussian noise to the confidence shift of every report
(simulating honest models that change slowly) to show how the detector degrades.
"""
import collections, json, sys
import numpy as np

if len(sys.argv) < 2:
    sys.exit(__doc__)

def load(path):
    d = json.load(open(path))["datasets"][0]
    A = [e for e in d["score_log_attack"] if len(e) >= 8 and e[3] is not None]
    B = [e for e in d["score_log_benign"] if len(e) >= 8 and e[3] is not None]
    return d.get("dataset"), A, B

data = [load(p) for p in sys.argv[1:]]
for name, A, B in data:
    print("%-14s attack reports=%d benign reports=%d (honest agents=%d)" %
          (name, len(A), len(B), sum(1 for e in B if e[0] == 0)))

rng = np.random.RandomState(0)
def rescore(e, tol, pcfs, noise):
    cd = e[3] + (rng.normal(0, noise) if noise else 0.0)
    pcf = e[4]; ds = e[5]
    return (0.45 * max(0.0, ds) + 0.25 * min(1.0, abs(cd) / tol)
            + 0.20 * (1.0 if e[2] else 0.0) + 0.10 * min(1.0, pcf / pcfs))

for tol, pcfs in ((0.02, 0.10), (0.03, 0.10), (0.02, 0.20), (0.01, 0.10)):
    print("\n=== conf_tol=%.3f pcf_scale=%.2f ===" % (tol, pcfs))
    allh = []
    for name, A, B in data:
        hs = [rescore(e, tol, pcfs, 0) for e in B]
        allh += hs
        print("  %-14s honest+byz-benign score  max=%.3f p99.9=%.3f  (n=%d)" %
              (name, max(hs), np.quantile(hs, .999), len(hs)))
    hmax = max(allh)
    print("  max benign score over datasets = %.3f" % hmax)
    for thr in sorted({round(hmax + 0.01, 2), 0.10, 0.12, 0.15}):
        print("  thr=%.2f" % thr)
        for name, A, B in data:
            by = collections.defaultdict(list)
            for e in A:
                by[e[0]].append(rescore(e, tol, pcfs, 0) >= thr or bool(e[2]))
            print("    %-14s " % name + "  ".join("%s=%.2f(n=%d)" % (k, np.mean(v), len(v)) for k, v in sorted(by.items(), key=lambda x: str(x[0]))))
        for noise in (0.0, 0.005, 0.01, 0.02):
            fp = []
            for name, A, B in data:
                fp.append(np.mean([rescore(e, tol, pcfs, noise) >= thr or bool(e[2]) for e in B]))
            cf = [np.mean([rescore(e, tol, pcfs, noise) >= thr for e in A if e[0] == "conf_falsify"] or [0]) for _, A, _ in data]
            print("      drift sigma=%.3f  benign FPR=%s   conf_falsify TPR=%s" %
                  (noise, "/".join("%.4f" % x for x in fp), "/".join("%.2f" % x for x in cf)))
