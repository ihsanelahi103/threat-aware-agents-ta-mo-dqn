#!/usr/bin/env python3
"""Feature distributions from the development runs (seed 7 only)."""
import json, sys
import numpy as np

def q(v):
    v = np.asarray(v, float)
    return "n=%d min=%.4f p5=%.4f p25=%.4f med=%.4f max=%.4f" % (
        len(v), v.min(), np.quantile(v, .05), np.quantile(v, .25), np.median(v), v.max())

for path in sys.argv[1:]:
    d = json.load(open(path))["datasets"][0]
    A = [e for e in d["score_log_attack"] if len(e) >= 8 and e[3] is not None]
    B = [e for e in d["score_log_benign"] if len(e) >= 8 and e[3] is not None]
    print("\n==", d["dataset"], path.split("/")[-1])
    print("benign |conf_delta|:", q([abs(e[3]) for e in B]), " frac exactly 0: %.3f" % np.mean([e[3] == 0 for e in B]))
    print("benign pcf:", q([e[4] for e in B]))
    for k in ("conf_falsify", "label_flip", "inconsistent", "replay"):
        S = [e for e in A if e[0] == k]
        if not S: continue
        zero = np.mean([(abs(e[3]) < 1e-9 and e[4] == 0 and e[5] <= 0) and not e[2] for e in S])
        print("%-13s |conf_delta| %s" % (k, q([abs(e[3]) for e in S])))
        print("%-13s pcf %s   dis_self med=%.3f   replay-flagged=%.2f   no-observable-change=%.2f" %
              ("", q([e[4] for e in S]), np.median([e[5] for e in S]), np.mean([e[2] for e in S]), zero))
