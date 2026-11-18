#!/usr/bin/env python3
"""Summarise one Agents result file: threat trajectory, per-strategy audit
results, detector score distributions and a threshold sweep.

Usage:  python3 analyze_scores.py results/<file>.json
"""
import collections
import json
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)

d = json.load(open(sys.argv[1]))["datasets"][0]
thr = d.get("detection_threshold")
print("file dataset=%s threshold=%s" % (d.get("dataset"), thr))

h = d["aap_stats"]["history"]
n = len(h)
first_crit = next((i for i, t in enumerate(h) if t > 0.6), None)
print("THREAT  n=%d mean=%.3f min=%.3f max=%.3f frac>0.3=%.2f frac>0.6=%.2f first_critical=%s"
      % (n, sum(h) / n, min(h), max(h),
         sum(t > 0.3 for t in h) / n, sum(t > 0.6 for t in h) / n, first_crit))
if n >= 10:
    k = max(1, n // 10)
    print("THREAT  mean per decile:", [round(sum(h[i:i + k]) / len(h[i:i + k]), 3) for i in range(0, n, k)][:10])

print("TP/FP/FN (per audit): %s/%s/%s" % (d["byz_tp"], d["byz_fp"], d["byz_fn"]))
print("AUDIT   base=%s window=%s forced=%s | actual attacks=%s examined=%s (%.1f%%) detected=%s (%.1f%%)"
      % (d.get("audit_base"), d.get("audit_window"), d.get("forced_audits"),
         d.get("byz_actual_attack_count"), d.get("attacks_examined"),
         100 * d.get("attack_examined_coverage", 0), d.get("attacks_detected"),
         100 * d.get("attack_detection_coverage", 0)))
print("PER-STRATEGY (audit-level):", json.dumps(d["bft_by_attack_type"]))
print("ACTIONS:", d["action_distribution"])

A = d.get("score_log_attack") or []
B = d.get("score_log_benign") or []
if not A and not B:
    sys.exit("no score log in this file")

# entries are (key, score[, replay_flag]); a repeated commitment hash
# (replay_flag=1) is flagged outright, independent of the score threshold.
def _rep(e): return e[2] if len(e) > 2 else 0
bs = collections.defaultdict(list)
for e in A:
    bs[e[0]].append((e[1], _rep(e)))
strategies = sorted(bs, key=str)
hv = sorted((e[1], _rep(e)) for e in B if e[0] == 0)
bv = sorted((e[1], _rep(e)) for e in B if e[0] == 1)
hvs = [x for x, _ in hv]; bvs = [x for x, _ in bv]
print("LOGGED  attack reports=%d  honest-agent benign=%d  byz-agent benign=%d" % (len(A), len(hv), len(bv)))

for s in strategies:
    v = sorted(x for x, _ in bs[s]); r = sum(f for _, f in bs[s])
    print("  attack %-14s n=%3d  min/med/max = %.3f / %.3f / %.3f   replay-flagged=%d" % (s, len(v), v[0], v[len(v) // 2], v[-1], r))
if hv:
    print("  honest benign        n=%d  median=%.3f p99=%.3f max=%.3f  replay-flagged=%d"
          % (len(hvs), hvs[len(hvs) // 2], hvs[int(len(hvs) * .99)], hvs[-1], sum(f for _, f in hv)))
if bv:
    print("  byz-agent benign     n=%d  median=%.3f p99=%.3f max=%.3f  replay-flagged=%d"
          % (len(bvs), bvs[len(bvs) // 2], bvs[int(len(bvs) * .99)], bvs[-1], sum(f for _, f in bv)))

print("\nthr   TPR_all  " + "  ".join("%-13s" % str(s) for s in strategies) + "  FPR_honest  FPR_byz_benign")
for t in (0.10, 0.15, 0.20, 0.23, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
    tot = sum(len(v) for v in bs.values())
    tpr = sum((x >= t) or f for v in bs.values() for x, f in v) / max(tot, 1)
    per = ["%-13.2f" % (sum((x >= t) or f for x, f in bs[s]) / len(bs[s])) for s in strategies]
    fpr = sum((x >= t) or f for x, f in hv) / max(len(hv), 1)
    fpb = sum((x >= t) or f for x, f in bv) / max(len(bv), 1)
    print("%.2f  %.2f     %s  %.4f      %.4f" % (t, tpr, "  ".join(per), fpr, fpb))