#!/usr/bin/env python3
"""Aggregate finished evaluation runs into the paper's main table.

Usage:  python3 aggregate_results.py [--thr 0.15] [--results results] [--csv results_summary.csv]

Reads results/fisco_v11_<DS>_ep100_seed<S>_thr<THR>_full_<mode>-sr_<time>.json, keeps the newest
file per (dataset, mode, seed), and prints mean +- sample std over seeds for each (dataset, mode),
with the number of seeds. Nothing is estimated: missing fields are shown as '-'.
"""
import argparse, csv, glob, json, os, re
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--thr", default="0.15")
ap.add_argument("--results", default="results")
ap.add_argument("--csv", default="results_summary.csv")
a = ap.parse_args()

pat = re.compile(r"fisco_v11_(?P<ds>.+?)_ep100_seed(?P<seed>\d+)_thr%s_full_(?P<mode>[^_]+?)-sr_" % re.escape(a.thr))

def aid(x):
    """agent id from an int or a string such as 'agent_014' / '14'."""
    return int(re.findall(r"\d+", str(x))[-1])

def thr_stats(d):
    h = d.get("aap_stats", {}).get("history") or []
    return (float(np.mean(h)), float(np.mean(np.array(h) > 0.6))) if h else (None, None)

def metrics(d):
    q = d.get("aap_stats", {}).get("quarantined") or []
    n_h = d.get("n_honest", 14)
    tm, tc = thr_stats(d)
    tp, fp, fn = d.get("byz_tp"), d.get("byz_fp"), d.get("byz_fn")
    return {
        "test_acc_%": 100 * d["test_acc"] if "test_acc" in d else None,
        "audit_precision": (tp / (tp + fp)) if tp is not None and (tp + fp) else None,
        "audit_recall": (tp / (tp + fn)) if tp is not None and (tp + fn) else None,
        "false_pos_audits": fp,
        "attacks_examined_%": 100 * d["attack_examined_coverage"] if "attack_examined_coverage" in d else None,
        "attacks_detected_all_%": 100 * d["attack_detection_coverage"] if "attack_detection_coverage" in d else None,
        "effective_detected_%": 100 * d["effective_detection_coverage"] if "effective_detection_coverage" in d else None,
        "byz_quarantined": sum(1 for x in q if aid(x) >= n_h),
        "honest_quarantined": sum(1 for x in q if aid(x) < n_h),
        "threat_mean": tm, "threat_frac_critical": tc,
        "forced_audits": d.get("forced_audits"),
        "tx_ok": d.get("tx_ok"), "tps": d.get("tps"), "e2e_ms": d.get("e2e_ms"),
        "avg_gas": d.get("avg_gas"), "runtime_h": d["runtime_s"] / 3600 if "runtime_s" in d else None,
    }

newest = {}
for f in sorted(glob.glob(os.path.join(a.results, "fisco_v11_*.json"))):
    m = pat.search(os.path.basename(f))
    if not m: continue
    try:
        d = json.load(open(f))["datasets"]
    except Exception:
        continue
    if len(d) != 1: continue
    newest[(m["ds"], m["mode"], int(m["seed"]))] = (f, d[0])   # later timestamp wins (sorted)

groups = defaultdict(list)
for (ds, mode, seed), (f, d) in newest.items():
    groups[(ds, mode)].append((seed, metrics(d)))

def fmt(vals, nd=2):
    v = [x for x in vals if x is not None]
    if not v: return "-"
    if len(v) == 1: return ("%." + str(nd) + "f") % v[0]
    return ("%." + str(nd) + "f +- %." + str(nd) + "f") % (np.mean(v), np.std(v, ddof=1))

rows = []
keys = None
for (ds, mode), lst in sorted(groups.items()):
    lst.sort()
    keys = list(lst[0][1].keys())
    row = {"dataset": ds, "mode": mode, "seeds": ",".join(str(s) for s, _ in lst), "n": len(lst)}
    for k in keys:
        row[k] = fmt([m[k] for _, m in lst], 3 if k.startswith("audit_") else 2)
    rows.append(row)

if not rows:
    raise SystemExit("no finished runs found for thr=%s in %s/" % (a.thr, a.results))
cols = ["dataset", "mode", "n", "seeds"] + keys
with open(a.csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

for r in rows:
    print("\n== %s / %s   seeds %s (n=%d)" % (r["dataset"], r["mode"], r["seeds"], r["n"]))
    for k in keys: print("   %-24s %s" % (k, r[k]))
print("\nwrote", a.csv)