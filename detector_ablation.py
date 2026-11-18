#!/usr/bin/env python3
"""Offline detector-component ablation from the logged score features (no GPU, no re-run).
Usage: python3 -I detector_ablation.py [glob]   (default: results/*thr0.15_full_*-sr_*.json)
Attack log tuple : (strategy, score, is_replay, conf_delta, pcf, dis_self, pop_z, pop_dis, effective)
Benign log tuple : (is_byz, score, is_replay, conf_delta, pcf, dis_self, pop_z, pop_dis)
A report is flagged iff score >= thr OR is_replay (same rule as the main runs)."""
import json, glob, sys, re, collections
import numpy as np

CT, PS = 0.02, 0.10
def sc_self(r, w):
    cd, pcf, ds, rp = r['cd'], r['pcf'], r['ds'], r['rp']
    if cd is None:                      # warm-up: only hard replay evidence
        return 0.20 * rp * w['rp']
    return (0.45*w['ds']*max(0.0, ds) + 0.25*w['cf']*min(1.0, abs(cd)/CT)
            + 0.20*w['rp']*rp + 0.10*w['pcf']*min(1.0, pcf/PS))
def sc_pop(r, w):                       # v1-style population-referenced score
    return (0.45*r['pd'] + 0.25*min(1.0, r['pz']/3.0) + 0.20*r['rp'])
W0 = dict(ds=1, cf=1, rp=1, pcf=1)
VARIANTS = collections.OrderedDict([
    ('full (self-ref, thr .15)',       (sc_self, W0, 0.15)),
    ('- confidence term',              (sc_self, {**W0, 'cf': 0}, 0.15)),
    ('- label-disagreement term',      (sc_self, {**W0, 'ds': 0}, 0.15)),
    ('- consistency (pcf) term',       (sc_self, {**W0, 'pcf': 0}, 0.15)),
    ('- replay term/flag',             (sc_self, {**W0, 'rp': 0}, 0.15)),
    ('population-ref (thr .15)',       (sc_pop,  W0, 0.15)),
    ('population-ref (thr .23)',       (sc_pop,  W0, 0.23)),
])
def rows(d):
    A = [dict(st=a[0], rp=a[2], cd=a[3], pcf=a[4], ds=a[5], pz=a[6] or 0, pd=a[7] or 0, eff=a[8]) for a in d['score_log_attack']]
    B = [dict(byz=b[0], rp=b[2], cd=b[3], pcf=b[4], ds=b[5], pz=b[6] or 0, pd=b[7] or 0) for b in d['score_log_benign']]
    return A, B
def evaluate(A, B, fn, w, thr):
    def flag(r, rp_on):
        return (fn(r, w) >= thr) or (bool(r['rp']) and rp_on)
    rp_on = w['rp'] == 1
    allv = [flag(r, rp_on) for r in A]
    eff = [flag(r, rp_on) for r in A if r['eff']]
    hon = [flag(r, rp_on) for r in B if r['byz'] == 0]
    byzb = [flag(r, rp_on) for r in B if r['byz'] == 1]
    per = {}
    for s in sorted({r['st'] for r in A}):
        v = [flag(r, rp_on) for r in A if r['st'] == s]; per[s] = float(np.mean(v)) if v else float('nan')
    m = lambda v: float(np.mean(v)) if v else float('nan')
    return dict(all=m(allv), eff=m(eff), fpr_h=m(hon), fpr_b=m(byzb), **{'st_'+k: v for k, v in per.items()})

pat = sys.argv[1] if len(sys.argv) > 1 else 'results/*thr0.15_full_*-sr_*.json'
res = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(pat)):
    m = re.search(r'fisco_v11_(.+?)_ep100_seed(\d+)_thr0\.15_full_(\w+?)-sr', f)
    if not m: continue
    ds, seed, mode = m.groups()
    if mode in ('byz0',): continue
    d = json.load(open(f))['datasets'][0]
    A, B = rows(d)
    for name, (fn, w, thr) in VARIANTS.items():
        res[(ds, mode)][name].append(evaluate(A, B, fn, w, thr))
for (ds, mode), vv in sorted(res.items()):
    n = len(next(iter(vv.values())))
    print(f"\n== {ds} / {mode}  (n={n} seeds)")
    print(f"{'variant':32s} {'eff-det':>12s} {'all-det':>12s} {'FPR honest':>11s} {'FPR byz-benign':>15s}  per-strategy detection")
    for name, lst in vv.items():
        g = lambda k: np.array([x[k] for x in lst], float)
        ms = lambda k: f"{100*np.nanmean(g(k)):5.1f}±{100*np.nanstd(g(k), ddof=1) if len(lst)>1 else 0:3.1f}"
        sts = sorted({k for x in lst for k in x if k.startswith('st_')})
        st = "  ".join(f"{k[3:][:6]}={100*np.nanmean(g(k)):.0f}" for k in sts)
        print(f"{name:32s} {ms('eff'):>12s} {ms('all'):>12s} {100*np.nanmean(g('fpr_h')):10.2f}% {100*np.nanmean(g('fpr_b')):14.2f}%  {st}")
