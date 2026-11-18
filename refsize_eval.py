#!/usr/bin/env python3
"""Offline: effect of the reference-batch SIZE on our detection rule (real ReferenceBatchDetector, thr 0.15, frozen).
Uses a logged report stream; scores only a subset of the 20 reference samples (labels, preds, confs all restricted).
usage: python3 refsize_eval.py LOG.jsonl --engine byzantine_engine.py [--draws 5] [--out refsize.json]
Place next to replay_eval.py (it imports its loader and scorer). Detection rule only; no audit policy."""
import argparse, json, numpy as np
import replay_eval as R

ap = argparse.ArgumentParser()
ap.add_argument('log'); ap.add_argument('--engine', default='byzantine_engine.py')
ap.add_argument('--draws', type=int, default=5); ap.add_argument('--out', default='refsize.json')
a = ap.parse_args()
eng = R.load_engine(a.engine); L = R.load_log(a.log)
n = len(L['labels']); rng = np.random.default_rng(0); res = {}
effm = (L['eff'] & L['have']).sum()
print(f'log {a.log}: {n} reference samples, effective attack reports {effm}')
for k in (3, 5, 10, 15, n):
    if k > n: continue
    draws = 1 if k == n else a.draws
    rows = []
    for d in range(draws):
        idx = np.sort(rng.choice(n, k, replace=False)) if k < n else np.arange(n)
        L2 = dict(L); L2['labels'] = L['labels'][idx]; L2['P'] = L['P'][:, :, idx]; L2['C'] = L['C'][:, :, idx]
        sc = R.ours(L2, eng)
        flag = np.maximum((sc >= 0.15).astype(float), L['rep'].astype(float))
        fpr, det = R.quick(L2, flag); rows.append((fpr, det))
        print(f'  k={k:2d} draw {d}: honest false-flag {100*fpr:.2f}%  effective detection {100*det:.2f}%', flush=True)
    r = np.array(rows) * 100
    res[k] = dict(fpr_mean=float(r[:, 0].mean()), fpr_sd=float(r[:, 0].std()), det_mean=float(r[:, 1].mean()), det_sd=float(r[:, 1].std()), draws=draws)
print('\n| reference samples | draws | honest false flags % | effective detection % |\n|---|---|---|---|')
for k, v in res.items():
    print(f"| {k} | {v['draws']} | {v['fpr_mean']:.2f} +/- {v['fpr_sd']:.2f} | {v['det_mean']:.1f} +/- {v['det_sd']:.1f} |")
json.dump(res, open(a.out, 'w'), indent=1)
