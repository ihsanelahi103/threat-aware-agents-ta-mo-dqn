#!/usr/bin/env python3
"""claims_check.py -- per-file table of the numbers behind the abstract's claims (read-only).
Usage: python3 claims_check.py [glob]   default: results/fisco_v11_*_thr0.15_full_*-sr_*.json"""
import json, glob, re, sys, statistics as st
from collections import defaultdict
pat = sys.argv[1] if len(sys.argv) > 1 else 'results/fisco_v11_*_thr0.15_full_*-sr_*.json'
rows = []; dump = None
for f in sorted(glob.glob(pat)):
    m = re.search(r'fisco_v11_(.+?)_ep(\d+)_seed(\d+)_thr[\d.]+_(\w+?)_(\w+)-sr_(\d+_\d+)\.json', f)
    if not m: continue
    ds, ep, seed, abl, mode, ts = m.groups()
    d = json.load(open(f))['datasets'][0]
    if dump is None: dump = {k: d.get(k) for k in ('byz_detection', 'aap_stats', 'bft_by_attack_type', 'dp_note')}
    g = lambda k: d.get(k)
    q = (d.get('aap_stats') or {}).get('quarantined') or []
    nh_ = d.get('n_honest') or 0
    def _id(x):
        try: return int(str(x).split('_')[-1])
        except Exception: return -1
    nqb = sum(1 for x in q if _id(x) >= nh_)          # Byzantine ids are the last (20 - n_honest) agents
    nqh = len(q) - nqb
    nbyz = 20 - nh_
    rows.append(dict(file=f.split('/')[-1], ds=ds, ep=int(ep), seed=int(seed), mode=mode, ts=ts, acc=g('test_acc'),
                     fp=g('byz_fp'), fq=g('false_quarantine_count'), nh=g('n_honest'),
                     eff_n=g('byz_effective_attack_count'), eff_det=g('effective_detected'),
                     eff_cov=g('effective_detection_coverage'), all_cov=g('attack_detection_coverage'),
                     exam=g('attack_examined_coverage'), n_att=g('byz_actual_attack_count'), qb=nqb, qh=nqh, nbyz=nbyz))
print('| dataset | setting | seed | episodes | test acc | false audit flags (byz_fp) | effective attacks | effective detected | eff. coverage | all-attack coverage | false quarantines (honest) | Byzantine quarantined |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for r in sorted(rows, key=lambda r: (r['ds'], r['mode'], r['seed'], r['ts'])):
    f = lambda x, p=3: ('n/a' if x is None else (f'{x:.{p}f}' if isinstance(x, float) else str(x)))
    print(f"| {r['ds']} | {r['mode']} | {r['seed']} | {r['ep']} | {f(r['acc'])} | {f(r['fp'])} | {f(r['eff_n'])} | {f(r['eff_det'])} | {f(r['eff_cov'])} | {f(r['all_cov'])} | {f(r['fq'])} | {r['qb']}/{r['nbyz']} |")
print('\n== per (dataset, setting): files, seeds, min / mean effective coverage, total false audit flags ==')
grp = defaultdict(list)
for r in rows: grp[(r['ds'], r['mode'])].append(r)
for k, v in sorted(grp.items()):
    cov = [r['eff_cov'] for r in v if isinstance(r['eff_cov'], (int, float))]
    fps = [r['fp'] for r in v if isinstance(r['fp'], (int, float))]
    print(k, 'files', len(v), 'seeds', sorted({r['seed'] for r in v}),
          'eff_cov min/mean', (round(min(cov), 4), round(st.mean(cov), 4)) if cov else 'n/a', 'sum byz_fp', sum(fps) if fps else 'n/a')
print()
for k, v in sorted(grp.items()):
    bad = [(r['seed'], r['qb'], r['nbyz']) for r in v if r['nbyz'] and r['qb'] != r['nbyz']]
    print(k, 'all Byzantine agents quarantined in every run:', 'YES' if not bad else f'NO {bad}')
dups = [k for k, v in grp.items() if len({r['seed'] for r in v}) != len(v)]
print('\nDUPLICATE seed files in groups:', dups or 'none')
print('\n== raw fields to locate the "quarantined every Byzantine agent" count (first file only) ==')
for k, v in (dump or {}).items(): print(k, '=', str(v)[:700])
