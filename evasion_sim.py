#!/usr/bin/env python3
"""
evasion_sim.py -- OFFLINE white-box evasion test of the self-referenced detector (the REAL ReferenceBatchDetector).

What it is: the attacker knows the detector (weights, thr 0.15, 40-report median baseline, tolerances) and chooses a
report strategy to stay under the threshold. The attacker's 'truthful' report stream is the REAL logged stream of an
HONEST agent; the attack modifies it (flip k correct predictions to wrong labels at FIXED positions, shift mean
confidence by delta) and the modified reports are scored with the real detector, which also updates the agent's own
history with the modified reports (so the baseline can be poisoned).
What it is NOT: no live run, no audit policy, no quorum, no DQN. It measures the detection RULE only, and says
nothing about attackers that also know the audit schedule. 'Damage' = number of wrong labels out of 20 per report.

Usage: python3 evasion_sim.py --log results/report_log_MNIST_seed42_static_ep100.jsonl --engine byzantine_engine.py --out evasion_results.json
"""
import argparse, json, importlib.util
import numpy as np

THR = 0.15

def load_engine(path):
    spec = importlib.util.spec_from_file_location('byzantine_engine', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def load_streams(path):
    hdr, recs = None, []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get('header'): hdr = r
            else: recs.append(r)
    labels = np.array(hdr['ref_labels'])
    streams = {}
    for r in recs:
        if not r['byz']:
            streams.setdefault(r['agent'], {})[r['step']] = (np.array(r['preds']), np.array(r['confs'], float))
    out = {}
    for a, d in streams.items():
        steps = sorted(d); out[a] = [d[s] for s in steps]
    return labels, out

def attack(preds, confs, labels, k, delta, pos):
    p, c = preds.copy(), confs.copy()
    if k > 0:
        ok = [i for i in range(len(labels)) if p[i] == labels[i]]
        sel = [i for i in pos if i in ok][:k]
        extra = [i for i in ok if i not in sel]
        sel = (sel + extra)[:k]
        for i in sel: p[i] = (labels[i] + 1) % 10
    c = np.clip(c + delta, 0.0, 0.999)
    return p, c

def run_agent(eng, stream, labels, kfun, dfun, W, T, aid):
    """kfun(t_att)->k flips, dfun(t_att)->delta at attack step t_att (0-based). Returns list of (t_att, k, flagged)."""
    d = eng.ReferenceBatchDetector(ref_size=len(labels), threshold=THR, seed=0)
    pos = list(range(len(labels)))
    res = []
    for t in range(min(W + T, len(stream))):
        p, c = stream[t]
        if t >= W:
            ta = t - W; k = kfun(ta); dl = dfun(ta)
            if k > 0 or dl != 0.0: p, c = attack(p, c, labels, k, dl, pos)
        d.observe(aid, p, c, labels)
        sf = d.self_features(aid, p, c, labels)
        sc = d.score(aid, t, p, c, labels, 0.0, 1.0, False, False, 0.0, self_feats=sf, selfref=True)
        if t >= W: res.append((t - W, kfun(t - W), bool(sc >= THR)))
    return res

def summarize(allres, early=20):
    """flagged fraction in the first `early` attack steps and afterwards; first-flag step (median over agents)."""
    e = [f for r in allres for (ta, k, f) in r if ta < early]
    l = [f for r in allres for (ta, k, f) in r if ta >= early]
    ff = []
    for r in allres:
        idx = [ta for (ta, k, f) in r if f]
        ff.append(idx[0] if idx else None)
    never = sum(x is None for x in ff) / len(ff)
    firsts = [x for x in ff if x is not None]
    return dict(flag_early=float(np.mean(e)) if e else float('nan'), flag_late=float(np.mean(l)) if l else float('nan'),
                never_flagged=float(never), median_first_flag=(float(np.median(firsts)) if firsts else None))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--log', required=True); ap.add_argument('--engine', required=True)
    ap.add_argument('--out', default='evasion_results.json'); ap.add_argument('--warm', type=int, default=60)
    ap.add_argument('--T', type=int, default=300)
    a = ap.parse_args()
    eng = load_engine(a.engine); labels, streams = load_streams(a.log)
    agents = sorted(streams); print(f'honest streams: {len(agents)} agents, {min(len(s) for s in streams.values())} reports min; '
                                    f'warm-up {a.warm}, attack length {a.T}')
    out = {'fixed': {}, 'ramp': {}, 'conf_only': {}}
    print('\n[E1] FIXED budget, attack every step from warm-up (k wrong labels of 20, confidence shift delta)')
    print('  k  delta  flagged(first 20 attack steps)  flagged(later)  never flagged (share of agents)  median first-flag step')
    for k in [0, 1, 2, 3, 4, 6, 8]:
        for dl in [0.0, 0.005, 0.01, 0.02]:
            res = [run_agent(eng, streams[x], labels, lambda t, k=k: k, lambda t, dl=dl: dl, a.warm, a.T, x) for x in agents]
            s = summarize(res); out['fixed'][f'k{k}_d{dl}'] = s
            print(f'  {k}  {dl:5.3f}  {100*s["flag_early"]:6.1f}%  {100*s["flag_late"]:6.1f}%  {100*s["never_flagged"]:6.1f}%  {s["median_first_flag"]}')
    print('\n[E2] RAMP: add one wrong label every `every` attack steps up to K (poisons the 40-report baseline), delta=0')
    print('  K  every  flagged(first 100 attack steps)  flagged(after)  never flagged  median first-flag step')
    for K in [3, 5, 8]:
        for every in [5, 10, 20]:
            kf = lambda t, K=K, every=every: min(K, 1 + t // every)
            res = [run_agent(eng, streams[x], labels, kf, lambda t: 0.0, a.warm, a.T, x) for x in agents]
            s = summarize(res, early=100); out['ramp'][f'K{K}_every{every}'] = s
            print(f'  {K}  {every:4d}  {100*s["flag_early"]:6.1f}%  {100*s["flag_late"]:6.1f}%  {100*s["never_flagged"]:6.1f}%  {s["median_first_flag"]}')
    print('\n[E3] SLOW CONFIDENCE DRIFT: lower confidence by `step` per attack step down to a total shift of 0.10')
    print('  step    flagged(first 100)  flagged(after)  never flagged  median first-flag step')
    for st in [0.0005, 0.001, 0.002, 0.005]:
        df = lambda t, st=st: -min(0.10, st * (t + 1))
        res = [run_agent(eng, streams[x], labels, lambda t: 0, df, a.warm, a.T, x) for x in agents]
        s = summarize(res, early=100); out['conf_only'][f'step{st}'] = s
        print(f'  {st:6.4f}  {100*s["flag_early"]:6.1f}%  {100*s["flag_late"]:6.1f}%  {100*s["never_flagged"]:6.1f}%  {s["median_first_flag"]}')
    json.dump(out, open(a.out, 'w'), indent=1); print('\nwrote', a.out)

if __name__ == '__main__':
    main()
