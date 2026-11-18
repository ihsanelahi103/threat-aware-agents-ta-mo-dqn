#!/usr/bin/env python3
"""
evasion_frozen.py -- OFFLINE test of two bounded-damage variants of the self-referenced detector, against the
same white-box evasion attacks as evasion_sim.py (E1 fixed budget, E2 ramp, E3 slow confidence drift), plus the
honest-side cost (false flags on unattacked honest streams, with and without honest drift).

Variants (all use the REAL ReferenceBatchDetector; nothing is re-implemented):
  V0  sliding   : the detector as in the paper (baseline = median of the last 40 own reports).
  V1  frozen    : the agent's history stops being updated after F reports (default 40), so the baseline is the
                  median of its first F reports and cannot be poisoned by later reports.
                  ASSUMES the first F reports of every agent are honest (a clean enrolment period).
  V2  sliding+cap : V0, plus a hard cap: flag a report whose label errors on the reference batch reach L_abs.
                  L_abs = (largest label-error count seen on any honest report of the DEV log) + 1; dev log = seed 7.
  V3  frozen+cap  : both.
What it is NOT: no live run, no audit policy, no quorum. It measures detection rules only.
The attack positions, warm-up (60) and attack length (300) are the same as evasion_sim.py.

Usage (same folder as evasion_sim.py):
  python3 evasion_frozen.py --log results/report_log_MNIST_seed42_static_ep100.jsonl \
      --dev_log <report log of seed 7> --engine byzantine_engine.py --out evasion_frozen.json
"""
import argparse, json
import numpy as np
from evasion_sim import load_engine, load_streams, attack, THR


def run_agent_v(eng, stream, labels, kfun, dfun, W, T, aid, freeze, cap, honest_fn=None, rng=None, start=None):
    """Like evasion_sim.run_agent, with optional history freezing, label cap and honest-side perturbation.
    honest_fn(t_after_warmup, rng, conf)->new conf, applied to ALL reports after warm-up (honest drift / noise).
    Returns [(t_att, k, flagged)] for t >= W."""
    d = eng.ReferenceBatchDetector(ref_size=len(labels), threshold=THR, seed=0)
    pos = list(range(len(labels)))
    res = []
    n_obs = 0
    last = min(W + T, len(stream))
    for t in range(last):
        p, c = stream[t]
        if t >= W:
            ta = t - W
            k = kfun(ta); dl = dfun(ta)
            if k > 0 or dl != 0.0:
                p, c = attack(p, c, labels, k, dl, pos)
            if honest_fn is not None:
                c = np.clip(honest_fn(ta, rng, c), 0.0, 0.999)
        if freeze is None or n_obs < freeze:
            d.observe(aid, p, c, labels); n_obs += 1
        sf = d.self_features(aid, p, c, labels)
        sc = d.score(aid, t, p, c, labels, 0.0, 1.0, False, False, 0.0, self_feats=sf, selfref=True)
        flagged = bool(sc >= THR)
        if cap is not None and int(np.sum(p != labels)) >= cap:
            flagged = True
        if t >= W:
            res.append((t - W, kfun(t - W), flagged))
    return res


def summarize(allres, early):
    e = [f for r in allres for (ta, k, f) in r if ta < early]
    l = [f for r in allres for (ta, k, f) in r if ta >= early]
    never = float(np.mean([not any(f for (_, _, f) in r) for r in allres]))
    return dict(early=float(np.mean(e)) if e else float('nan'), late=float(np.mean(l)) if l else float('nan'),
                never=never, all=float(np.mean(e + l)) if (e or l) else float('nan'))


def error_counts(streams, labels):
    return np.array([int(np.sum(p != labels)) for s in streams.values() for (p, c) in s])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True); ap.add_argument('--dev_log', required=True)
    ap.add_argument('--engine', required=True); ap.add_argument('--out', default='evasion_frozen.json')
    ap.add_argument('--warm', type=int, default=60); ap.add_argument('--T', type=int, default=300)
    ap.add_argument('--only_honest', action='store_true'); ap.add_argument('--freeze', type=int, default=40); ap.add_argument('--honestT', type=int, default=1500)
    a = ap.parse_args()
    eng = load_engine(a.engine)
    labels, streams = load_streams(a.log); dlabels, dstreams = load_streams(a.dev_log)
    agents = sorted(streams)
    ec_dev = error_counts(dstreams, dlabels); ec_test = error_counts(streams, labels)
    cap = int(ec_dev.max()) + 1
    print(f'honest label errors on the 20-sample batch: DEV(seed 7) max {ec_dev.max()}, mean {ec_dev.mean():.2f}; '
          f'TEST(seed 42) max {ec_test.max()}, mean {ec_test.mean():.2f}')
    print('error-count histogram (TEST honest reports):', {int(v): int((ec_test == v).sum()) for v in np.unique(ec_test)})
    print(f'cap L_abs = {cap} (flag when label errors >= {cap}); freeze after {a.freeze} reports; warm-up {a.warm}, '
          f'attack length {a.T}, honest-side length {a.honestT}')
    variants = {'V0 sliding': (None, None), 'V1 frozen': (a.freeze, None),
                'V2 sliding+cap': (None, cap), 'V3 frozen+cap': (a.freeze, cap)}
    out = {'cap': cap, 'freeze': a.freeze, 'scenarios': {}}
    rng = np.random.default_rng(0)

    def scenario(name, kf, df, early, T, honest_fn=None):
        out['scenarios'][name] = {}
        row = f'{name:34s}'
        for vn, (fz, cp) in variants.items():
            res = [run_agent_v(eng, streams[x], labels, kf, df, a.warm, T, x, fz, cp, honest_fn, rng) for x in agents]
            s = summarize(res, early); out['scenarios'][name][vn] = s
            row += f' | {vn}: {100*s["early"]:5.1f} {100*s["late"]:5.1f} {100*s["never"]:5.1f}'
        print(row)

    hdr = '(each cell: early% late% never%)   early = first 20 attack steps (E1) or first 100 (E2, E3, honest)'
    print('\n[H] HONEST agents, no attack (cell: flag rate% = early/late split is by step, never% = agents never flagged)')
    print(hdr)
    zero = lambda t: 0
    zerod = lambda t: 0.0
    scenario('honest, no perturbation', zero, zerod, 100, a.honestT)
    for sg in (0.005, 0.01, 0.02):
        scenario(f'honest iid conf noise sd {sg}', zero, zerod, 100, a.honestT,
                 lambda t, r, c, sg=sg: c + r.normal(0, sg))
    for st in (0.0001, 0.0005):
        scenario(f'honest slow drift {st}/report (max 0.10)', zero, zerod, 100, a.honestT,
                 lambda t, r, c, st=st: c - min(0.10, st * (t + 1)))
    if a.only_honest:
        json.dump(out, open(a.out, 'w'), indent=1); print('\nwrote', a.out); return
    print('\n[E1] fixed budget, delta=0')
    for k in (1, 2, 3, 4, 8):
        scenario(f'E1 k={k} delta=0', lambda t, k=k: k, zerod, 20, a.T)
    print('\n[E2] ramp (one more wrong label every `every` steps up to K), delta=0')
    for K in (3, 5, 8):
        for every in (5, 10, 20):
            scenario(f'E2 K={K} every={every}', lambda t, K=K, e=every: min(K, 1 + t // e), zerod, 100, a.T)
    print('\n[E3] slow confidence drift (total shift capped at 0.10)')
    for st in (0.0005, 0.001, 0.002, 0.005):
        scenario(f'E3 step={st}', zero, lambda t, st=st: -min(0.10, st * (t + 1)), 100, a.T)
    json.dump(out, open(a.out, 'w'), indent=1); print('\nwrote', a.out)


if __name__ == '__main__':
    main()
