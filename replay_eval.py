#!/usr/bin/env python3
"""
replay_eval.py -- offline comparison of report-level DETECTION RULES on logged raw reports.

Scope (state this in the paper): every method scores every logged report from the same fixed stream.
There is no feedback from detection to later reports, no DQN, no forced-audit policy, no privacy noise,
no live quarantine. This evaluates detection rules, not complete autonomous policies.

Methods
  B0    no defense                       flags nothing
  OURS  self-referenced detector v2      the REAL ReferenceBatchDetector class (byzantine_engine.py), thr 0.15,
                                         flag = score>=thr OR repeated step-bound commitment
  TROD  trusted-reference-only           score = max(0, label error vs ground truth - h0), h0 = honest median error (dev)
  PCD-L peer consensus, labels only      leave-one-out plurality label of the OTHER agents (ties -> lower label)
  PCD-LC peer consensus, labels+conf     PCD-L OR |batch-mean conf - median of OTHER agents' batch-mean conf| >= tc
  CPV   collaborative peer validation    k=5 validators from the other agents; honest validator flags if its own report
                                         differs from the sender's (label disagreement >= tl OR |mean conf diff| >= tc);
                                         Byzantine validators adversarial (flag honest senders, clear Byzantine senders);
                                         sender flagged if >= 3 of 5 flag. Reported as the EXACT expected flag probability
                                         (hypergeometric over the validator draw), so rates are expectations.
  OUB   oracle upper bound               knows the attack labels (reported separately, not ranked)
Baselines are also reported with "+H": flag |= repeated step-bound commitment (a protocol primitive of OUR design;
without it content-only baselines cannot see replay by construction).

Threshold rule (identical for every tuned method, fixed BEFORE looking at test logs):
  on the dev log choose the grid cell with the highest effective-attack detection among cells whose false-flag rate on
  benign reports (honest agents) is <= FPR_CAP (default 1%); ties -> lower false-flag rate, then larger thresholds.
  OURS keeps its frozen threshold 0.15 (not retuned).

Usage
  python3 replay_eval.py --dev dev.jsonl --test t42.jsonl t43.jsonl t44.jsonl --engine /path/to/byzantine_engine.py
"""
import argparse, json, sys, itertools, importlib.util, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
from collections import deque, defaultdict
import numpy as np

FPR_CAP = 0.01
CPV_K, CPV_MAJ = 5, 3


def load_engine(path):
    spec = importlib.util.spec_from_file_location('byzantine_engine', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def load_log(path):
    hdr, recs = None, []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get('header'): hdr = r
            else: recs.append(r)
    labels = np.array(hdr['ref_labels'])
    steps = sorted({r['step'] for r in recs}); agents = sorted({r['agent'] for r in recs})
    si = {s: i for i, s in enumerate(steps)}; ai = {a: i for i, a in enumerate(agents)}
    S, A, R = len(steps), len(agents), len(labels)
    P = -np.ones((S, A, R), int); C = np.zeros((S, A, R)); have = np.zeros((S, A), bool)
    byz = np.zeros(A, bool); att = np.zeros((S, A), bool); eff = np.zeros((S, A), bool)
    rep = np.zeros((S, A), bool); strat = np.full((S, A), '', dtype=object)
    for r in recs:
        s, a = si[r['step']], ai[r['agent']]
        P[s, a] = r['preds']; C[s, a] = r['confs']; have[s, a] = True
        byz[a] = bool(r['byz']); att[s, a] = bool(r['attacking']); eff[s, a] = bool(r['effective'])
        rep[s, a] = bool(r['replay']); strat[s, a] = r['strategy'] or ''
    return dict(labels=labels, P=P, C=C, have=have, byz=byz, att=att, eff=eff, rep=rep, strat=strat,
                steps=steps, agents=agents)


# ---------------------------------------------------------------- scorers (return score matrix S x A, hard-flag matrix)
def ours(L, eng, thr=0.15):
    d = eng.ReferenceBatchDetector(ref_size=len(L['labels']), threshold=thr, seed=0)
    S, A = L['have'].shape; sc = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            if not L['have'][s, a]: continue
            p, c = L['P'][s, a], L['C'][s, a]
            d.observe(L['agents'][a], p, c, L['labels'])           # report enters own history first (as live)
            sf = d.self_features(L['agents'][a], p, c, L['labels'])
            sc[s, a] = d.score(L['agents'][a], L['steps'][s], p, c, L['labels'], 0.0, 1.0,
                               bool(L['rep'][s, a]), False, 0.0, self_feats=sf, selfref=True)
    return sc


def err(L):  # label error rate per report
    return (L['P'] != L['labels'][None, None, :]).mean(2)


def meanconf(L):
    return L['C'].mean(2)


def trod_score(L, h0):
    return np.maximum(0.0, err(L) - h0)


def pcd_feats(L):
    """per report: label disagreement with the leave-one-out plurality of the other agents (ties -> lower label),
    and |batch-mean conf - median of the others' batch-mean conf|."""
    S, A, R = L['P'].shape; Ld = np.zeros((S, A)); Cd = np.zeros((S, A)); mc = meanconf(L)
    for s in range(S):
        idx = np.where(L['have'][s])[0]
        if len(idx) < 3: continue
        P = L['P'][s, idx]; cnt = np.zeros((len(idx), R, 10), int)
        for lab in range(10): cnt[:, :, lab] = (P == lab).sum(0)[None, :] - (P == lab)   # exclude own vote
        cons = cnt.argmax(2)                                    # argmax returns the lowest label on ties
        Ld[s, idx] = (P != cons).mean(1)
        for k, a in enumerate(idx):
            others = np.delete(mc[s, idx], k); Cd[s, a] = abs(mc[s, a] - np.median(others))
    return Ld, Cd


from math import comb
_HG = np.zeros((40, 40))            # _HG[n, m] = P(>= CPV_MAJ flagging validators among CPV_K drawn from n others, m of which would flag)
for _n in range(CPV_K, 40):
    for _m in range(0, _n + 1):
        _HG[_n, _m] = sum(comb(_m, i) * comb(_n - _m, CPV_K - i) for i in range(CPV_MAJ, CPV_K + 1)) / comb(_n, CPV_K)


def cpv_cache(L):
    """pairwise label disagreement D[s,a,j] and |mean-conf difference| DC[s,a,j] (computed once per log)."""
    S, A, R = L['P'].shape; mc = meanconf(L)
    D = np.zeros((S, A, A), np.float32); DC = np.abs(mc[:, :, None] - mc[:, None, :]).astype(np.float32)
    for s in range(S):
        P = L['P'][s]; D[s] = (P[:, None, :] != P[None, :, :]).mean(2)
    return D, DC


def cpv_flags(L, cache, tl, tc, adversarial=True):
    """EXACT probability that a report is flagged by CPV (hypergeometric over the uniform draw of k=5 validators from the
    other agents). Each potential validator j votes deterministically given the sender a: an honest validator flags if
    label disagreement >= tl or |mean conf diff| >= tc; a Byzantine validator (adversarial worst case) flags iff the
    sender is honest. A sender is flagged if >= 3 of the 5 drawn validators flag."""
    D, DC = cache; S, A = L['have'].shape
    vote = (D >= tl) | (DC >= tc)                                     # S,A(sender),A(validator)
    if adversarial:
        byzv = L['byz'][None, None, :]; byzs = L['byz'][None, :, None]
        vote = np.where(byzv, ~byzs, vote)
    valid = L['have'][:, None, :] & L['have'][:, :, None] & ~np.eye(A, dtype=bool)[None]
    m = (vote & valid).sum(2); n = valid.sum(2)
    out = _HG[np.clip(n, 0, 39), np.clip(m, 0, 39)]
    out[~L['have']] = 0.0
    return out


# ---------------------------------------------------------------- metrics
def metrics(L, flag, hard=None):
    """flag: bool S x A (already thresholded; CPV passes probabilities rounded as expectation -> use float matrix)."""
    f = flag.astype(float)
    if hard is not None: f = np.maximum(f, hard.astype(float))
    have = L['have']; byz = L['byz'][None, :] & have
    honest = (~L['byz'])[None, :] & have
    benign = have & ~L['att']
    effm = L['eff'] & have; attm = L['att'] & have
    def rate(m): return float((f * m).sum() / m.sum()) if m.sum() else float('nan')
    res = dict(n_eff=int(effm.sum()), n_att=int(attm.sum()), n_honest=int(honest.sum()), honest_flags=float((f * honest).sum()),
               eff_detect=rate(effm), all_detect=rate(attm), honest_fpr=rate(honest),
               byz_benign_fpr=rate(byz & ~L['att']), benign_fpr=rate(benign))
    strat = {}
    for sname in ('label_flip', 'conf_falsify', 'replay', 'inconsistent'):
        m = effm & (L['strat'] == sname); strat[sname] = (rate(m), int(m.sum()))
    res['by_strategy'] = strat
    # detection latency: steps from an agent's first effective attack to its first flagged effective attack
    lat = []
    for a in range(L['have'].shape[1]):
        ss = np.where(effm[:, a])[0]
        if len(ss) == 0: continue
        hit = [s for s in ss if f[s, a] >= 0.5]
        if hit: lat.append(hit[0] - ss[0])
    res['latency_median_steps'] = float(np.median(lat)) if lat else float('nan')
    return res


# ---------------------------------------------------------------- tuning (dev) and evaluation (test)
def pick(cells, key_fpr='honest_fpr'):
    """cells: list of (params, metrics). Rule documented in the module docstring."""
    ok = [(p, m) for p, m in cells if m[key_fpr] <= FPR_CAP]
    if not ok: ok = [min(cells, key=lambda x: x[1][key_fpr])]
    best = sorted(ok, key=lambda x: (-x[1]['eff_detect'], x[1][key_fpr], tuple(-v for v in x[0])))[0]
    return best[0]


def tune(dev, eng):
    G = {}
    h0 = float(np.median(err(dev)[dev['have'] & ~dev['byz'][None, :]]))
    sc = trod_score(dev, h0); cells = []
    for t in np.round(np.arange(0.01, 0.51, 0.01), 3):
        cells.append(((float(t),), metrics(dev, sc >= t, dev['rep'])))
    G['TROD'] = dict(h0=h0, thr=pick(cells)[0])
    Ld, Cd = pcd_feats(dev); cells = []
    for t in np.round(np.arange(0.05, 1.0, 0.05), 3):
        cells.append(((float(t),), metrics(dev, Ld >= t, dev['rep'])))
    G['PCD-L'] = dict(tl=pick(cells)[0])
    cells = []
    for tl in np.round(np.arange(0.05, 1.0, 0.05), 3):
        for tc in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3):
            cells.append(((float(tl), float(tc)), metrics(dev, (Ld >= tl) | (Cd >= tc), dev['rep'])))
    b = pick(cells); G['PCD-LC'] = dict(tl=b[0], tc=b[1])
    cache = cpv_cache(dev); cells = []
    for tl in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
        for tc in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2):
            fl = cpv_flags(dev, cache, tl, tc)
            cells.append(((tl, tc), metrics(dev, fl, dev['rep'])))
    b = pick(cells); G['CPV'] = dict(tl=b[0], tc=b[1])
    cells = []
    for tl in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
        for tc in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2):
            fl = cpv_flags(dev, cache, tl, tc, adversarial=False)
            cells.append(((tl, tc), metrics(dev, fl, dev['rep'])))
    b = pick(cells); G['CPV-H'] = dict(tl=b[0], tc=b[1])
    return G


def evaluate(L, eng, G, seed=0):
    out = {}
    S, A = L['have'].shape; none = np.zeros((S, A), bool)
    out['B0'] = metrics(L, none)
    out['OURS'] = metrics(L, ours(L, eng) >= 0.15, L['rep'])
    t = G['TROD']; sc = trod_score(L, t['h0'])
    out['TROD'] = metrics(L, sc >= t['thr']); out['TROD+H'] = metrics(L, sc >= t['thr'], L['rep'])
    Ld, Cd = pcd_feats(L)
    out['PCD-L'] = metrics(L, Ld >= G['PCD-L']['tl']); out['PCD-L+H'] = metrics(L, Ld >= G['PCD-L']['tl'], L['rep'])
    g = G['PCD-LC']; fl = (Ld >= g['tl']) | (Cd >= g['tc'])
    out['PCD-LC'] = metrics(L, fl); out['PCD-LC+H'] = metrics(L, fl, L['rep'])
    g = G['CPV']; fl = cpv_flags(L, cpv_cache(L), g['tl'], g['tc'])
    out['CPV'] = metrics(L, fl); out['CPV+H'] = metrics(L, fl, L['rep'])
    g = G['CPV-H']; fl = cpv_flags(L, cpv_cache(L), g['tl'], g['tc'], adversarial=False)
    out['CPV-H'] = metrics(L, fl); out['CPV-H+H'] = metrics(L, fl, L['rep'])
    out['OUB'] = metrics(L, L['att'] & L['have'])
    return out


# ---------------------------------------------------------------- operating points tuned on the TEST log (best case for baselines)
CAPS = (0.0, 0.001, 0.01, 0.05, 0.10)


def quick(L, f):
    have = L['have']; honest = (~L['byz'])[None, :] & have; effm = L['eff'] & have
    return float((f * honest).sum() / honest.sum()), float((f * effm).sum() / effm.sum())


def oracle_points(L, eng):
    """For each method: the highest effective-attack detection reachable at each honest false-flag cap when the
    threshold is chosen ON THIS LOG ITSELF (an optimistic upper bound for the baselines; no dev tuning).
    Baselines use the +H variant (repeated commitment as hard evidence) to give them every advantage."""
    rep = L['rep'].astype(float); curves = {}
    sc = trod_score(L, 0.0)
    curves['TROD+H'] = [quick(L, np.maximum((sc >= t).astype(float), rep)) for t in np.round(np.arange(0.01, 1.001, 0.01), 3)]
    Ld, Cd = pcd_feats(L)
    curves['PCD-LC+H'] = [quick(L, np.maximum(((Ld >= tl) | (Cd >= tc)).astype(float), rep))
                          for tl in np.round(np.arange(0.05, 1.001, 0.05), 3) for tc in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3)]
    cache = cpv_cache(L)
    for name, adv in (('CPV+H', True), ('CPV-H+H', False)):
        curves[name] = [quick(L, np.maximum(cpv_flags(L, cache, tl, tc, adversarial=adv), rep))
                        for tl in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5) for tc in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2)]
    so = ours(L, eng)
    curves['OURS'] = [quick(L, np.maximum((so >= t).astype(float), rep)) for t in np.round(np.arange(0.02, 0.61, 0.02), 3)]
    out = {}
    for k, cells in curves.items():
        out[k] = {}
        for cap in CAPS:
            ok = [e for fpr, e in cells if fpr <= cap]
            out[k][cap] = max(ok) if ok else float('nan')
    return out


# ---------------------------------------------------------------- honest drift stress (simulation on top of the logs)
def perturb(L, steps, sigma_c, p_flip, seed):
    """First `steps` steps of the log with simulated honest-model drift applied to EVERY report (honest and Byzantine):
    a COHERENT confidence shift N(0, sigma_c) per report (added to all 20 confidences of that report, i.e. the batch-mean
    confidence jitters with std sigma_c; models do not drift sample by sample), and probability p_flip per sample of
    replacing a label by a random wrong label.
    attack / effective / replay flags are unchanged (commitments are unaffected by content noise)."""
    rng = np.random.RandomState(seed)
    M = {k: (v[:steps] if hasattr(v, 'shape') and v.ndim >= 2 else v) for k, v in L.items()}
    M['steps'] = L['steps'][:steps]
    P = M['P'].copy(); C = M['C'].copy(); have = M['have']
    C = np.clip(C + sigma_c * rng.randn(C.shape[0], C.shape[1], 1), 0.01, 0.9999)
    if p_flip > 0:
        flip = (rng.rand(*P.shape) < p_flip) & have[:, :, None]
        P = np.where(flip, (P + rng.randint(1, 10, size=P.shape)) % 10, P)
    M['P'] = P; M['C'] = C
    return M


def stress(L, eng, G, steps=1500, seed=5):
    rows = []
    for sigma_c, p_flip in ((0.0, 0.0), (0.005, 0.0), (0.01, 0.0), (0.02, 0.0), (0.0, 0.02), (0.01, 0.02)):
        M = perturb(L, steps, sigma_c, p_flip, seed)
        rep = M['rep'].astype(float); r = {}
        r['OURS'] = quick(M, np.maximum((ours(M, eng) >= 0.15).astype(float), rep))
        t = G['TROD']; r['TROD+H'] = quick(M, np.maximum((trod_score(M, t['h0']) >= t['thr']).astype(float), rep))
        Ld, Cd = pcd_feats(M); g = G['PCD-LC']
        r['PCD-LC+H'] = quick(M, np.maximum(((Ld >= g['tl']) | (Cd >= g['tc'])).astype(float), rep))
        cache = cpv_cache(M)
        g = G['CPV']; r['CPV+H'] = quick(M, np.maximum(cpv_flags(M, cache, g['tl'], g['tc']), rep))
        g = G['CPV-H']; r['CPV-H+H'] = quick(M, np.maximum(cpv_flags(M, cache, g['tl'], g['tc'], adversarial=False), rep))
        rows.append(((sigma_c, p_flip), r))
    return rows


def fmt(x): return 'n/a' if x != x else f'{100*x:.1f}'


def table(results, names):
    keys = ['B0', 'TROD', 'TROD+H', 'PCD-L', 'PCD-L+H', 'PCD-LC', 'PCD-LC+H', 'CPV', 'CPV+H', 'CPV-H', 'CPV-H+H', 'OURS', 'OUB']
    lines = ['| method | effective-attack detection % | all-attack % | honest false flags % (expected count / honest reports) | latency (median steps) | label_flip | conf_falsify | replay | inconsistent |',
             '|---|---|---|---|---|---|---|---|---|']
    for k in keys:
        rs = [r[k] for r in results]
        def mean(f): return np.nanmean([f(r) for r in rs]) if rs else float('nan')
        sd = lambda f: np.nanstd([f(r) for r in rs])
        row = [k, f"{fmt(mean(lambda r: r['eff_detect']))} +/- {100*sd(lambda r: r['eff_detect']):.1f}",
               fmt(mean(lambda r: r['all_detect'])), f"{fmt(mean(lambda r: r['honest_fpr']))} ({np.mean([r['honest_flags'] for r in rs]):.0f} / {int(rs[0]['n_honest'])})",
               f"{np.nanmean([r['latency_median_steps'] for r in rs]):.1f}"]
        for sname in ('label_flip', 'conf_falsify', 'replay', 'inconsistent'):
            row.append(fmt(np.nanmean([r['by_strategy'][sname][0] for r in rs])))
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev', required=True); ap.add_argument('--test', nargs='+', required=True)
    ap.add_argument('--engine', required=True); ap.add_argument('--out', default='replay_results.json')
    ap.add_argument('--stress-steps', type=int, default=1500); ap.add_argument('--no-extras', action='store_true')
    a = ap.parse_args()
    eng = load_engine(a.engine)
    dev = load_log(a.dev); G = tune(dev, eng)
    print('tuned on dev (rule: max effective detection subject to honest false-flag <= %.0f%%):' % (100 * FPR_CAP))
    print(json.dumps(G, indent=1))
    allres = []
    for i, p in enumerate(a.test):
        L = load_log(p); r = evaluate(L, eng, G, seed=i); allres.append(r)
        print('\n==', p, ' effective attacks:', r['OURS']['n_eff'], ' honest reports:', r['OURS']['n_honest'])
    print('\n' + table(allres, a.test))
    extras = {}
    if not a.no_extras:
        pts = [oracle_points(load_log(p), eng) for p in a.test]
        print('\nOperating points tuned ON THE TEST LOGS (optimistic upper bound for baselines): max effective-attack detection %'
              ' at honest false-flag cap (mean over test logs); baselines use the +H variant')
        print('| method | ' + ' | '.join(f'cap {100*c:g}%' for c in CAPS) + ' |'); print('|---|' + '---|' * len(CAPS))
        for k in pts[0]:
            print(f'| {k} | ' + ' | '.join((lambda v: 'unreachable' if v != v else fmt(v))(np.nanmean([q[k][c] for q in pts])) for c in CAPS) + ' |')
        extras['oracle_points'] = pts
        rows = stress(load_log(a.test[0]), eng, G, steps=a.stress_steps)
        print(f'\nSimulated honest drift on {a.test[0]} (first {a.stress_steps} steps): (effective-attack detection % / honest false-flag %); thresholds from dev, OURS frozen 0.15')
        meths = list(rows[0][1].keys())
        print('| sigma_conf (per-report shift), p_label_flip | ' + ' | '.join(meths) + ' |'); print('|---|' + '---|' * len(meths))
        for (sc_, pf_), r in rows:
            print(f'| {sc_}, {pf_} | ' + ' | '.join(f'{fmt(r[m][1])} / {fmt(r[m][0])}' for m in meths) + ' |')
        extras['stress'] = [dict(sigma_c=a_[0], p_flip=a_[1], res=r) for a_, r in rows]
    json.dump(dict(tuned=G, per_log=allres, extras=extras), open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
