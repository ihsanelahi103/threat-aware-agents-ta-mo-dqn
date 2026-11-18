"""Attribute-inference attack on uploaded records (offline).
Input: JSONL written by the patched run (UPLOAD_LOG=...), one record per honest SendData upload:
  {agent, step, true_conf, noisy_conf, acc, eps}
Attacker model: sees the uploaded records (they are on the ledger). Task: given T records from an unknown
honest agent, decide which of two candidate agents produced them (pairwise agent linking / attribute inference
on the agent's confidence level), using
  - noisy_conf : what the system uploads (Laplace noise)
  - true_conf  : what would be uploaded WITHOUT noise (no-DP reference)
  - acc        : the local-accuracy field, which the system uploads WITHOUT noise
Chance = 0.50. Also reports basic-composition privacy loss per agent.
"""
import json, sys
import numpy as np
from itertools import combinations
if len(sys.argv) < 2: sys.exit('usage: python3 privacy_attack.py upload_log.jsonl')
rec = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
agents = sorted({r['agent'] for r in rec})
F = ('true_conf', 'noisy_conf', 'acc', 'eps')
by = {a: {k: np.array([r[k] for r in rec if r['agent'] == a], float) for k in F} for a in agents}
n_per = [len(by[a]['eps']) for a in agents]
print(f'records={len(rec)} agents={len(agents)} uploads/agent min={min(n_per)} max={max(n_per)}')
eps_all = np.concatenate([by[a]['eps'] for a in agents])
print(f'per-release epsilon: mean={eps_all.mean():.3f} min={eps_all.min():.3f} max={eps_all.max():.3f}')
print('basic-composition cumulative epsilon per agent over the run: '
      f'mean={np.mean([by[a]["eps"].sum() for a in agents]):.0f}')
nz = np.concatenate([by[a]['noisy_conf'] - by[a]['true_conf'] for a in agents])
print(f'noise: std={nz.std():.2f} (true confidence is in [0,1]; between-agent spread of mean true conf = '
      f'{np.std([by[a]["true_conf"].mean() for a in agents]):.4f})')
rng = np.random.default_rng(0)
pairs = list(combinations(agents, 2))
def link_acc(field, T, trials=4000):
    ok = tot = 0
    for _ in range(trials):
        i, j = pairs[rng.integers(len(pairs))]
        if rng.random() < .5: i, j = j, i
        xi, xj = by[i][field], by[j][field]
        if len(xi) < 2 * T or len(xj) < T: continue
        p = rng.permutation(len(xi)); u = xi[p[:T]].mean(); ri = xi[p[T:2*T]].mean()
        rj = xj[rng.choice(len(xj), T, replace=False)].mean()
        di, dj = abs(u - ri), abs(u - rj)
        ok += (1.0 if di < dj else 0.5 if di == dj else 0.0); tot += 1
    return ok / tot if tot else float('nan')
Ts = [1, 5, 20, 100, 400, 1000]
print('\nPairwise agent-linking accuracy (chance 0.50) vs number of observed uploads T')
print(f'{"T":>6} {"noisy conf (uploaded)":>24} {"true conf (no noise)":>22} {"acc field (not noised)":>24}')
for T in Ts:
    if T * 2 > min(n_per): break
    print(f'{T:>6} {link_acc("noisy_conf",T):>24.3f} {link_acc("true_conf",T):>22.3f} {link_acc("acc",T):>24.3f}')
mt = np.array([by[a]['true_conf'].mean() for a in agents]); mn = np.array([by[a]['noisy_conf'].mean() for a in agents])
ma = np.array([by[a]['acc'].mean() for a in agents])
from numpy import corrcoef
print(f'\nAcross agents, correlation of run-mean uploaded value with run-mean true confidence: '
      f'noisy_conf r={corrcoef(mt,mn)[0,1]:.2f}, acc field r={corrcoef(mt,ma)[0,1]:.2f}')
