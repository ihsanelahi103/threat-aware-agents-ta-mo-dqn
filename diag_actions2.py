from collections import Counter, defaultdict
import environment.edge_mining_env as e
C = defaultdict(Counter)
orig = e.EdgeMiningEnvironment.step
def step(self, actions):
    b = (self.current_episode - 1) // 50
    for aid, a in actions.items():
        key = ('H', b) if aid < self.num_honest else ('S', b)
        C[key][int(a)] += 1
    return orig(self, actions)
e.EdgeMiningEnvironment.step = step
from run import run_quick_test
run_quick_test(scale='medium', episodes=200, dataset='yelp', attack='static', seed=42)
for k in sorted(C): print('ACTIONS', k[0], 'ep', k[1]*50+1, '-', k[1]*50+50, dict(sorted(C[k].items())))
