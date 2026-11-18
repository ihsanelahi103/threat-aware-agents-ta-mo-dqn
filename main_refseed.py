"""
main_fisco_real_v11.py
--------------------------------------------------------------
-track revision of main_fisco_real.py (v10).

WHAT CHANGED FROM v10, AND WHY
================================
1. Byzantine agents now have real trained models and actually run
   inference, instead of being empty placeholders skipped in the loop.
   They corrupt their output via byzantine_engine.ByzantineAgent using
   one of four realistic strategies (label flip, confidence
   falsification, replay, inconsistency) — not silence.

2. VerifyData / DetectByzantine now compute a real statistical
   suspicion score (byzantine_engine.ReferenceBatchDetector) using a
   small shared reference batch with known ground-truth labels, rather
   than a fixed `random.random() < 0.85` coin flip. This makes
   precision/recall/F1 an actual measured result instead of an
   artifact guaranteed by construction.

3. Byzantine agents call chain.uploadData() when attacking, so their
   falsified commitments are genuinely on-chain and inspectable —
   consistent with the paper's threat model ("Byzantine agents inject
   falsified inference results through uploadData()").

4. Four previously-hardcoded state-vector dimensions are now computed
   from real measurements instead of fixed constants:
     s3  comm_ratio   : real bandwidth usage vs a nominal per-step budget
     s5  dp_base       : this agent's actual initial epsilon (was a fixed 0.5)
     s7  mem_usage     : real psutil memory reading (was a fixed 0.75)
     s9  peer_trust    : running trust score updated by real verification
                         outcomes (was a fixed 0.7)

5. Ground-truth bookkeeping for BFT precision/recall/F1 is now derived
   from (a) whether the reported agent is actually Byzantine and (b)
   whether it was actually attacking on the specific query being
   checked — not from a global counter capped at n_attacks.

WHAT DIDN'T CHANGE
===================
EdgeCNN architecture, training procedure, dataset partitioning
mechanics, ACV hashing (still used, now actually checked), AAP
threat-level thresholds, DP mechanism, FISCORealChain / smart contract
interaction, and the overall episode/step loop structure. This keeps
the revision an extension of the existing, working system rather than
a rewrite from scratch.

KNOWN SIMPLIFICATIONS (be upfront about these in the paper / limitations
section — do not silently claim more than this delivers):
  - Byzantine agents use a fixed (non-learned) attack policy: attack
    when in the sampled Omega schedule, honest otherwise. They are not
    yet adaptive/adversarial to the detector itself (a stronger
    adaptive-attacker experiment is future work, as before).
  - The reference batch is small (ref_size=20 by default) and reused
    across steps within a dataset run; a larger or per-episode-refreshed
    batch is a reasonable follow-up if compute allows.
  - "peer_trust" is a simple exponential update, not a formally modeled
    reputation system with proofs of convergence.
"""

import argparse, hashlib, json, os, random, subprocess, sys, time, tracemalloc
import numpy as np
from collections import deque

from web3 import Web3
from eth_account import Account

from byzantine_engine import ByzantineAgent, ReferenceBatchDetector, commitment_hash, AttackStrategy

try:
    import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    import torchvision, torchvision.transforms as transforms
    print(f"[SDK] PyTorch {torch.__version__} loaded")
except ImportError:
    print("[ERROR] PyTorch not installed."); sys.exit(1)

try:
    import psutil; PSUTIL_AVAILABLE=True; print("[SDK] psutil loaded")
except ImportError:
    PSUTIL_AVAILABLE=False; print("[WARN] psutil not available")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config'))
try:
    from experiment_config import (RANDOM_SEED, N_AGENTS, N_BYZANTINE, N_EPISODES,
        N_STEPS, RESULTS_DIR, INFERENCE_TIME_MS, COMM_TIME_MS, INFEDGE_GAS_PER_CALL)
except ImportError:
    RANDOM_SEED=42; N_AGENTS=20; N_BYZANTINE=6; N_EPISODES=15; N_STEPS=50
    INFERENCE_TIME_MS=156.0; COMM_TIME_MS=145.0; INFEDGE_GAS_PER_CALL=4_300_000
    RESULTS_DIR="results"

DATASETS = {
    'MNIST':         {'n_attacks': 30, 'model': 'edgecnn', 'num_classes': 10, 'channels': 1, 'img_size': 28},
    'Fashion-MNIST': {'n_attacks': 20, 'model': 'edgecnn', 'num_classes': 10, 'channels': 1, 'img_size': 28},
}

FISCO_CHAIN_ID = 20200
BYZ_SCORE_DECAY = 0.998   # per global step; half-life ~346 steps
DEFAULT_RPC    = os.environ.get('RPC_URL', 'http://172.25.0.10:8545')
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Device] {DEVICE}" + (f" — {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))

ACTIONS   = ['SendData', 'VerifyData', 'AdjustPrivacy', 'OptimizeResource', 'DetectByzantine']
N_ACTIONS = len(ACTIONS); STATE_DIM = 15

# ── Ablation control (NEW) ──────────────────────────────────
# Matches the original paper's Table IV/V ablation configurations.
# Set via --ablation; 'full' runs every component as normal.
# ── Ablation control (NEW, multi-select) ────────────────────
# Matches the original paper's Table IV/V ablation configurations, plus
# support for combinations (e.g. --ablation no_aap,no_dp) for the
# comprehensive MNIST ablation set.
ABLATION_COMPONENTS = ['no_acv', 'no_aap', 'no_dp', 'no_byz_dqn', 'no_blockchain']
ABLATION = frozenset()  # empty = 'full' (nothing disabled)
UPLOAD_LOG = os.environ.get('UPLOAD_LOG','')  # optional JSONL log of uploaded records (privacy analysis; no effect on the run)
REPORT_LOG = os.environ.get('REPORT_LOG','')  # optional JSONL log of every reference-batch report (baseline comparison; no effect on the run)
def _log_report(rec):
    if REPORT_LOG:
        with open(REPORT_LOG,'a') as _f: _f.write(json.dumps(rec)+'\n')
def _rep(step, agent, byz, preds, confs, h, replay, attacking, strategy, effective):
    return {'step':int(step),'agent':int(agent),'byz':bool(byz),'preds':[int(x) for x in preds],
            'confs':[round(float(x),6) for x in confs],'hash':(h.hex() if isinstance(h,(bytes,bytearray)) else str(h)),
            'replay':bool(replay),'attacking':bool(attacking),
            'strategy':(None if strategy is None else str(getattr(strategy,'value',strategy))),'effective':bool(effective)}

# ── Adaptive attacker + collusion controls (NEW, item 8/9) ──
ADAPTIVE_ATTACKER = False
COLLUSION = False

# ── Threat-scaled audit mechanism (NEW) ──
# Auditing is a designed protocol component, not something the DQN must
# rediscover: each honest agent, every step, audits a random peer with
# probability AUDIT_BASE * AUDIT_MULT[security level] (the DQN keeps control
# of all other steps). An audit inspects the suspect's last AUDIT_WINDOW
# committed reference-batch reports, so an attack is catchable for
# AUDIT_WINDOW steps after it happens instead of only in the step it occurs.
# AUDIT_BASE=0 disables forced audits (pure learned policy); AUDIT_WINDOW=1
# reproduces single-report inspection. Under 'no_aap' the rate is fixed at
# AUDIT_BASE (no threat scaling).
AUDIT_BASE   = 0.10
AUDIT_WINDOW = 10
AUDIT_MULT   = {'normal': 1.0, 'elevated': 2.0, 'critical': 3.5}
FLAG_SCALE   = 30.0   # accepted reports per 250-step window that saturate the attack signal

# Detector / reference-batch config (new)
REF_BATCH_SIZE   = 20
DETECTOR_MODE    = 'selfref'   # 'selfref' (agent compared with its own history) | 'popref' (v1, honest-population baseline)
CONF_TOL         = 0.02
PCF_SCALE        = 0.10
DETECT_THRESHOLD = 0.55
BANDWIDTH_BUDGET_PER_STEP_BYTES = 50_000  # nominal per-step budget for comm_ratio normalization

ABI_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EdgeCommunication_abi.json')
if os.path.exists(ABI_FILE):
    with open(ABI_FILE) as f: CONTRACT_ABI = json.load(f)
else:
    CONTRACT_ABI = [
        {"name":"registerDevice","type":"function","inputs":[{"name":"agentId","type":"string"}],"outputs":[{"name":"","type":"bool"}],"stateMutability":"nonpayable"},
        {"name":"uploadData","type":"function","inputs":[{"name":"dataHash","type":"bytes32"},{"name":"metadata","type":"string"}],"outputs":[{"name":"","type":"bytes32"}],"stateMutability":"nonpayable"},
        {"name":"validateData","type":"function","inputs":[{"name":"dataHash","type":"bytes32"},{"name":"isValid","type":"bool"}],"outputs":[{"name":"","type":"bool"}],"stateMutability":"nonpayable"},
        {"name":"reportByzantine","type":"function","inputs":[{"name":"evidenceHash","type":"bytes32"},{"name":"suspectId","type":"string"}],"outputs":[{"name":"","type":"bool"}],"stateMutability":"nonpayable"},
        {"name":"getReportCount","type":"function","inputs":[{"name":"agentId","type":"string"}],"outputs":[{"name":"","type":"uint256"}],"stateMutability":"view"},
        {"name":"totalAgents","type":"function","inputs":[],"outputs":[{"name":"","type":"uint256"}],"stateMutability":"view"},
    ]

def agent_private_key(aid, seed=RANDOM_SEED):
    return "0x" + hashlib.sha256(f"Agents_agent_{seed}_{aid:04d}".encode()).hexdigest()

def model_label(cfg):
    return {'edgecnn':'EdgeCNN','resnet18':'ResNet-18','resnet50':'ResNet-50'}.get(cfg['model'], cfg['model'])

# ── EdgeCNN (unchanged) ─────────────────────────────────────
class EdgeCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2   = nn.Conv2d(16, 32, 3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.fc1     = nn.Linear(32*7*7, 128)
        self.fc2     = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32*7*7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
    def memory_bytes(self): return sum(p.numel()*p.element_size() for p in self.parameters())

def build_model(cfg):
    return EdgeCNN(num_classes=cfg['num_classes'])

# ── Dataset loader (unchanged) ────────────────────────────────
def load_dataset(name):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(d, exist_ok=True)
    cls = torchvision.datasets.FashionMNIST if 'Fashion' in name else torchvision.datasets.MNIST
    return (cls(root=d, train=True,  download=True, transform=transform),
            cls(root=d, train=False, download=True, transform=transform))

def partition_dataset(dataset, n, seed=RANDOM_SEED):
    """Now partitions across ALL N_AGENTS (honest + Byzantine), not just
    the honest count — Byzantine agents need real trained models too so
    their corrupted output has a plausible honest baseline to corrupt."""
    rng = np.random.RandomState(seed); idx = list(range(len(dataset))); rng.shuffle(idx)
    cs  = len(idx) // n
    return [Subset(dataset, idx[i*cs:(i+1)*cs if i<n-1 else len(idx)]) for i in range(n)]

def train_local_model(model, subset, epochs=3, lr=0.001, bs=64):
    model.to(DEVICE)
    opt    = optim.Adam(model.parameters(), lr=lr)
    crit   = nn.CrossEntropyLoss()
    loader = DataLoader(subset, batch_size=bs, shuffle=True, num_workers=0)
    for _ in range(epochs):
        model.train()
        for img, lbl in loader:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(img), lbl)
            loss.backward()
            opt.step()
    model.eval(); tc = ts = 0
    with torch.no_grad():
        for img, lbl in loader:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            tc += (model(img).argmax(1)==lbl).sum().item()
            ts += lbl.size(0)
    return tc/ts if ts > 0 else 0.0

def evaluate_model(model, test_ds, bs=128):
    model.to(DEVICE); model.eval()
    loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)
    c = t  = 0
    with torch.no_grad():
        for img, lbl in loader:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            c += (model(img).argmax(1)==lbl).sum().item()
            t += lbl.size(0)
    return c/t if t > 0 else 0.0

def run_inference(model, test_ds, n=10):
    model.to(DEVICE); model.eval()
    idx    = random.sample(range(len(test_ds)), min(n, len(test_ds)))
    loader = DataLoader(Subset(test_ds, idx), batch_size=n, shuffle=False)
    with torch.no_grad():
        for img, lbl in loader:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            out        = model(img)
            probs      = F.softmax(out, dim=1)
            conf, pred = torch.max(probs, dim=1)
            acc        = (pred==lbl).sum().item()/lbl.size(0)
            return pred.cpu().numpy(), conf.cpu().numpy(), acc, conf.mean().item()
    return np.array([]), np.array([]), 0.0, 0.0

def run_inference_on_indices(model, test_ds, idx):
    """New: run real inference on a FIXED set of indices (the shared
    reference batch), returning predictions + confidences + true labels.
    This is what makes real detection possible — verifiers can compare
    a reported prediction against a known ground truth."""
    model.to(DEVICE); model.eval()
    loader = DataLoader(Subset(test_ds, list(idx)), batch_size=len(idx), shuffle=False)
    with torch.no_grad():
        for img, lbl in loader:
            img = img.to(DEVICE)
            out   = model(img)
            probs = F.softmax(out, dim=1)
            conf, pred = torch.max(probs, dim=1)
            return pred.cpu().numpy(), conf.cpu().numpy(), lbl.numpy()
    return np.array([]), np.array([]), np.array([])

# ── Algorithm 1: ACV (unchanged) ──────────────────────────────
class AdaptiveCryptographicVerification:
    THREAT_LOW=0; THREAT_MEDIUM=1; THREAT_HIGH=2
    def __init__(self): self.count=0; self.threat=self.THREAT_LOW
    def assess_threat(self, byz_score, attacks, n_agents):
        if 'no_acv' in ABLATION:
            # w/o ACV: cryptographic strength no longer adapts to threat —
            # always use the cheapest, weakest (single SHA3-256) mode.
            self.threat = self.THREAT_LOW
            return self.threat
        r = attacks/max(n_agents,1)
        if byz_score>0.6 or r>0.3:   self.threat=self.THREAT_HIGH
        elif byz_score>0.3 or r>0.1: self.threat=self.THREAT_MEDIUM
        else:                          self.threat=self.THREAT_LOW
        return self.threat
    def generate_hash(self, aid, step, preds, conf, noise):
        self.count += 1
        pred_str = "_".join(str(int(p)) for p in preds[:10])
        base = f"{aid}_{step}_{conf:.6f}_{noise:.6f}_{pred_str}"
        if self.threat == self.THREAT_LOW:
            h = hashlib.sha3_256(base.encode()).digest()
            m = "SHA3-256"
        elif self.threat == self.THREAT_MEDIUM:
            nonce = hashlib.sha256(f"{aid}_{step}_{time.time()}".encode()).hexdigest()[:8]
            h = hashlib.sha3_256(
                    hashlib.sha3_256(f"{base}_{nonce}".encode()).digest()
                    + nonce.encode()).digest()
            m = "SHA3-256-double"
        else:
            phs = [hashlib.sha3_256(f"{aid}_{step}_{i}_{p}".encode()).digest()
                   for i, p in enumerate(preds[:10])]
            h = hashlib.sha3_256(b''.join(phs) + base.encode()).digest()
            m = "SHA3-256-merkle"
        return h, {'method': m, 'threat': self.threat, 'count': self.count}

# ── Algorithm 2: BT-MO-DQN (unchanged) ───────────────────────
class BTMODQN:
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=128, lr=0.001, gamma=0.95, tau=0.01):
        self.gamma=gamma; self.tau=tau; self.n_actions=n_actions
        self.q_net      = nn.Sequential(nn.Linear(state_dim,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,n_actions)).to(DEVICE)
        self.target_net = nn.Sequential(nn.Linear(state_dim,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,n_actions)).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay     = deque(maxlen=2000); self.batch_size=32; self.min_replay=64
        self.obj_w      = {'accuracy':0.30,'security':0.30,'privacy':0.20,'efficiency':0.20}
        self.byz_threat = 0.0; self.training_steps=0; self.losses=[]
    def select_action(self, state, epsilon):
        adj_eps = min(1.0, epsilon+self.byz_threat*0.1)
        if random.random()<adj_eps:
            if self.byz_threat>0.5 and random.random()<0.3: return ACTIONS.index('DetectByzantine')
            return random.randrange(self.n_actions)
        with torch.no_grad(): return int(self.q_net(torch.FloatTensor(state).unsqueeze(0).to(DEVICE)).argmax(1).item())
    def compute_reward(self, action, result, ag_state, is_attack, det_byz):
        ra=rs=rp=re=0.0
        if action=='SendData' and result['success']:       ra=0.2+ag_state['test_acc']*0.6
        elif action=='VerifyData' and result['success']:   ra=0.15+ag_state['test_acc']*0.2; rs=0.3
        if action=='DetectByzantine' and det_byz:          rs=1.5
        elif action=='DetectByzantine' and is_attack and not det_byz: rs=-0.3
        if action=='AdjustPrivacy':
            # FIXED: under no_dp, no real privacy budget is being
            # maintained (zero noise is ever injected), so this reward
            # must be zero regardless of the (now-frozen, see
            # AAP.adaptive_dp) epsilon bookkeeping value — matching the
            # original paper's own stated ablation design ("rpriv...
            # explicitly set to zero in this configuration since no
            # privacy budget is maintained when Laplace noise injection
            # is disabled").
            rp = 0.0 if 'no_dp' in ABLATION else (
                0.3*(1.0-ag_state['dp_eps']/3.5) if self.byz_threat>0.3 else 0.1)
        elif action=='SendData' and result['success']:     rp=-0.05
        if action=='OptimizeResource':
            if ag_state['cpu']>0.7: re=0.4
            else:                    re=-0.1
        elif result.get('gas',0)>0:   re=-0.05*min(result['gas']/50000,1.0)
        if not result['success'] and action not in ('AdjustPrivacy','OptimizeResource'): ra-=0.1
        return self.obj_w['accuracy']*ra+self.obj_w['security']*rs+self.obj_w['privacy']*rp+self.obj_w['efficiency']*re
    def store(self, s,a,r,ns,done=False): self.replay.append((s,a,r,ns,done))
    def update_byz(self, score):
        if 'no_byz_dqn' in ABLATION:
            # w/o Byz-DQN: standard DQN behavior — objective weights stay
            # uniform and exploration is not biased toward DetectByzantine
            # regardless of threat, isolating BT-MO-DQN's security-aware
            # weighting/exploration as the thing under test.
            self.byz_threat = 0.0
            return
        self.byz_threat=min(1.0,score)
        if self.byz_threat>0.5: self.obj_w={'accuracy':0.25,'security':0.40,'privacy':0.20,'efficiency':0.15}
        else:                   self.obj_w={'accuracy':0.30,'security':0.30,'privacy':0.20,'efficiency':0.20}
    def train_step(self):
        if len(self.replay)<self.min_replay: return 0.0
        batch=random.sample(self.replay,self.batch_size); s,a,r,ns,d=zip(*batch)
        st=torch.FloatTensor(np.array(s)).to(DEVICE); at=torch.LongTensor(a).to(DEVICE)
        rt=torch.FloatTensor(r).to(DEVICE); nst=torch.FloatTensor(np.array(ns)).to(DEVICE); dt=torch.FloatTensor(d).to(DEVICE)
        cq=self.q_net(st).gather(1,at.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            na=self.q_net(nst).argmax(dim=1); nq=self.target_net(nst).gather(1,na.unsqueeze(1)).squeeze(1)
            tq=rt+self.gamma*nq*(1-dt)
        loss=F.smooth_l1_loss(cq,tq); self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(),1.0); self.optimizer.step()
        for tp,op in zip(self.target_net.parameters(),self.q_net.parameters()):
            tp.data.copy_(self.tau*op.data+(1-self.tau)*tp.data)
        self.training_steps+=1; lv=loss.item(); self.losses.append(lv); return lv
    def memory_bytes(self):
        return sum(p.numel()*p.element_size() for p in self.q_net.parameters())*2+len(self.replay)*STATE_DIM*4*2

# ── Algorithm 4: AAP (unchanged) ─────────────────────────────
class AAP:
    def __init__(self, n_agents, threshold=3):
        self.threshold=threshold; self.reports={}; self.quarantined=set()
        # FIXED (audit item 6): quarantine used to fire on a raw report
        # COUNT, not distinct reporters — meaning a single agent could
        # reach the threshold by itself across a few steps, with no
        # quorum/reporter-diversity requirement at all. That's the exact
        # false-quarantine vulnerability the collusion experiment (item 9)
        # needs a real mechanism to test against. Now genuinely requires
        # >=threshold DISTINCT reporters per suspect.
        self.reporters_by_suspect = {}
        self.false_quarantine_count = 0  # set externally by ground-truth check when collusion targets an honest agent
        self.global_threat=0.0; self.security_level='normal'; self.ver_freq=0.2; self.threat_history=[]
        self.optimize_count={}; self.step_count={}
        self.flag_steps=deque()
    def record_report(self, reporter, suspect, on_chain):
        reporters = self.reporters_by_suspect.setdefault(suspect, set())
        was_already_quarantined = suspect in self.quarantined
        reporters.add(reporter)
        self.reports[suspect] = len(reporters)  # kept for backward-compatible stats
        if len(reporters) >= self.threshold and not was_already_quarantined:
            self.quarantined.add(suspect)
            return True
        return False
    def track_optimize(self, agent_id, selected_optimize, cpu):
        aid=str(agent_id)
        self.optimize_count[aid]=self.optimize_count.get(aid,0)+(1 if selected_optimize else 0)
        self.step_count[aid]=self.step_count.get(aid,0)+1
        ratio=self.optimize_count[aid]/max(self.step_count[aid],1)
        # FIXED: only a sustained resource-hogging pattern (OptimizeResource
        # chosen for >50% of an agent's actions, after a warm-up of 50
        # actions) counts as suspicious. The previous extra rule "+0.05 on
        # every OptimizeResource with cpu<0.5" fired for ordinary honest
        # behaviour, never decayed, and drove every honest agent's byz_score
        # to 1.0 within ~50 steps -> security level 'critical' in ALL runs,
        # including runs with zero Byzantine agents.
        if selected_optimize and self.step_count[aid]>=50 and ratio>0.5: return 0.15
        return 0.0
    def audit_rate(self):
        if AUDIT_BASE <= 0: return 0.0
        if 'no_aap' in ABLATION: return AUDIT_BASE
        return min(1.0, AUDIT_BASE * AUDIT_MULT[self.security_level])
    def record_flag(self, step):
        # system-observable evidence: a DetectByzantine report that was
        # accepted on-chain (no ground truth involved).
        self.flag_steps.append(step)
    def attack_signal(self, step, window=250, scale=None):
        if scale is None: scale = FLAG_SCALE
        # recent accepted reports in a sliding window, normalised to [0,1].
        # Replaces the cumulative ground-truth byz_tp, which saturated the
        # threat in any attack run and leaked oracle information.
        while self.flag_steps and self.flag_steps[0] < step-window: self.flag_steps.popleft()
        return min(1.0, len(self.flag_steps)/scale)
    def assess(self, byz_scores, attacks):
        # byz_scores: scores of ALL agents (a deployed system cannot know
        # which ones are honest); attacks: attack_signal() in [0,1].
        mx=max(byz_scores) if byz_scores else 0; av=np.mean(byz_scores) if byz_scores else 0
        self.global_threat=min(1.0,av*0.4+mx*0.4+attacks*0.2); self.threat_history.append(self.global_threat)
        if self.global_threat>0.6:   self.security_level='critical';  self.ver_freq=0.5
        elif self.global_threat>0.3: self.security_level='elevated';  self.ver_freq=0.35
        else:                         self.security_level='normal';    self.ver_freq=0.2
        return self.security_level
    def adaptive_dp(self, base):
        if ABLATION & {'no_aap', 'no_dp'}:
            # w/o AAP: privacy budget no longer adapts — this is the
            # configuration the paper uses to demonstrate the ~8x DP
            # degradation (epsilon drifts upward uncontrolled) that AAP
            # exists to prevent.
            # w/o DP (FIXED): this path was previously only guarded for
            # no_aap, so under no_dp epsilon still converged to the 0.30
            # floor via THIS method even though add_noise() injects zero
            # actual noise — a misleading bookkeeping value with no real
            # privacy meaning. Now correctly frozen under no_dp too.
            return base
        if self.security_level=='critical':  return max(0.3, base*0.92)
        elif self.security_level=='elevated': return max(0.5, base*0.95)
        return base
    def adaptive_explore(self, base):
        if 'no_aap' in ABLATION:
            return base
        if self.security_level=='critical':  return min(1.0, base+0.15)
        elif self.security_level=='elevated': return min(1.0, base+0.05)
        return base
    def get_stats(self):
        return {'quarantined':list(self.quarantined),'threat':round(self.global_threat,4),
                'level':self.security_level,'history':[round(t,4) for t in self.threat_history]}

class DPMechanism:
    def __init__(self, epsilon=0.5, delta=1e-5, sensitivity=1.0):
        self.epsilon=epsilon; self.delta=delta; self.sensitivity=sensitivity
        self.noise_total=0.0; self.queries=0
        # FIXED (audit finding): true basic composition requires summing
        # the per-release epsilon AT THE TIME each release happened
        # (Sum_t eps_t), not final_epsilon * total_queries. Since epsilon
        # tightens over a run (converges toward the 0.30 floor as threat
        # rises), the old end-point formula systematically UNDERESTIMATED
        # cumulative privacy loss — early releases happened under a
        # looser (higher) epsilon that the old formula never captured.
        # That error direction means the old number overclaimed privacy,
        # which is the wrong direction to be wrong in for a DP paper.
        self.epsilon_sum = 0.0
    def add_noise(self, value):
        if 'no_dp' in ABLATION:
            # w/o DP: Laplace noise injection disabled entirely — used to
            # isolate DP's contribution to the privacy reward term and
            # measure the resulting utility/reward change.
            self.queries+=1
            self.epsilon_sum += self.epsilon
            return value, 0.0
        scale=self.sensitivity/self.epsilon; noise=np.random.laplace(0,scale)
        self.noise_total+=abs(noise); self.queries+=1
        self.epsilon_sum += self.epsilon  # record epsilon AT release time
        return value+noise, abs(noise)
    def adapt(self, threat):
        if ABLATION & {'no_aap', 'no_dp'}:
            # no_aap: privacy budget must not respond to threat (see
            # AAP.adaptive_dp docstring — this is the second of two
            # threat-driven epsilon paths and needs disabling too).
            # no_dp: this whole mechanism is bypassed at the noise-source
            # anyway (see add_noise), but guard here too for clarity.
            return
        if threat>0.6:  self.epsilon=max(0.3, self.epsilon*0.95)
        elif threat<0.2: self.epsilon=min(2.5, self.epsilon*1.03)
    def cumulative_epsilon_basic_composition(self):
        """
        Honest cumulative privacy loss under basic composition:
        eps_total = Sum_{t=1}^{T} eps_t, using the ACTUAL epsilon in
        force at each release, not the final epsilon extrapolated across
        all releases. Reported alongside (not instead of) per-release
        epsilon. Do NOT claim a single global epsilon=0.30 without this.
        """
        return self.epsilon_sum
    def get_stats(self):
        return {'epsilon':round(self.epsilon,4),'delta':self.delta,
                'noise':round(self.noise_total,4),'queries':self.queries,
                'avg_epsilon_per_release':round(self.epsilon_sum/max(self.queries,1),4),
                'cumulative_epsilon_basic_composition':round(self.cumulative_epsilon_basic_composition(),4)}


class ResourceMonitor:
    def __init__(self):
        self.proc=psutil.Process() if PSUTIL_AVAILABLE else None; self.cpu_h=[]; self.mem_h=[]
    def measure(self):
        if self.proc:
            cpu=self.proc.cpu_percent(interval=0.01)/100.0
            mem=self.proc.memory_percent()/100.0
            mb =self.proc.memory_info().rss/(1024*1024)
        else: cpu=random.uniform(0.2,0.8); mem=0.4; mb=50
        self.cpu_h.append(cpu); self.mem_h.append(mem); return cpu,mem,mb
    def get_stats(self):
        return {'avg_cpu':round(float(np.mean(self.cpu_h)),4) if self.cpu_h else 0,
                'avg_mem':round(float(np.mean(self.mem_h)),4) if self.mem_h else 0,
                'max_cpu':round(float(np.max(self.cpu_h)),4) if self.cpu_h else 0}

# ── FISCORealChain (unchanged) ────────────────────────────────
class FISCORealChain:
    def __init__(self, rpc_url, contract_address):
        self.rpc_url=rpc_url; self.addr=contract_address; self.tx_count=0; self.tx_fail=0
        self.gas_list=[]; self.lat_list=[]; self.blocks=[]; self.gas_by_function={}
        self.lat_by_function={}; self.tx_by_agent={}; self.gas_by_agent={}
        self.all_latencies=[]; self.tx_timestamps=[]; self.bandwidth_bytes=0
        self.w3=Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout':30}))
        if not self.w3.is_connected(): raise ConnectionError(f"Cannot connect to {rpc_url}")
        print(f"[Chain] Connected — Block #{self.w3.eth.block_number}  Chain ID: {self.w3.eth.chain_id}")
        self.contract=self.w3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=CONTRACT_ABI)
        self._accts={}
        print(f"[Chain] Contract: {contract_address}")
    def get_agent_account(self, aid):
        if aid not in self._accts: self._accts[aid]=Account.from_key(agent_private_key(aid))
        return self._accts[aid]
    def _send_tx(self, aid, fn, *args):
        t0=time.time()
        try:
            acct=self.get_agent_account(aid); func=getattr(self.contract.functions,fn)
            nonce=self.w3.eth.get_transaction_count(acct.address)
            tx=func(*args).build_transaction({'from':acct.address,'gas':300_000,
                'gasPrice':self.w3.to_wei('300','gwei'),'nonce':nonce,'chainId':FISCO_CHAIN_ID})
            self.bandwidth_bytes+=len(json.dumps(tx))
            signed=self.w3.eth.account.sign_transaction(tx,acct.key)
            txh=self.w3.eth.send_raw_transaction(signed.raw_transaction)
            rcpt=self.w3.eth.wait_for_transaction_receipt(txh,timeout=30)
            gas=rcpt.get('gasUsed',0); blk=rcpt.get('blockNumber',0); ok=(rcpt.get('status',1)==1)
            lat=(time.time()-t0)*1000
            if not ok and fn=='registerDevice': return {'success':True,'gas':0,'block':blk,'latency_ms':lat}
            if ok:
                self.tx_count+=1; self.gas_list.append(gas); self.lat_list.append(lat)
                self.blocks.append(blk); self.all_latencies.append(lat); self.tx_timestamps.append(time.time())
                self.gas_by_function.setdefault(fn,[]).append(gas); self.lat_by_function.setdefault(fn,[]).append(lat)
                self.tx_by_agent[aid]=self.tx_by_agent.get(aid,0)+1; self.gas_by_agent.setdefault(aid,[]).append(gas)
            else: self.tx_fail+=1
            return {'success':ok,'gas':gas,'block':blk,'latency_ms':lat}
        except Exception as e:
            self.tx_fail+=1; lat=(time.time()-t0)*1000
            if "already" in str(e).lower(): return {'success':True,'gas':0,'block':0,'latency_ms':lat}
            return {'success':False,'gas':0,'block':0,'latency_ms':lat}
    def registerDevice(self,aid,s):   return self._send_tx(aid,'registerDevice',s)
    def uploadData(self,aid,dh,m):    return self._send_tx(aid,'uploadData',dh[:32].ljust(32,b'\x00'),m)
    def validateData(self,aid,dh,v):  return self._send_tx(aid,'validateData',dh[:32].ljust(32,b'\x00'),v)
    def reportByzantine(self,aid,eh,s): return self._send_tx(aid,'reportByzantine',eh[:32].ljust(32,b'\x00'),s)
    def get_stats(self):
        return {'total_tx':self.tx_count,'failed_tx':self.tx_fail,
                'avg_gas_per_tx':int(np.mean(self.gas_list)) if self.gas_list else 0,
                'avg_latency_ms':float(np.mean(self.lat_list)) if self.lat_list else 0,
                'current_block':self.blocks[-1] if self.blocks else 0}
    def get_enhanced_stats(self):
        tps=len(self.tx_timestamps)/(self.tx_timestamps[-1]-self.tx_timestamps[0]) if len(self.tx_timestamps)>=2 and (self.tx_timestamps[-1]-self.tx_timestamps[0])>0 else 0
        la=np.array(self.all_latencies) if self.all_latencies else np.array([0])
        fg={fn:{'avg':int(np.mean(gl)),'count':len(gl),'total':int(np.sum(gl))} for fn,gl in self.gas_by_function.items()}
        fl={fn:{'avg_ms':round(float(np.mean(ll)),1),'count':len(ll)} for fn,ll in self.lat_by_function.items()}
        return {'tps':round(tps,3),
                'latency_p50_ms':round(float(np.percentile(la,50)),1),
                'latency_p95_ms':round(float(np.percentile(la,95)),1),
                'latency_p99_ms':round(float(np.percentile(la,99)),1),
                'latency_std_ms':round(float(np.std(la)),1),
                'all_latencies':[round(float(x),1) for x in self.all_latencies],
                'gas_by_function':fg,'latency_by_function':fl,
                'total_bandwidth_kb':round(self.bandwidth_bytes/1024,2),
                'avg_bandwidth_per_tx_bytes':round(self.bandwidth_bytes/max(self.tx_count,1),1)}

def local_log_result():
    """
    Used by the no_blockchain ablation: simulates the outcome of a
    successful on-chain call WITHOUT touching the real chain — no PBFT
    round, no gas, no confirmation latency. This isolates blockchain
    accountability's cost (the paper's ~16-18x runtime reduction w/o
    blockchain) while keeping every other mechanism (ACV, AAP, DP,
    BT-MO-DQN) running exactly as in the full configuration.
    """
    return {'success': True, 'gas': 0, 'block': 0, 'latency_ms': 0.5}


def deploy_fresh_contract():
    print("\n  Deploying fresh contract...")
    r=subprocess.run(['python3','deploy_contract_python.py'],capture_output=True,text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),env={**os.environ})
    for l in r.stdout.strip().split('\n'): print(f"    {l}")
    if r.returncode!=0: print(f"  ERROR:\n{r.stderr}"); return None
    af=os.path.join(os.path.dirname(os.path.abspath(__file__)),'contract_address.txt')
    if os.path.exists(af):
        with open(af) as f: a=f.read().strip()
        print(f"  Contract: {a}\n"); return a
    return None


def run_real_experiment(chain, dataset_name, dataset_cfg, n_episodes):
    mlabel=model_label(dataset_cfg)
    print(f"\n{'─'*60}\n  Dataset: {dataset_name}  Model: {mlabel}  Episodes: {n_episodes}  Agents: {N_AGENTS}\n{'─'*60}")
    tracemalloc.start(); rm=ResourceMonitor(); n_attacks=dataset_cfg['n_attacks']
    print(f"  Loading {dataset_name}...")
    train_ds,test_ds=load_dataset(dataset_name)

    # CHANGED: partition across ALL agents (honest + Byzantine) so Byzantine
    # agents have real trained models to corrupt, not just placeholders.
    partitions=partition_dataset(train_ds,N_AGENTS)
    print(f"  {len(train_ds)} train / {len(test_ds)} test / {N_AGENTS} partitions (all agents)")

    aap=AAP(N_AGENTS); agents=[]
    print(f"  Training {N_AGENTS} EdgeCNN models (honest + Byzantine)...")
    train_epochs = 10 if 'Fashion' in dataset_name else 5

    # NEW: shared reference batch with known ground-truth labels, used by
    # the real detector. Built once per dataset run.
    detector = ReferenceBatchDetector(ref_size=REF_BATCH_SIZE, threshold=DETECT_THRESHOLD, seed=int(os.environ.get("REF_SEED", RANDOM_SEED)))
    detector.conf_tol = CONF_TOL; detector.pcf_scale = PCF_SCALE
    def _sf(aid, preds, confs):
        # self-referenced features (None while the agent has too little history or in popref mode)
        return detector.self_features(aid, preds, confs, ref_labels) if DETECTOR_MODE == 'selfref' else None
    ref_idx = detector.build_reference_batch(len(test_ds))
    ref_labels = np.array([test_ds[int(i)][1] for i in ref_idx])
    _log_report({'header':True,'ref_idx':[int(i) for i in ref_idx],'ref_labels':[int(x) for x in ref_labels]})

    # Rolling pool of honest agents' reference-batch confidence means,
    # used as the population baseline for anomaly scoring (replaces
    # the old fixed-probability detection).
    honest_ref_conf_history = deque(maxlen=200)
    # NEW: tracks the honest population's own reference-batch error rate,
    # so detection scores "excess" disagreement rather than raw disagreement
    # (fixes a false-positive blowup on lower-accuracy datasets — see
    # byzantine_engine.ReferenceBatchDetector.score docstring for why).
    honest_ref_disagreement_history = deque(maxlen=200)

    for i in range(N_AGENTS):
        is_byz=i>=N_AGENTS-N_BYZANTINE
        m=build_model(dataset_cfg); t0=time.time()
        tr_acc=train_local_model(m,partitions[i],epochs=train_epochs)
        te_acc=evaluate_model(m,test_ds); tt=time.time()-t0
        ag={'id':i,'is_byz':is_byz,'registered':False,'tx_count':0,'total_inf_ms':0.0,'cum_reward':0.0,
            'model':m,'train_acc':tr_acc,'test_acc':te_acc,
            'dp':DPMechanism(epsilon=1.0+random.uniform(-0.2,0.3)),
            'acv':AdaptiveCryptographicVerification(),
            'byz_score':0.0,
            'ref_log':deque(maxlen=AUDIT_WINDOW),   # last committed reference reports (audit window)
            # NEW real-signal fields (replace hardcoded state constants):
            'peer_trust':0.5,
            'last_ref':None,        # (preds, confs, hash, is_replay, was_attacking)
        }
        ag['dp_base']=ag['dp'].epsilon  # store once, used for real s5 (was hardcoded 0.5/3.5)
        if not is_byz:
            ag.update({'dqn':BTMODQN(),'epsilon':1.0})
        else:
            # Byzantine agents: fixed (non-learned) adversarial policy —
            # attack when scheduled, honest otherwise. No DQN needed since
            # they are not optimizing a learned objective in this design
            # (an adaptive/learned adversary is noted as future work).
            ag.update({'dqn':None,'epsilon':0,
                       'byz_engine':ByzantineAgent(agent_id=i, model=m, seed=RANDOM_SEED,
                                                    adaptive=ADAPTIVE_ATTACKER)})
        role = "BYZANTINE" if is_byz else "honest"
        print(f"    agent_{i:03d} [{role}]: train={tr_acc:.4f} test={te_acc:.4f} ({tt:.1f}s)")
        agents.append(ag)

    honest=[a for a in agents if not a['is_byz']]
    byzantine=[a for a in agents if a['is_byz']]
    print(f"  Avg honest train={float(np.mean([a['train_acc'] for a in honest])):.4f}  test={float(np.mean([a['test_acc'] for a in honest])):.4f}")
    print("  Registering agents (honest + Byzantine — Byzantine devices are")
    print("  legitimately-registered but compromised, per the threat model)...")
    reg_ok=0
    for ag in agents:
        r=chain.registerDevice(ag['id'],f"agent_{ag['id']:03d}")
        if r['success']: ag['registered']=True; reg_ok+=1
    print(f"  Registered: {reg_ok}/{len(agents)}")

    total_steps=n_episodes*N_STEPS
    # FIXED: n_attacks was a fixed absolute count (e.g. 30 for MNIST)
    # regardless of episode count. Since attack steps are sampled from
    # range(total_steps), longer runs diluted attack density toward zero
    # (e.g. 30/750=4.0% at 15 episodes vs 30/10000=0.3% at 200 episodes),
    # making genuine detection opportunities vanishingly rare at realistic
    # episode counts. We now scale n_attacks to preserve a minimum attack
    # density (ATTACK_DENSITY) while never going below the dataset's
    # original calibrated count, so short test runs keep prior behavior.
    ATTACK_DENSITY = 0.02  # 2% of steps are attack-active; tune deliberately,
                           # report this as an explicit experimental parameter
    n_attacks = max(n_attacks, int(ATTACK_DENSITY * total_steps))
    atk_steps=set(random.sample(range(total_steps),n_attacks))
    # NEW (item 9): a separate schedule for collusion / false-quarantine
    # attempts. Uses the same density so it scales with run length the
    # same way the main attack schedule does. Kept disjoint from
    # atk_steps only incidentally (random sampling) — a collusion attempt
    # doesn't need to coincide with an individual-attack step, since it's
    # a distinct threat (coordinated false reporting), not a data-quality
    # attack.
    COLLUSION_DENSITY = 0.01  # collusion attempts are rarer than individual attacks
    n_collusion = max(3, int(COLLUSION_DENSITY * total_steps)) if COLLUSION else 0
    collusion_steps = set(random.sample(range(total_steps), min(n_collusion, total_steps))) if COLLUSION else set()
    collusion_attempts = 0; false_quarantine_count = 0
    # NEW: time/attempts-to-full-compromise tracking. false_quarantine_count
    # alone saturates at len(honest) once every honest agent has been hit
    # at least once (50 collusion attempts over a small honest population
    # reliably collects all of them — see paper discussion), which makes a
    # single final count/rate uninformative across seeds. This records the
    # attempt index (1-based) at which EACH honest agent is first falsely
    # quarantined, giving a real compromise curve: fraction of the honest
    # population falsely quarantined vs. number of collusion attempts so
    # far. attempts_to_full_compromise is the attempt index at which the
    # LAST honest agent gets hit (None if the run ends before all are).
    quarantine_order = []  # list of {'attempt': int, 'gs': int, 'victim': str}
    # Ground-truth bookkeeping is now derived per-observation, not a global
    # counter capped at n_attacks.
    byz_tp=byz_fp=byz_fn=0
    # NEW (item 7): per-attack-type breakdown, so results can report
    # P/R/F1 per attack strategy (label_flip, conf_falsify, replay,
    # inconsistent, adaptive) rather than only one aggregate number.
    byz_tp_by_type = {}; byz_fn_by_type = {}
    # NEW: attack-level coverage. (agent_id, step) pairs of ACTUAL attacks that
    # an audit examined / detected-and-reported. Unlike TP/FN (counted per
    # audit, conditional on the attack being examined), these give the
    # fraction of ALL attacks the system caught.
    attacks_examined = set(); attacks_detected = set()
    # 'effective' attacks: the committed report differs from the agent's truthful output
    # (changed predictions/confidences) or re-uses a stale commitment. Attacks that leave the
    # report identical to the truthful one (e.g. the first answer to a query under the
    # 'inconsistent' strategy) change nothing and cannot be detected by any statistic.
    byz_effective_attack_count = 0; effective_examined = set(); effective_detected = set()
    forced_audits = 0
    audit_rng = random.Random(random.getrandbits(32))
    seen_entries = set(); score_log_attack = []; score_log_benign = []
    log_rng = random.Random(7)   # separate stream: logging must not perturb audit decisions
    # NEW (diagnostic): direct visibility into the adaptive attacker's
    # actual behavior, since the aggregate TP/FN alone can't distinguish
    # "attacker successfully evaded every investigation" from "attacker
    # essentially never attacked at all" — these have very different
    # implications and need to be told apart from real data, not inferred.
    byz_scheduled_count = 0   # how many (agent, scheduled-step) pairs occurred
    byz_actual_attack_count = 0  # how many of those actually manifested as an attack
    t_start=time.time()
    act_counts={a:0 for a in ACTIONS}; total_rew=0.0; dqn_losses=[]; ep_metrics=[]

    for ep in range(n_episodes):
        ep_start=time.time(); ep_gl=[]; ep_ll=[]; ep_rew=0.0; ep_bw0=chain.bandwidth_bytes
        for step in range(N_STEPS):
            gs=ep*N_STEPS+step; is_atk=(gs in atk_steps)
            # FIXED: byz_score decays (half-life ~350 steps) so false flags
            # do not accumulate forever, and the threat is computed over ALL
            # agents with a windowed, non-oracle attack signal.
            for a in agents: a['byz_score'] *= BYZ_SCORE_DECAY
            byz_sc=[a['byz_score'] for a in agents]; aap.assess(byz_sc,aap.attack_signal(gs))

            # ── Byzantine agents act FIRST each step: they run real
            # inference on the shared reference batch, then honestly or
            # dishonestly report depending on the schedule. This is what
            # verifiers will later inspect. ──
            for bg in byzantine:
                true_preds, true_confs, _ = run_inference_on_indices(bg['model'], test_ds, ref_idx)
                # NEW (item 8): decide_attack lets an adaptive attacker
                # skip a scheduled attack (or attack more cautiously) when
                # it perceives elevated threat — see byzantine_engine.py
                # ByzantineAgent.decide_attack docstring. For non-adaptive
                # agents this reduces to the original "always attack when
                # scheduled" behavior. Ground truth (was_attacking) is set
                # to whether the agent ACTUALLY attacked this step, not
                # merely whether it was scheduled to — this is what makes
                # the evasion measurable in TP/FN.
                should_attack, eff_intensity = bg['byz_engine'].decide_attack(
                    is_atk, observed_threat=aap.global_threat)
                if is_atk:
                    byz_scheduled_count += 1
                    if should_attack: byz_actual_attack_count += 1
                if should_attack:
                    preds, confs, strategy, extra = bg['byz_engine'].produce_output(
                        true_preds, true_confs, gs, query_id=f"ref_{gs % 5}",
                        intensity_override=eff_intensity)
                else:
                    preds, confs, strategy, extra = true_preds, true_confs, None, {'replay': False}
                ch = commitment_hash(bg['id'], gs, preds, float(np.mean(confs)))
                # A REPLAY attacker re-submits its previous (stale) commitment.
                # The detector sees only the repeated commitment hash —
                # it is NOT told the attacker's internal replay flag (that
                # was an oracle leak in earlier versions).
                if extra.get('replay') and bg.get('last_hash') is not None:
                    ch = bg['last_hash']
                is_replay = detector.check_replay(bg['id'], gs, ch)
                bg['last_hash'] = ch
                _changed = bool(should_attack and (np.any(np.asarray(preds) != np.asarray(true_preds))
                                or not np.allclose(confs, true_confs, atol=1e-9)))
                _effective = bool(should_attack and (_changed or is_replay))
                if _effective: byz_effective_attack_count += 1
                bg['last_ref'] = {'preds': preds, 'confs': confs, 'hash': ch,
                                   'is_replay': is_replay, 'was_attacking': should_attack,
                                   'strategy': strategy, 'effective': _effective}
                bg['ref_log'].append(dict(bg['last_ref'], step=gs))
                _log_report(_rep(gs,bg['id'],True,preds,confs,ch,is_replay,should_attack,strategy,_effective))
                detector.observe(bg['id'], preds, confs, ref_labels)
                # Byzantine agents still generate real on-chain transactions
                # when attacking, consistent with the threat model.
                if should_attack and bg['registered']:
                    meta = json.dumps({'agent': bg['id'], 'step': gs, 'byz': True})
                    if 'no_blockchain' not in ABLATION:
                        chain.uploadData(bg['id'], ch, meta)

            # NEW (item 9): collusion / false-quarantine attack. A subset
            # of Byzantine agents coordinate to submit matching false
            # reports against a genuinely HONEST victim, attempting to
            # reach the quarantine threshold through pure numbers rather
            # than legitimate evidence. This directly tests the quorum
            # fix above (audit item 6) — before that fix, a single
            # colluder repeating its own report would have been enough;
            # now it genuinely requires >= threshold DISTINCT reporters.
            if COLLUSION and gs in collusion_steps and len(byzantine) >= 3:
                collusion_attempts += 1
                victim = random.choice(honest)
                victim_label = f"agent_{victim['id']:03d}"
                colluders = random.sample(byzantine, 3)
                was_quarantined_before = victim_label in aap.quarantined
                for colluder in colluders:
                    evid = hashlib.sha3_256(
                        f"collusion_{colluder['id']}_{victim['id']}_{gs}".encode()).digest()
                    result = (local_log_result() if 'no_blockchain' in ABLATION
                              else chain.reportByzantine(colluder['id'], evid, victim_label))
                    if result['success']:
                        colluder['tx_count'] += 1
                        aap.record_report(colluder['id'], victim_label, True)
                if not was_quarantined_before and victim_label in aap.quarantined:
                    false_quarantine_count += 1
                    quarantine_order.append({'attempt': collusion_attempts, 'gs': gs,
                                              'victim': victim_label})

            for ag in honest:
                cpu,mem,mb=rm.measure()
                t_inf=time.time(); preds,confs,inf_acc,avg_conf=run_inference(ag['model'],test_ds,n=10)
                inf_ms=(time.time()-t_inf)*1000; ag['total_inf_ms']+=inf_ms
                if gs % 50 == 0:
                    ag['test_acc'] = evaluate_model(ag['model'], test_ds)
                else:
                    ag['test_acc'] = ag['test_acc']*0.95+inf_acc*0.05

                # Real reference-batch pass for this honest agent too, so
                # the detector has an honest-population baseline and this
                # agent can itself be verified (for false-positive tracking).
                ref_preds, ref_confs, _ = run_inference_on_indices(ag['model'], test_ds, ref_idx)
                honest_ref_conf_history.append(float(np.mean(ref_confs)))
                honest_ref_disagreement_history.append(float(np.mean(ref_preds != ref_labels)))
                _h = commitment_hash(ag['id'], gs, ref_preds, float(np.mean(ref_confs)))
                ag['last_ref'] = {'preds': ref_preds, 'confs': ref_confs,
                                   'hash': _h,
                                   'is_replay': detector.check_replay(ag['id'], gs, _h),
                                   'was_attacking': False, 'strategy': None, 'effective': False}
                ag['ref_log'].append(dict(ag['last_ref'], step=gs))
                _log_report(_rep(gs,ag['id'],False,ref_preds,ref_confs,_h,ag['last_ref']['is_replay'],False,None,False))
                detector.observe(ag['id'], ref_preds, ref_confs, ref_labels)

                honest_conf_mean = float(np.mean(honest_ref_conf_history)) if honest_ref_conf_history else 0.85
                honest_conf_std  = float(np.std(honest_ref_conf_history)) + 1e-6 if honest_ref_conf_history else 0.06
                honest_disagreement_mean = float(np.mean(honest_ref_disagreement_history)) if honest_ref_disagreement_history else 0.05

                # CALIBRATION DIAGNOSTIC (safe to remove once threshold is tuned):
                # prints the real EdgeCNN confidence stats so DETECT_THRESHOLD
                # can be checked against actual data, not the synthetic guess
                # used in validate_detection.py.
                if gs % 50 == 0 and ag['id'] == honest[0]['id']:
                    print(f"    [calib] step={gs} honest_conf_mean={honest_conf_mean:.4f} honest_conf_std={honest_conf_std:.4f}")
                    if ADAPTIVE_ATTACKER:
                        rate = byz_actual_attack_count/max(byz_scheduled_count,1)
                        print(f"    [adaptive] step={gs} global_threat={aap.global_threat:.4f} "
                              f"scheduled={byz_scheduled_count} actual_attacks={byz_actual_attack_count} "
                              f"(rate={rate:.3f})")

                # CHANGED state vector — s3, s5, s7, s9 now use real signal:
                comm_ratio = min(chain.bandwidth_bytes / max(BANDWIDTH_BUDGET_PER_STEP_BYTES*(gs+1),1), 1.0)  # s3, was 0.5
                dp_base_norm = ag['dp_base']/3.5                                                              # s5, was 0.5/3.5
                # s7 mem now uses the REAL psutil reading `mem` (was hardcoded 0.75)
                # s9 peer_trust now uses REAL running trust score (was hardcoded 0.7)
                state=np.array([ag['test_acc'],float(np.mean([a['train_acc'] for a in honest])),
                    inf_ms/2000, comm_ratio, ag['dp'].epsilon/3.5, dp_base_norm, cpu, mem,
                    ag['byz_score'], ag['peer_trust'],
                    avg_conf,aap.global_threat,float(chain.tx_count%20)/20,aap.ver_freq,min((gs+1)/750,1.0)],dtype=np.float32)

                ag['dqn'].update_byz(ag['byz_score'])
                aidx=ag['dqn'].select_action(state,aap.adaptive_explore(ag['epsilon']))
                # Threat-scaled audit schedule overrides the learned action
                # with probability aap.audit_rate() (see AUDIT_BASE above).
                if audit_rng.random() < aap.audit_rate():
                    aidx = ACTIONS.index('DetectByzantine'); forced_audits += 1
                action=ACTIONS[aidx]; act_counts[action]+=1
                result={'success':False,'gas':0,'latency_ms':0,'block':0}; det_b=False

                if action=='SendData' and ag['registered']:
                    # attack rate = windowed accepted-report signal in [0,1]
                    # (expressed as a count out of N_AGENTS to keep the
                    # r=attacks/n_agents thresholds in assess_threat intact)
                    ag['acv'].assess_threat(ag['byz_score'],aap.attack_signal(gs)*N_AGENTS,N_AGENTS)
                    ag['dp'].epsilon=aap.adaptive_dp(ag['dp'].epsilon)
                    noisy_conf,nm=ag['dp'].add_noise(avg_conf)
                    if UPLOAD_LOG:
                        with open(UPLOAD_LOG,'a') as _ul:
                            _ul.write(json.dumps({'agent':ag['id'],'step':gs,'true_conf':float(avg_conf),'noisy_conf':float(noisy_conf),'acc':round(float(ag['test_acc']),6),'eps':float(ag['dp'].epsilon)})+'\n')
                    dh,acv_info=ag['acv'].generate_hash(ag['id'],gs,preds,noisy_conf,nm)
                    meta=json.dumps({'agent':ag['id'],'step':gs,'acc':round(float(ag['test_acc']),6),
                        'conf':round(float(noisy_conf),4),'acv':acv_info['method'],'dp_eps':round(ag['dp'].epsilon,4)})
                    result=local_log_result() if 'no_blockchain' in ABLATION else chain.uploadData(ag['id'],dh,meta)
                    if result['success']: ag['tx_count']+=1; ep_gl.append(result['gas']); ep_ll.append(result['latency_ms'])

                elif action=='VerifyData':
                    # CHANGED: peers now include Byzantine agents (previously
                    # excluded, which made real verification impossible).
                    peers=[a for a in agents if a['id']!=ag['id'] and a['registered'] and a['last_ref'] is not None]
                    if peers:
                        peer=random.choice(peers)
                        lr = peer['last_ref']
                        score = detector.score(
                            agent_id=peer['id'], step=gs,
                            reported_preds=lr['preds'], reported_confs=lr['confs'],
                            true_labels=ref_labels,
                            honest_conf_mean=honest_conf_mean, honest_conf_std=honest_conf_std,
                            replay_flag=lr['is_replay'], inconsistency_flag=False,
                            honest_disagreement_mean=honest_disagreement_mean,
                            self_feats=_sf(peer['id'], lr['preds'], lr['confs']),
                            selfref=(DETECTOR_MODE == 'selfref'))
                        flagged = score >= DETECT_THRESHOLD
                        dh=peer['last_ref']['hash']
                        result=local_log_result() if 'no_blockchain' in ABLATION else chain.validateData(ag['id'],dh,not flagged)
                        if result['success']:
                            ag['tx_count']+=1; ep_gl.append(result['gas']); ep_ll.append(result['latency_ms'])
                            if flagged:
                                peer['byz_score']=min(1.0,peer['byz_score']+0.15)
                                peer['peer_trust']=max(0.0,peer['peer_trust']-0.08)
                            else:
                                peer['peer_trust']=min(1.0,peer['peer_trust']+0.02)

                elif action=='AdjustPrivacy': ag['dp'].adapt(aap.global_threat)

                elif action=='OptimizeResource':
                    opt_batch = 5 if cpu > 0.7 else 10
                    res_data  = f"{ag['id']}_{gs}_{cpu:.4f}_{mem:.4f}_{opt_batch}"
                    res_hash  = hashlib.sha3_256(res_data.encode()).digest()
                    res_meta  = json.dumps({'agent':ag['id'],'step':gs,'cpu':round(cpu,4),
                                           'mem':round(mem,4),'batch':opt_batch,'action':'optimize'})
                    result    = local_log_result() if 'no_blockchain' in ABLATION else chain.uploadData(ag['id'],res_hash,res_meta)
                    if result['success']:
                        ag['tx_count']+=1; ep_gl.append(result['gas']); ep_ll.append(result['latency_ms'])
                    byz_inc = aap.track_optimize(ag['id'], True, cpu)
                    if byz_inc > 0:
                        ag['byz_score'] = min(1.0, ag['byz_score'] + byz_inc)

                elif action=='DetectByzantine':
                    # CHANGED: real detection via ReferenceBatchDetector,
                    # not `random.random() < 0.85`. Suspect chosen by
                    # highest current byz_score among all OTHER agents
                    # (a realistic "investigate the most suspicious peer"
                    # policy), falling back to random if scores are tied.
                    candidates=[a for a in agents if a['id']!=ag['id'] and len(a['ref_log'])>0]
                    if candidates:
                        # FIXED (was a bug): sorting by byz_score and taking the
                        # top entry degenerates when all scores tie at 0.0 (true
                        # at the start of every run) — Python's stable sort then
                        # always returns the same low-ID agent, and since
                        # Byzantine agents have the highest IDs, a Byzantine
                        # agent could NEVER be investigated (its byz_score can
                        # only rise after being investigated — a deadlock).
                        # Weighted random sampling fixes this: uniform-ish
                        # exploration early on (all weights near-equal), shifting
                        # toward already-suspicious agents as scores diverge.
                        weights = np.array([a['byz_score'] + 0.05 for a in candidates])
                        weights = weights / weights.sum()
                        sus = candidates[np.random.choice(len(candidates), p=weights)]
                        # AUDIT WINDOW: inspect the suspect's last AUDIT_WINDOW
                        # committed reports (all scored by the same detector).
                        # Ground truth (was_attacking) is used ONLY for the
                        # TP/FP/FN and coverage bookkeeping, never for the
                        # flagging decision.
                        tp_entries = []; attack_entries = []; flagged_other_step = None
                        for lr in list(sus['ref_log']):
                            _sfeat = _sf(sus['id'], lr['preds'], lr['confs'])
                            sc = detector.score(
                                agent_id=sus['id'], step=lr['step'],
                                reported_preds=lr['preds'], reported_confs=lr['confs'],
                                true_labels=ref_labels,
                                honest_conf_mean=honest_conf_mean, honest_conf_std=honest_conf_std,
                                replay_flag=lr['is_replay'], inconsistency_flag=False,
                                honest_disagreement_mean=honest_disagreement_mean,
                                self_feats=_sfeat, selfref=(DETECTOR_MODE == 'selfref'))
                            # raw feature record so tolerances / thresholds can be
                            # re-derived offline from development runs without re-running
                            _pz = abs(float(np.mean(lr['confs'])) - honest_conf_mean) / honest_conf_std
                            _fx = ((round(_sfeat['conf_delta'], 5), round(_sfeat['pcf'], 4), round(_sfeat['dis_self'], 4))
                                   if _sfeat else (None, None, None)) + (round(_pz, 3),
                                   round(max(0.0, float(np.mean(lr['preds'] != ref_labels)) - honest_disagreement_mean), 4))
                            # A repeated commitment hash is hard, on-chain-verifiable
                            # evidence (a truthful report always carries a fresh
                            # step-bound commitment), so it is flagged outright
                            # rather than relying on the statistical score alone.
                            fl = (sc >= DETECT_THRESHOLD) or bool(lr['is_replay'])
                            # Score log (first sighting of each (agent, step)
                            # report) for offline ROC / per-strategy analysis.
                            # Benign reports are sub-sampled and capped.
                            _k = (sus['id'], lr['step'])
                            if _k not in seen_entries:
                                seen_entries.add(_k)
                                if sus['is_byz'] and lr['was_attacking']:
                                    score_log_attack.append((lr.get('strategy'), round(float(sc), 4), int(bool(lr['is_replay']))) + _fx + (int(bool(lr.get('effective'))),))
                                elif len(score_log_benign) < 20000 and log_rng.random() < 0.25:
                                    score_log_benign.append((1 if sus['is_byz'] else 0, round(float(sc), 4), int(bool(lr['is_replay']))) + _fx)
                            if sus['is_byz'] and lr['was_attacking']:
                                attack_entries.append(lr)
                                attacks_examined.add((sus['id'], lr['step']))
                                if lr.get('effective'):
                                    effective_examined.add((sus['id'], lr['step']))
                                    if fl: effective_detected.add((sus['id'], lr['step']))
                                if fl: tp_entries.append(lr)
                            elif fl and flagged_other_step is None:
                                flagged_other_step = lr['step']
                        flagged = bool(tp_entries) or flagged_other_step is not None
                        reported_ok = False

                        if flagged:
                            ev_step = tp_entries[0]['step'] if tp_entries else flagged_other_step
                            evid=hashlib.sha3_256(f"evidence_{sus['id']}_{ev_step}".encode()).digest()
                            result=local_log_result() if 'no_blockchain' in ABLATION else chain.reportByzantine(ag['id'],evid,f"agent_{sus['id']:03d}")
                            if result['success']:
                                reported_ok = True
                                ag['tx_count']+=1; ep_gl.append(result['gas']); ep_ll.append(result['latency_ms'])
                                sus['byz_score']=min(1.0,sus['byz_score']+0.35)
                                aap.record_report(ag['id'],f"agent_{sus['id']:03d}",True)
                                aap.record_flag(gs)
                                det_b=True

                        if tp_entries:
                            if reported_ok:
                                byz_tp+=1
                                for e in tp_entries: attacks_detected.add((sus['id'], e['step']))
                                attack_type = tp_entries[0].get('strategy')
                                if attack_type: byz_tp_by_type[attack_type] = byz_tp_by_type.get(attack_type, 0) + 1
                        elif attack_entries:
                            byz_fn+=1
                            attack_type = attack_entries[0].get('strategy')
                            if attack_type: byz_fn_by_type[attack_type] = byz_fn_by_type.get(attack_type, 0) + 1
                        if flagged_other_step is not None and reported_ok:
                            byz_fp+=1

                if action != 'OptimizeResource':
                    aap.track_optimize(ag['id'], False, cpu)

                rew=ag['dqn'].compute_reward(action,result,
                    {'test_acc':ag['test_acc'],'dp_eps':ag['dp'].epsilon,'cpu':cpu},is_atk,det_b)
                ag['cum_reward']+=rew; ep_rew+=rew; total_rew+=rew
                nc,nm2,_=rm.measure()
                ns=np.array([ag['test_acc'],float(np.mean([a['train_acc'] for a in honest])),
                    inf_ms/2000, comm_ratio, ag['dp'].epsilon/3.5, dp_base_norm, nc, nm2,
                    ag['byz_score'], ag['peer_trust'],
                    avg_conf,aap.global_threat,float(chain.tx_count%20)/20,aap.ver_freq,min((gs+1)/750,1.0)],dtype=np.float32)
                ag['dqn'].store(state,aidx,rew,ns); loss=ag['dqn'].train_step()
                if loss>0: dqn_losses.append(loss)
                ag['epsilon']=max(0.05,ag['epsilon']*0.990)

        et=time.time()-ep_start
        atr=float(np.mean([a['train_acc'] for a in honest]))*100
        ate=float(np.mean([a['test_acc']  for a in honest]))*100
        adp=float(np.mean([a['dp'].epsilon for a in honest]))
        dr=byz_tp/max(byz_tp+byz_fn,1)
        ebw=(chain.bandwidth_bytes-ep_bw0)/1024; rs=rm.get_stats()
        ep_metrics.append({'episode':ep+1,
            'train_accuracy':round(atr/100,6),'test_accuracy':round(ate/100,6),
            'gap':round((atr-ate)/100,6),'avg_cpu':rs['avg_cpu'],'avg_mem':rs['avg_mem'],
            'avg_dp_epsilon':round(adp,4),'byz_detection_rate':round(dr,4),
            'ep_reward':round(ep_rew,2),'cum_reward':round(total_rew,2),
            'bandwidth_kb':round(ebw,2),'security_level':aap.security_level,
            'dqn_loss':round(float(np.mean(dqn_losses[-50:])),6) if dqn_losses else 0,
            'runtime_s':round(et,1)})
        if (ep+1)%max(1,n_episodes//5)==0:
            s=chain.get_stats()
            print(f"  Ep {ep+1:3d}/{n_episodes} train={atr:.2f}% test={ate:.2f}% dp={adp:.2f} rew={ep_rew:.0f} "
                  f"tx={s['total_tx']} bw={ebw:.0f}KB sec={aap.security_level} "
                  f"BFT[TP={byz_tp} FP={byz_fp} FN={byz_fn}]")

    ft=float(np.mean([a['train_acc'] for a in honest]))
    fe=float(np.mean([a['test_acc']  for a in honest]))
    fg=ft-fe
    s=chain.get_stats(); es=chain.get_enhanced_stats()
    e2e=s['avg_latency_ms']+INFERENCE_TIME_MS+COMM_TIME_MS
    rt=time.time()-t_start
    cm,pm=tracemalloc.get_traced_memory(); tracemalloc.stop()

    bp=byz_tp/max(byz_tp+byz_fp,1); br=byz_tp/max(byz_tp+byz_fn,1)
    bf=2*bp*br/max(bp+br,1e-8) if (bp+br)>0 else 0
    # NEW (item 7): per-attack-type recall (precision isn't meaningful
    # per-type since a false positive has no "type" — the suspect wasn't
    # actually attacking). Recall by type tells us which attack strategies
    # are harder to catch.
    bft_by_attack_type = {}
    for atype in set(list(byz_tp_by_type.keys()) + list(byz_fn_by_type.keys())):
        tp_t = byz_tp_by_type.get(atype, 0); fn_t = byz_fn_by_type.get(atype, 0)
        bft_by_attack_type[atype] = {'tp': tp_t, 'fn': fn_t,
            'recall': round(tp_t/max(tp_t+fn_t,1), 4)}
    gs_pct=(1-s['avg_gas_per_tx']/INFEDGE_GAS_PER_CALL)*100
    rs=rm.get_stats(); aap_s=aap.get_stats()

    ar=[]
    for ag in honest:
        ds=ag['dp'].get_stats()
        ar.append({'agent_id':ag['id'],'train_acc':round(ag['train_acc'],6),'test_acc':round(ag['test_acc'],6),
            'dp_epsilon':ds['epsilon'],'dp_noise':ds['noise'],
            'dp_cumulative_epsilon_basic_composition':ds['cumulative_epsilon_basic_composition'],
            'tx_count':ag['tx_count'],
            'reward':round(ag['cum_reward'],2),'cnn_kb':round(ag['model'].memory_bytes()/1024,2),
            'dqn_kb':round(ag['dqn'].memory_bytes()/1024,2),
            'dqn_steps':ag['dqn'].training_steps,'acv_count':ag['acv'].count,
            'final_peer_trust':round(ag['peer_trust'],4)})

    print(f"\n  ✓ {dataset_name} complete:")
    print(f"    Train:{ft*100:.2f}%  Test:{fe*100:.2f}%  Gap:{fg*100:.2f}%")
    print(f"    E2E:{e2e:.0f}ms  BW:{es['total_bandwidth_kb']:.1f}KB  TPS:{es['tps']:.2f}")
    print(f"    BFT (REAL, not scripted):  P={bp:.3f}  R={br:.3f}  F1={bf:.3f}  TP={byz_tp} FP={byz_fp} FN={byz_fn}")
    print(f"    AUDIT: base={AUDIT_BASE} window={AUDIT_WINDOW} forced={forced_audits}  "
          f"attacks actual={byz_actual_attack_count} examined={len(attacks_examined)} "
          f"({len(attacks_examined)/max(byz_actual_attack_count,1):.1%}) detected={len(attacks_detected)} "
          f"({len(attacks_detected)/max(byz_actual_attack_count,1):.1%}) | "
          f"effective attacks={byz_effective_attack_count} examined={len(effective_examined)} "
          f"detected={len(effective_detected)} ({len(effective_detected)/max(byz_effective_attack_count,1):.1%})")
    print(f"    DP:ε={float(np.mean([a['dp'].epsilon for a in honest])):.4f}  "
          f"(cumulative under basic composition — see per-agent breakdown, do not report as a single global value)")
    print(f"    DQN:reward={total_rew:.0f}  loss={float(np.mean(dqn_losses[-50:])):.6f}  steps={sum(a['dqn'].training_steps for a in honest)}")
    print(f"    Runtime:{rt:.1f}s")

    return {'dataset':dataset_name,'model':mlabel,'train_acc':round(ft,6),'test_acc':round(fe,6),
        'gap':round(fg,6),'e2e_ms':round(e2e,1),'inference_ms':INFERENCE_TIME_MS,
        'communication_ms':COMM_TIME_MS,'blockchain_ms':round(s['avg_latency_ms'],1),
        'latency_p50_ms':es['latency_p50_ms'],'latency_p95_ms':es['latency_p95_ms'],
        'latency_p99_ms':es['latency_p99_ms'],'latency_std_ms':es['latency_std_ms'],
        'all_latencies':es['all_latencies'],'total_bandwidth_kb':es['total_bandwidth_kb'],
        'avg_bandwidth_per_tx_bytes':es['avg_bandwidth_per_tx_bytes'],
        'byz_detection':round(br,4),'byz_tp':byz_tp,'byz_fp':byz_fp,'byz_fn':byz_fn,
        'n_attacks':n_attacks,'bft_precision':round(bp,4),'bft_recall':round(br,4),'bft_f1':round(bf,4),
        'bft_by_attack_type':bft_by_attack_type,
        'collusion_attempts':collusion_attempts,'false_quarantine_count':false_quarantine_count,
        'byz_scheduled_count':byz_scheduled_count,'byz_actual_attack_count':byz_actual_attack_count,
        'byz_actual_attack_rate':round(byz_actual_attack_count/max(byz_scheduled_count,1),4),
        # NEW: audit mechanism + attack-level coverage (share of ALL actual
        # attacks examined / detected-and-reported), alongside the
        # conditional recall (bft_recall = detected / examined-but-attack).
        'audit_base':AUDIT_BASE,'audit_window':AUDIT_WINDOW,'forced_audits':forced_audits,
        'flag_scale':FLAG_SCALE,
        # detector scores per distinct audited report: attack reports as
        # (strategy, score); benign as (is_byzantine_agent, score), 25% sample.
        'score_log_attack':score_log_attack,'score_log_benign':score_log_benign,
        'attacks_examined':len(attacks_examined),'attacks_detected':len(attacks_detected),
        'byz_effective_attack_count':byz_effective_attack_count,
        'effective_examined':len(effective_examined),'effective_detected':len(effective_detected),
        'effective_detection_coverage':round(len(effective_detected)/max(byz_effective_attack_count,1),4),
        'attack_examined_coverage':round(len(attacks_examined)/max(byz_actual_attack_count,1),4),
        'attack_detection_coverage':round(len(attacks_detected)/max(byz_actual_attack_count,1),4),
        'false_quarantine_rate':round(false_quarantine_count/max(collusion_attempts,1),4) if COLLUSION else None,
        # NEW: time/attempts-to-full-compromise curve. quarantine_order is
        # the sequence of {attempt, gs, victim} events, one per DISTINCT
        # honest agent first falsely quarantined — this is what should be
        # reported/plotted instead of (or alongside) the saturating
        # false_quarantine_rate above, since that rate reliably saturates
        # once enough attempts have been made to cover the whole honest
        # population (see run discussion). n_honest is included so the
        # curve's denominator doesn't depend on reading N_AGENTS elsewhere.
        'quarantine_order':quarantine_order if COLLUSION else None,
        'n_honest':len(honest),
        'attempts_to_full_compromise': (quarantine_order[-1]['attempt']
            if COLLUSION and len(quarantine_order) == len(honest) else None),
        'dp_epsilon_avg':round(float(np.mean([a['dp'].epsilon for a in honest])),4),
        'dp_epsilon_min':round(float(np.min([a['dp'].epsilon for a in honest])),4),
        'dp_epsilon_max':round(float(np.max([a['dp'].epsilon for a in honest])),4),
        'dp_delta':1e-5,'dp_noise_avg':round(float(np.mean([a['dp'].noise_total for a in honest])),4),
        'dp_note':'per-release epsilon reported; see agent_resources[*].dp_cumulative_epsilon_basic_composition for cumulative loss',
        'total_reward':round(total_rew,2),
        'avg_reward_per_agent':round(float(np.mean([a['cum_reward'] for a in honest])),2),
        'dqn_avg_loss':round(float(np.mean(dqn_losses[-50:])),6) if dqn_losses else 0,
        'dqn_total_steps':sum(a['dqn'].training_steps for a in honest),
        'tx_ok':s['total_tx'],'tx_fail':s['failed_tx'],'avg_gas':s['avg_gas_per_tx'],'total_gas':s['avg_gas_per_tx']*s['total_tx'],
        'current_block':s['current_block'],'gas_saving_pct':round(gs_pct,1),'tps':es['tps'],
        'cpu_avg':rs['avg_cpu'],'cpu_max':rs['max_cpu'],'mem_avg':rs['avg_mem'],
        'peak_memory_mb':round(pm/1024/1024,2),'agents_registered':reg_ok,'runtime_s':round(rt,1),
        'gas_by_function':es['gas_by_function'],'latency_by_function':es['latency_by_function'],
        'agent_resources':ar,'action_distribution':act_counts,
        'episode_convergence':ep_metrics,'aap_stats':aap_s,
        'detection_threshold':DETECT_THRESHOLD,'reference_batch_size':REF_BATCH_SIZE,
        'detector':DETECTOR_MODE,'conf_tol':CONF_TOL,'pcf_scale':PCF_SCALE}


def main():
    global DETECT_THRESHOLD, ABLATION, ADAPTIVE_ATTACKER, COLLUSION, N_BYZANTINE, AUDIT_BASE, AUDIT_WINDOW, FLAG_SCALE, DETECTOR_MODE, CONF_TOL, PCF_SCALE
    parser=argparse.ArgumentParser(description='Agents v11 — revision (real Byzantine + real detection)')
    parser.add_argument('--rpc',     type=str, default=DEFAULT_RPC)
    parser.add_argument('--dataset', type=str, default=None, choices=list(DATASETS.keys()))
    parser.add_argument('--episodes',type=int, default=N_EPISODES)
    parser.add_argument('--seed',    type=int, default=RANDOM_SEED)
    parser.add_argument('--threshold', type=float, default=DETECT_THRESHOLD,
                         help='Detection score threshold (default 0.55). Tune this against '
                              'real confidence stats from [calib] log lines before committing '
                              'to a full multi-seed campaign.')
    parser.add_argument('--ablation', type=str, default='full',
                         help="Comma-separated components to disable, e.g. 'no_aap,no_dp' for "
                              "a combined ablation, or a single value like 'no_acv'. "
                              "'full' (default) disables nothing. Valid components: "
                              + ",".join(ABLATION_COMPONENTS) + ". 'all' disables everything "
                              "(maximally-degraded baseline).")
    parser.add_argument('--adaptive_attacker', action='store_true',
                         help='(item 8) Byzantine agents become evasive: they observe the AAP '
                              'global threat signal and reduce attack frequency/intensity as '
                              'perceived detection risk rises, instead of always attacking on '
                              'every scheduled step at fixed intensity.')
    parser.add_argument('--collusion', action='store_true',
                         help='(item 9) Enables periodic collusion attempts: 3 distinct '
                              'Byzantine agents coordinate false reports against a genuinely '
                              'honest victim, testing whether the quorum-based quarantine '
                              'mechanism (fixed per audit item 6) resists coordinated false '
                              'reporting. Reports collusion_attempts / false_quarantine_count.')
    parser.add_argument('--byz_ratio', type=float, default=None,
                         help='(item 10) Overrides the Byzantine agent ratio, e.g. 0.0, 0.30, '
                              '0.40. N_BYZANTINE = round(byz_ratio * N_AGENTS). If omitted, '
                              "uses the dataset config's default (30%%, i.e. 6/20).")
    parser.add_argument('--audit_base', type=float, default=AUDIT_BASE,
                         help='Base per-step probability that an honest agent performs a forced '
                              'audit (DetectByzantine). Scaled x2 at elevated and x3.5 at critical '
                              'threat; fixed under no_aap. 0 disables forced audits.')
    parser.add_argument('--audit_window', type=int, default=AUDIT_WINDOW,
                         help='Number of the suspect\'s most recent committed reports an audit '
                              'inspects (1 = single latest report).')
    parser.add_argument('--flag_scale', type=float, default=FLAG_SCALE,
                         help='Accepted DetectByzantine reports per 250-step window at which the '
                              'attack-signal term of the AAP threat saturates.')
    parser.add_argument('--detector', choices=['selfref', 'popref'], default=DETECTOR_MODE,
                         help='selfref: each report is scored against the agent\'s own robust history '
                              '(v2). popref: against the honest-population spread (v1, kept for comparison).')
    parser.add_argument('--conf_tol', type=float, default=CONF_TOL,
                         help='selfref: mean-confidence shift that saturates the confidence term.')
    parser.add_argument('--pcf_scale', type=float, default=PCF_SCALE,
                         help='selfref: fraction of changed predictions that saturates the consistency term.')
    args=parser.parse_args()
    FLAG_SCALE = args.flag_scale
    DETECTOR_MODE = args.detector; CONF_TOL = args.conf_tol; PCF_SCALE = args.pcf_scale
    if not (0.0 <= args.audit_base <= 1.0): parser.error('--audit_base must be in [0,1]')
    if args.audit_window < 1: parser.error('--audit_window must be >= 1')
    AUDIT_BASE = args.audit_base; AUDIT_WINDOW = args.audit_window
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
    DETECT_THRESHOLD = args.threshold
    ADAPTIVE_ATTACKER = args.adaptive_attacker
    COLLUSION = args.collusion
    if args.byz_ratio is not None:
        if not (0.0 <= args.byz_ratio < 1.0):
            parser.error(f"--byz_ratio must be in [0.0, 1.0), got {args.byz_ratio}")
        N_BYZANTINE = round(args.byz_ratio * N_AGENTS)
        if COLLUSION and N_BYZANTINE < 3:
            parser.error(f"--collusion requires at least 3 Byzantine agents "
                         f"(--byz_ratio {args.byz_ratio} gives only {N_BYZANTINE}).")
    if args.ablation.strip().lower() == 'full':
        ABLATION = frozenset()
    elif args.ablation.strip().lower() == 'all':
        ABLATION = frozenset(ABLATION_COMPONENTS)
    else:
        tokens = frozenset(t.strip() for t in args.ablation.split(',') if t.strip())
        invalid = tokens - set(ABLATION_COMPONENTS)
        if invalid:
            parser.error(f"Invalid ablation component(s): {sorted(invalid)}. "
                         f"Valid: {ABLATION_COMPONENTS} (or 'full' / 'all').")
        ABLATION = tokens
    ds={args.dataset:DATASETS[args.dataset]} if args.dataset else DATASETS
    print("="*65)
    print("  Agents v11 — Revision")
    print("  Byzantine agents: REAL models + REAL falsification strategies")
    print("  Detection: REAL statistical test vs reference batch (not scripted)")
    print(f"  Agents: {N_AGENTS} ({N_AGENTS-N_BYZANTINE} honest + {N_BYZANTINE} Byzantine)")
    print(f"  RPC: {args.rpc}  Device: {DEVICE}")
    print(f"  Datasets: {list(ds.keys())}  Episodes: {args.episodes}  Seed: {args.seed}")
    ablation_label = 'full' if not ABLATION else '+'.join(sorted(ABLATION))
    # FIXED: attacker mode / Byzantine ratio were not recorded anywhere in the
    # filename or JSON, so static, adaptive, collusion and ratio-sweep runs
    # would all have been labeled plain 'full'. Encode them explicitly.
    mode_tags = []
    if ADAPTIVE_ATTACKER: mode_tags.append('adaptive')
    if COLLUSION: mode_tags.append('collusion')
    if args.byz_ratio is not None: mode_tags.append(f'byz{int(round(args.byz_ratio*100))}')
    if not mode_tags: mode_tags.append('static')
    if DETECTOR_MODE == 'selfref': mode_tags.append('sr')
    if AUDIT_BASE != 0.10 or AUDIT_WINDOW != 10:
        mode_tags.append(f'aud{int(round(AUDIT_BASE*100))}w{AUDIT_WINDOW}')
    if FLAG_SCALE != 30.0:
        mode_tags.append(f'fs{int(round(FLAG_SCALE))}')
    ablation_label = ablation_label + '_' + '-'.join(mode_tags)
    print(f"  Ablation: {ablation_label}")
    print("="*65)
    results=[]; dl=list(ds.items())
    for i,(dn,dc) in enumerate(dl):
        print(f"\n{'='*65}\n  [{i+1}/{len(dl)}] {dn}\n{'='*65}")
        ca=deploy_fresh_contract()
        if not ca: continue
        try: ch=FISCORealChain(rpc_url=args.rpc,contract_address=ca)
        except Exception as e: print(f"  ERROR: {e}"); continue
        r=run_real_experiment(ch,dn,dc,args.episodes); r['contract']=ca; results.append(r)
    os.makedirs(RESULTS_DIR,exist_ok=True)
    # FIXED: filename now encodes dataset(s), episode count, threshold, and a
    # timestamp — previously every run saved to the same fisco_real_results_
    # seed{N}.json regardless of episodes/threshold/dataset, so a later run
    # (even a FAILED one, e.g. the empty-containers deployment error) would
    # silently overwrite a previous good result. This also lets multiple
    # seeds/configs coexist on disk for later aggregation into mean±std.
    ds_tag = "_".join(ds.keys()) if len(ds) <= 2 else f"{len(ds)}datasets"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out=os.path.join(RESULTS_DIR,
        f'fisco_v11_{ds_tag}_ep{args.episodes}_seed{args.seed}_thr{args.threshold}_{ablation_label}_{timestamp}.json')
    payload = {'mode':'Agents v11 (real BFT)','seed':args.seed,
        'episodes':args.episodes,'threshold':args.threshold,'ablation':ablation_label,'adaptive_attacker':ADAPTIVE_ATTACKER,'collusion':COLLUSION,
        'n_byzantine':N_BYZANTINE,'n_agents':N_AGENTS,'datasets':results}
    try:
        with open(out,'w') as f: json.dump(payload,f,indent=2)
    except OSError as e:
        # CRITICAL SAFETY NET: a completed run (potentially many hours of
        # compute) must never be silently lost just because the disk was
        # full at the very last save step. If the primary write fails,
        # dump the SUMMARY (not the full per-step latency arrays, to keep
        # this small) to stdout so it can be recovered from the terminal
        # log/scrollback, and try a couple of fallback locations.
        print(f"\n  !!! DISK WRITE FAILED ({e}) — RESULTS FILE NOT SAVED !!!")
        print("  Attempting fallback save locations...")
        summary_only = {'mode':payload['mode'],'seed':payload['seed'],'episodes':payload['episodes'],
            'threshold':payload['threshold'],
            'datasets':[{k:v for k,v in d.items() if k not in ('all_latencies','episode_convergence')}
                        for d in results]}
        saved = False
        for fallback in ['/tmp/Agents_emergency_results.json', os.path.expanduser('~/Agents_emergency_results.json')]:
            try:
                with open(fallback,'w') as f: json.dump(summary_only,f,indent=2)
                print(f"  Fallback summary saved to: {fallback}")
                saved = True
                break
            except OSError:
                continue
        if not saved:
            print("  ALL fallback writes also failed. Printing summary JSON below —")
            print("  COPY THIS OUTPUT MANUALLY before closing the terminal:")
            print(json.dumps(summary_only, indent=2))
    if not results:
        print("\n  ⚠ WARNING: NO DATASETS COMPLETED — deployment or connection failed for all datasets.")
        print(f"  An (empty) results file was still written to {out} for the record,")
        print("  but this run produced NO usable data. Check the RPC/Docker connection and retry.")
    print(f"\n  Results: {out}\n  Done.")

if __name__=='__main__': main()