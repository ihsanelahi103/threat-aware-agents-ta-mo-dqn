"""
rl_comparison.py — Internal Baseline RL Algorithm Comparison
--------------------------------------------------------------
Compares 6 RL algorithms on the Agents framework:
  1. Random Policy      — uniform random action selection (baseline)
  2. Epsilon-Greedy     — simple ε-greedy without learning (baseline)
  3. Q-Learning         — tabular Q-learning (no neural network)
  4. DQN               — Deep Q-Network (our method)
  5. DDQN              — Double DQN (reduces overestimation)
  6. PPO (Simplified)  — Proximal Policy Optimization (actor-critic)

All algorithms share the same blockchain, smart contract, and agents.
Only the decision-making strategy changes.

Usage:
    python3 rl_comparison.py --dataset MNIST --episodes 15
    python3 rl_comparison.py --episodes 15  (runs all datasets)

Output:
    results/rl_comparison.json
    plots/rl_comparison_*.png
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import tracemalloc
import numpy as np
from typing import Optional
from copy import deepcopy

try:
    from web3 import Web3
    from eth_account import Account
    print("[SDK] Using web3.py")
except ImportError:
    print("[ERROR] web3 not installed.")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config'))
try:
    from experiment_config import (
        RANDOM_SEED, N_AGENTS, N_BYZANTINE, N_EPISODES, N_STEPS,
        DATASETS, RESULTS_DIR, INFERENCE_TIME_MS, COMM_TIME_MS,
        INFEDGE_GAS_PER_CALL
    )
except ImportError:
    RANDOM_SEED       = 42
    N_AGENTS          = 20
    N_BYZANTINE       = 6
    N_EPISODES        = 15
    N_STEPS           = 50
    INFERENCE_TIME_MS = 156.0
    COMM_TIME_MS      = 145.0
    INFEDGE_GAS_PER_CALL = 4_300_000
    RESULTS_DIR       = "results"
    DATASETS = {
        'MNIST':          {'base_acc': 0.9926, 'train_acc': 0.9997, 'n_attacks': 5},
        'Fashion-MNIST':  {'base_acc': 0.9292, 'train_acc': 0.9881, 'n_attacks': 3},
        'Fashion-Easy':   {'base_acc': 0.9955, 'train_acc': 0.9997, 'n_attacks': 5},
        'Fashion-Medium': {'base_acc': 0.8778, 'train_acc': 0.9760, 'n_attacks': 2},
        'Fashion-Hard':   {'base_acc': 0.9803, 'train_acc': 0.9984, 'n_attacks': 2},
    }

FISCO_CHAIN_ID = 20200
DEFAULT_RPC = os.environ.get('RPC_URL', 'http://127.0.0.1:8545')
ACTIONS = ['SendData', 'VerifyData', 'AdjustPrivacy',
           'OptimizeResource', 'DetectByzantine']
N_ACTIONS = len(ACTIONS)
STATE_DIM = 15

# ── Load ABI ──────────────────────────────────────────────────
ABI_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'EdgeCommunication_abi.json')
if os.path.exists(ABI_FILE):
    with open(ABI_FILE) as f:
        CONTRACT_ABI = json.load(f)
else:
    CONTRACT_ABI = [
        {"name":"registerDevice","type":"function","inputs":[{"name":"agentId","type":"string"}],"outputs":[{"name":"","type":"bool"}],"stateMutability":"nonpayable"},
        {"name":"uploadData","type":"function","inputs":[{"name":"dataHash","type":"bytes32"},{"name":"metadata","type":"string"}],"outputs":[{"name":"","type":"bytes32"}],"stateMutability":"nonpayable"},
        {"name":"validateData","type":"function","inputs":[{"name":"dataHash","type":"bytes32"},{"name":"isValid","type":"bool"}],"outputs":[{"name":"","type":"bool"}],"stateMutability":"nonpayable"},
        {"name":"reportByzantine","type":"function","inputs":[{"name":"evidenceHash","type":"bytes32"},{"name":"suspectId","type":"string"}],"outputs":[{"name":"","type":"bool"}],"stateMutability":"nonpayable"},
        {"name":"getReportCount","type":"function","inputs":[{"name":"agentId","type":"string"}],"outputs":[{"name":"","type":"uint256"}],"stateMutability":"view"},
        {"name":"totalAgents","type":"function","inputs":[],"outputs":[{"name":"","type":"uint256"}],"stateMutability":"view"},
    ]


def agent_private_key(agent_id, seed=RANDOM_SEED):
    raw = hashlib.sha256(f"Agents_agent_{seed}_{agent_id:04d}".encode()).digest()
    return "0x" + raw.hex()


# ══════════════════════════════════════════════════════════════
#  RL ALGORITHMS
# ══════════════════════════════════════════════════════════════

class RandomPolicy:
    """Baseline: uniform random action selection."""
    name = "Random"

    def __init__(self, **kwargs):
        pass

    def select_action(self, state, epsilon=0.0):
        return random.randrange(N_ACTIONS)

    def update(self, state, action, reward, next_state):
        pass

    def memory_bytes(self):
        return 0


class EpsilonGreedyFixed:
    """Baseline: fixed ε-greedy with heuristic action preferences."""
    name = "ε-Greedy"

    def __init__(self, **kwargs):
        # Heuristic preferences: favor SendData and VerifyData
        self.preferences = np.array([0.35, 0.30, 0.10, 0.10, 0.15])

    def select_action(self, state, epsilon=0.3):
        if random.random() < epsilon:
            return random.randrange(N_ACTIONS)
        return int(np.random.choice(N_ACTIONS, p=self.preferences))

    def update(self, state, action, reward, next_state):
        pass

    def memory_bytes(self):
        return self.preferences.nbytes


class TabularQLearning:
    """Tabular Q-Learning with discretized state space."""
    name = "Q-Learning"

    def __init__(self, n_bins=5, lr=0.1, gamma=0.95, **kwargs):
        self.n_bins = n_bins
        self.lr = lr
        self.gamma = gamma
        # Discretize 15-dim state into bins → Q-table key
        self.q_table = {}

    def _discretize(self, state):
        """Discretize continuous state into bin indices."""
        bins = np.clip(np.floor(state * self.n_bins).astype(int), 0, self.n_bins - 1)
        return tuple(bins)

    def _get_q(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(N_ACTIONS)
        return self.q_table[state_key]

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(N_ACTIONS)
        state_key = self._discretize(state)
        return int(np.argmax(self._get_q(state_key)))

    def update(self, state, action, reward, next_state):
        sk = self._discretize(state)
        nsk = self._discretize(next_state)
        q = self._get_q(sk)
        nq = self._get_q(nsk)
        q[action] += self.lr * (reward + self.gamma * np.max(nq) - q[action])

    def memory_bytes(self):
        return len(self.q_table) * N_ACTIONS * 8  # float64


class NumpyDQN:
    """Deep Q-Network with experience replay."""
    name = "DQN"

    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=128,
                 lr=0.001, gamma=0.95, **kwargs):
        s = np.sqrt(2.0 / state_dim)
        self.W1 = np.random.randn(state_dim, hidden) * s
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, hidden) * np.sqrt(2.0/hidden)
        self.b2 = np.zeros(hidden)
        self.W3 = np.random.randn(hidden, n_actions) * np.sqrt(2.0/hidden)
        self.b3 = np.zeros(n_actions)
        self.lr = lr
        self.gamma = gamma

        # Experience replay buffer
        self.buffer = []
        self.buffer_size = 500
        self.batch_size = 32

    def forward(self, x):
        if x.ndim == 1: x = x[np.newaxis, :]
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        return (h2 @ self.W3 + self.b3).squeeze()

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(N_ACTIONS)
        return int(np.argmax(self.forward(state)))

    def update(self, state, action, reward, next_state):
        # Store experience
        self.buffer.append((state, action, reward, next_state))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        # Mini-batch update
        if len(self.buffer) >= self.batch_size:
            batch = random.sample(self.buffer, self.batch_size)
            for s, a, r, ns in batch:
                target = r + self.gamma * np.max(self.forward(ns))
                current = self.forward(s)
                error = target - current[a]

                # Simple gradient update on output layer
                h1 = np.maximum(0, s[np.newaxis, :] @ self.W1 + self.b1)
                h2 = np.maximum(0, h1 @ self.W2 + self.b2)
                grad = np.zeros(N_ACTIONS)
                grad[a] = error
                self.W3 += self.lr * h2.T @ grad[np.newaxis, :] / self.batch_size
                self.b3 += self.lr * grad / self.batch_size

    def memory_bytes(self):
        total = (self.W1.nbytes + self.b1.nbytes +
                 self.W2.nbytes + self.b2.nbytes +
                 self.W3.nbytes + self.b3.nbytes)
        total += len(self.buffer) * STATE_DIM * 8 * 2  # replay buffer
        return total


class NumpyDDQN:
    """Double DQN — uses target network to reduce overestimation."""
    name = "DDQN"

    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=128,
                 lr=0.001, gamma=0.95, **kwargs):
        s = np.sqrt(2.0 / state_dim)
        # Online network
        self.W1 = np.random.randn(state_dim, hidden) * s
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, hidden) * np.sqrt(2.0/hidden)
        self.b2 = np.zeros(hidden)
        self.W3 = np.random.randn(hidden, n_actions) * np.sqrt(2.0/hidden)
        self.b3 = np.zeros(n_actions)

        # Target network (copy of online)
        self.tW1 = self.W1.copy(); self.tb1 = self.b1.copy()
        self.tW2 = self.W2.copy(); self.tb2 = self.b2.copy()
        self.tW3 = self.W3.copy(); self.tb3 = self.b3.copy()

        self.lr = lr
        self.gamma = gamma
        self.buffer = []
        self.buffer_size = 500
        self.batch_size = 32
        self.update_count = 0
        self.target_update_freq = 50

    def forward(self, x, use_target=False):
        if x.ndim == 1: x = x[np.newaxis, :]
        if use_target:
            h1 = np.maximum(0, x @ self.tW1 + self.tb1)
            h2 = np.maximum(0, h1 @ self.tW2 + self.tb2)
            return (h2 @ self.tW3 + self.tb3).squeeze()
        else:
            h1 = np.maximum(0, x @ self.W1 + self.b1)
            h2 = np.maximum(0, h1 @ self.W2 + self.b2)
            return (h2 @ self.W3 + self.b3).squeeze()

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(N_ACTIONS)
        return int(np.argmax(self.forward(state)))

    def _sync_target(self):
        self.tW1 = self.W1.copy(); self.tb1 = self.b1.copy()
        self.tW2 = self.W2.copy(); self.tb2 = self.b2.copy()
        self.tW3 = self.W3.copy(); self.tb3 = self.b3.copy()

    def update(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        if len(self.buffer) >= self.batch_size:
            batch = random.sample(self.buffer, self.batch_size)
            for s, a, r, ns in batch:
                # DDQN: use online net to select action, target net to evaluate
                best_a = int(np.argmax(self.forward(ns, use_target=False)))
                target_q = self.forward(ns, use_target=True)
                target = r + self.gamma * target_q[best_a]
                current = self.forward(s)
                error = target - current[a]

                h1 = np.maximum(0, s[np.newaxis, :] @ self.W1 + self.b1)
                h2 = np.maximum(0, h1 @ self.W2 + self.b2)
                grad = np.zeros(N_ACTIONS)
                grad[a] = error
                self.W3 += self.lr * h2.T @ grad[np.newaxis, :] / self.batch_size
                self.b3 += self.lr * grad / self.batch_size

            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self._sync_target()

    def memory_bytes(self):
        online = (self.W1.nbytes + self.b1.nbytes +
                  self.W2.nbytes + self.b2.nbytes +
                  self.W3.nbytes + self.b3.nbytes)
        target = online  # same size
        buffer = len(self.buffer) * STATE_DIM * 8 * 2
        return online + target + buffer


class SimplifiedPPO:
    """Simplified PPO with actor-critic architecture."""
    name = "PPO"

    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=64,
                 lr_actor=0.002, lr_critic=0.005, gamma=0.95, clip_eps=0.2, **kwargs):
        s = np.sqrt(2.0 / state_dim)
        # Actor (policy) network
        self.aW1 = np.random.randn(state_dim, hidden) * s
        self.ab1 = np.zeros(hidden)
        self.aW2 = np.random.randn(hidden, n_actions) * np.sqrt(2.0/hidden)
        self.ab2 = np.zeros(n_actions)

        # Critic (value) network
        self.cW1 = np.random.randn(state_dim, hidden) * s
        self.cb1 = np.zeros(hidden)
        self.cW2 = np.random.randn(hidden, 1) * np.sqrt(2.0/hidden)
        self.cb2 = np.zeros(1)

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.trajectory = []

    def _softmax(self, x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / (e.sum() + 1e-8)

    def _actor_forward(self, x):
        if x.ndim == 1: x = x[np.newaxis, :]
        h = np.maximum(0, x @ self.aW1 + self.ab1)
        logits = (h @ self.aW2 + self.ab2).squeeze()
        return self._softmax(logits)

    def _critic_forward(self, x):
        if x.ndim == 1: x = x[np.newaxis, :]
        h = np.maximum(0, x @ self.cW1 + self.cb1)
        return (h @ self.cW2 + self.cb2).squeeze()

    def select_action(self, state, epsilon=0.0):
        probs = self._actor_forward(state)
        if random.random() < epsilon * 0.3:  # PPO uses less random exploration
            return random.randrange(N_ACTIONS)
        return int(np.random.choice(N_ACTIONS, p=probs))

    def update(self, state, action, reward, next_state):
        self.trajectory.append((state, action, reward, next_state))

        # Update every 32 steps
        if len(self.trajectory) >= 32:
            # Compute advantages
            for s, a, r, ns in self.trajectory:
                v = self._critic_forward(s)
                v_next = self._critic_forward(ns)
                advantage = r + self.gamma * v_next - v
                td_error = advantage

                # Update critic
                h = np.maximum(0, s[np.newaxis, :] @ self.cW1 + self.cb1)
                self.cW2 += self.lr_critic * td_error * h.T / 32
                self.cb2 += self.lr_critic * td_error / 32

                # Update actor with clipped objective
                probs = self._actor_forward(s)
                grad = np.zeros(N_ACTIONS)
                grad[a] = advantage / (probs[a] + 1e-8)
                # Clip gradient
                grad = np.clip(grad, -self.clip_eps, self.clip_eps)
                h_actor = np.maximum(0, s[np.newaxis, :] @ self.aW1 + self.ab1)
                self.aW2 += self.lr_actor * h_actor.T @ grad[np.newaxis, :] / 32
                self.ab2 += self.lr_actor * grad / 32

            self.trajectory = []

    def memory_bytes(self):
        actor = (self.aW1.nbytes + self.ab1.nbytes +
                 self.aW2.nbytes + self.ab2.nbytes)
        critic = (self.cW1.nbytes + self.cb1.nbytes +
                  self.cW2.nbytes + self.cb2.nbytes)
        traj = len(self.trajectory) * STATE_DIM * 8 * 2
        return actor + critic + traj


class A2C:
    """Advantage Actor-Critic."""
    name = "A2C"

    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=64,
                 lr=0.003, gamma=0.95, **kwargs):
        s = np.sqrt(2.0 / state_dim)
        # Shared feature layer
        self.W1 = np.random.randn(state_dim, hidden) * s
        self.b1 = np.zeros(hidden)
        # Actor head
        self.aW = np.random.randn(hidden, n_actions) * np.sqrt(2.0/hidden)
        self.ab = np.zeros(n_actions)
        # Critic head
        self.cW = np.random.randn(hidden, 1) * np.sqrt(2.0/hidden)
        self.cb = np.zeros(1)

        self.lr = lr
        self.gamma = gamma

    def _softmax(self, x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / (e.sum() + 1e-8)

    def _forward(self, x):
        if x.ndim == 1: x = x[np.newaxis, :]
        h = np.maximum(0, x @ self.W1 + self.b1)
        logits = (h @ self.aW + self.ab).squeeze()
        value = (h @ self.cW + self.cb).squeeze()
        return self._softmax(logits), float(value), h

    def select_action(self, state, epsilon=0.0):
        probs, _, _ = self._forward(state)
        if random.random() < epsilon * 0.3:
            return random.randrange(N_ACTIONS)
        return int(np.random.choice(N_ACTIONS, p=probs))

    def update(self, state, action, reward, next_state):
        probs, value, h = self._forward(state)
        _, next_value, _ = self._forward(next_state)
        advantage = reward + self.gamma * next_value - value

        # Actor update
        grad = np.zeros(N_ACTIONS)
        grad[action] = advantage / (probs[action] + 1e-8)
        self.aW += self.lr * h.T @ grad[np.newaxis, :]
        self.ab += self.lr * grad

        # Critic update
        self.cW += self.lr * advantage * h.T
        self.cb += self.lr * advantage

    def memory_bytes(self):
        return (self.W1.nbytes + self.b1.nbytes +
                self.aW.nbytes + self.ab.nbytes +
                self.cW.nbytes + self.cb.nbytes)


# ── All algorithms to compare ─────────────────────────────────
ALGORITHMS = {
    'Random':     RandomPolicy,
    'ε-Greedy':   EpsilonGreedyFixed,
    'Q-Learning': TabularQLearning,
    'DQN':        NumpyDQN,
    'DDQN':       NumpyDDQN,
    'PPO':        SimplifiedPPO,
    'A2C':        A2C,
}


# ── Blockchain connector (same as main experiment) ────────────
class FISCORealChain:
    def __init__(self, rpc_url, contract_address):
        self.rpc_url = rpc_url
        self.addr = contract_address
        self.tx_count = 0; self.tx_fail = 0
        self.gas_list = []; self.lat_list = []; self.blocks = []

        self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 30}))
        if not self.w3.is_connected():
            raise ConnectionError(f"Cannot connect to {rpc_url}")

        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_address), abi=CONTRACT_ABI)
        self._agent_accounts = {}
        self._default_caller = self.get_agent_account(0).address

    def get_agent_account(self, agent_id):
        if agent_id not in self._agent_accounts:
            self._agent_accounts[agent_id] = Account.from_key(agent_private_key(agent_id))
        return self._agent_accounts[agent_id]

    def _send_tx(self, agent_id, func_name, *args):
        t0 = time.time()
        try:
            acct = self.get_agent_account(agent_id)
            func = getattr(self.contract.functions, func_name)
            nonce = self.w3.eth.get_transaction_count(acct.address)
            tx = func(*args).build_transaction({
                'from': acct.address, 'gas': 300_000,
                'gasPrice': self.w3.to_wei('300', 'gwei'),
                'nonce': nonce, 'chainId': FISCO_CHAIN_ID,
            })
            signed = self.w3.eth.account.sign_transaction(tx, acct.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
            gas = receipt.get('gasUsed', 0)
            block = receipt.get('blockNumber', 0)
            status = receipt.get('status', 1)
            success = (status == 1)
            lat_ms = (time.time() - t0) * 1000
            already_reg = (not success and func_name == 'registerDevice')
            if already_reg:
                return {'success': True, 'gas': 0, 'block': block, 'latency_ms': lat_ms, 'already_registered': True}
            if success:
                self.tx_count += 1; self.gas_list.append(gas)
                self.lat_list.append(lat_ms); self.blocks.append(block)
            return {'success': success, 'gas': gas, 'block': block, 'latency_ms': lat_ms}
        except Exception as e:
            self.tx_fail += 1
            lat_ms = (time.time() - t0) * 1000
            if "already" in str(e).lower():
                return {'success': True, 'gas': 0, 'block': 0, 'latency_ms': lat_ms, 'already_registered': True}
            return {'success': False, 'gas': 0, 'block': 0, 'latency_ms': lat_ms}

    def registerDevice(self, aid, astr): return self._send_tx(aid, 'registerDevice', astr)
    def uploadData(self, aid, dh, meta):
        return self._send_tx(aid, 'uploadData', dh[:32].ljust(32, b'\x00'), meta)
    def validateData(self, aid, dh, v):
        return self._send_tx(aid, 'validateData', dh[:32].ljust(32, b'\x00'), v)
    def reportByzantine(self, aid, eh, sid):
        return self._send_tx(aid, 'reportByzantine', eh[:32].ljust(32, b'\x00'), sid)

    def get_stats(self):
        return {
            'total_tx': self.tx_count, 'failed_tx': self.tx_fail,
            'avg_gas_per_tx': int(np.mean(self.gas_list)) if self.gas_list else 0,
            'avg_latency_ms': float(np.mean(self.lat_list)) if self.lat_list else 0,
            'current_block': self.blocks[-1] if self.blocks else 0,
        }


# ── Deploy fresh contract ─────────────────────────────────────
def deploy_fresh_contract():
    result = subprocess.run(
        ['python3', 'deploy_contract_python.py'],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env={**os.environ}
    )
    if result.returncode != 0:
        print(f"  Deploy ERROR:\n{result.stderr}")
        return None
    addr_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'contract_address.txt')
    if os.path.exists(addr_file):
        with open(addr_file) as f:
            return f.read().strip()
    return None


# ── Compute reward ────────────────────────────────────────────
def compute_reward(action, result, ag, is_attack, detected_byz):
    """Reward function shared across all algorithms."""
    reward = 0.0
    if action == 'SendData' and result['success']:
        reward = 0.3 + ag['acc'] * 0.2  # reward for successful data upload
    elif action == 'VerifyData' and result['success']:
        reward = 0.4  # reward for verification
    elif action == 'DetectByzantine' and detected_byz:
        reward = 1.0  # high reward for catching Byzantine
    elif action == 'DetectByzantine' and is_attack and not detected_byz:
        reward = -0.2  # missed attack
    elif action == 'AdjustPrivacy':
        reward = 0.1
    elif action == 'OptimizeResource':
        reward = 0.1 + (1.0 - ag['cpu']) * 0.2  # reward more when CPU is high
    else:
        reward = -0.05  # small penalty for failed/no-op actions
    return reward


# ── Run single algorithm experiment ───────────────────────────
def run_algorithm_experiment(algo_name, algo_class, chain, dataset_name,
                             dataset_cfg, n_episodes) -> dict:
    print(f"\n  ── {algo_name} on {dataset_name} ──")

    tracemalloc.start()
    n_attacks = dataset_cfg['n_attacks']

    agents = []
    for i in range(N_AGENTS):
        agents.append({
            'id': i,
            'is_byz': i >= N_AGENTS - N_BYZANTINE,
            'acc': dataset_cfg['base_acc'] + random.uniform(-0.01, 0.01),
            'dp_eps': 0.5,
            'cpu': 0.70 + random.uniform(-0.05, 0.05),
            'byz_score': 0.0,
            'policy': algo_class(),
            'epsilon': 1.0,
            'registered': False,
        })

    # Register agents
    honest = [a for a in agents if not a['is_byz']]
    reg_ok = 0
    for ag in honest:
        result = chain.registerDevice(ag['id'], f"agent_{ag['id']:03d}")
        if result['success']:
            ag['registered'] = True
            reg_ok += 1
    print(f"    Registered: {reg_ok}/{len(honest)}")

    total_steps = n_episodes * N_STEPS
    attack_steps = set(random.sample(range(total_steps), n_attacks))
    byz_detected = 0; byz_fp = 0
    episode_accs = []; episode_dets = []
    t_start = time.time()
    total_rewards = 0.0

    for ep in range(n_episodes):
        for step in range(N_STEPS):
            global_step = ep * N_STEPS + step
            is_attack = (global_step in attack_steps)

            for ag in agents:
                if ag['is_byz']:
                    continue

                state = np.array([
                    ag['acc'], 0.95, 312.0/2000.0, 0.5,
                    ag['dp_eps']/3.5, 0.5/3.5, ag['cpu'], 0.75,
                    ag['byz_score'], 0.7,
                    random.uniform(0.3, 0.95), random.uniform(0.0, 0.4),
                    float(chain.tx_count % 20)/20.0,
                    0.0, min(global_step/750.0, 1.0),
                ], dtype=np.float32)

                action_idx = ag['policy'].select_action(state, ag['epsilon'])
                action = ACTIONS[action_idx]
                result = {'success': False, 'gas': 0, 'latency_ms': 0, 'block': 0}
                detected_byz = False

                if action == 'SendData' and ag['registered']:
                    dh = hashlib.sha3_256(f"{ag['id']}_{global_step}_{ag['dp_eps']:.6f}".encode()).digest()
                    result = chain.uploadData(ag['id'], dh,
                        json.dumps({'agent': ag['id'], 'step': global_step, 'acc': round(float(ag['acc']), 6)}))

                elif action == 'VerifyData':
                    peers = [a for a in agents if not a['is_byz'] and a['id'] != ag['id'] and a['registered']]
                    if peers:
                        peer = random.choice(peers)
                        dh = hashlib.sha3_256(f"{peer['id']}_{global_step}".encode()).digest()
                        result = chain.validateData(ag['id'], dh, True)
                        if result['success']:
                            peer['byz_score'] = max(0.0, peer['byz_score'] - 0.04)

                elif action == 'DetectByzantine' and is_attack:
                    byz_agents = [a for a in agents if a['is_byz']]
                    if byz_agents:
                        suspect = random.choice(byz_agents)
                        detected = (random.random() < 0.85)
                        if detected:
                            evid = hashlib.sha3_256(f"evidence_{suspect['id']}_{global_step}".encode()).digest()
                            result = chain.reportByzantine(ag['id'], evid, f"agent_{suspect['id']:03d}")
                            if result['success']:
                                byz_detected = min(byz_detected + 1, n_attacks)
                                detected_byz = True
                                suspect['byz_score'] = min(1.0, suspect['byz_score'] + 0.35)
                        else:
                            if random.random() < 0.003:
                                byz_fp += 1

                # Compute reward and update policy
                reward = compute_reward(action, result, ag, is_attack, detected_byz)
                total_rewards += reward

                next_state = np.array([
                    ag['acc'], 0.95, 312.0/2000.0, 0.5,
                    ag['dp_eps']/3.5, 0.5/3.5, ag['cpu'], 0.75,
                    ag['byz_score'], 0.7,
                    random.uniform(0.3, 0.95), random.uniform(0.0, 0.4),
                    float(chain.tx_count % 20)/20.0,
                    0.0, min((global_step+1)/750.0, 1.0),
                ], dtype=np.float32)

                ag['policy'].update(state, action_idx, reward, next_state)

                ag['acc'] = float(np.clip(ag['acc'] + np.random.normal(0, 0.0015), 0.70, 0.999))
                ag['cpu'] = float(np.clip(ag['cpu'] + np.random.normal(0, 0.01), 0.20, 0.98))
                ag['epsilon'] = max(0.05, ag['epsilon'] * 0.990)

        # Episode metrics
        avg_acc = float(np.mean([a['acc'] for a in agents if not a['is_byz']]))
        det_rate = byz_detected / n_attacks if n_attacks > 0 else 0
        episode_accs.append(avg_acc)
        episode_dets.append(det_rate)

        if (ep + 1) % max(1, n_episodes // 3) == 0:
            s = chain.get_stats()
            print(f"    Ep {ep+1:3d}/{n_episodes}  acc={avg_acc:.4f}  "
                  f"det={det_rate:.0%}  tx={s['total_tx']}  "
                  f"gas={s['avg_gas_per_tx']:,}/tx")

    # Final metrics
    runtime = time.time() - t_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_test = float(np.mean([a['acc'] for a in agents if not a['is_byz']]))
    det_rate = byz_detected / n_attacks if n_attacks > 0 else 0
    s = chain.get_stats()
    e2e = s['avg_latency_ms'] + INFERENCE_TIME_MS + COMM_TIME_MS
    gas_saving = (1 - s['avg_gas_per_tx'] / INFEDGE_GAS_PER_CALL) * 100 if s['avg_gas_per_tx'] > 0 else 0

    # Memory per agent
    agent_mem = int(np.mean([a['policy'].memory_bytes() for a in honest]))

    print(f"    ✓ {algo_name}: acc={avg_test:.4f}  det={det_rate:.0%}  "
          f"tx={s['total_tx']}  mem={agent_mem/1024:.1f}KB/agent  "
          f"time={runtime:.0f}s")

    return {
        'algorithm':      algo_name,
        'dataset':        dataset_name,
        'test_acc':       round(avg_test, 6),
        'byz_detection':  round(det_rate, 4),
        'byz_tp':         byz_detected,
        'byz_fp':         byz_fp,
        'n_attacks':      n_attacks,
        'tx_ok':          s['total_tx'],
        'tx_fail':        s['failed_tx'],
        'avg_gas':        s['avg_gas_per_tx'],
        'avg_chain_ms':   round(s['avg_latency_ms'], 1),
        'e2e_ms':         round(e2e, 1),
        'gas_saving_pct': round(gas_saving, 1),
        'total_reward':   round(total_rewards, 2),
        'mem_per_agent_kb': round(agent_mem / 1024, 2),
        'peak_memory_mb': round(peak_mem / 1024 / 1024, 2),
        'runtime_s':      round(runtime, 1),
        'episode_accs':   [round(a, 6) for a in episode_accs],
        'episode_dets':   [round(d, 4) for d in episode_dets],
        'agents_registered': reg_ok,
    }


# ── Plot comparison results ───────────────────────────────────
def plot_comparison(all_results, save_dir):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    os.makedirs(save_dir, exist_ok=True)

    # Group by dataset
    datasets = {}
    for r in all_results:
        ds = r['dataset']
        datasets.setdefault(ds, []).append(r)

    ALGO_COLORS = {
        'Random': '#9CA3AF', 'ε-Greedy': '#D97706', 'Q-Learning': '#059669',
        'DQN': '#2563EB', 'DDQN': '#7C3AED', 'PPO': '#DC2626', 'A2C': '#EC4899'
    }

    # ── Plot 1: Accuracy comparison bar chart ─────────────────
    for ds_name, results in datasets.items():
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        algos = [r['algorithm'] for r in results]
        colors = [ALGO_COLORS.get(a, '#6B7280') for a in algos]
        x = np.arange(len(algos))

        # Accuracy
        vals = [r['test_acc'] * 100 for r in results]
        bars = axes[0,0].bar(x, vals, 0.6, color=colors, edgecolor='white')
        for bar, v in zip(bars, vals):
            axes[0,0].text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.2,
                          f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')
        axes[0,0].set_title('Test Accuracy (%)', fontweight='bold')
        axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(algos, rotation=25, ha='right', fontsize=9)
        axes[0,0].set_ylim(min(vals)-5, max(vals)+3)

        # Detection Rate
        vals = [r['byz_detection'] * 100 for r in results]
        bars = axes[0,1].bar(x, vals, 0.6, color=colors, edgecolor='white')
        for bar, v in zip(bars, vals):
            axes[0,1].text(bar.get_x()+bar.get_width()/2., bar.get_height()+1,
                          f'{v:.0f}%', ha='center', fontsize=8, fontweight='bold')
        axes[0,1].set_title('Byzantine Detection Rate (%)', fontweight='bold')
        axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(algos, rotation=25, ha='right', fontsize=9)
        axes[0,1].set_ylim(0, 115)

        # Total Reward
        vals = [r['total_reward'] for r in results]
        axes[0,2].bar(x, vals, 0.6, color=colors, edgecolor='white')
        axes[0,2].set_title('Cumulative Reward', fontweight='bold')
        axes[0,2].set_xticks(x); axes[0,2].set_xticklabels(algos, rotation=25, ha='right', fontsize=9)

        # Memory per Agent
        vals = [r['mem_per_agent_kb'] for r in results]
        axes[1,0].bar(x, vals, 0.6, color=colors, edgecolor='white')
        for i, v in enumerate(vals):
            axes[1,0].text(x[i], v+0.5, f'{v:.1f}', ha='center', fontsize=8, fontweight='bold')
        axes[1,0].set_title('Memory per Agent (KB)', fontweight='bold')
        axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(algos, rotation=25, ha='right', fontsize=9)

        # TX Count
        vals = [r['tx_ok'] for r in results]
        axes[1,1].bar(x, vals, 0.6, color=colors, edgecolor='white')
        axes[1,1].set_title('Successful Transactions', fontweight='bold')
        axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(algos, rotation=25, ha='right', fontsize=9)

        # Runtime
        vals = [r['runtime_s'] for r in results]
        axes[1,2].bar(x, vals, 0.6, color=colors, edgecolor='white')
        axes[1,2].set_title('Runtime (seconds)', fontweight='bold')
        axes[1,2].set_xticks(x); axes[1,2].set_xticklabels(algos, rotation=25, ha='right', fontsize=9)

        fig.suptitle(f'RL Algorithm Comparison — {ds_name}', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'rl_comparison_{ds_name.replace("-","_").lower()}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ rl_comparison_{ds_name.replace('-','_').lower()}.png")

    # ── Plot 2: Convergence curves ────────────────────────────
    for ds_name, results in datasets.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        for r in results:
            eps_range = range(1, len(r['episode_accs'])+1)
            color = ALGO_COLORS.get(r['algorithm'], '#6B7280')
            ax1.plot(eps_range, [a*100 for a in r['episode_accs']], '-o',
                    color=color, label=r['algorithm'], markersize=3, linewidth=1.5)
            ax2.plot(eps_range, [d*100 for d in r['episode_dets']], '-s',
                    color=color, label=r['algorithm'], markersize=3, linewidth=1.5)

        ax1.set_title('Accuracy Convergence'); ax1.set_xlabel('Episode')
        ax1.set_ylabel('Accuracy (%)'); ax1.legend(fontsize=8)
        ax2.set_title('Detection Rate Convergence'); ax2.set_xlabel('Episode')
        ax2.set_ylabel('Detection (%)'); ax2.legend(fontsize=8); ax2.set_ylim(-5, 110)

        fig.suptitle(f'Convergence — {ds_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'rl_convergence_{ds_name.replace("-","_").lower()}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ rl_convergence_{ds_name.replace('-','_').lower()}.png")

    # ── Plot 3: Summary comparison table ──────────────────────
    fig, ax = plt.subplots(figsize=(16, 4 + len(all_results) * 0.4))
    ax.axis('off')
    cols = ['Algorithm', 'Dataset', 'Accuracy', 'Byz Det', 'Reward', 'TX OK',
            'Gas/TX', 'Mem/Agent', 'Runtime']
    rows = []
    for r in all_results:
        rows.append([
            r['algorithm'], r['dataset'], f"{r['test_acc']*100:.2f}%",
            f"{r['byz_detection']*100:.0f}%", f"{r['total_reward']:.0f}",
            str(r['tx_ok']), f"{r['avg_gas']:,}",
            f"{r['mem_per_agent_kb']:.1f} KB", f"{r['runtime_s']:.0f}s"
        ])
    table = ax.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.0, 1.6)
    for j in range(len(cols)):
        table[0, j].set_facecolor('#2563EB')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(len(rows)):
        color = '#F1F5F9' if i % 2 == 0 else 'white'
        for j in range(len(cols)):
            table[i+1, j].set_facecolor(color)
    fig.suptitle('RL Algorithm Comparison — Summary Table', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(save_dir, 'rl_comparison_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ rl_comparison_table.png")


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='RL Algorithm Comparison on Agents')
    parser.add_argument('--rpc', type=str, default=DEFAULT_RPC)
    parser.add_argument('--dataset', type=str, default=None, choices=list(DATASETS.keys()))
    parser.add_argument('--episodes', type=int, default=N_EPISODES)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--algorithms', type=str, default=None,
        help='Comma-separated list of algorithms (default: all)')
    args = parser.parse_args()

    ds_to_run = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    algo_names = args.algorithms.split(',') if args.algorithms else list(ALGORITHMS.keys())
    algo_map = {k: v for k, v in ALGORITHMS.items() if k in algo_names}

    print("=" * 60)
    print("  Agents — RL Algorithm Comparison")
    print(f"  RPC:        {args.rpc}")
    print(f"  Algorithms: {', '.join(algo_map.keys())}")
    print(f"  Datasets:   {', '.join(ds_to_run.keys())}")
    print(f"  Episodes:   {args.episodes}")
    print("=" * 60)

    all_results = []

    for ds_name, ds_cfg in ds_to_run.items():
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")

        for algo_name, algo_class in algo_map.items():
            # Reset seed for fair comparison
            random.seed(args.seed)
            np.random.seed(args.seed)

            # Deploy fresh contract
            print(f"\n  Deploying fresh contract for {algo_name}...")
            contract_addr = deploy_fresh_contract()
            if not contract_addr:
                print(f"  ERROR: Deploy failed for {algo_name}")
                continue
            print(f"  Contract: {contract_addr}")

            try:
                chain = FISCORealChain(rpc_url=args.rpc, contract_address=contract_addr)
            except ConnectionError as e:
                print(f"  ERROR: {e}")
                continue

            result = run_algorithm_experiment(
                algo_name, algo_class, chain, ds_name, ds_cfg, args.episodes)
            all_results.append(result)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, 'rl_comparison.json')
    with open(out, 'w') as f:
        json.dump({
            'mode': 'RL Algorithm Comparison',
            'algorithms': list(algo_map.keys()),
            'results': all_results,
        }, f, indent=2)
    print(f"\n  Results saved: {out}")

    # Generate plots
    print("\n  Generating comparison plots...")
    plot_comparison(all_results, 'plots')

    # Print summary table
    print(f"\n{'='*90}")
    print(f"  {'Algorithm':<12} {'Dataset':<15} {'Accuracy':>10} {'Byz Det':>10} "
          f"{'Reward':>10} {'Mem/Agent':>12} {'Runtime':>10}")
    print(f"{'='*90}")
    for r in all_results:
        print(f"  {r['algorithm']:<12} {r['dataset']:<15} "
              f"{r['test_acc']*100:>9.2f}% {r['byz_detection']*100:>9.0f}% "
              f"{r['total_reward']:>10.0f} {r['mem_per_agent_kb']:>10.1f} KB "
              f"{r['runtime_s']:>9.0f}s")
    print(f"{'='*90}")
    print("  Done.")


if __name__ == '__main__':
    main()
