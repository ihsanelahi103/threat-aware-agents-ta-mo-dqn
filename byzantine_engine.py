"""
byzantine_engine.py
--------------------------------------------------------------
Real (non-scripted) Byzantine behavior and detection for Agents.

WHY THIS EXISTS
================
In main_fisco_real.py, Byzantine agents currently do nothing (no model,
no forward pass, `if ag['is_byz']: continue`), and "detection" is a
scripted coin flip (`random.random() < 0.85`) that is independent of
any signal a real agent could observe. That means:
  - The reported 100% F1 / zero-FP-FN result is close to guaranteed by
    construction, not discovered by learning or genuine verification.
  - The ablations (w/o ACV, w/o AAP) can't meaningfully change detection
    quality, because ACV's hash isn't actually being *checked* against
    anything — it's generated and uploaded, never compared.

This module replaces that with:
  1. ByzantineAgent — actually runs inference and then corrupts its
     output using one of several realistic attack strategies.
  2. ReferenceBatchDetector — a real statistical test that VerifyData /
     DetectByzantine can call, using signal that is actually present
     (confidence deviation from a shared reference batch, hash/nonce
     mismatch, replay detection) rather than a fixed probability.

DESIGN PRINCIPLE
=================
Every attack must leave a *detectable but non-trivial* trace — i.e.
something a verifier could plausibly notice by comparing agent outputs
to a reference, but not something that is always caught (that would
just move the "guaranteed by construction" problem elsewhere). This is
what makes precision/recall/F1 an actual result instead of an artifact
of the simulation's constants.
"""

import hashlib
import random
import numpy as np
from typing import Optional


# ── Attack strategies ──────────────────────────────────────────────
class AttackStrategy:
    LABEL_FLIP = "label_flip"           # reports a wrong class prediction
    CONFIDENCE_FALSIFY = "conf_falsify" # inflates/deflates reported confidence
    REPLAY = "replay"                   # resubmits a stale prior commitment
    INCONSISTENT = "inconsistent"       # gives different answers to the same query

    ALL = [LABEL_FLIP, CONFIDENCE_FALSIFY, REPLAY, INCONSISTENT]


class ByzantineAgent:
    """
    Wraps a real (trained or randomly-initialized) model and corrupts
    its output according to an assigned attack strategy. Unlike the
    current implementation, this agent actually runs inference and
    actually produces falsified — but non-random-noise — output that a
    verifier could, in principle, detect.

    strategy_mix: dict mapping AttackStrategy -> probability of using it
                  on any given attack step (agent still behaves honestly
                  on non-attack steps to mimic an intermittent adversary,
                  consistent with the paper's existing |Omega| schedule).
    """

    def __init__(self, agent_id: int, model, strategy_mix: Optional[dict] = None,
                 intensity: float = 0.6, seed: int = 0, adaptive: bool = False):
        self.agent_id = agent_id
        self.model = model  # a real EdgeCNN, or None -> falls back to synthetic logits
        self.rng = np.random.RandomState(seed + agent_id)
        self.strategy_mix = strategy_mix or {
            AttackStrategy.LABEL_FLIP: 0.35,
            AttackStrategy.CONFIDENCE_FALSIFY: 0.30,
            AttackStrategy.REPLAY: 0.20,
            AttackStrategy.INCONSISTENT: 0.15,
        }
        self.intensity = intensity  # how strongly the corruption deviates (0-1)
        self._last_commitment = None  # for REPLAY strategy
        self._query_answers = {}      # query_id -> answer, for INCONSISTENT strategy
        # NEW (item 8): adaptive/evasive attacker. When True, the agent
        # observes the AAP global threat signal (a value the threat model
        # already broadcasts to "all agents" per the paper's own AAP
        # description — a legitimately-registered-but-compromised device
        # can plausibly receive it too) and modulates BOTH whether it
        # attacks on a scheduled step and how strongly, becoming more
        # cautious as perceived detection risk rises. This operationalizes
        # the paper's own stated future-work formulation:
        #   P(attack) = f(threat_level, detection_probability)
        # An attacker with adaptive=False (the default) is the static,
        # non-evasive attacker used in the main campaign — adaptive=True
        # is a separate, explicit experimental condition.
        self.adaptive = adaptive

    def decide_attack(self, is_scheduled: bool, observed_threat: float = 0.0):
        """
        Returns (should_attack: bool, effective_intensity: float) for
        THIS step. Non-adaptive agents always attack on scheduled steps
        at fixed intensity (the original behavior). Adaptive agents
        probabilistically skip attacking, and reduce intensity, as
        observed_threat rises — trading attack frequency/strength for a
        lower chance of detection, which is exactly the evasive behavior
        an S&P reviewer will ask "what happens if the attacker adapts?"
        about.
        """
        if not is_scheduled:
            return False, self.intensity
        if not self.adaptive:
            return True, self.intensity
        observed_threat = float(np.clip(observed_threat, 0.0, 1.0))
        # Floor of 0.15 so the attacker never goes fully silent (still a
        # meaningful adversary, just a cautious one); scales down toward
        # that floor as threat -> 1.0.
        p_attack = max(0.15, 1.0 - observed_threat)
        should_attack = self.rng.random() < p_attack
        effective_intensity = max(0.15, self.intensity * (1.0 - observed_threat * 0.7))
        return should_attack, effective_intensity

    def _pick_strategy(self):
        strategies = list(self.strategy_mix.keys())
        probs = np.array(list(self.strategy_mix.values()))
        probs = probs / probs.sum()
        return self.rng.choice(strategies, p=probs)

    def produce_output(self, true_preds: np.ndarray, true_confs: np.ndarray,
                        step: int, query_id: Optional[str] = None,
                        intensity_override: Optional[float] = None):
        """
        Given the agent's *true* model output (computed the same way an
        honest agent's would be — this agent still runs real inference,
        it just decides to misreport afterward), return a falsified
        (preds, confs, commitment_extra) tuple plus the strategy used,
        for logging / eventual ground truth comparison.

        commitment_extra carries strategy-specific metadata (e.g. a
        stale hash for REPLAY) needed by the detector.

        intensity_override: if given (used by adaptive attackers via
        decide_attack's effective_intensity), overrides self.intensity
        for THIS call only, without mutating the agent's base setting.
        """
        effective_intensity = self.intensity if intensity_override is None else intensity_override
        strategy = self._pick_strategy()
        preds = true_preds.copy()
        confs = true_confs.copy()
        commitment_extra = {"replay": False, "prior_hash": None}

        if strategy == AttackStrategy.LABEL_FLIP:
            # Flip a fraction of predictions to a plausible-but-wrong class.
            n_flip = max(1, int(len(preds) * effective_intensity))
            flip_idx = self.rng.choice(len(preds), size=n_flip, replace=False)
            for i in flip_idx:
                preds[i] = (preds[i] + self.rng.randint(1, 9)) % 10
            # Confidence stays high despite being wrong -> detectable via
            # cross-check against reference batch statistics.
            confs[flip_idx] = np.clip(confs[flip_idx] + 0.15, 0, 0.999)

        elif strategy == AttackStrategy.CONFIDENCE_FALSIFY:
            # Report implausibly high confidence regardless of true value.
            delta = effective_intensity * self.rng.uniform(0.2, 0.4, size=len(confs))
            confs = np.clip(confs + delta, 0, 0.999)

        elif strategy == AttackStrategy.REPLAY:
            # Resubmit the previous step's commitment instead of a fresh one.
            if self._last_commitment is not None:
                preds, confs = self._last_commitment
                commitment_extra["replay"] = True
            # else: fall through as honest this one time (no history yet)

        elif strategy == AttackStrategy.INCONSISTENT:
            # Give a different answer than previously given to the same
            # reference query (only detectable when queries repeat).
            if query_id is not None:
                if query_id in self._query_answers:
                    prev_preds, _ = self._query_answers[query_id]
                    # deliberately diverge from own prior answer
                    preds = (prev_preds + self.rng.randint(1, 9, size=len(preds))) % 10
                self._query_answers[query_id] = (preds.copy(), confs.copy())

        self._last_commitment = (preds.copy(), confs.copy())
        return preds, confs, strategy, commitment_extra


# ── Real (statistical) detection ────────────────────────────────────
class ReferenceBatchDetector:
    """
    Replaces the scripted `random.random() < 0.85` detection with a
    real statistical test. Honest verifiers periodically query all
    agents (honest and Byzantine) on a small SHARED reference batch
    with known ground truth. Because Byzantine agents don't know which
    queries are "reference" queries versus ordinary ones (a realistic
    assumption under the paper's threat model — agents can't distinguish
    verification traffic from normal traffic), their attack strategies
    still fire on reference queries, producing a detectable deviation.

    Detection score combines:
      - disagreement with the reference ground-truth label
      - confidence anomaly (z-score vs the honest-agent population)
      - replay flag (identical commitment across two different steps)
      - inconsistency flag (different answer to a repeated query)
    """

    def __init__(self, ref_size: int = 20, threshold: float = 0.55, seed: int = 0):
        self.ref_size = ref_size
        self.threshold = threshold
        self.rng = np.random.RandomState(seed)
        # Self-referenced baseline (detector v2): each agent's own history of
        # committed reference-batch reports. An agent is compared with ITS OWN
        # past behaviour (robust median), not with the spread of the honest
        # population, so the statistic does not depend on how different the
        # honest models are from each other.
        self.self_hist_len = 40
        self.self_min_hist = 5
        self.conf_tol = 0.02      # |mean-conf shift| that saturates the conf term
        self.pcf_scale = 0.10     # fraction of changed predictions that saturates the consistency term
        self._self_hist = {}      # agent_id -> deque of (mean_conf, preds, disagreement)
        self._history = {}  # agent_id -> list of (step, pred, conf, commitment_hash)

    def build_reference_batch(self, test_ds_len: int):
        return self.rng.choice(test_ds_len, size=self.ref_size, replace=False)

    def observe(self, agent_id: int, reported_preds, reported_confs, true_labels):
        """Record one committed report in the agent's own history (call once per
        report, honest or not, at submission time)."""
        from collections import deque
        h = self._self_hist.setdefault(agent_id, deque(maxlen=self.self_hist_len))
        preds = np.asarray(reported_preds).copy()
        h.append((float(np.mean(reported_confs)), preds,
                  float(np.mean(preds != np.asarray(true_labels)))))

    def self_features(self, agent_id: int, reported_preds, reported_confs, true_labels):
        """Deviation of a report from the agent's own robust baseline.
        Returns None until the agent has self_min_hist reports."""
        h = self._self_hist.get(agent_id)
        if h is None or len(h) < self.self_min_hist:
            return None
        means = np.array([x[0] for x in h]); diss = np.array([x[2] for x in h])
        P = np.stack([x[1] for x in h])                       # (H, ref_size)
        counts = (P[:, :, None] == np.arange(10)[None, None, :]).sum(0)
        modal = counts.argmax(1)
        preds = np.asarray(reported_preds)
        return {
            'conf_delta': float(np.mean(reported_confs) - np.median(means)),   # signed
            'pcf': float(np.mean(preds != modal)),                              # changed-prediction fraction
            'dis_self': float(np.mean(preds != np.asarray(true_labels)) - np.median(diss)),
        }

    def score(self, agent_id: int, step: int, reported_preds: np.ndarray,
              reported_confs: np.ndarray, true_labels: np.ndarray,
              honest_conf_mean: float, honest_conf_std: float,
              replay_flag: bool, inconsistency_flag: bool,
              honest_disagreement_mean: float = 0.0,
              self_feats=None, selfref: bool = False) -> float:
        """
        Returns a suspicion score in [0, 1]. This is a genuine function
        of observable signal, not a fixed constant.

        honest_disagreement_mean: the reference-batch error rate typical
        of currently-honest agents on THIS dataset/model (e.g. ~0.05 for
        a 95%-accurate MNIST model, ~0.16 for an 84%-accurate Fashion-MNIST
        model). Without this baseline, raw disagreement conflates "wrong
        because the task is hard" with "wrong because the agent lied" —
        which caused a large false-positive blowup on the harder dataset
        (P collapsed to 0.043 in testing) because honest agents' normal
        error rate alone was often enough to cross the threshold. Scoring
        EXCESS disagreement beyond the honest baseline generalizes across
        datasets of different difficulty instead of assuming near-perfect
        honest accuracy.
        """
        # Detector v2 warm-up: until an agent has a baseline of its own, the only
        # admissible evidence is a repeated commitment (hard, on-chain). The
        # population-referenced statistics are NOT used as a fallback: they are what
        # made scores depend on the spread between honest models.
        if selfref and self_feats is None:
            return float(0.20 if replay_flag else 0.0)

        # 1. Label disagreement rate against known ground truth on the
        #    reference batch, normalized against the honest population's
        #    own typical error rate on this dataset (see docstring above).
        raw_disagreement = float(np.mean(reported_preds != true_labels))
        disagreement = max(0.0, raw_disagreement - honest_disagreement_mean)

        # 2. Confidence anomaly: how many std-devs the reported mean
        #    confidence is from the honest-agent population.
        if honest_conf_std > 1e-6:
            z = abs(np.mean(reported_confs) - honest_conf_mean) / honest_conf_std
            conf_anomaly = min(1.0, z / 3.0)  # saturate at 3 std-devs
        else:
            conf_anomaly = 0.0

        replay_component = 1.0 if replay_flag else 0.0
        inconsistency_component = 1.0 if inconsistency_flag else 0.0

        # Detector v2: when the agent has a history, every term is measured
        # against the agent's OWN baseline instead of the honest population.
        if self_feats is not None:
            disagreement = max(0.0, self_feats['dis_self'])
            conf_anomaly = min(1.0, abs(self_feats['conf_delta']) / self.conf_tol)
            inconsistency_component = min(1.0, self_feats['pcf'] / self.pcf_scale)

        score = (0.45 * disagreement + 0.25 * conf_anomaly +
                 0.20 * replay_component + 0.10 * inconsistency_component)
        return float(np.clip(score, 0.0, 1.0))

    def detect(self, *args, **kwargs) -> bool:
        return self.score(*args, **kwargs) >= self.threshold

    def check_replay(self, agent_id: int, step: int, commitment_hash: bytes) -> bool:
        """Detects an exact repeat of a prior commitment hash for this agent."""
        hist = self._history.setdefault(agent_id, [])
        is_replay = any(h == commitment_hash for _, h in hist)
        hist.append((step, commitment_hash))
        if len(hist) > 50:
            hist.pop(0)
        return is_replay


def commitment_hash(agent_id, step, preds, conf) -> bytes:
    pred_str = "_".join(str(int(p)) for p in preds[:10])
    # The round/step IS bound into the commitment (as a challenge nonce would
    # be in a deployed protocol). Previously the step was excluded, so an
    # honest agent with a fixed model reporting on a fixed reference batch
    # produced the SAME hash every step; exact-repeat "replay" detection
    # then flagged truthful repeats (and could not distinguish them from a
    # real replay). Now a truthful report never repeats a commitment, and a
    # real replay (re-submitting the stale commitment) always does.
    base = f"{agent_id}_{step}_{conf:.6f}_{pred_str}"
    return hashlib.sha3_256(base.encode()).digest()