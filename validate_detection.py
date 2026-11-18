"""
validate_detection.py
--------------------------------------------------------------
Sanity-checks byzantine_engine.py WITHOUT needing torch, FISCO, or a
GPU. Simulates a simplified version of the agent loop using synthetic
(but realistic-shaped) confidence distributions, so we can confirm the
detection logic:
  (a) is NOT trivially 100% (i.e. it can miss stealthy attacks), and
  (b) IS clearly better than random guessing, and
  (c) responds sensibly to intensity (stronger attacks -> higher recall).

Run: python3 validate_detection.py
"""

import numpy as np
from byzantine_engine import ByzantineAgent, ReferenceBatchDetector, commitment_hash, AttackStrategy


def simulate(n_honest=14, n_byzantine=6, n_steps=500, ref_size=20,
             intensity=0.6, threshold=0.55, seed=42, verbose=True):
    rng = np.random.RandomState(seed)
    n_classes = 10

    # Honest agents: confidence ~ N(0.85, 0.06), clipped. Predictions
    # correct 90% of the time (mimics a well-trained EdgeCNN).
    honest_conf_mean, honest_conf_std = 0.85, 0.06

    byzantine_agents = [
        ByzantineAgent(agent_id=100 + i, model=None, intensity=intensity, seed=seed)
        for i in range(n_byzantine)
    ]
    detector = ReferenceBatchDetector(ref_size=ref_size, threshold=threshold, seed=seed)

    y_true = []   # 1 if this observation is actually from a Byzantine agent on an attack step
    y_pred = []   # 1 if detector flagged it

    # Attack steps: Byzantine agents are only adversarial on ~30/500 steps
    # (mirrors the paper's intermittent |Omega| schedule), otherwise honest.
    n_attacks = 30
    attack_steps = set(rng.choice(n_steps, size=n_attacks, replace=False))

    for step in range(n_steps):
        true_labels = rng.randint(0, n_classes, size=ref_size)
        is_attack_step = step in attack_steps

        # --- Honest agents on the reference batch (ground truth for the population stats)
        honest_confs_pool = []
        for _ in range(n_honest):
            correct_mask = rng.random(ref_size) < 0.90
            preds = np.where(correct_mask, true_labels, rng.randint(0, n_classes, ref_size))
            confs = np.clip(rng.normal(honest_conf_mean, honest_conf_std, ref_size), 0, 0.999)
            honest_confs_pool.append(confs.mean())
        honest_conf_mean_obs = float(np.mean(honest_confs_pool))
        honest_conf_std_obs = float(np.std(honest_confs_pool)) + 1e-6

        # --- Byzantine agents
        for ba in byzantine_agents:
            correct_mask = rng.random(ref_size) < 0.90
            true_preds = np.where(correct_mask, true_labels, rng.randint(0, n_classes, ref_size))
            true_confs = np.clip(rng.normal(honest_conf_mean, honest_conf_std, ref_size), 0, 0.999)

            if is_attack_step:
                preds, confs, strategy, extra = ba.produce_output(true_preds, true_confs, step, query_id=f"ref_{step % 5}")
                ch = commitment_hash(ba.agent_id, step, preds, confs.mean())
                is_replay = detector.check_replay(ba.agent_id, step, ch) or extra["replay"]
                label = 1  # ground truth: this IS an attack observation
            else:
                preds, confs = true_preds, true_confs
                ch = commitment_hash(ba.agent_id, step, preds, confs.mean())
                is_replay = detector.check_replay(ba.agent_id, step, ch)
                label = 0

            score = detector.score(
                agent_id=ba.agent_id, step=step,
                reported_preds=preds, reported_confs=confs,
                true_labels=true_labels,
                honest_conf_mean=honest_conf_mean_obs,
                honest_conf_std=honest_conf_std_obs,
                replay_flag=is_replay, inconsistency_flag=False,
            )
            pred_label = 1 if score >= threshold else 0

            y_true.append(label)
            y_pred.append(pred_label)

        # --- Honest agents also scored, to measure false-positive rate
        for _ in range(n_honest):
            correct_mask = rng.random(ref_size) < 0.90
            preds = np.where(correct_mask, true_labels, rng.randint(0, n_classes, ref_size))
            confs = np.clip(rng.normal(honest_conf_mean, honest_conf_std, ref_size), 0, 0.999)
            score = detector.score(
                agent_id=-1, step=step,
                reported_preds=preds, reported_confs=confs,
                true_labels=true_labels,
                honest_conf_mean=honest_conf_mean_obs,
                honest_conf_std=honest_conf_std_obs,
                replay_flag=False, inconsistency_flag=False,
            )
            y_true.append(0)
            y_pred.append(1 if score >= threshold else 0)

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    if verbose:
        print(f"intensity={intensity:.2f} threshold={threshold:.2f} -> "
              f"TP={tp} FP={fp} FN={fn} TN={tn} | "
              f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")
    return precision, recall, f1


if __name__ == "__main__":
    print("=== Sanity check: does detection quality respond to attack intensity? ===")
    print("(If F1 were trivially 1.000 regardless of intensity, that would mean")
    print(" the detector is STILL guaranteed-by-construction, not a real test.)\n")
    for intensity in [0.1, 0.3, 0.5, 0.6, 0.8, 1.0]:
        simulate(intensity=intensity, seed=42)

    print("\n=== Sanity check: threshold sweep at fixed intensity=0.6 ===")
    for threshold in [0.30, 0.45, 0.55, 0.65, 0.80]:
        simulate(intensity=0.6, threshold=threshold, seed=42)

    print("\n=== Multi-seed stability at intensity=0.6, threshold=0.55 ===")
    f1s = []
    for seed in range(10):
        p, r, f1 = simulate(intensity=0.6, threshold=0.55, seed=seed, verbose=False)
        f1s.append(f1)
    import numpy as np
    print(f"F1 over 10 seeds: mean={np.mean(f1s):.3f} std={np.std(f1s):.3f} "
          f"min={np.min(f1s):.3f} max={np.max(f1s):.3f}")