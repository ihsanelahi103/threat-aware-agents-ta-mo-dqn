#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
mkdir -p logs results results_reportlog
n=$(docker ps --format '{{.Names}}' | grep -c '^fisco-node'); [ "$n" -eq 4 ] || { echo "ABORT: $n nodes"; exit 1; }
touch logs/.marker_puredqn
echo "[puredqn] START $(date)"
REPORT_LOG="results/report_log_MNIST_seed42_audit0_ep50.jsonl" UPLOAD_LOG="" python3 -u main_fisco_real_v11.py \
  --dataset MNIST --episodes 50 --seed 42 --threshold 0.15 --ablation full --audit_base 0 > logs/puredqn.log 2>&1 \
 && { find results -maxdepth 1 -name 'fisco_v11_*.json' -newer logs/.marker_puredqn -exec mv -t results_reportlog {} + ; echo "[puredqn] DONE $(date)"; } \
 || echo "[puredqn] FAILED (see logs/puredqn.log)"
