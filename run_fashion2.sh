#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
while pgrep -f "run_extra.sh" >/dev/null || pgrep -f "run_privfix.sh" >/dev/null || pgrep -f "main_privfix.py" >/dev/null || pgrep -f "main_refseed.py" >/dev/null || pgrep -f "main_fisco_real_v11.py" >/dev/null; do sleep 60; done
n=$(docker ps --format '{{.Names}}' | grep -c '^fisco-node')
[ "$n" -eq 4 ] || { echo "[fashion2] ABORT: only $n fisco nodes up"; exit 1; }
touch logs/.marker_fashion2
echo "[fashion2] START Fashion-MNIST seed42 static ep=50 $(date)"
REPORT_LOG="results/report_log_FashionMNIST_seed42_static_ep50.jsonl" UPLOAD_LOG="" python3 -u main_fisco_real_v11.py \
    --dataset Fashion-MNIST --episodes 50 --seed 42 --threshold 0.15 --ablation full > logs/fashion2.log 2>&1 \
  && { find results -maxdepth 1 -name 'fisco_v11_*.json' -newer logs/.marker_fashion2 -exec mv -t results_reportlog {} + ; touch logs/fashion2.done; echo "[fashion2] DONE $(date)"; } \
  || echo "[fashion2] FAILED (see logs/fashion2.log)"
