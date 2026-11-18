#!/usr/bin/env bash
# run_refseed.sh -- ONE extra run on a DIFFERENT reference batch (MNIST seed 42 static, 50 episodes, frozen thr 0.15, full system).
# Uses a COPY of the main script (main_refseed.py, REF_SEED env var); the main queue and main script are untouched.
# Waits until run_reportlog.sh has exited and no experiment is running, then starts. Result JSON goes to results_reportlog/.
cd "$(dirname "$0")" || exit 1
REF_SEED="${REF_SEED:-1234}"
[ -f main_refseed.py ] || { echo "[refseed] ABORT: main_refseed.py missing"; exit 1; }
grep -q REF_SEED main_refseed.py || { echo "[refseed] ABORT: main_refseed.py has no REF_SEED"; exit 1; }
mkdir -p logs results results_reportlog
while pgrep -f "run_reportlog.sh" >/dev/null || pgrep -f "main_fisco_real_v11.py" >/dev/null; do sleep 60; done
n=$(docker ps --format '{{.Names}}' | grep -c '^fisco-node')
[ "$n" -eq 4 ] || { echo "[refseed] ABORT: only $n fisco nodes up"; exit 1; }
touch logs/.marker_refseed
echo "[refseed] START refB seed42 static ep=50 REF_SEED=$REF_SEED $(date)"
REF_SEED="$REF_SEED" REPORT_LOG="results/report_log_MNIST_seed42_static_refB_ep50.jsonl" UPLOAD_LOG="" python3 -u main_refseed.py \
    --dataset MNIST --episodes 50 --seed 42 --threshold 0.15 --ablation full > logs/refseed_MNIST42.log 2>&1 \
  && { find results -maxdepth 1 -name 'fisco_v11_*.json' -newer logs/.marker_refseed -exec mv -t results_reportlog {} + ; touch logs/refseed.done; echo "[refseed] DONE $(date)"; } \
  || echo "[refseed] FAILED (see logs/refseed_MNIST42.log)"
