#!/usr/bin/env bash
# run_privfix.sh -- ONE run with the unnoised accuracy field REMOVED from the public ledger metadata
# (MNIST seed 42 static, 50 episodes, frozen thr 0.15, full system), with the upload log for the linking attack.
# Uses a COPY of the main script (main_privfix.py). Waits until the extra queue and any experiment have exited.
cd "$(dirname "$0")" || exit 1
[ -f main_privfix.py ] || { echo "[privfix] ABORT: main_privfix.py missing (run: python3 make_privfix.py)"; exit 1; }
grep -q "dp_eps':round" main_privfix.py && { echo "[privfix] ABORT: dp_eps still public in main_privfix.py"; exit 1; }
grep -q UPLOAD_LOG main_privfix.py || { echo "[privfix] ABORT: UPLOAD_LOG patch missing"; exit 1; }
mkdir -p logs results results_reportlog
while pgrep -f "run_extra.sh" >/dev/null || pgrep -f "run_fashion.sh" >/dev/null || pgrep -f "run_refseed.sh" >/dev/null || pgrep -f "run_reportlog.sh" >/dev/null || pgrep -f "main_refseed.py" >/dev/null || pgrep -f "main_fisco_real_v11.py" >/dev/null; do sleep 60; done
n=$(docker ps --format '{{.Names}}' | grep -c '^fisco-node')
[ "$n" -eq 4 ] || { echo "[privfix] ABORT: only $n fisco nodes up"; exit 1; }
touch logs/.marker_privfix
echo "[privfix] START MNIST seed42 static ep=50, acc field removed $(date)"
UPLOAD_LOG="results/upload_log_MNIST_seed42_privfix_ep50.jsonl" REPORT_LOG="" python3 -u main_privfix.py \
    --dataset MNIST --episodes 50 --seed 42 --threshold 0.15 --ablation full > logs/privfix_MNIST42.log 2>&1 \
  && { find results -maxdepth 1 -name 'fisco_v11_*.json' -newer logs/.marker_privfix -exec mv -t results_reportlog {} + ; touch logs/privfix.done; echo "[privfix] DONE $(date)"; } \
  || echo "[privfix] FAILED (see logs/privfix_MNIST42.log)"
