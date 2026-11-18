#!/usr/bin/env bash
# run_extra.sh -- two extra 50-episode report-log runs, sequential:
#   1) FashionMNIST seed 42 static, original reference batch  (main_fisco_real_v11.py, patched)
#   2) MNIST seed 43 static, DIFFERENT reference batch REF_SEED=777 (main_refseed.py)
# Waits until no experiment process is running. Resumable via logs/extra_*.done. Result JSONs go to results_reportlog/.
cd "$(dirname "$0")" || exit 1
mkdir -p logs results results_reportlog
[ "$(grep -c REPORT_LOG main_fisco_real_v11.py)" -ge 3 ] || { echo "[extra] ABORT: REPORT_LOG patch missing"; exit 1; }
[ -f main_refseed.py ] && grep -q REF_SEED main_refseed.py || { echo "[extra] ABORT: main_refseed.py missing/unpatched"; exit 1; }
while pgrep -f "run_reportlog.sh" >/dev/null || pgrep -f "run_refseed.sh" >/dev/null || pgrep -f "main_fisco_real_v11.py" >/dev/null || pgrep -f "main_refseed.py" >/dev/null; do sleep 60; done

run_one() {  # tag script dataset seed reflog refseed
  tag="$1"; script="$2"; ds="$3"; seed="$4"; rl="$5"; rs="$6"
  [ -f "logs/extra_${tag}.done" ] && { echo "[extra] SKIP $tag"; return 0; }
  n=$(docker ps --format '{{.Names}}' | grep -c '^fisco-node')
  [ "$n" -eq 4 ] || { echo "[extra] ABORT: only $n fisco nodes up"; return 1; }
  touch "logs/.marker_extra_${tag}"
  echo "[extra] START $tag $(date)"
  REF_SEED="$rs" REPORT_LOG="$rl" UPLOAD_LOG="" python3 -u "$script" \
      --dataset "$ds" --episodes 50 --seed "$seed" --threshold 0.15 --ablation full > "logs/extra_${tag}.log" 2>&1 \
    && { find results -maxdepth 1 -name 'fisco_v11_*.json' -newer "logs/.marker_extra_${tag}" -exec mv -t results_reportlog {} + ; touch "logs/extra_${tag}.done"; echo "[extra] DONE  $tag $(date)"; } \
    || { echo "[extra] FAILED $tag (see logs/extra_${tag}.log)"; return 1; }
}
run_one FashionMNIST_seed42_static main_fisco_real_v11.py FashionMNIST 42 "results/report_log_FashionMNIST_seed42_static_ep50.jsonl" ""
run_one MNIST_seed43_static_refC main_refseed.py MNIST 43 "results/report_log_MNIST_seed43_static_refC_ep50.jsonl" 777
echo "[extra] ALL DONE $(date)"
