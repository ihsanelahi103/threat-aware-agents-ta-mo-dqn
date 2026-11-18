#!/usr/bin/env bash
# run_reportlog.sh (v3) -- raw-report logging runs for the baseline comparison, thr 0.15, full system.
# Horizon: EP episodes (default 100) for test runs; DEV_EP (default 50) for the dev run. Priority order, least important last:
#   MNIST 42 static, MNIST 7 dev, MNIST 43 static, MNIST 44 static, MNIST 42/43/44 adaptive (optional Fashion with ADD_FASHION=1).
# No new run starts after NO_START_AFTER (machine clock). Resumable via logs/reportlog_<tag>.done.
# Result JSONs of these runs are moved to results_reportlog/ so aggregation globs over results/ never see them.
cd "$(dirname "$0")" || exit 1
EP="${EP:-100}"; DEV_EP="${DEV_EP:-50}"
FASHION_NAME="${FASHION_NAME:-FashionMNIST}"      # must match the --dataset value your script accepts
NO_START_AFTER="${NO_START_AFTER:-2026-11-16 20:30}"
MAIN=main_fisco_real_v11.py
mkdir -p logs results results_reportlog
grep -q REPORT_LOG "$MAIN" || { echo "[reportlog] ABORT: report_log.patch not applied"; exit 1; }
grep -q UPLOAD_LOG "$MAIN" || { echo "[reportlog] ABORT: upload_log.patch not applied"; exit 1; }
grep -q "$FASHION_NAME" "$MAIN" || echo "[reportlog] WARNING: '$FASHION_NAME' not found in $MAIN; check the dataset name (Fashion runs will fail otherwise)"
while pgrep -f "$MAIN" >/dev/null; do sleep 60; done
# dataset|seed|mode|flags|episodes
RUNS=("MNIST|42|static||$EP" "MNIST|7|static||$DEV_EP" "MNIST|43|static||$EP" "MNIST|44|static||$EP"
      "MNIST|42|adaptive|--adaptive_attacker|$EP" "MNIST|43|adaptive|--adaptive_attacker|$EP" "MNIST|44|adaptive|--adaptive_attacker|$EP")
# Optional Fashion-MNIST runs are NOT in the default plan; add with ADD_FASHION=1 (dev seed 7, then seed 42 static)
[ "${ADD_FASHION:-0}" = "1" ] && RUNS+=("$FASHION_NAME|7|static||$DEV_EP" "$FASHION_NAME|42|static||$EP")
LIMIT=$(date -d "$NO_START_AFTER" +%s)
for R in "${RUNS[@]}"; do
  IFS='|' read -r DS SEED MODE FLAGS E <<< "$R"
  TAG="${DS}_seed${SEED}_${MODE}"
  [ -f "logs/reportlog_${TAG}.done" ] && { echo "[reportlog] SKIP $TAG"; continue; }
  [ "$(date +%s)" -gt "$LIMIT" ] && { echo "[reportlog] STOP: past NO_START_AFTER, not starting $TAG"; exit 0; }
  n=$(docker ps --format '{{.Names}}' | grep -c '^fisco-node')
  [ "$n" -eq 4 ] || { echo "[reportlog] ABORT: only $n fisco nodes up"; exit 1; }
  RL="results/report_log_${TAG}_ep${E}.jsonl"; rm -f "$RL"
  touch "logs/.marker_${TAG}"
  echo "[reportlog] START $TAG ep=$E $(date)"
  REPORT_LOG="$RL" UPLOAD_LOG="" python3 -u "$MAIN" \
      --dataset "$DS" --episodes "$E" --seed "$SEED" --threshold 0.15 --ablation full $FLAGS \
      > "logs/reportlog_${TAG}.log" 2>&1 \
    && { find results -maxdepth 1 -name 'fisco_v11_*.json' -newer "logs/.marker_${TAG}" -exec mv -t results_reportlog {} + ;
         touch "logs/reportlog_${TAG}.done"; echo "[reportlog] DONE  $TAG $(date)"; } \
    || { echo "[reportlog] FAILED $TAG (see logs/reportlog_${TAG}.log)"; exit 1; }
done
echo "[reportlog] ALL DONE"
