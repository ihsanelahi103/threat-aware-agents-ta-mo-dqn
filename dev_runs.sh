#!/usr/bin/env bash
# Development runs for detector v2 (seed 7, 20 episodes, both datasets).
# Run detached:  nohup bash dev_runs.sh > logs/dev_all.log 2>&1 &
set -u -o pipefail
cd "$(dirname "$0")"
source venv/bin/activate
export RPC_URL=${RPC_URL:-http://127.0.0.1:8545}
mkdir -p logs results
# Guard: refuse to run unless the detector-v2 code is really in this folder.
if ! grep -q "DETECTOR_MODE" main_fisco_real_v11.py || ! grep -q "self_hist" byzantine_engine.py; then
  echo "[dev] detector-v2 code NOT found in this folder (v1 files in place?). Aborting."; exit 2
fi
if pgrep -f "python3 -u main_fisco_real_v11" >/dev/null; then
  echo "[dev] another experiment is running; aborting"; exit 1
fi
for DS in MNIST Fashion-MNIST; do
  if ls results/fisco_v11_${DS}_ep20_seed7_thr0.12_full_static-sr_*.json >/dev/null 2>&1; then
    echo "[dev] $DS already done, skipping"; continue
  fi
  echo "[dev] $(date '+%F %T') START $DS"
  python3 -u main_fisco_real_v11.py --dataset "$DS" --episodes 20 --seed 7 \
      --threshold 0.12 --ablation full 2>&1 | tee "logs/dev_${DS}_seed7.log"
  rc=${PIPESTATUS[0]}
  echo "[dev] $(date '+%F %T') END $DS rc=$rc"
  [ "$rc" -eq 0 ] || { echo "[dev] failed, stopping"; exit "$rc"; }
done
echo "[dev] ALL DONE"