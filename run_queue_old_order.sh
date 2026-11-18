#!/usr/bin/env bash
# run_queue.sh -- resumable queue for ALL remaining paper runs (frozen settings).
#
# Run from the activated venv, in the foreground:
#     bash run_queue.sh
# See what is left without running anything:
#     DRY_RUN=1 bash run_queue.sh
#
# Safe to re-run after any interruption: a run whose VALID result file exists
# is skipped, so at most the run in progress is lost.
#
# Before every run it checks: 4 fisco nodes up, RPC answering, >=20G free,
# no other main_fisco / fedrec_baselines process running (it would skew
# timing and double-use the chain). Any failure stops the queue.
#
# Frozen settings: --threshold 0.23, audit_base 0.10, audit_window 10,
# flag_scale 30 (all script defaults except the threshold, passed here).

set -u -o pipefail
shopt -s nullglob
cd "$(dirname "$0")"
mkdir -p logs results

export RPC_URL="${RPC_URL:-http://127.0.0.1:8545}"
# Detector v2 (self-referenced). The threshold is set ONLY after the development
# runs (seed 7) have been analysed; pass it explicitly:  THR=0.xx bash run_queue.sh
THR=${THR:-}
if [ -z "$THR" ]; then echo '[queue] set THR=<frozen threshold> first (see dev-run analysis)'; exit 1; fi
# Guard: the detector-v2 code with the 'effective attack' bookkeeping must be in this folder.
if ! grep -q "effective_detected" main_fisco_real_v11.py || ! grep -q "self_hist" byzantine_engine.py; then
  echo '[queue] detector-v2 code NOT found in this folder (old files in place?). Aborting.'; exit 2
fi

# dataset | seed | result-tag | extra flags
# ORDER = PRIORITY. Stop the queue at any tier boundary and the result set is
# still coherent for the paper.
#   Tier 1: headline static runs on both datasets + MNIST adaptive/collusion
#   Tier 2: Fashion-MNIST seed 42 adaptive/collusion + full MNIST ratio sweep
#   Tier 3: Fashion-MNIST seed 42 ratio sweep
#   Tier 4: remaining Fashion-MNIST seeds (43, 44) for adaptive/collusion/sweep
QUEUE=(
  # ---- Tier 1 ----
  "MNIST|42|static|"
  "MNIST|43|static|"
  "MNIST|44|static|"
  "Fashion-MNIST|42|static|"
  "Fashion-MNIST|43|static|"
  "Fashion-MNIST|44|static|"
  "MNIST|42|adaptive|--adaptive_attacker"
  "MNIST|43|adaptive|--adaptive_attacker"
  "MNIST|44|adaptive|--adaptive_attacker"
  "MNIST|42|collusion|--collusion"
  "MNIST|43|collusion|--collusion"
  "MNIST|44|collusion|--collusion"
  # ---- Tier 2 ----
  "Fashion-MNIST|42|adaptive|--adaptive_attacker"
  "Fashion-MNIST|42|collusion|--collusion"
  "MNIST|42|byz40|--byz_ratio 0.40"
  "MNIST|43|byz40|--byz_ratio 0.40"
  "MNIST|44|byz40|--byz_ratio 0.40"
  "MNIST|42|byz0|--byz_ratio 0.0"
  "MNIST|43|byz0|--byz_ratio 0.0"
  "MNIST|44|byz0|--byz_ratio 0.0"
  # ---- Tier 3 ----
  "Fashion-MNIST|42|byz40|--byz_ratio 0.40"
  "Fashion-MNIST|42|byz0|--byz_ratio 0.0"
  # ---- Tier 4 ----
  "Fashion-MNIST|43|adaptive|--adaptive_attacker"
  "Fashion-MNIST|44|adaptive|--adaptive_attacker"
  "Fashion-MNIST|43|collusion|--collusion"
  "Fashion-MNIST|44|collusion|--collusion"
  "Fashion-MNIST|43|byz40|--byz_ratio 0.40"
  "Fashion-MNIST|44|byz40|--byz_ratio 0.40"
  "Fashion-MNIST|43|byz0|--byz_ratio 0.0"
  "Fashion-MNIST|44|byz0|--byz_ratio 0.0"
)

preflight() {
  local n avail
  n=$(docker ps --format '{{.Names}}' | grep -c '^fisco-node')
  if [ "$n" -ne 4 ]; then echo "PREFLIGHT FAIL: $n/4 fisco nodes running"; return 1; fi
  if ! curl -s -m 10 -X POST \
        --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
        "$RPC_URL" | grep -q '"result"'; then
    echo "PREFLIGHT FAIL: RPC not answering at $RPC_URL"; return 1
  fi
  avail=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "${avail:-0}" -lt 20 ]; then echo "PREFLIGHT FAIL: only ${avail}G free on /"; return 1; fi
  if pgrep -f "main_fisco_real_v11" >/dev/null; then
    echo "PREFLIGHT FAIL: another main_fisco run is already running."; return 1
  fi
  if [ "${ALLOW_CONTENTION:-0}" != "1" ] && pgrep -f fedrec_baselines >/dev/null; then
    echo "PREFLIGHT FAIL: fedrec_baselines.py is running (skews timing). ALLOW_CONTENTION=1 to override."
    return 1
  fi
  return 0
}

# A result file only counts if it holds exactly one dataset entry AND was run
# with the frozen threshold (filename carries thr0.23).
valid_result() {
  python3 -I -c '
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if len(d.get("datasets", [])) == 1 else 1)
' "$1" 2>/dev/null
}

find_done() {  # dataset seed tag
  local f
  for f in results/fisco_v11_"$1"_ep100_seed"$2"_thr${THR}_full_"$3"-sr_*.json; do
    if valid_result "$f"; then echo "$f"; return 0; fi
  done
  return 1
}

for entry in "${QUEUE[@]}"; do
  IFS='|' read -r DS SEED TAG FLAGS <<< "$entry"

  if done_file=$(find_done "$DS" "$SEED" "$TAG"); then
    echo "[queue] SKIP  $DS seed=$SEED $TAG (done: $done_file)"
    continue
  fi
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[queue] WOULD RUN  $DS seed=$SEED $TAG  [$FLAGS]"
    continue
  fi
  if ! preflight; then
    echo "[queue] STOPPED before $DS seed=$SEED $TAG"; exit 1
  fi

  echo "[queue] $(date '+%F %T') START $DS seed=$SEED $TAG"
  LOG="logs/queue_${DS}_seed${SEED}_${TAG}_$(date +%Y%m%d_%H%M%S).log"
  # shellcheck disable=SC2086
  python3 -u main_fisco_real_v11.py --dataset "$DS" --episodes 100 \
      --seed "$SEED" --threshold "$THR" --ablation full $FLAGS 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}

  if [ "$rc" -ne 0 ]; then
    echo "[queue] FAIL: exit code $rc for $DS seed=$SEED $TAG (log: $LOG)"; exit 1
  fi
  if ! find_done "$DS" "$SEED" "$TAG" >/dev/null; then
    echo "[queue] FAIL: no valid result file for $DS seed=$SEED $TAG"
    for f in results/fisco_v11_"$DS"_ep100_seed"$SEED"_thr${THR}_full_"$TAG"-sr_*.json; do
      mv "$f" "$f.failed" && echo "[queue] moved invalid file to $f.failed"
    done
    exit 1
  fi
  echo "[queue] $(date '+%F %T') DONE  $DS seed=$SEED $TAG"
done

echo "[queue] ALL DONE"