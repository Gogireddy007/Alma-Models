#!/bin/zsh
# Alma-Live Gas shadow loop (PAPER ONLY): one hash-chained book per model.
#   tools/live/shadow.sh init|decide|settle            live books (decide: 15:30-15:50 ET; settle: after 18:00 ET + px update)
#   tools/live/shadow.sh decide --sim --day 2026-10-30 dry run (sim books under runs/live/paper_sim; never count)
cd "$(dirname "$0")/../.." || exit 1
cmd="$1"; shift
MODELS=(${=${SHADOW_MODELS:-buy_hold short_ung_carry tsmom_20 seasonal weather_rev storage_surprise fc_ewma fc_hist}})
for m in $MODELS; do
  nice -n 10 .venv/bin/python tools/live/paper.py "$cmd" --book "shadow_$m" --model "$m" --instruments UNG "$@" || echo "shadow_$m: $cmd failed"
done
