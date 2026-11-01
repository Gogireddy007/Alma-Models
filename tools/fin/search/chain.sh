#!/bin/zsh
# sequential command queue (one job at a time): each non-comment line of $1 is a command run from the repo root
cd /Users/srimanarayana/Timeseries
n=0
while read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  n=$((n+1))
  echo "=== $(date) [$n] $line"
  eval "nice -n 10 $line" > runs/fin/search/logs/$(basename $1 .txt)_$n.log 2>&1
  echo "--- exit $? ; $(grep -hE 'P1 all|SR_heldout|S_C|\{' runs/fin/search/logs/$(basename $1 .txt)_$n.log | tail -2)"
done < $1
echo "=== CHAIN DONE $(date)"
