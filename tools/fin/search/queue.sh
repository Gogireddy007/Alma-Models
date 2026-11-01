#!/bin/zsh
# sequential queue: each line of $1 is the argument list of train_wm.py (one job at a time)
cd /Users/srimanarayana/Timeseries
while read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  tag=$(echo $line | tr ' -' '__')
  echo "=== $(date) $line"
  nice -n 10 .venv/bin/python tools/fin/search/train_wm.py ${=line} > runs/fin/search/logs/wm${tag}.log 2>&1
  tail -1 runs/fin/search/logs/wm${tag}.log
done < $1
