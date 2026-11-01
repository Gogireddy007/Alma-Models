#!/bin/bash
# usage: q.sh queue_file [parallel]; each line = args to run.py (logs in runs/power/search/logs/<name>.log)
cd "$(dirname "$0")/../../.."
P=${2:-2}
grep -v '^#' "$1" | grep -v '^$' | while read -r line; do
  name=$(echo "$line" | awk '{print $2}')
  while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 2; done
  nice -n 10 .venv/bin/python tools/power/search/run.py $line > runs/power/search/logs/$name.log 2>&1 &
done
wait
