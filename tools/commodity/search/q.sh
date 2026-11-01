#!/bin/bash
# run experiment lines from a file sequentially: each line = args to run_exp.py
cd /Users/srimanarayana/Timeseries
LOG=runs/commodity/search/logs; mkdir -p $LOG
while IFS= read -r line; do
  [ -z "$line" ] && continue; [[ "$line" == \#* ]] && continue
  name=$(echo "$line" | sed -E 's/.*--name ([^ ]+).*/\1/')
  echo "$(date +%H:%M) start $name"
  eval nice -n 10 python3 tools/commodity/search/run_exp.py $line > $LOG/$name.log 2>&1
  tail -1 $LOG/$name.log
done < "$1"
