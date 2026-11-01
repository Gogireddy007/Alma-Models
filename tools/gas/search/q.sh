#!/bin/zsh
# Sequential queue: each line "tag<TAB>seed<TAB>cfg-json". One training job at a time; waits while swap > 5.2 GB
# and free memory is low. Every run (crash included) is logged by run.py to runs/gas/search/results.tsv.
cd /Users/srimanarayana/Timeseries
while IFS=$'\t' read -r tag seed cfg; do
  [[ -z "$tag" || "$tag" == \#* ]] && continue
  nice -n 10 .venv/bin/python -W ignore tools/gas/search/run.py bt --tag "$tag" --seed "$seed" --cfg "$cfg" \
    > runs/gas/search/logs/$tag.log 2>&1
  tail -1 runs/gas/search/logs/$tag.log | cut -c1-400
done < "$1"
