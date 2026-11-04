#!/bin/bash
cd "/Users/yugandharreddygogireddy/Documents/Alma Models /repo"
while IFS='|' read -r seed cfg tag; do
  echo "OMP_NUM_THREADS=3 nice -n 10 .venv/bin/python tools/gas/search/run.py bt --cfg '$cfg' --seed $seed --tag $tag > ../audit/logs/$tag.log 2>&1; echo DONE $tag \$(grep -c Traceback ../audit/logs/$tag.log)"
done < ../audit/gas_jobs.txt | xargs -P 3 -I{} bash -c "{}"
