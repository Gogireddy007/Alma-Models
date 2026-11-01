#!/bin/zsh
cd /Users/srimanarayana/Timeseries
L=runs/fin/search/logs
PY="nice -n 10 .venv/bin/python"
until grep -q "CHAIN3 DONE" $L/chain3.out; do sleep 15; done
${=PY} tools/fin/search/decide.py wm > $L/decide_wm.out 2>&1
WK=$(python3 -c "import json;print(json.load(open('runs/fin/search/wm_final.json'))['key'])")
echo "=== $(date -u) wm final $WK"
${=PY} tools/fin/search/finalize.py holdout_cl > $L/holdout_cl.out 2>&1; echo "EXIT $?" >> $L/holdout_cl.out
${=PY} tools/fin/search/decide.py cl > $L/decide_cl.out 2>&1
${=PY} tools/fin/search/finalize.py holdout_rl > $L/holdout_rl.out 2>&1; echo "EXIT $?" >> $L/holdout_rl.out
${=PY} tools/fin/search/decide.py rl > $L/decide_rl.out 2>&1
CK=$(python3 -c "import json;print(json.load(open('runs/fin/search/cl_final.json'))['key'])")
RK=$(python3 -c "import json;print(json.load(open('runs/fin/search/rl_final.json'))['key'])")
echo "=== $(date -u) cl $CK rl $RK"
${=PY} tools/fin/search/finalize.py seeds_wm --key $WK > $L/seeds_wm.out 2>&1; echo "EXIT $?" >> $L/seeds_wm.out
echo "=== $(date -u) seeds_wm done"
${=PY} tools/fin/search/finalize.py refit_wm --key $WK > $L/refit_wm.out 2>&1; echo "EXIT $?" >> $L/refit_wm.out
echo "=== $(date -u) refit_wm done"
${=PY} tools/fin/search/finalize.py seeds_cl --key $CK > $L/seeds_cl.out 2>&1; echo "EXIT $?" >> $L/seeds_cl.out
${=PY} tools/fin/search/finalize.py seeds_rl --key $RK > $L/seeds_rl.out 2>&1; echo "EXIT $?" >> $L/seeds_rl.out
echo "=== $(date -u) seeds cl/rl done"
${=PY} tools/fin/search/finalize.py refit_rl --key $RK > $L/refit_rl.out 2>&1; echo "EXIT $?" >> $L/refit_rl.out
echo "CHAIN4 DONE $(date -u)"
