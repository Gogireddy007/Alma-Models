#!/bin/zsh
# Am2 sequence (UTC stops). One job at a time.
cd /Users/srimanarayana/Timeseries
L=runs/fin/search/logs
until ! pgrep -f "pillar cl" >/dev/null; do sleep 10; done
echo "=== $(date -u) leakgate dow" 
CK=$(nice -n 10 .venv/bin/python tools/fin/search/leakgate_dow.py 2>$L/leakgate_dow.err | tail -1)
echo "ckpt $CK"
nice -n 10 .venv/bin/python tools/fin/search/test_leak.py --family rssm --ckpt $CK > $L/leak_rssm_dow.out 2>&1
rc=$?
cp runs/fin/search/test_leak_rssm.txt runs/fin/search/test_leak_rssm_dow.txt
echo "leak rssm dow exit $rc $(tail -1 $L/leak_rssm_dow.out)"
if [[ $rc -eq 0 ]]; then
  echo "=== $(date -u) WM extension"
  nice -n 10 .venv/bin/python tools/fin/search/loop.py --pillar wm --budget 30 --until 07:05 --utc --force dow=True > $L/loop_wm_ext.out 2>&1; echo "EXIT $?" >> $L/loop_wm_ext.out
fi
echo "=== $(date -u) RL loop"
nice -n 10 .venv/bin/python tools/fin/search/loop.py --pillar rl --budget 80 --until 07:35 --utc > $L/loop_rl.out 2>&1; echo "EXIT $?" >> $L/loop_rl.out
echo "=== $(date -u) agent refs F1"
nice -n 10 .venv/bin/python tools/fin/search/refs.py --models buy_hold,tsmom_1,tsmom_7,tsmom_30,meanrev_1,voltgt_tsmom,lgbm_policy > $L/refs.out 2>&1; echo "EXIT $?" >> $L/refs.out
echo "=== $(date -u) holdout_wm"
nice -n 10 .venv/bin/python tools/fin/search/finalize.py holdout_wm > $L/holdout_wm.out 2>&1; echo "EXIT $?" >> $L/holdout_wm.out
echo "CHAIN3 DONE $(date -u)"
