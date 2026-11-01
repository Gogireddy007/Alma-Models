#!/bin/zsh
# clock stops in local time; DST ended at 02:00 EDT, local clock is now EST (02:35 EDT = 01:35 EST, 03:05 EDT = 02:05 EST)
cd /Users/srimanarayana/Timeseries
nice -n 10 .venv/bin/python tools/fin/search/loop.py --pillar cl --budget 80 --until 01:30 > runs/fin/search/logs/loop_cl.out 2>&1; echo "EXIT $?" >> runs/fin/search/logs/loop_cl.out
nice -n 10 .venv/bin/python tools/fin/search/loop.py --pillar rl --budget 80 --until 01:55 > runs/fin/search/logs/loop_rl.out 2>&1; echo "EXIT $?" >> runs/fin/search/logs/loop_rl.out
nice -n 10 .venv/bin/python tools/fin/search/refs.py > runs/fin/search/logs/refs.out 2>&1; echo "EXIT $?" >> runs/fin/search/logs/refs.out
nice -n 10 .venv/bin/python tools/fin/search/finalize.py holdout_wm > runs/fin/search/logs/holdout_wm.out 2>&1; echo "EXIT $?" >> runs/fin/search/logs/holdout_wm.out
echo "AFTER_WM2 DONE $(date)" >> runs/fin/search/logs/after_wm.out
