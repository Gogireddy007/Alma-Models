#!/bin/zsh
cd /Users/srimanarayana/Timeseries
until grep -q "^EXIT" runs/fin/search/logs/loop_wm.out; do sleep 20; done
nice -n 10 .venv/bin/python tools/fin/search/loop.py --pillar cl --budget 80 --until 02:35 > runs/fin/search/logs/loop_cl.out 2>&1; echo "EXIT $?" >> runs/fin/search/logs/loop_cl.out
nice -n 10 .venv/bin/python tools/fin/search/loop.py --pillar rl --budget 80 --until 03:05 > runs/fin/search/logs/loop_rl.out 2>&1; echo "EXIT $?" >> runs/fin/search/logs/loop_rl.out
nice -n 10 .venv/bin/python tools/fin/search/refs.py > runs/fin/search/logs/refs.out 2>&1; echo "EXIT $?" >> runs/fin/search/logs/refs.out
echo "AFTER_WM DONE $(date)" >> runs/fin/search/logs/after_wm.out
