#!/bin/bash
# usage: run_ssm.sh <abl none|direct> <seed>   (recipe = stored cfg of data/ckpt/wm_ssm_none_s0.pt)
cd "/Users/yugandharreddygogireddy/Documents/Alma Models /repo"
env TSAR_AUG=0.5 TSAR_BENCH=1 TSAR_GIFT_CLEAN=1 TSAR_RUN=aud_ssm_$1_s$2 TSAR_SEED=$2 TSAR_STEPS=2500 TSAR_SYN3=1 TSAR_SYN5=1 \
    TSAR_SYN_CURR=0.9 TSAR_SYN_FRAC=0.3 TSAR_WM=ssm TSAR_WM_ABL=$1 \
    /usr/bin/time -p .venv/bin/python train_wm.py >> ../audit/logs/aud_ssm_$1_s$2.log 2>&1
