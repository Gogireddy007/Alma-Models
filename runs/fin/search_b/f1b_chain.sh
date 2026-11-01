cd /Users/srimanarayana/Timeseries
CK=runs/fin/search_b/ckpt/c6_b2_moe4_s300_F1.pt
CFG='{"steps": 300, "bs": 32, "lr": 0.002, "d": 32, "enc_d": "mlp", "coin_drop": 0.5, "head": "moe", "K": 4}'
nice -n 10 .venv/bin/python tools/fin/search_b/holdout_b.py --name p_vt_k1_c4.0 --ckpt $CK --cfg "$CFG" --agent '{"k": 1, "c": 4.0, "band": 0.0}'
nice -n 10 .venv/bin/python tools/fin/search_b/holdout_b.py --name p_vt_k1_c2.0 --ckpt $CK --cfg "$CFG" --agent '{"k": 1, "c": 2.0, "band": 0.0}'
nice -n 10 .venv/bin/python tools/fin/search_b/test_leak_b.py --ckpts runs/fin/search_b/ckpt/c6_b2_moe4_s300_F1.pt,runs/fin/search_b/ckpt/c6_b2_hconv_ens3_F1.pt --agent $CK --cl $CK
echo LEAKDONE
nice -n 10 .venv/bin/python tools/fin/search_b/cl2_b.py runs/fin/search_b/q_cl2s.jsonl
echo CLDONE
