cd /Users/srimanarayana/Timeseries
B='"bs": 32, "lr": 0.002, "d": 32, "enc_d": "mlp", "coin_drop": 0.5'
nice -n 10 .venv/bin/python tools/fin/search_b/holdout_b.py --name c6_b2_hconv_ens3 --seeds 3 --cfg "{\"steps\": 400, $B, \"enc_h\": \"conv\"}"
nice -n 10 .venv/bin/python tools/fin/search_b/holdout_b.py --name c6_b2_moe4_s300 --seeds 3 --cfg "{\"steps\": 300, $B, \"head\": \"moe\", \"K\": 4}"
nice -n 10 .venv/bin/python tools/fin/search_b/holdout_b.py --name c6_b2_hconv_s300 --seeds 3 --cfg "{\"steps\": 300, $B, \"enc_h\": \"conv\"}"
echo F1DONE
