"""B freeze-refit cache invariance (auditor): rebuild refit4_2025-06-30 into scratch with every open row dated
> 2025-06-30 (VAL) AND every held-out-coin row randomised; training inputs/targets of the 16 training coins must equal
the cache used for wm_b_*_refit.pt. DEV only."""
import os, sys
import torch  # noqa: F401
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_rt as R
sys.path.insert(0, os.path.join(R.FIN, "search_b")); sys.path.insert(0, os.path.join(R.FIN, "search"))
import feats_b as FB
ref = np.load(os.path.join(FB.CACHEB, "refit4_2025-06-30.npz"))
FB.CACHEB = os.environ.get("RT_SCRATCH", "/tmp") + "/rt_b_refit"
os.makedirs(FB.CACHEB, exist_ok=True)
p = os.path.join(FB.CACHEB, "refit4_2025-06-30.npz")
if os.path.exists(p):
    os.remove(p)
import harness as H
_orig = R.DT.load
rng = np.random.default_rng(5)
_, ho = R.DT.universe()
import pandas as pd
def load(kind="crypto_daily", allow_test=False):
    df = _orig(kind, False)
    rows = df["symbol"].isin(ho).to_numpy().copy() if kind.startswith("crypto") else np.zeros(len(df), bool)
    c = pd.Timestamp("2025-06-30")
    rows |= ((df["t"] >= c + pd.Timedelta(days=1)) if kind == "crypto_hourly" else (df["t"] > c)).to_numpy()
    return R.corrupt(df, rows, rng, "rand")
R.DT.load = load
D = FB.build_cache(end="2025-06-30")
tm = ref["train_mask"]
bad = []
for k in ("Xtr", "tl", "y", "blk", "mblk", "days"):
    a, b = ref[k], D[k]
    if k in ("tl", "y", "blk", "mblk"):
        a, b = a[:, tm], b[:, tm]
    if not np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else not np.array_equal(a, b):
        bad.append(k)
for k in ("hz", "har", "aux", "A", "ok"):
    if not np.array_equal(ref[k][:, tm], D[k][:, tm], equal_nan=(ref[k].dtype.kind == "f")):
        bad.append(k + "(train cols)")
print(("OK   " if not bad else "FAIL ") + f"B refit cache: training-coin inputs/targets unchanged under VAL + held-out randomisation {bad}")
