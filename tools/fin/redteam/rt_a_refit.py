"""A freeze-refit training set (auditor): rebuild ds_w28_2025-06-30 (finalize.dataset(DEV_FIT) path) into scratch with
every open row > 2025-06-30 (VAL) AND every held-out-coin row randomised -> must equal the cached file used for
frozen_*.pt. Also I1 on frozen ckpts present. DEV only."""
import os, sys, glob
import torch  # noqa: F401
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_rt as R
import rt_arch as RA
sys.path.insert(0, os.path.join(R.FIN, "search"))
import common as C
import wm_core as WC
tr, _ = R.DT.universe()
fn = f"ds_w{WC.W_MAX}_2025-06-30.npz"
ref = np.load(os.path.join(C.CACHE, fn))
C.CACHE = os.path.join(os.environ.get("RT_SCRATCH", "/tmp"), "rt_a_refit"); os.makedirs(C.CACHE, exist_ok=True)
p = os.path.join(C.CACHE, fn)
if os.path.exists(p):
    os.remove(p)
_orig = R.DT.load
rng = np.random.default_rng(8)
_, ho = R.DT.universe()
import pandas as pd
def load(kind="crypto_daily", allow_test=False):
    df = _orig(kind, False)
    rows = df["symbol"].isin(ho).to_numpy().copy() if kind.startswith("crypto") else np.zeros(len(df), bool)
    c = pd.Timestamp("2025-06-30")
    rows |= ((df["t"] >= c + pd.Timedelta(days=1)) if kind == "crypto_hourly" else (df["t"] > c)).to_numpy()
    return R.corrupt(df, rows, rng, "rand")
R.DT.load = load
D = C.build_dataset(C.train_panel("2025-06-30"), "2025-06-30", coins=tr, wmax=WC.W_MAX)
R.DT.load = _orig
bad = [k for k in ref.files if k in D and not RA._eq(ref[k], D[k])]
print(("OK   " if not bad else "FAIL ") + f"A refit dataset {fn} unchanged under VAL + held-out randomisation (keys {len(ref.files)}) {bad}")
