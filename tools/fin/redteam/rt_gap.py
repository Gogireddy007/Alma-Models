"""RT-15 follow-up (auditor): what explains B's F1 edge beyond the weekday term? Pooled OLS / logistic on DEV
training coins, fit <= 2024-06-30, scored on F1 (2024-07..2025-06), all inputs known at D 00:00. DEV only."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_rt as R
DT = R.DT
tr, ho = DT.universe(); syms = tr + ho
d = pd.read_parquet(DT.FILES["crypto_daily"])
cal = pd.date_range(d.t.min(), "2025-06-30", freq="D")
piv = lambda c: d.pivot(index="t", columns="symbol", values=c).reindex(cal)[syms]
lr = np.log(piv("rv").where(piv("rv") > 0))
h = pd.read_parquet(DT.FILES["crypto_hourly"], columns=["symbol", "t", "rv5"])
h = h[h.t < pd.Timestamp("2025-07-01")]
h["day"] = h.t.dt.floor("D"); h["blk"] = h.t.dt.hour // 6
blk = h.pivot_table(index="day", columns=["symbol", "blk"], values="rv5", aggfunc="sum")
lastblk = np.log(4 * blk.xs(3, axis=1, level="blk").reindex(cal)[syms].clip(lower=1e-12))   # 18-24h of day D-1 (shifted below)
hr23 = np.log(24 * h[h.t.dt.hour == 23].pivot(index="day", columns="symbol", values="rv5").reindex(cal)[syms].clip(lower=1e-12))
ref = lr.rolling(7, min_periods=7).mean().shift(1)
y = (lr > ref).astype(float).where(lr.notna() & ref.notna() & (lr != ref))
dow = cal.dayofweek
us_hol = pd.to_datetime(["2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29", "2023-06-19", "2023-07-04",
                         "2023-09-04", "2023-11-23", "2023-12-25", "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29",
                         "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25", "2025-01-01",
                         "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19",
                         "2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30", "2022-06-20", "2022-07-04", "2022-09-05",
                         "2022-11-24", "2022-12-26", "2021-07-05", "2021-09-06", "2021-11-25", "2021-12-24",
                         "2021-01-18", "2021-02-15", "2021-04-02", "2021-05-31", "2020-07-03", "2020-09-07",
                         "2020-11-26", "2020-12-25", "2021-01-01", "2022-01-01"])
hol = np.isin(cal, us_hol).astype(float)
feats = {"har": [lr.shift(1), lr.shift(1).rolling(5, min_periods=3).mean(), lr.shift(1).rolling(22, min_periods=15).mean()],
         "lastblk": [lastblk.shift(1)], "hr23": [hr23.shift(1)]}
def X_of(rows, cols, keys, use_dow, use_hol):
    ii = cal.get_indexer(rows)
    X = [np.ones(len(rows) * len(cols))]
    for k in keys:
        X += [f.loc[rows, cols].to_numpy().ravel() for f in feats[k]]
    X.append(ref.loc[rows, cols].to_numpy().ravel())
    if use_dow:
        X.append(np.repeat(np.eye(7)[dow[ii]][:, 1:], len(cols), 0))
    if use_hol:
        X.append(np.repeat(hol[ii], len(cols)))
    return np.column_stack(X)
trn, tst = cal[(cal >= "2020-06-01") & (cal <= "2024-06-30")], cal[cal >= "2024-07-01"]
Y = y.loc[tst, syms].to_numpy(); hoj = [syms.index(s) for s in ho]
for keys, ud, uh in ((["har"], False, False), (["har"], True, False), (["har"], True, True), (["har", "lastblk"], True, False),
                     (["har", "lastblk", "hr23"], True, False), (["har", "lastblk", "hr23"], True, True)):
    Xa, ya = X_of(trn, tr, keys, ud, uh), lr.loc[trn, tr].to_numpy().ravel()
    ok = np.isfinite(Xa).all(1) & np.isfinite(ya)
    b = np.linalg.lstsq(Xa[ok], ya[ok], rcond=None)[0]
    mu = (X_of(tst, syms, keys, ud, uh) @ b).reshape(len(tst), len(syms))
    pr = (mu > ref.loc[tst, syms].to_numpy()).astype(float)
    okk = np.isfinite(Y)
    pr = np.where(okk & ~np.isfinite(mu), 0.0, pr)
    a = ((pr == Y) & okk).sum() / okk.sum(); ah = ((pr == Y) & okk)[:, hoj].sum() / okk[:, hoj].sum()
    print(f"{'+'.join(keys)}{' +dow' if ud else ''}{' +hol' if uh else ''}: F1 P1 {a:.4f} held-out {ah:.4f}", flush=True)
