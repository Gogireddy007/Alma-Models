"""P2 re-derivation (auditor): tsmom_30 and buy_hold net P&L on the DEV fold 2024-07-01..2025-06-30 recomputed from
the raw daily parquet (own fill/funding/cost code), compared with harness.run + harness.pnl. DEV only."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_rt as R
H, DT = R.H, R.DT
sys.path.insert(0, os.path.join(R.FIN, "search"))
import common as C
import models as M

d = pd.read_parquet(DT.FILES["crypto_daily"])
tr, ho = DT.universe(); syms = tr + ho
cal = pd.date_range(d.t.min(), "2025-06-30", freq="D")
piv = lambda c: d.pivot(index="t", columns="symbol", values=c).reindex(cal)[syms]
close, fill, fund = piv("close"), piv("fill"), piv("funding_sum").fillna(0)
ret = np.log(close).diff()
cost = np.array([0.0007 if s in ("BTCUSDT", "ETHUSDT") else 0.0010 for s in syms])
days = cal[(cal >= "2024-07-01") & (cal <= "2025-06-30")]
P = C.dev_panel("crypto")
idx = C.bt_days(P)
assert (pd.DatetimeIndex(P.days[idx]) == days).all()
for name, posf in (("tsmom_30", lambda D: np.sign(ret.loc[D - pd.Timedelta(days=30):D - pd.Timedelta(days=1)].sum(min_count=1).fillna(0).to_numpy())),
                   ("buy_hold", lambda D: np.ones(len(syms)))):
    prev = np.zeros(len(syms)); net = []
    for D in days:
        p = posf(D)
        p = np.where(np.isfinite(fill.loc[D].to_numpy()), p, 0.0)
        nxt = D + pd.Timedelta(days=1)
        f1 = fill.loc[nxt].to_numpy() if nxt in fill.index else np.full(len(syms), np.nan)
        f1 = np.where(np.isfinite(f1), f1, close.loc[D].to_numpy())
        r = f1 / fill.loc[D].to_numpy() - 1
        n = p * np.nan_to_num(r) - p * fund.loc[D].to_numpy() - np.abs(p - prev) * cost
        net.append(n); prev = p
    net = np.array(net)
    res = H.run(P, M.get(name), idx, online=False)
    pl = H.pnl(P, idx, res["pos"])
    hoj = [syms.index(s) for s in ho]
    sh = lambda x: x.mean() / x.std(ddof=1) * np.sqrt(365)
    mine_ho, h_ho = net[:, hoj].mean(1), pl["net_asset"][:, hoj].mean(1)
    ok = np.allclose(net.mean(1), pl["net"], atol=1e-12) and np.allclose(mine_ho, h_ho, atol=1e-12)
    print(f"{'OK  ' if ok else 'FAIL'} {name}: Sharpe all mine {sh(net.mean(1)):.4f} harness {sh(pl['net']):.4f}; "
          f"held-out mine {sh(mine_ho):.4f} harness {sh(h_ho):.4f}; max|diff| {np.abs(net.mean(1)-pl['net']).max():.2e}")
