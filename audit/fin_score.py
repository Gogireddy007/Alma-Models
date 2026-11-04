"""Own labels + metrics for audit/fin_test/crypto/*.npz (see fin_check.py)."""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.join(HERE, "..", "repo")
QS = np.array([0.025,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975]); I9 = slice(2, 11)
d = pd.concat([pd.read_parquet(f"{REPO}/data/fin/crypto_daily.parquet"), pd.read_parquet(f"{REPO}/data/fin/crypto_daily_SEALED.parquet")])
d["day"] = pd.to_datetime(d.t).dt.tz_localize(None).dt.normalize()
rv = d.pivot_table(index="day", columns="symbol", values="rv")
nl = d.pivot_table(index="day", columns="symbol", values="n_lab")
lrv = np.log(rv.where(nl >= 274))
ref = lrv.shift(1).rolling(7).mean()      # mean of D-7..D-1 (needs all 7)
ref = ref.where(lrv.shift(1).rolling(7).count() == 7)
y = (lrv > ref).astype(float).where(lrv.notna() & ref.notna())
held = set(json.load(open(f"{REPO}/data/fin/universe.json"))["heldout"])
rows = []
for f in sorted(os.listdir(f"{HERE}/fin_test/crypto")):
    z = np.load(f"{HERE}/fin_test/crypto/{f}", allow_pickle=True)
    days = pd.to_datetime(z["dates"]); syms = list(z["symbols"])
    Y = y.reindex(index=days, columns=syms).to_numpy(); L = lrv.reindex(index=days, columns=syms).to_numpy()
    for grp, cols in (("train", [j for j, s in enumerate(syms) if s not in held]), ("heldout", [j for j, s in enumerate(syms) if s in held]), ("all", list(range(len(syms))))):
        p = z["p_up"][:, cols]; yy = Y[:, cols]; ok = np.isfinite(p) & np.isfinite(yy)
        acc = np.mean((p[ok] > 0.5) == (yy[ok] == 1))
        q = z["q"][:, cols][..., I9]; ll = L[:, cols]; okq = np.isfinite(q).all(-1) & np.isfinite(ll)
        dd = ll[..., None] - q; pin = np.maximum(QS[I9] * dd, (QS[I9] - 1) * dd).mean(-1)
        rows.append(dict(model=f[:-4], group=grp, n=int(ok.sum()), P1=acc, QS=np.nanmean(np.where(okq, pin, np.nan))))
out = pd.DataFrame(rows)
print(out.pivot(index="model", columns="group", values=["P1", "QS"]).round(4).sort_values(("P1", "all"), ascending=False).to_string())
out.to_csv(f"{HERE}/fin_score.csv", index=False)

# direction from the median quantile vs the label reference (for models without p_up)
print("\nP1 from median > reference (all models):")
rows = []
for f in sorted(os.listdir(f"{HERE}/fin_test/crypto")):
    z = np.load(f"{HERE}/fin_test/crypto/{f}", allow_pickle=True)
    days = pd.to_datetime(z["dates"]); syms = list(z["symbols"])
    Y = y.reindex(index=days, columns=syms).to_numpy(); RF = ref.reindex(index=days, columns=syms).to_numpy()
    pm = z["q"][..., 6] > RF
    r = {"model": f[:-4]}
    for grp, cols in (("train", [j for j, s in enumerate(syms) if s not in held]), ("heldout", [j for j, s in enumerate(syms) if s in held])):
        ok = np.isfinite(Y[:, cols]) & np.isfinite(z["q"][:, cols, 6])
        r[grp] = np.mean(pm[:, cols][ok] == (Y[:, cols][ok] == 1))
    rows.append(r)
print(pd.DataFrame(rows).round(4).sort_values("heldout", ascending=False).to_string(index=False))
