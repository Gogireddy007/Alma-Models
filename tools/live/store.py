"""
Alma-Live Gas point-in-time store.

    from store import Store
    S = Store()                         # loads data/live/pit/*.csv (append-only record files)
    ctx = S.asof(t)                     # everything knowable at UTC time t (published_at <= t)
    ctx = S.asof(t, live=True)          # paper/live: additionally fetched_at <= t (what was actually in hand)
    S.panel("px")                       # raw long table (common.COLS) with parsed timestamps
    S.labels(days)                      # realized outcomes (evaluation only; never inside ctx)

Rules (docs/live_leakage.md; tests tools/live/test_infoset.py):
  * a record is visible at t iff published_at <= t (and fetched_at <= t when live=True);
  * for each (series, obs) the LATEST VISIBLE vintage is returned (a revision published after t is invisible;
    first prints are what you see before the revision's publication);
  * forecasts (cpc_fcst) are returned per issue (vintage = issue), never merged with observations;
  * prices are stored unadjusted; ctx re-applies only splits with ex-date <= t's session;
  * no labels / outcomes in ctx. The decision clock is common.decision_time(day) (15:30 ET).

ctx keys: t, day, decision_time, fill_time, split ('dev'|'val'|'exposed'), px {sym: DataFrame[date x open, high,
low, close, volume, published_at]}, ng_roll {active contract, yahoo switch days}, storage DataFrame[week x
<series> levels, <series>.chg_printed, released_at], cpc_fcst DataFrame (last 30 issues, long: issue, day, hdd, cdd,
published_at), cpc_obs DataFrame[date x hdd, cdd] (last ~11 years, realized), om_fcst (long, optional, from
2021-04), cot DataFrame[tuesday x fields, published_at], edgar_filings (last 400 days), edgar_facts, curve,
features (dict of floats, see features()), meta {n_records, max_published_at, max_fetched_at, input_sha256}.
"""
import os
import sys
import json
import hashlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

# core sources (manager re-scope 2026-11-01): storage, prices, CPC forecasts (+ obs for normals / scoring).
# Paused (code and PIT files kept, not loaded by default): cot (LRT-10/11), edgar_*, om_fcst, curve.
SOURCES = ["storage", "px", "cpc_fcst", "cpc_obs"]
ALL_SOURCES = SOURCES + ["om_fcst", "cot", "edgar_filing", "edgar_fact", "curve"]
PX_MAIN = ["UNG", "BOIL", "NG=F"]
PX_DAYS = 800               # daily bars handed to models
OBS_DAYS = 11 * 366
FCST_ISSUES = 30
STOR_SERIES = ["l48", "east", "midwest", "mountain", "pacific", "south_central"]


class Store:
    def __init__(self, pit_dir=None, frames=None, sources=SOURCES):
        self.pit_dir = pit_dir or C.PIT
        self.raw = {}
        for s in sources:
            if frames is not None:
                df = frames.get(s)
                if df is None:
                    continue
                df = df.copy()
            else:
                p = os.path.join(self.pit_dir, f"{s}.csv")
                if not os.path.exists(p):
                    continue
                df = pd.read_csv(p, dtype={"series": str, "obs": str, "vintage": str, "extra": str},
                                 keep_default_na=False, na_values={"value": ["", "nan", "NaN"]})
            self._add(s, df)

    def _add(self, s, df):
        df = df.copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["pa"] = pd.to_datetime(df["published_at"], utc=True).dt.as_unit("ns").astype("int64")
        df["fa"] = pd.to_datetime(df["fetched_at"], utc=True).dt.as_unit("ns").astype("int64")
        df = df.sort_values(["pa", "fa"], kind="mergesort").reset_index(drop=True)
        self.raw[s] = df

    # ------------------------------------------------------------------ primitives
    def panel(self, source):
        df = self.raw[source].copy()
        df["published_at"] = pd.to_datetime(df["pa"], utc=True)
        df["fetched_at"] = pd.to_datetime(df["fa"], utc=True)
        return df.drop(columns=["pa", "fa"])

    def visible(self, source, t, live=False, key=("series", "obs")):
        """Records of `source` visible at t; latest visible vintage per key."""
        if source not in self.raw:
            return self.raw.get(source, pd.DataFrame(columns=C.COLS + ["pa", "fa"])).iloc[:0]
        df = self.raw[source]
        tn = pd.Timestamp(t).value
        n = int(np.searchsorted(df["pa"].to_numpy(), tn, side="right"))
        v = df.iloc[:n]
        if live:
            v = v[v["fa"].to_numpy() <= tn]
        if key:
            v = v.drop_duplicates(subset=list(key), keep="last")
        return v

    # ------------------------------------------------------------------ context
    def asof(self, t, live=False, with_features=True):
        t = pd.Timestamp(t)
        if t.tzinfo is None:
            raise ValueError("asof(t) needs a tz-aware UTC timestamp (use common.decision_time(day))")
        t = t.tz_convert("UTC")
        day = t.tz_convert(C.ET).normalize().tz_localize(None)
        ctx = {"t": t, "day": day, "decision_time": C.decision_time(day), "fill_time": C.fill_time(day),
               "split": C.split_of(day), "live": live}
        used = []

        # prices
        v = self.visible("px", t, live)
        used.append(v)
        px = {}
        for sym in sorted(set(s.rsplit(".", 1)[0] for s in v["series"].unique())):
            sub = v[v["series"].str.startswith(sym + ".")]
            if not len(sub):
                continue
            fld = sub["series"].str.rsplit(".", n=1).str[1]
            bars = sub[fld != "split"]
            w = bars.assign(f=fld[fld != "split"]).pivot(index="obs", columns="f", values="value")
            pa = bars.assign(f=fld[fld != "split"]).pivot(index="obs", columns="f", values="pa")
            w.index = pd.DatetimeIndex(pd.to_datetime(w.index))
            w = w.sort_index()
            w["published_at"] = pd.to_datetime(pa.max(axis=1).reindex(w.index.strftime("%Y-%m-%d")).to_numpy(), utc=True)
            sp = sub[fld == "split"]
            for _, r in sp.iterrows():                         # splits known at t: adjust earlier bars
                sd = pd.Timestamp(r["obs"])
                m = w.index < sd
                for c in ("open", "high", "low", "close"):
                    if c in w:
                        w.loc[m, c] = w.loc[m, c] / r["value"]
                if "volume" in w:
                    w.loc[m, "volume"] = w.loc[m, "volume"] * r["value"]
            px[sym] = w.iloc[-PX_DAYS:]
        ctx["px"] = px
        act = C.ng_active_contract(day)
        ctx["ng_roll"] = {"active": act["symbol"], "active_last_trade": act["last_trade"],
                          "yahoo_switch_days": C.yahoo_front_switch_days(day - pd.Timedelta(days=PX_DAYS * 1.5), day)}

        # storage
        v = self.visible("storage", t, live)
        used.append(v)
        if len(v):
            st = v.pivot(index="obs", columns="series", values="value")
            st.index = pd.DatetimeIndex(pd.to_datetime(st.index))
            rel = v[v["series"] == "l48"].set_index("obs")["pa"]
            first_pa = self.raw["storage"]
            fr = first_pa[(first_pa["series"] == "l48") & (first_pa["pa"] <= t.value)].drop_duplicates("obs", keep="first")
            st["released_at"] = pd.to_datetime(fr.set_index("obs")["pa"].reindex(st.index.strftime("%Y-%m-%d")).to_numpy(), utc=True)
            ctx["storage"] = st.sort_index()
            nw = ctx["storage"].index[-1] + pd.Timedelta(days=7)
            ctx["next_storage"] = {"week": nw, "expectation_l48": storage_expectation(ctx["storage"], nw)}
        else:
            ctx["storage"] = pd.DataFrame()
            ctx["next_storage"] = {"week": None, "expectation_l48": np.nan}

        # weather forecasts (per issue)
        v = self.visible("cpc_fcst", t, live, key=("series", "obs", "vintage"))
        if len(v):
            iss = v["vintage"].unique()[-FCST_ISSUES * 2:]
            keep = sorted(set(iss), key=lambda s: s)[-FCST_ISSUES:]
            v = v[v["vintage"].isin(keep)]
            used.append(v)
            f = v.pivot_table(index=["vintage", "obs"], columns="series", values="value", aggfunc="last").reset_index()
            pa = v.groupby("vintage")["pa"].max()
            f["issue"] = pd.to_datetime(f["vintage"].str[6:])
            f["day"] = pd.to_datetime(f["obs"])
            f["published_at"] = pd.to_datetime(f["vintage"].map(pa), utc=True)
            # a 'forecast' for a valid day that had already ended when the issue was published is not a forecast:
            # dropped (never substitute analysis / realized weather for forecasts)
            f = f[f["day"].dt.tz_localize("UTC") + pd.Timedelta(days=1) > f["published_at"]]
            ctx["cpc_fcst"] = f[["issue", "day"] + [c for c in ("hdd", "cdd") if c in f] + ["published_at"]] \
                .sort_values(["issue", "day"]).reset_index(drop=True)
        else:
            ctx["cpc_fcst"] = pd.DataFrame(columns=["issue", "day", "hdd", "cdd", "published_at"])

        # realized weather (observations only)
        v = self.visible("cpc_obs", t, live)
        if len(v):
            v = v[pd.to_datetime(v["obs"]) >= day - pd.Timedelta(days=OBS_DAYS)]
            used.append(v)
            o = v.pivot(index="obs", columns="series", values="value")
            o.index = pd.DatetimeIndex(pd.to_datetime(o.index))
            ctx["cpc_obs"] = o.sort_index()
        else:
            ctx["cpc_obs"] = pd.DataFrame(columns=["hdd", "cdd"])

        # optional Open-Meteo previous-runs forecasts (latest visible per region/var/lead/day, last 30 days)
        v = self.visible("om_fcst", t, live)
        if len(v):
            v = v[pd.to_datetime(v["obs"]) >= day - pd.Timedelta(days=30)]
            used.append(v)
        ctx["om_fcst"] = v[["series", "obs", "value"]].assign(published_at=pd.to_datetime(v["pa"], utc=True)) \
            .reset_index(drop=True) if len(v) else pd.DataFrame(columns=["series", "obs", "value", "published_at"])

        # COT
        v = self.visible("cot", t, live)
        used.append(v)
        if len(v):
            c = v.pivot(index="obs", columns="series", values="value")
            c.index = pd.DatetimeIndex(pd.to_datetime(c.index))
            c["published_at"] = pd.to_datetime(v.groupby("obs")["pa"].max().reindex(c.index.strftime("%Y-%m-%d")).to_numpy(), utc=True)
            ctx["cot"] = c.sort_index()
        else:
            ctx["cot"] = pd.DataFrame()

        # EDGAR
        v = self.visible("edgar_filing", t, live, key=("series", "obs", "extra"))
        if len(v):
            v = v[pd.to_datetime(v["pa"], utc=True) >= t - pd.Timedelta(days=400)]
            used.append(v)
        ctx["edgar_filings"] = pd.DataFrame({"series": v["series"].to_numpy(), "published_at": pd.to_datetime(v["pa"].to_numpy(), utc=True),
                                             "extra": v["extra"].to_numpy()}) if len(v) else pd.DataFrame(columns=["series", "published_at", "extra"])
        v = self.visible("edgar_fact", t, live, key=("series", "obs", "extra"))
        if len(v):
            used.append(v)
        ctx["edgar_facts"] = v[["series", "obs", "value", "extra"]].assign(published_at=pd.to_datetime(v["pa"], utc=True)) \
            .reset_index(drop=True) if len(v) else pd.DataFrame(columns=["series", "obs", "value", "extra", "published_at"])
        v = self.visible("curve", t, live)
        ctx["curve"] = v[["series", "obs", "value"]].assign(published_at=pd.to_datetime(v["pa"], utc=True)) \
            .reset_index(drop=True) if len(v) else pd.DataFrame(columns=["series", "obs", "value", "published_at"])
        if len(v):
            used.append(v)

        allv = [u for u in used if len(u)]
        ctx["meta"] = {"n_records": int(sum(len(u) for u in allv)),
                       "max_published_at": pd.Timestamp(max(int(u["pa"].max()) for u in allv), tz="UTC") if allv else None,
                       "max_fetched_at": pd.Timestamp(max(int(u["fa"].max()) for u in allv), tz="UTC") if allv else None}
        assert ctx["meta"]["max_published_at"] is None or ctx["meta"]["max_published_at"] <= t
        if with_features:
            ctx["features"] = features(ctx)
        ctx["meta"]["input_sha256"] = ctx_hash(ctx)
        return ctx

    # ------------------------------------------------------------------ outcomes (evaluation only)
    def labels(self, days=None):
        """Realized outcomes per decision day d (NEVER part of ctx). Uses the latest snapshot of prices.
        Columns: ung_ret, boil_ret (close(d) -> close(d+1) log return: fill at the 16:00 close of d, held one session),
        ng_cc (NG=F close(d)->close(d+1), NaN when the interval crosses a Yahoo contract switch), ng_oo (NG=F open(d+1)
        -> open(d+2), masked likewise), ung_rv5 (sum of squared UNG close-to-close log returns d+1..d+5),
        available_at (UTC time the label is complete)."""
        v = self.raw["px"].drop_duplicates(["series", "obs"], keep="last")
        out = {}
        for sym in PX_MAIN:
            sub = v[v["series"].str.startswith(sym + ".")]
            fld = sub["series"].str.rsplit(".", n=1).str[1]
            w = sub[fld != "split"].assign(f=fld[fld != "split"]).pivot(index="obs", columns="f", values="value")
            w.index = pd.DatetimeIndex(pd.to_datetime(w.index))
            w = w.sort_index()
            for _, r in sub[fld == "split"].iterrows():
                m = w.index < pd.Timestamp(r["obs"])
                for c in ("open", "high", "low", "close"):
                    w.loc[m, c] = w.loc[m, c] / r["value"]
            out[sym] = w
        idx = out["UNG"].index
        L = pd.DataFrame(index=idx)
        for sym, col in (("UNG", "ung_ret"), ("BOIL", "boil_ret")):
            c = out[sym]["close"].reindex(idx)
            L[col] = np.log(c.shift(-1) / c)
        ng = out["NG=F"]
        sw = set(C.yahoo_front_switch_days(ng.index.min(), ng.index.max() + pd.Timedelta(days=10)))
        nidx = ng.index
        nxt = pd.Series(nidx[1:].tolist() + [pd.NaT], index=nidx)
        nxt2 = nxt.shift(-1)
        cc = np.log(ng["close"].shift(-1) / ng["close"])
        cc[[n in sw for n in nxt]] = np.nan                       # close(d)->close(d+1) spans the switch at d+1
        oo = np.log(ng["open"].shift(-2) / ng["open"].shift(-1))
        oo[[n2 in sw for n2 in nxt2]] = np.nan                    # open(d+1)->open(d+2) spans the switch at d+2
        L["ng_cc"] = cc.reindex(idx)
        L["ng_oo"] = oo.reindex(idx)
        r = np.log(out["UNG"]["close"] / out["UNG"]["close"].shift(1)).reindex(idx)
        L["ung_rv5"] = sum(r.shift(-k) ** 2 for k in range(1, 6))
        nd = pd.Series(idx[1:].tolist() + [pd.NaT], index=idx)
        L["available_at"] = [C.et(x, 16, 0) if pd.notna(x) else pd.NaT for x in nd]
        if days is not None:
            L = L.reindex(pd.DatetimeIndex(days))
        return L


def storage_labels(S, weeks=None):
    """Storage target per report week W (evaluation only): dev(W) = printed net change(W) - naive expectation (mean
    printed change of the same week in the 5 previous years, as known just before W's first print), the release time
    of W's first print, and the forecast origin day = last trading day whose decision time is before that release."""
    raw = S.raw["storage"]
    fp = raw[(raw["series"] == "l48.chg_printed")].drop_duplicates("obs", keep="first")
    out = []
    for _, r in fp.iterrows():
        w = pd.Timestamp(r["obs"])
        if weeks is not None and w not in weeks:
            continue
        rel = pd.Timestamp(r["pa"], tz="UTC")
        ctxst = S.visible("storage", rel - pd.Timedelta(seconds=1))
        st = ctxst.pivot(index="obs", columns="series", values="value")
        st.index = pd.DatetimeIndex(pd.to_datetime(st.index))
        e = storage_expectation(st, w)
        d = rel.tz_convert(C.ET).normalize().tz_localize(None)
        while C.decision_time(d) >= rel or not C.is_bday(d):
            d -= pd.Timedelta(days=1)
        out.append({"week": w, "chg_printed": r["value"], "expectation": e, "dev": r["value"] - e,
                    "released_at": rel, "origin_day": d})
    return pd.DataFrame(out).set_index("week")


# ---------------------------------------------------------------------- features (pure function of ctx)
FEATURES = ["ung_r1", "ung_r5", "ung_r20", "ung_r60", "ung_vol20", "ng_r1", "ng_r5", "ng_r20", "ng_r60",
            "boil_r5", "stor_surprise", "stor_dev5_lvl", "stor_chg", "days_since_stor", "wx_fcst_dd7",
            "wx_fcst_anom7", "wx_rev", "m_sin", "m_cos", "dow"]


def _logret(s, k):
    s = s.dropna()
    if len(s) <= k:
        return np.nan
    return float(np.log(s.iloc[-1] / s.iloc[-1 - k]))


def _ng_series(ctx):
    """NG=F closes with roll jumps removed (returns on Yahoo switch days set to 0) -> roll-free log-price index."""
    ng = ctx["px"].get("NG=F")
    if ng is None or not len(ng):
        return pd.Series(dtype=float)
    r = np.log(ng["close"] / ng["close"].shift(1))
    sw = set(ctx["ng_roll"]["yahoo_switch_days"])
    r[[d in sw for d in r.index]] = 0.0
    return np.exp(r.fillna(0).cumsum())


def storage_expectation(st, week, series="l48"):
    """Naive expectation of the printed change of `week`: mean printed change of the same report week (+-3 days)
    in the 5 previous years, from the storage frame of ctx (values as known at t)."""
    col = f"{series}.chg_printed"
    if col not in st:
        return np.nan
    vals = []
    for y in range(1, 6):
        target = week - pd.Timedelta(days=364 * y)
        m = st.index[(abs((st.index - target).days) <= 3)]
        if len(m):
            vals.append(st.at[m[0], col])
    vals = [x for x in vals if np.isfinite(x)]
    return float(np.mean(vals)) if len(vals) == 5 else np.nan


def weather_summary(ctx):
    """(7-day forecast HDD+CDD total of the latest usable CPC issue, anomaly vs 10-yr observed normal of the same
    days, revision = latest minus previous issue over their common days)."""
    f = ctx["cpc_fcst"]
    if not len(f):
        return np.nan, np.nan, np.nan
    iss = sorted(f["issue"].unique())
    a = f[f["issue"] == iss[-1]].set_index("day")
    dd = (a["hdd"] + a["cdd"])
    tot = float(dd.sum())
    o = ctx["cpc_obs"]
    anom = np.nan
    if len(o):
        od = o["hdd"] + o["cdd"]
        norms = []
        for d in dd.index:
            past = [d - pd.DateOffset(years=y) for y in range(1, 11)]
            vals = od.reindex(past).dropna()
            norms.append(vals.mean() if len(vals) >= 8 else np.nan)
        if np.all(np.isfinite(norms)):
            anom = tot - float(np.sum(norms))
    rev = np.nan
    if len(iss) >= 2:
        b = f[f["issue"] == iss[-2]].set_index("day")
        common = a.index.intersection(b.index)
        if len(common):
            rev = float((a.loc[common, "hdd"] + a.loc[common, "cdd"]).sum() - (b.loc[common, "hdd"] + b.loc[common, "cdd"]).sum())
    return tot, anom, rev


def features(ctx):
    F = {k: np.nan for k in FEATURES}
    u = ctx["px"].get("UNG")
    if u is not None and len(u):
        c = u["close"]
        for k in (1, 5, 20, 60):
            F[f"ung_r{k}"] = _logret(c, k)
        r = np.log(c / c.shift(1)).dropna()
        F["ung_vol20"] = float(r.iloc[-20:].std(ddof=1) * np.sqrt(252)) if len(r) >= 20 else np.nan
    b = ctx["px"].get("BOIL")
    if b is not None and len(b):
        F["boil_r5"] = _logret(b["close"], 5)
    ng = _ng_series(ctx)
    for k in (1, 5, 20, 60):
        F[f"ng_r{k}"] = _logret(ng, k) if len(ng) else np.nan
    st = ctx["storage"]
    if len(st) and "l48.chg_printed" in st:
        w = st.index[-1]
        exp = storage_expectation(st, w)
        F["stor_chg"] = float(st.at[w, "l48.chg_printed"])
        F["stor_surprise"] = F["stor_chg"] - exp if np.isfinite(exp) else np.nan
        lv = [st.at[m[0], "l48"] for y in range(1, 6)
              for m in [st.index[abs((st.index - (w - pd.Timedelta(days=364 * y))).days) <= 3]] if len(m)]
        F["stor_dev5_lvl"] = float(st.at[w, "l48"] - np.mean(lv)) if len(lv) == 5 else np.nan
        ra = st["released_at"].iloc[-1]
        F["days_since_stor"] = float((ctx["t"] - ra).total_seconds() / 86400) if pd.notna(ra) else np.nan
    F["wx_fcst_dd7"], F["wx_fcst_anom7"], F["wx_rev"] = weather_summary(ctx)
    m = ctx["day"].month + ctx["day"].day / 31.0
    F["m_sin"], F["m_cos"] = float(np.sin(2 * np.pi * m / 12)), float(np.cos(2 * np.pi * m / 12))
    F["dow"] = float(ctx["day"].weekday())
    return F


def _canon_bytes(df):
    """Canonical bytes of a frame: numeric columns as float64 (int/float parsing invariant, red-team LRT-12),
    datetimes as int64 ns UTC, everything else as str; columns sorted by name."""
    out = []
    for c in sorted(df.columns, key=str):
        x = df[c]
        if pd.api.types.is_datetime64_any_dtype(x):
            x = x.dt.tz_convert("UTC") if getattr(x.dt, "tz", None) is not None else x
            x = x.astype("datetime64[ns, UTC]" if getattr(x.dt, "tz", None) is not None else "datetime64[ns]").astype("int64")
        elif pd.api.types.is_numeric_dtype(x):
            x = x.astype("float64")
        else:
            x = x.astype(str)
        out.append(str(c).encode())
        out.append(pd.util.hash_pandas_object(x, index=False).values.tobytes())
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        out.append(idx.as_unit("ns").asi8.tobytes())
    else:
        out.append(pd.util.hash_pandas_object(pd.Series(idx.astype(str)), index=False).values.tobytes())
    return b"".join(out)


def ctx_hash(ctx):
    """sha256 over a canonical serialization of every data frame in ctx (+ t). Logged with every decision."""
    h = hashlib.sha256()
    h.update(C.iso(ctx["t"]).encode())
    for k in sorted(ctx):
        v = ctx[k]
        if isinstance(v, dict) and k == "px":
            for s in sorted(v):
                h.update(s.encode())
                h.update(_canon_bytes(v[s]))
        elif isinstance(v, pd.DataFrame):
            h.update(k.encode())
            if len(v):
                h.update(_canon_bytes(v))
    return h.hexdigest()


if __name__ == "__main__":
    import time
    S = Store()
    day = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("2024-03-14")
    t0 = time.time()
    ctx = S.asof(C.decision_time(day))
    print(f"asof({C.iso(ctx['t'])}) in {time.time() - t0:.2f}s; records {ctx['meta']['n_records']}; "
          f"max_published_at {ctx['meta']['max_published_at']}")
    print({k: (round(v, 4) if np.isfinite(v) else v) for k, v in ctx["features"].items()})
    print("px last:", {s: str(d.index[-1].date()) for s, d in ctx["px"].items()}, "storage last:",
          ctx["storage"].index[-1].date() if len(ctx["storage"]) else None)
