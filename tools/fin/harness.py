"""
Alma-Fin evaluation harness (docs/fin_prereg.md sections 2-5 are binding; interface: tools/fin/README.md).

Clock (crypto). Decision day D: the decision is taken at D 00:00 UTC. A context holds daily rows of days <= D-1 and
hourly rows of hours ending <= D 00:00, nothing else. The fill is the 00:05-00:10 bar of D at (O+H+L+C)/4 (one full
5-min bar after the decision; never the decision bar). The position is held until the fill of D+1. Funding settled
in (D 00:00, D+1 00:00] is charged to it. Labels of D (RV of D, return of D) are complete at D+1 00:00 and are
revealed to an online model only AFTER its decision for D and BEFORE its decision for D+1.
Clock (ETF). Decision for session D at the close of session D-1; fill = open of D; held to the open of D+1.

Three model kinds, one runner:
  forecaster.predict(ctx) -> {"p_up": [m] P(y_vol = 1), "q": [m, len(QS)] quantiles of log RV(D),
                              optional "p_ret_up": [m]}
  agent.act(ctx)          -> positions [m] in [-1, 1] (fraction of each asset's equal 1/m sleeve; clipped)
  online (either kind) additionally has observe(rec) -> int (number of parameter updates it made); rec holds day
  D's completed row (all fields dated <= D+1 00:00). The runner logs update counts.

Guards (enforced, tested by tools/fin/test_infoset.py):
  - contexts are fresh read-only copies built by build_context, with assertions on timestamps;
  - while a model runs (predict/act/observe), a sys.addaudithook guard raises on any attempt to open a file under
    data/fin/ (models get data only through the arguments);
  - the runner never passes the panel; day D's outcome is materialised only after the decision for D is recorded.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import data as DT       # noqa: E402

QS = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
CTX_DAYS = 400                      # daily context length passed to models
CTX_HOURS = 24 * 28                 # hourly context length (crypto)
LOOKBACK = 7                        # label reference window (days)
FEE = 0.0005                        # taker fee per side (crypto)
SLIP = {"BTCUSDT": 0.0002, "ETHUSDT": 0.0002}
SLIP_DEFAULT = 0.0005
ETF_COST = 0.0002                   # per side, ETFs (half-spread + slippage; no commission)
MAX_POS = 1.0
DAILY_FIELDS = ["open", "high", "low", "close", "quote_volume", "taker_ratio", "rv", "logrv", "ret", "funding",
                "fill", "n5"]
ETF_FIELDS = ["open", "high", "low", "close", "volume", "rv", "logrv", "ret", "fill"]
HOURLY_FIELDS = ["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_ratio", "rv5", "n5",
                 "funding", "ret"]
SPLITS = {"dev": (pd.Timestamp("2022-07-01"), DT.TRAIN_END), "val": (DT.VAL_START, DT.VAL_END),
          "test": (DT.TEST_START, DT.TEST_END)}
DATA_DIR = os.path.realpath(DT.DATA)
GUARDED = [DATA_DIR, os.path.realpath(os.path.join(ROOT, "runs", "fin"))]   # RT-5: labels/caches/forecasts too


# --------------------------------------------------------------------------- data-access guard
class _Guard:
    active = False
    violations = []


def _audit(event, args):
    if _Guard.active and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        try:
            rp = os.path.realpath(p)
        except Exception:
            rp = p
        if any(rp.startswith(g) for g in GUARDED) and "/ckpt/" not in rp:
            _Guard.violations.append(rp)
            raise PermissionError(f"model tried to open {rp}: models get data only through the context "
                                  "(data/fin/** and runs/fin/** are guarded; */ckpt/ dirs are allowed)")


sys.addaudithook(_audit)


class no_data_access:
    def __enter__(self):
        _Guard.active = True

    def __exit__(self, *a):
        _Guard.active = False


# --------------------------------------------------------------------------- panel
class Panel:
    """Dense arrays over (day, symbol). market 'crypto' (universe.json, train + heldout) or 'etf'."""

    def __init__(self, market="crypto", allow_test=False, frames=None):
        self.market = market
        if market == "crypto":
            tr, ho = DT.universe()
            self.symbols = tr + ho
            self.group = {s: "train" for s in tr} | {s: "heldout" for s in ho}
            d = frames["daily"] if frames else DT.load("crypto_daily", allow_test)
            h = frames["hourly"] if frames else DT.load("crypto_hourly", allow_test)
        else:
            self.symbols = list(DT.ETFS)
            self.group = {s: "etf" for s in self.symbols}
            d = frames["daily"] if frames else DT.load("etf_daily", allow_test)
            h = None
        d = d[d["symbol"].isin(self.symbols)].copy()
        self.days = np.array(sorted(d["t"].unique()), dtype="datetime64[ns]")
        if market == "crypto":            # complete UTC calendar: a day missing from the archive is a NaN row
            self.days = pd.date_range(self.days[0], self.days[-1], freq="D").to_numpy("datetime64[ns]")
        n, m = len(self.days), len(self.symbols)
        di = {t: i for i, t in enumerate(self.days)}
        si = {s: j for j, s in enumerate(self.symbols)}
        d["i"], d["j"] = d["t"].map(lambda x: di[np.datetime64(x, "ns")]), d["symbol"].map(si)
        if market == "crypto":
            d["taker_ratio"] = d["taker_buy_quote_volume"] / d["quote_volume"].replace(0, np.nan)
            d["funding"] = d["funding_sum"]
            d["ret"] = np.log(d["close"]).groupby(d["symbol"]).diff()
            fields = DAILY_FIELDS
        else:
            d["ret"] = np.log(d["adjclose"]).groupby(d["symbol"]).diff()
            d["fill"] = d["fill"] * d["adj_factor"]            # dividend-adjusted fills for P&L
            fields = ETF_FIELDS
        d["logrv"] = np.log(np.maximum(d["rv"], 1e-12))
        d.loc[~np.isfinite(d["rv"]) | (d["rv"] <= 0), "logrv"] = np.nan
        self.daily = {f: np.full((n, m), np.nan) for f in fields}
        for f in fields:
            self.daily[f][d["i"].to_numpy(), d["j"].to_numpy()] = d[f].to_numpy(float)
        self.last_bar = np.zeros((n, m), bool)
        self.last_bar[d["i"].to_numpy(), d["j"].to_numpy()] = d["last_bar"].to_numpy(bool)
        self.alive = np.isfinite(self.daily["close"])
        self.hourly = None
        if h is not None:
            h = h[h["symbol"].isin(self.symbols)].copy()
            h["taker_ratio"] = h["taker_buy_quote_volume"] / h["quote_volume"].replace(0, np.nan)
            h["ret"] = np.log(h["close"]).groupby(h["symbol"]).diff()
            t0 = self.days[0]
            hi = ((h["t"].to_numpy("datetime64[ns]") - t0) // np.timedelta64(1, "h")).astype(int)
            self.hourly = {f: np.full((24 * n, m), np.nan, np.float32) for f in HOURLY_FIELDS}
            hj = h["symbol"].map(si).to_numpy()
            for f in HOURLY_FIELDS:
                self.hourly[f][hi, hj] = h[f].to_numpy(np.float32)
        self.cost = np.array([SLIP.get(s, SLIP_DEFAULT) + FEE if market == "crypto" else ETF_COST for s in self.symbols])
        self.periods = 365 if market == "crypto" else 252

    def index(self, day):
        return int(np.searchsorted(self.days, np.datetime64(pd.Timestamp(day), "ns")))

    def split_days(self, split):
        a, b = SPLITS[split]
        sel = (self.days >= np.datetime64(a, "ns")) & (self.days <= np.datetime64(b, "ns"))
        return np.nonzero(sel)[0]

    def decision_time(self, i):
        """Crypto: D 00:00 UTC. ETF: the previous session's date (decision at its close)."""
        if self.market == "crypto":
            return pd.Timestamp(self.days[i])
        return pd.Timestamp(self.days[i - 1]) + pd.Timedelta(hours=21)        # ~16:00 ET close of D-1, in UTC

    # ------------------------------------------------------------------ labels (harness-side only)
    def labels(self, i):
        lr = self.daily["logrv"]
        ref = lr[i - LOOKBACK:i].mean(0) if i >= LOOKBACK else np.full(len(self.symbols), np.nan)
        y = np.where(np.isfinite(lr[i]) & np.isfinite(ref), (lr[i] > ref).astype(float), np.nan)
        y[np.isfinite(lr[i]) & np.isfinite(ref) & (lr[i] == ref)] = np.nan
        ret = self.daily["ret"][i]
        yr = np.where(np.isfinite(ret) & (ret != 0), (ret > 0).astype(float), np.nan)
        return {"y_vol": y, "logrv": lr[i].copy(), "ref": ref, "y_ret": yr, "ret": ret.copy()}

    def asset_return(self, i):
        """Simple return of a position taken at fill(i) and closed at fill(i+1) (or at close(i) if i is the asset's
        last day / the next fill is missing), plus the funding rate charged (crypto)."""
        f0 = self.daily["fill"][i]
        if i + 1 < len(self.days):
            f1 = self.daily["fill"][i + 1].copy()
        else:
            f1 = np.full_like(f0, np.nan)
        exit_close = ~np.isfinite(f1) | self.last_bar[i]
        f1 = np.where(exit_close, self.daily["close"][i] if self.market == "crypto" else f1, f1)
        r = f1 / f0 - 1
        fund = self.daily["funding"][i] if self.market == "crypto" else np.zeros_like(r)
        return r, np.nan_to_num(fund), exit_close


def build_context(P, i, ctx_days=CTX_DAYS, ctx_hours=CTX_HOURS, symbols=None):
    """Context for decision day index i: daily rows i-ctx_days..i-1, hourly rows ending <= decision time.
    Fresh float copies, read-only. Asserts the information set with explicit timestamps."""
    js = np.arange(len(P.symbols)) if symbols is None else np.array([P.symbols.index(s) for s in symbols])
    t_dec = P.decision_time(i)
    a = max(0, i - ctx_days)
    days = P.days[a:i]
    assert len(days) == 0 or pd.Timestamp(days[-1]) < pd.Timestamp(P.days[i]), "context includes the decision day"
    if P.market == "crypto":
        assert len(days) == 0 or pd.Timestamp(days[-1]) + pd.Timedelta(days=1) <= t_dec, \
            f"daily row {days[-1]} not complete by {t_dec}"
    else:
        assert len(days) == 0 or pd.Timestamp(days[-1]) <= t_dec, f"ETF row {days[-1]} after decision {t_dec}"
    daily = {}
    for f, arr in P.daily.items():
        x = arr[a:i][:, js].astype(np.float64, copy=True)
        x.setflags(write=False)
        daily[f] = x
    ctx = {"market": P.market, "decision_time": t_dec, "day": pd.Timestamp(P.days[i]),
           "symbols": [P.symbols[j] for j in js], "groups": [P.group[P.symbols[j]] for j in js],
           "days": days.copy(), "daily": daily, "quantiles": list(QS), "lookback": LOOKBACK}
    if P.hourly is not None:
        h1 = 24 * i                                 # hour index of D 00:00 = first hour NOT allowed
        h0 = max(0, h1 - ctx_hours)
        hours = P.days[0] + np.arange(h0, h1) * np.timedelta64(1, "h")
        assert len(hours) == 0 or pd.Timestamp(hours[-1]) + pd.Timedelta(hours=1) <= t_dec, "hourly row after decision"
        hourly = {}
        for f, arr in P.hourly.items():
            x = arr[h0:h1][:, js].astype(np.float32, copy=True)
            x.setflags(write=False)
            hourly[f] = x
        ctx["hourly"], ctx["hours"] = hourly, hours
    return ctx


def outcome_record(P, i, symbols=None):
    """Day i's completed row for online models (revealed after the decision for i, before the decision for i+1)."""
    js = np.arange(len(P.symbols)) if symbols is None else np.array([P.symbols.index(s) for s in symbols])
    rec = {"day": pd.Timestamp(P.days[i]), "symbols": [P.symbols[j] for j in js],
           "row": {f: arr[i, js].astype(np.float64, copy=True) for f, arr in P.daily.items()},
           "available_from": pd.Timestamp(P.days[i]) + pd.Timedelta(days=1) if P.market == "crypto"
           else pd.Timestamp(P.days[i]) + pd.Timedelta(hours=21)}
    if i + 1 < len(P.days):
        assert rec["available_from"] <= P.decision_time(i + 1), "outcome revealed after next decision time"
    if P.hourly is not None:
        rec["hourly"] = {f: arr[24 * i:24 * (i + 1), js].astype(np.float32, copy=True) for f, arr in P.hourly.items()}
    return rec


# --------------------------------------------------------------------------- runner
def run(P, model, days, symbols=None, kind=None, online=None, log=None):
    """Walk forward over decision-day indices `days` (ascending). Returns a dict of arrays.
    kind: 'forecaster' | 'agent' (default: from model.kind or presence of act()). online: call model.observe after
    each decision (default: hasattr(model, 'observe'))."""
    kind = kind or getattr(model, "kind", None) or ("agent" if hasattr(model, "act") else "forecaster")
    online = hasattr(model, "observe") if online is None else online
    syms = symbols or P.symbols
    js = np.array([P.symbols.index(s) for s in syms])
    m = len(syms)
    out = {"days": np.asarray(days), "symbols": list(syms), "kind": kind, "online": bool(online),
           "updates": np.zeros(len(days), int)}
    if kind == "forecaster":
        out["p_up"] = np.full((len(days), m), np.nan)
        out["q"] = np.full((len(days), m, len(QS)), np.nan)
        out["p_ret_up"] = np.full((len(days), m), np.nan)
    else:
        out["pos"] = np.full((len(days), m), np.nan)
    t0 = time.time()
    for k, i in enumerate(days):
        ctx = build_context(P, int(i), symbols=syms)
        with no_data_access():
            res = model.predict(ctx) if kind == "forecaster" else model.act(ctx)
        if kind == "forecaster":
            for key in ("p_up", "p_ret_up"):
                if res.get(key) is not None:
                    out[key][k] = np.asarray(res[key], float)
            if res.get("q") is not None:
                q = np.asarray(res["q"], float)
                assert q.shape == (m, len(QS)), f"q shape {q.shape}"
                out["q"][k] = q
        else:
            pos = np.asarray(res, float)
            assert pos.shape == (m,), f"positions shape {pos.shape}"
            out["pos"][k] = np.clip(np.nan_to_num(pos), -MAX_POS, MAX_POS)
        if online:
            rec = outcome_record(P, int(i), symbols=syms)        # materialised only now
            with no_data_access():
                n_up = model.observe(rec)
            out["updates"][k] = int(n_up or 0)
        if log and (k % 50 == 0 or k == len(days) - 1):
            log(f"  {k + 1}/{len(days)} {pd.Timestamp(P.days[i]).date()} {time.time() - t0:.0f}s")
    del js
    return out


# --------------------------------------------------------------------------- scoring helpers
def pnl(P, days, pos, symbols=None, cost_mult=1.0):
    """Net daily portfolio return of positions pos [n_days, m] (equal 1/m sleeves; fixed denominator m).
    cost = (fee + slippage) * |pos_D - pos_{D-1}| (pos before the first day = 0; after an asset's last day its
    position is forced to 0 and the closing trade is charged). Funding: -pos * rate (long pays positive rate).
    The last decision day of a split is dropped if its exit fill lies outside the split (handled by caller)."""
    syms = symbols or P.symbols
    js = np.array([P.symbols.index(s) for s in syms])
    pos = np.asarray(pos, float).copy()
    n, m = pos.shape
    gross = np.zeros((n, m))
    fund = np.zeros((n, m))
    tc = np.zeros((n, m))
    turn = np.zeros(n)
    prev = np.zeros(m)
    dead = np.zeros(m, bool)
    for k, i in enumerate(days):
        r, f, exit_close = P.asset_return(int(i))
        r, f, ex = r[js], f[js], exit_close[js]
        alive = np.isfinite(P.daily["fill"][int(i), js]) & ~dead
        p = np.where(alive, pos[k], 0.0)
        pos[k] = p
        d = np.abs(p - prev)
        tc[k] = d * P.cost[js] * cost_mult
        gross[k] = np.where(alive, p * np.nan_to_num(r), 0.0)
        fund[k] = np.where(alive, -p * f, 0.0) if P.market == "crypto" else 0.0
        turn[k] = d.mean()
        prev = p
        lb = P.last_bar[int(i), js]
        if lb.any():                                    # delisted after this day: close at close(i), charge exit
            tc[k] += np.abs(np.where(lb, p, 0.0)) * P.cost[js] * cost_mult
            prev = np.where(lb, 0.0, prev)
            dead |= lb
    net = gross + fund - tc
    return {"net": net.mean(1), "gross": gross.mean(1), "funding": fund.mean(1), "costs": tc.mean(1),
            "turnover": turn, "net_asset": net, "pos": pos}
