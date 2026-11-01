"""
Alma-Power data: raw Energy-Charts JSON -> hourly local-delivery-day panel, plus the information-set-safe window
builder (docs/power_prereg.md section 2 is binding; tools/power/README.md describes the interface).

    nice -n 10 .venv/bin/python tools/power/data.py build      # writes data/power/hourly.csv, data/power/gaps.json

Panel convention
  * Day d = a local (Europe/Berlin) delivery day of the day-ahead auction. Slot s = local clock hour 0..23.
  * Each slot holds the mean over the UTC hour(s) whose local start hour is s (all timestamps handled in UTC):
      - normal day: 24 UTC hours -> 24 slots;
      - spring-forward day (23 h): slot 2 does not exist -> NaN, flag "dst_gap" (never filled in targets);
      - fall-back day (25 h): local hour 2 occurs twice -> slot 2 = mean of the two UTC hours, flag "dst_dup".
  * 15-min values are averaged to the UTC hour; an hour needs ALL its expected sub-hourly values (4 for 15-min data,
    1 for hourly data), otherwise it is NaN and recorded in gaps.json. No forward fill anywhere.
  * Price zone: DE-AT-LU for delivery days <= 2018-09-30, DE-LU from 2018-10-01 (zone split; recorded).
    Day-ahead price resolution: hourly products until 2025-09-30, 15-min products from 2025-10-01 (hourly = mean
    of the four quarter-hours).
"""
import os
import sys
import glob
import json
import hashlib

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "power", "raw")
HOURLY = os.path.join(ROOT, "data", "power", "hourly.csv")
GAPS = os.path.join(ROOT, "data", "power", "gaps.json")
TZ = "Europe/Berlin"

# ---------------------------------------------------------------- splits (target = delivery day)
FIRST_DAY = pd.Timestamp("2015-01-01")
ZONE_SPLIT = pd.Timestamp("2018-10-01")         # first DE-LU delivery day
QH_SWITCH = pd.Timestamp("2025-10-01")          # first delivery day with 15-min day-ahead products
TRAIN_END = pd.Timestamp("2025-06-30")          # DEV: every target day <= this
DEV_BT_START = pd.Timestamp("2024-07-01")       # harness DEV backtest target days 2024-07-01 .. 2025-06-30
VAL_START, VAL_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2025-10-31")
TEST_START = pd.Timestamp("2025-11-01")

COVS = ["load", "wind_onshore", "wind_offshore", "solar"]          # actual public net values, MW
PP_NAMES = {"Load": "load", "Wind onshore": "wind_onshore", "Wind offshore": "wind_offshore", "Solar": "solar"}
COV_LAG_DAYS = 2        # fundamentals for day D may only come from days <= D-2 (conservative publication rule)
PRICE_LAG_DAYS = 1      # prices for day D may only come from days <= D-1 (cleared at noon D-2)


# ---------------------------------------------------------------- build
def _load_series(name, field=None):
    """All raw month files of one series -> pd.Series indexed by unix seconds (de-duplicated, sorted)."""
    parts = []
    for f in sorted(glob.glob(os.path.join(RAW, name, "*.json"))):
        d = json.load(open(f))
        u = d.get("unix_seconds") or []
        if not u:
            continue
        if field is None:
            v = d["price"]
            parts.append(pd.Series(v, index=u, dtype="float64"))
        else:
            for p in d["production_types"]:
                if p["name"] == field:
                    parts.append(pd.Series(p["data"], index=u, dtype="float64"))
    s = pd.concat(parts)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _to_hourly(s, label, gaps):
    """Sub-hourly/hourly series (unix s) -> hourly UTC mean. An hour is valid only if all expected samples exist
    and are non-null; the expected count is 3600 / spacing, where spacing is the median spacing in that UTC day."""
    t = pd.to_datetime(s.index, unit="s", utc=True)
    df = pd.DataFrame({"v": s.values}, index=t)
    hour = df.index.floor("h")
    step = pd.Series(np.r_[np.diff(s.index.values), np.nan], index=t)
    day = df.index.floor("D")
    spacing = step.groupby(day).median().reindex(day).values          # per UTC day
    spacing = np.where(np.isfinite(spacing), spacing, 3600)
    exp = pd.Series(3600 / spacing, index=t).groupby(hour).first()
    g = df["v"].groupby(hour)
    cnt, mean = g.count(), g.mean()
    ok = cnt >= exp.reindex(cnt.index).round()
    full = pd.date_range(hour.min(), hour.max(), freq="h", tz="UTC")
    out = mean.where(ok).reindex(full)
    for ts in out.index[out.isna()]:
        gaps.append({"series": label, "utc_hour": ts.strftime("%Y-%m-%dT%H:%MZ"),
                     "reason": "missing" if ts not in cnt.index else f"incomplete ({int(cnt[ts])} samples)"})
    return out


def build():
    gaps = []
    p_old = _load_series("price_de_at_lu")
    p_new = _load_series("price_de_lu")
    split_utc = ZONE_SPLIT.tz_localize(TZ).tz_convert("UTC").timestamp()
    p_old = p_old[p_old.index < split_utc]
    p_new = p_new[p_new.index >= split_utc]
    price = _to_hourly(pd.concat([p_old, p_new]).sort_index(), "price", gaps)
    cols = {"price": price}
    for raw_name, c in PP_NAMES.items():
        cols[c] = _to_hourly(_load_series("public_power", raw_name), c, gaps)
    idx = cols["price"].index
    for c in COVS:
        idx = idx.union(cols[c].index)
    H = pd.DataFrame({k: v.reindex(idx) for k, v in cols.items()})
    H.index.name = "utc"
    loc = H.index.tz_convert(TZ)
    H["date"] = loc.strftime("%Y-%m-%d")
    H["slot"] = loc.hour
    H = H[(H["date"] >= FIRST_DAY.strftime("%Y-%m-%d"))]
    # restrict to complete local days that have at least one price
    last_price_day = H.loc[H["price"].notna(), "date"].max()
    H = H[H["date"] <= last_price_day]
    H["zone"] = np.where(H["date"] < ZONE_SPLIT.strftime("%Y-%m-%d"), "DE-AT-LU", "DE-LU")
    H["price_res"] = np.where(H["date"] < QH_SWITCH.strftime("%Y-%m-%d"), "60min", "15min")
    H = H.reset_index()
    H["utc"] = H["utc"].dt.strftime("%Y-%m-%dT%H:%MZ")
    H.to_csv(HOURLY, index=False, float_format="%.4f")
    gaps = [g for g in gaps if g["utc_hour"] >= H["utc"].iloc[0] and g["utc_hour"] <= H["utc"].iloc[-1]]
    summ = {}
    for g in gaps:
        summ[g["series"]] = summ.get(g["series"], 0) + 1
    json.dump({"n_gap_hours": summ, "gaps": gaps}, open(GAPS, "w"), indent=0)
    sha = hashlib.sha256(open(HOURLY, "rb").read()).hexdigest()
    print(f"hourly.csv: {len(H)} rows, {H['date'].iloc[0]} .. {H['date'].iloc[-1]}, sha256 {sha}")
    print("gap hours:", summ)
    return sha


# ---------------------------------------------------------------- panel
_PANEL = {}


def panel():
    """-> dates [n] (datetime64[D]), P [n, 24] price EUR/MWh, C [n, 24, 4] covariates (COVS order), flags dict.
    Spring-forward slot 2 is NaN; fall-back slot 2 is the mean of the two hours."""
    if "p" in _PANEL:
        return _PANEL["p"]
    H = pd.read_csv(HOURLY)
    g = H.groupby(["date", "slot"])[["price"] + COVS].mean()      # fall-back day: mean of the two slot-2 hours
    n_per = H.groupby(["date", "slot"]).size()
    days = pd.to_datetime(sorted(H["date"].unique()))
    full = pd.MultiIndex.from_product([days.strftime("%Y-%m-%d"), range(24)], names=["date", "slot"])
    g = g.reindex(full)
    P = g["price"].to_numpy().reshape(len(days), 24)
    C = np.stack([g[c].to_numpy().reshape(len(days), 24) for c in COVS], -1)
    n_per = n_per.reindex(full).fillna(0).to_numpy().reshape(len(days), 24)
    flags = {"dst_gap": np.where((n_per == 0).any(1))[0], "dst_dup": np.where((n_per == 2).any(1))[0]}
    dates = days.values.astype("datetime64[D]")
    _PANEL["p"] = (dates, P, C, flags)
    return _PANEL["p"]


def day_index(dates, day):
    i = np.searchsorted(dates, np.datetime64(pd.Timestamp(day).date(), "D"))
    if i >= len(dates) or dates[i] != np.datetime64(pd.Timestamp(day).date(), "D"):
        raise KeyError(day)
    return int(i)


def forecast_time_utc(target_day):
    """The forecast for delivery day D is issued at the auction gate: 12:00 local on D-1."""
    d = pd.Timestamp(target_day) - pd.Timedelta(days=1)
    return pd.Timestamp(d.year, d.month, d.day, 12).tz_localize(TZ).tz_convert("UTC")


def day_start_utc(day):
    d = pd.Timestamp(day)
    return pd.Timestamp(d.year, d.month, d.day).tz_localize(TZ).tz_convert("UTC")


def interp_dst_gap(x):
    """Model INPUT only: linearly interpolate isolated NaN slots (the spring-forward slot 2) so contexts are regular.
    Targets are never interpolated. Longer gaps stay NaN."""
    x = np.array(x, float)
    bad = np.where(~np.isfinite(x))[0]
    for i in bad:
        if 0 < i < len(x) - 1 and np.isfinite(x[i - 1]) and np.isfinite(x[i + 1]):
            x[i] = 0.5 * (x[i - 1] + x[i + 1])
    return x


def build_context(i, ctx_days, dates=None, P=None, C=None, flags=None):
    """Information-set-safe inputs for target day index i (delivery day dates[i]).

    Returns dict(prices [ctx_days*24], covs [ctx_days*24, 4], ctx_dates [ctx_days], target_day).
      prices: slots of days i-ctx_days .. i-1 (oldest first), spring-forward slot interpolated (input only);
      covs:   same rows; rows of day i-1 are NaN (fundamentals only from days <= D-2).
    Assertions enforce the information set with explicit UTC timestamps."""
    if dates is None:
        dates, P, C, flags = panel()
    lo = i - ctx_days
    if lo < 0:
        raise ValueError(f"not enough history for day {dates[i]}")
    D = pd.Timestamp(dates[i])
    ft = forecast_time_utc(D)
    ctx_days_arr = dates[lo:i]
    # ---- price information set: every context price belongs to a day <= D-1, whose auction cleared at noon of
    # the day before it (<= noon D-2) -- strictly before the forecast time (noon D-1).
    assert (ctx_days_arr <= np.datetime64((D - pd.Timedelta(days=PRICE_LAG_DAYS)).date(), "D")).all()
    last_price_day = pd.Timestamp(ctx_days_arr[-1])
    assert forecast_time_utc(last_price_day) < ft, "price context published after forecast time"
    assert day_start_utc(last_price_day + pd.Timedelta(days=1)) <= day_start_utc(D), "price context overlaps target day"
    prices = interp_dst_gap(P[lo:i].reshape(-1))
    covs = C[lo:i].copy()
    if COV_LAG_DAYS > 1:
        covs[ctx_days - (COV_LAG_DAYS - 1):] = np.nan        # day D-1 unknown under the conservative rule
    covs = covs.reshape(-1, C.shape[-1])
    # ---- covariate information set: the last non-NaN covariate row must end at or before the start of D-1
    known = np.isfinite(covs).any(1)
    if known.any():
        last_row = np.where(known)[0][-1]
        last_cov_day = pd.Timestamp(ctx_days_arr[last_row // 24])
        assert last_cov_day <= D - pd.Timedelta(days=COV_LAG_DAYS), "covariate from a day after D-2"
        assert day_start_utc(last_cov_day + pd.Timedelta(days=1)) <= day_start_utc(D - pd.Timedelta(days=1)) < ft
    assert prices.shape == (ctx_days * 24,) and covs.shape == (ctx_days * 24, C.shape[-1])
    return {"prices": prices, "covs": covs, "ctx_dates": ctx_days_arr, "target_day": dates[i]}


def target(i, dates=None, P=None):
    if dates is None:
        dates, P, _, _ = panel()
    return P[i].copy()


# ---------------------------------------------------------------- training helpers (DEV only)
def train_panel(end=None):
    """Panel restricted to delivery days <= TRAIN_END (2025-06-30). Passing a later `end` raises."""
    end = TRAIN_END if end is None else pd.Timestamp(end)
    if end > TRAIN_END:
        raise ValueError(f"training data may not extend past {TRAIN_END.date()} (asked {end.date()})")
    dates, P, C, flags = panel()
    m = dates <= np.datetime64(end.date(), "D")
    out = (dates[m], P[m].copy(), C[m].copy())
    assert out[0].max() <= np.datetime64(TRAIN_END.date(), "D")
    return out


def train_windows(ctx_days=28, start=None, end=None, stride=1, skip_missing_target=True):
    """Materialised training windows from DEV data only.
    Returns X [n, ctx_days*24] (prices), XC [n, ctx_days*24, 4] (covs; last day NaN), Y [n, 24], target_dates [n].
    Asserts that no target day is after 2025-06-30. For long contexts use train_panel() + build_context() lazily."""
    dates, P, C = train_panel(end)
    flags = {"dst_gap": np.array([], int), "dst_dup": np.array([], int)}
    lo = ctx_days if start is None else max(ctx_days, int(np.searchsorted(dates, np.datetime64(pd.Timestamp(start).date(), "D"))))
    X, XC, Y, T = [], [], [], []
    for i in range(lo, len(dates), stride):
        y = P[i]
        if skip_missing_target and not np.isfinite(y).all():
            continue
        w = build_context(i, ctx_days, dates, P, C, flags)
        X.append(w["prices"]), XC.append(w["covs"]), Y.append(y), T.append(dates[i])
    T = np.array(T)
    assert len(T) == 0 or T.max() <= np.datetime64(TRAIN_END.date(), "D"), "training target after TRAIN_END"
    return np.array(X, np.float32), np.array(XC, np.float32), np.array(Y, np.float32), T


def split_days(split, dates=None):
    """Target day indices of a split. TEST ends at the last delivery day with a complete price vector."""
    if dates is None:
        dates, P, _, _ = panel()
    else:
        P = panel()[1]
    if split == "dev":
        lo, hi = DEV_BT_START, TRAIN_END
    elif split == "val":
        lo, hi = VAL_START, VAL_END
    elif split == "test":
        lo = TEST_START
        ok = np.where(np.isfinite(P).sum(1) >= 23)[0]
        hi = pd.Timestamp(dates[ok[-1]])
    else:
        raise ValueError(split)
    m = (dates >= np.datetime64(lo.date(), "D")) & (dates <= np.datetime64(hi.date(), "D"))
    return np.where(m)[0]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
