"""
Fix the Alma-Fin universe ONCE, using only data dated <= 2025-10-31 (docs/fin_research.md section 1, docs/fin_prereg.md
section 1). Inputs: data/fin/listing_um.json (S3 listing incl. delisted symbols) and the 1d USD-M klines of 2024-10,
2024-11 and 2025-10 (data/fin/raw/um/klines_1d). Output: data/fin/universe.json.

Rule (pre-registered):
  candidates = USDT-margined perpetuals (no '_' delivery suffix) with a 1d file for 2025-10, whose FIRST daily bar
               (open_time) is <= 2024-11-01 00:00 UTC; excluded: index products and stablecoins (EXCLUDE below).
  rank       = sum of quote_volume of the 30 daily bars 2025-10-02 .. 2025-10-31 (UTC), descending.
  universe   = top 24. Held-out coins = ranks 3, 6, 9, ..., 24 (every third); training coins = the other 16.
No later information (no TEST bar) is read: rows with open_time >= 2025-11-01 are asserted absent.
"""
import io
import os
import sys
import json
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "fin", "raw", "um", "klines_1d")
OUT = os.path.join(ROOT, "data", "fin", "universe.json")
EXCLUDE = {"BTCDOMUSDT", "DEFIUSDT", "FOOTBALLUSDT", "BLUEBIRDUSDT", "USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT",
           "USDPUSDT", "DAIUSDT", "EURUSDT", "AEURUSDT", "XAUUSDT", "XAGUSDT", "PAXGUSDT", "USD1USDT"}
LISTED_BY = pd.Timestamp("2024-11-01", tz="UTC")
W0, W1 = pd.Timestamp("2025-10-02", tz="UTC"), pd.Timestamp("2025-10-31", tz="UTC")
N, HELD_EVERY = 24, 3
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def read_zip(path):
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    first = raw.split(b"\n", 1)[0]
    df = pd.read_csv(io.BytesIO(raw), header=0 if first.startswith(b"open_time") else None)
    df.columns = COLS[:df.shape[1]]
    ot = df["open_time"].astype("int64")
    unit = "us" if ot.max() > 10 ** 14 else "ms"
    df["t"] = pd.to_datetime(ot, unit=unit, utc=True)
    return df


def main():
    cands = [l.strip() for l in open(os.path.join(ROOT, "data", "fin", "universe_candidates.txt")) if l.strip()]
    rows, notes = [], {}
    for s in cands:
        if s in EXCLUDE:
            notes[s] = "excluded (index/stablecoin/commodity)"
            continue
        f10 = os.path.join(RAW, s, f"{s}-1d-2025-10.zip")
        if not os.path.exists(f10):
            notes[s] = "no 2025-10 file"
            continue
        first = None
        for m in ("2024-10", "2024-11"):
            p = os.path.join(RAW, s, f"{s}-1d-{m}.zip")
            if os.path.exists(p):
                first = read_zip(p)["t"].min()
                break
        if first is None:
            notes[s] = "no 2024-10/11 file"
            continue
        if first > LISTED_BY and first.strftime("%Y-%m") != "2024-10":
            notes[s] = f"first bar {first} > 2024-11-01"
            continue
        if first.strftime("%Y-%m") == "2024-11" and first > LISTED_BY:
            notes[s] = f"first bar {first} > 2024-11-01"
            continue
        d = read_zip(f10)
        assert d["t"].max() < pd.Timestamp("2025-11-01", tz="UTC"), s
        w = d[(d["t"] >= W0) & (d["t"] <= W1)]
        rows.append({"symbol": s, "quote_volume_30d": float(w["quote_volume"].sum()), "n_days": int(len(w)),
                     "first_bar_in_2024_10_11": str(first)})
    r = pd.DataFrame(rows).sort_values("quote_volume_30d", ascending=False).reset_index(drop=True)
    r["rank"] = np.arange(1, len(r) + 1)
    top = r.head(N).copy()
    top["role"] = np.where(top["rank"] % HELD_EVERY == 0, "heldout", "train")
    out = {"rule": __doc__.strip(), "as_of": "2025-10-31", "n_candidates": len(cands), "n_eligible": len(r),
           "universe": top.to_dict("records"), "train": top[top.role == "train"].symbol.tolist(),
           "heldout": top[top.role == "heldout"].symbol.tolist(),
           "next_10": r.iloc[N:N + 10][["symbol", "rank", "quote_volume_30d"]].to_dict("records"),
           "notes": notes}
    json.dump(out, open(OUT, "w"), indent=1)
    print(top[["rank", "symbol", "quote_volume_30d", "n_days", "role"]].to_string(index=False))
    print(f"eligible {len(r)} of {len(cands)}; notes {len(notes)}")


if __name__ == "__main__":
    sys.exit(main())
