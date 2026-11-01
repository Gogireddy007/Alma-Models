"""Red-team attacks on the paper ledger (tools/live/paper.py). Runs ONLY on throw-away books under a temp root
(never touches runs/live/paper). Attacks:
  A1 edit a past line's target            -> verify must fail
  A2 reorder two lines                    -> verify must fail
  A3 truncate the tail (drop a losing day)-> verify must fail (needs an external anchor)
  A4 rewrite the tail with recomputed hashes (forged history) -> must fail
  A5 open a LIVE ledger as Book(mode='sim') and backfill a past day -> must be refused
  A6 settle across a split ex-date with raw (unadjusted) closes -> P&L must be ~market move, not the split ratio
    .venv/bin/python tools/live/redteam/rt_ledger.py
"""
import json
import os
import shutil
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402
import paper as PB  # noqa: E402


class M:
    name = "rt_long"
    meta = {}

    def __init__(self, w):
        self.w = w

    def act(self, ctx):
        return self.w


def ctx_for(day):
    return {"live": False, "t": C.decision_time(day), "meta": {"input_sha256": "x", "max_published_at": None,
                                                               "max_fetched_at": None}, "ng_roll": {"active": "NA"}}


def run_days(B, days, closes, w):
    for d in days:
        B.decide(d, M(w), ctx_for(d), now=C.decision_time(d))
        B.settle(d, closes[d], now=C.et(d, 18, 5))


def main():
    root = tempfile.mkdtemp(prefix="rt_ledger_", dir=os.environ.get("TMPDIR"))
    res = {}
    try:
        days = [pd.Timestamp(x) for x in ("2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29")]
        cl = {d: {"UNG": p, "BOIL": 30.0} for d, p in zip(days, (12.0, 12.5, 11.0, 11.2))}
        B = PB.Book("t", mode="sim", root=root)
        run_days(B, days, cl, {"UNG": 1.0})
        path = B.L.path

        def rechain(lines):
            prev, out = PB.GENESIS, []
            for x in lines:
                x = dict(x)
                x["prev"] = prev
                x.pop("hash", None)
                x["hash"] = PB.hashlib.sha256((prev + PB.canon(x)).encode()).hexdigest()
                prev = x["hash"]
                out.append(PB.canon(x))
            return out
        # sim books write genesis at the real wall clock and decisions at past times -> verify() always fails on a
        # sim book (LRT note); re-stamp genesis to 2026-10-01 so the untouched baseline verifies
        raw_ok = PB.Ledger(path).verify()
        L0 = [json.loads(x) for x in open(path).read().splitlines()]
        L0[0]["written_at"] = "2026-10-01T00:00:00Z"
        orig = rechain(L0)
        open(path, "w").write("\n".join(orig) + "\n")
        res["A0_baseline"] = PB.Ledger(path).verify()
        res["A0_sim_raw_verify"] = raw_ok

        def verify_text(lines):
            open(path, "w").write("\n".join(lines) + "\n")
            return PB.Ledger(path).verify()

        L = [json.loads(x) for x in orig]
        # A1 edit
        e = [dict(x) for x in L]
        k = next(i for i, x in enumerate(e) if x["kind"] == "decision")
        e[k]["target"] = {"UNG": -1.0}
        res["A1_edit"] = verify_text([PB.canon(x) for x in e])
        # A2 reorder
        r = orig[:]
        r[2], r[3] = r[3], r[2]
        res["A2_reorder"] = verify_text(r)
        # A3 truncate: drop the last day's decision/fill/mark (the losing day 10-28 -> keep up to 10-27)
        cut = max(i for i, x in enumerate(L) if x.get("day") == "2026-10-27") + 1
        committed = ("\n".join(orig) + "\n").encode()
        real_co = PB.subprocess.check_output
        PB.subprocess.check_output = lambda *a, **k: committed if "show" in a[0] else real_co(*a, **k)

        def verify_both(lines):
            ok1, m1 = verify_text(lines)
            ok2, m2 = PB.Ledger(path).verify_anchor() if hasattr(PB.Ledger, "verify_anchor") else (True, "no anchor")
            return ok1 and ok2, f"chain: {m1}; anchor: {m2}"
        res["A3_truncate"] = verify_both(orig[:cut])
        # A4 forge: change a decision and recompute all later hashes
        f = [dict(x) for x in L]
        prev = PB.GENESIS
        for i, x in enumerate(f):
            if x["kind"] == "decision" and x["day"] == "2026-10-28":
                x["target"] = {"UNG": 0.0}
            x["prev"] = prev
            x.pop("hash", None)
            x["hash"] = PB.hashlib.sha256((prev + PB.canon(x)).encode()).hexdigest()
            prev = x["hash"]
        res["A4_forge"] = verify_both([PB.canon(x) for x in f])
        PB.subprocess.check_output = real_co
        open(path, "w").write("\n".join(orig) + "\n")
        # A5 live ledger reopened in sim mode
        _anchor = getattr(PB.Ledger, "anchor", None)
        PB.Ledger.anchor = lambda self, msg: True          # never commit from a red-team test
        B2 = PB.Book("live_book", mode="live", root=root)
        gen = B2.L.lines[0]
        try:
            B3 = PB.Book("live_book", mode="sim", root=root)
            d = pd.Timestamp("2026-10-27")
            B3.decide(d, M({"UNG": 1.0}), ctx_for(d), now=C.utcnow())
            res["A5_sim_on_live"] = (False, f"backfill ACCEPTED into a ledger whose genesis mode={gen['mode']}")
        except SystemExit as ex:
            res["A5_sim_on_live"] = (True, f"refused: {ex}")
        if _anchor is not None:
            PB.Ledger.anchor = _anchor
        # A6 split: BOIL 1:2 reverse split ex-date 2026-05-28; raw close doubles while the market is flat
        B4 = PB.Book("split", mode="sim", root=root)
        d1, d2 = pd.Timestamp("2026-05-27"), pd.Timestamp("2026-05-28")
        sp = {"UNG": [], "BOIL": [("2026-05-28", 0.5)]}
        for d, c in ((d1, {"UNG": 12.0, "BOIL": 10.0}), (d2, {"UNG": 12.0, "BOIL": 20.0})):
            B4.decide(d, M({"BOIL": 0.5}), ctx_for(d), now=C.decision_time(d))
            try:
                B4.settle(d, c, now=C.et(d, 18, 5), splits=sp)
            except TypeError:
                B4.settle(d, c, now=C.et(d, 18, 5))
        mk = B4.L.of("mark")
        res["A6_split_pnl"] = (abs(mk[-1]["ret"]) < 0.05, f"ret on split day with flat market = {mk[-1]['ret']:+.2%}"
                               f"; halts {[h['reason'] for h in B4.L.of('halt')]}")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    for k, (ok, msg) in res.items():
        detected = (not ok) if k.startswith(("A1", "A2", "A3", "A4")) else ok

        print(f"{k:16s} {'DETECTED/OK' if detected else 'NOT DETECTED'}  ({msg})")
    return res


if __name__ == "__main__":
    main()
