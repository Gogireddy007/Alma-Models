"""Download every file listed in a committed data manifest that is missing on disk, and check its sha256 pin.

    .venv/bin/python tools/deploy/fetch_from_manifest.py data/fin/manifest.json [--only /um/]

Exit code 1 if any pinned file cannot be fetched or does not match its pin. Files are written exactly as served.
Growing files (current-year archives) are expected to differ; they are reported, not failed, when --allow-growing
names a substring of their path.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--only", default="")
    ap.add_argument("--allow-growing", default="/2026/,2026.json")
    a = ap.parse_args()
    man = json.load(open(os.path.join(ROOT, a.manifest)))["files"]
    items = [(k, v) for k, v in man.items() if a.only in k and "url" in v and "sha256" in v]
    grow = [g for g in a.allow_growing.split(",") if g]

    def one(item):
        key, meta = item
        dst = os.path.join(ROOT, key)
        if os.path.exists(dst):
            return key, "ok-cached" if hashlib.sha256(open(dst, "rb").read()).hexdigest() == meta["sha256"] else "diff-cached"
        for k in range(6):
            try:
                body = urllib.request.urlopen(urllib.request.Request(meta["url"], headers={"User-Agent": "alma-research"}),
                                              timeout=120).read()
                break
            except Exception:  # noqa: BLE001
                time.sleep(3 * (k + 1))
        else:
            return key, "FAILED"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "wb").write(body)
        return key, "ok" if hashlib.sha256(body).hexdigest() == meta["sha256"] else "diff"

    res = dict(ThreadPoolExecutor(8).map(one, items))
    bad = {k: s for k, s in res.items() if not s.startswith("ok") and not any(g in k for g in grow)}
    print(f"{a.manifest}: {dict(Counter(res.values()))}; failures: {len(bad)}")
    for k, s in list(bad.items())[:20]:
        print("  ", s, k)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
