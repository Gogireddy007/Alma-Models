"""Re-download every Alma-Fin raw file listed in data/fin/manifest.json and check its sha256 against the pin."""
import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "repo")
man = json.load(open(os.path.join(REPO, "data/fin/manifest.json")))["files"]


def one(item):
    key, meta = item
    dst = os.path.join(REPO, key)
    if os.path.exists(dst) and hashlib.sha256(open(dst, "rb").read()).hexdigest() == meta["sha256"]:
        return key, "ok-cached"
    for k in range(6):
        try:
            req = urllib.request.Request(meta["url"], headers={"User-Agent": "Mozilla/5.0"})
            body = urllib.request.urlopen(req, timeout=120).read()
            break
        except Exception as e:  # noqa: BLE001
            time.sleep(3 * (k + 1))
    else:
        return key, "FAILED"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "wb").write(body)
    return key, "ok" if hashlib.sha256(body).hexdigest() == meta["sha256"] else "SHA-DIFF"


res = {}
with ThreadPoolExecutor(8) as ex:
    for n, (k, s) in enumerate(ex.map(one, man.items())):
        res[k] = s
        if n % 500 == 0:
            print(n, k, s, flush=True)
bad = {k: s for k, s in res.items() if not s.startswith("ok")}
print(f"done {len(res)} files, {len(bad)} not matching")
for k, s in list(bad.items())[:40]:
    print(s, k)
json.dump(bad, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fin_fetch_bad.json"), "w"), indent=1)
sys.exit(0)
