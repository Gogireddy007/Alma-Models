"""Alma-Fin agent `dqn` (architect): model-free DQN twin (5 sizes) on the same observations, real DEV transitions only. See tools/fin/agents/_rl.py and docs/fin_search.md."""
import os
import sys
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
FIN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(FIN))
sys.path.insert(0, FIN)
_spec = importlib.util.spec_from_file_location("fin_agents_rl", os.path.join(HERE, "_rl.py"))
RL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RL)
import wm_fin as WF      # noqa: E402

# filled at freeze; the factory refuses a policy checkpoint whose sha256 differs
FROZEN_CKPT = ("runs/fin/search/ckpt/frozen_dqn.pt", "97a9da54fd7ace7d69a46eaf2722fd9f74d322ad0033cd6c64bae998c30597be")


def make():
    rel, sh = FROZEN_CKPT
    p = os.path.join(ROOT, rel)
    got = WF._sha(p)
    if sh is not None and got != sh:
        raise RuntimeError(f"dqn: policy checkpoint {rel} sha256 {got} != frozen {sh}")
    members, wmeta = WF.load_frozen("wm_fin")
    pols, meta = RL.load_policies(p)
    return RL.RLAgent("dqn", members, pols, dict(meta, ckpt=rel, ckpt_sha256=got, wm_ckpt=wmeta["ckpt"],
                                                    wm_ckpt_sha256=wmeta["ckpt_sha256"]))


def register(R):
    R["dqn"] = make
