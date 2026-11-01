"""Alma-Fin agent `wm_agent` (architect): actor-critic trained in latent imagination of the frozen wm_fin world model (claim model). See tools/fin/agents/_rl.py and docs/fin_search.md."""
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
FROZEN_CKPT = ("runs/fin/search/ckpt/frozen_wm_agent.pt", "acf5559970505e1aa31dd3e1ca1e596d716cfa548845e09adddb208e9bb01594")


def make():
    rel, sh = FROZEN_CKPT
    p = os.path.join(ROOT, rel)
    got = WF._sha(p)
    if sh is not None and got != sh:
        raise RuntimeError(f"wm_agent: policy checkpoint {rel} sha256 {got} != frozen {sh}")
    members, wmeta = WF.load_frozen("wm_fin")
    pols, meta = RL.load_policies(p)
    return RL.RLAgent("dreamer", members, pols, dict(meta, ckpt=rel, ckpt_sha256=got, wm_ckpt=wmeta["ckpt"],
                                                    wm_ckpt_sha256=wmeta["ckpt_sha256"]))


def register(R):
    R["wm_agent"] = make
