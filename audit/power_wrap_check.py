import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "repo", "tools", "power"))
import run_eval as R
R.RUN_DIR["test"] = os.path.join(HERE, "power_test"); R.forecast("test", ["wm_power_blend"])
a = np.load(f"{HERE}/power_test/wm_power.npz")["q"]; b = np.load(f"{HERE}/power_test/lear.npz")["q"]; c = np.load(f"{HERE}/power_test/wm_power_blend.npz")["q"]
print("wrapper == offline blend:", np.allclose(c, np.sort(0.5 * a + 0.5 * b, 1)), np.abs(c - np.sort(0.5 * a + 0.5 * b, 1)).max())
