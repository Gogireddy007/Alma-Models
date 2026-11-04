import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "repo", "tools", "fin"))
import run_eval as R
R.RUN_DIR["val"] = os.path.join(HERE, "fin_val")
R.forecast("val", "crypto", ["wm_fin", "wm_direct", "har_intra_dow", "har_dow", "lgbm_dow"])
