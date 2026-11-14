"""Print mean +/- std across seeds for every (experiment, model, split) in results.csv.

Usage: python src/summarize.py [experiment-substring]"""
import sys
from pathlib import Path

import pandas as pd

d = pd.read_csv(Path(__file__).resolve().parents[1] / "results.csv")
if len(sys.argv) > 1:
    d = d[d["experiment"].str.contains(sys.argv[1])]
metrics = ["f1", "recall", "precision", "roc_auc", "pr_auc", "rmse", "mae", "nasa_score", "r2",
           "rmse_capped125", "frac_late", "train_f1", "train_rmse", "train_time_s",
           "infer_ms_single", "infer_us_per_row_batch", "peak_mem_mb", "n_params"]
pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 500)
for (exp, task), g in d.groupby(["experiment", "task"], sort=False):
    cols = [m for m in metrics if g[m].notna().any()]
    agg = g.groupby(["model", "split"], sort=False)[cols].agg(["mean", "std", "count"])
    out = pd.DataFrame(index=agg.index)
    out["n_seeds"] = agg[(cols[0], "count")]
    for m in cols:
        mu, sd = agg[(m, "mean")], agg[(m, "std")].fillna(0)
        out[m] = [f"{a:.4g} ± {b:.2g}" if pd.notna(a) else "" for a, b in zip(mu, sd)]
    print(f"\n=== {exp} [{task}] ===")
    print(out.to_string())
