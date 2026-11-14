"""Mean +/- std over seeds for an ablation experiment, keyed by window and feature set.

Usage: python src/ablation_table.py <experiment>"""
import json
import sys
from pathlib import Path

import pandas as pd

d = pd.read_csv(Path(__file__).resolve().parents[1] / "results.csv")
d = d[d["experiment"] == sys.argv[1]].copy()
d["window"] = d["config"].map(lambda c: json.loads(c)["window"])
d["kinds"] = d["config"].map(lambda c: "+".join(json.loads(c)["kinds"]))
cols = ["f1", "recall", "precision", "pr_auc", "train_f1"]
g = d.groupby(["split", "model", "window", "kinds"], sort=False)[cols].agg(["mean", "std"])
out = pd.DataFrame({c: [f"{m:.3f} ± {s:.3f}" if pd.notna(m) else "" for m, s in zip(g[(c, "mean")], g[(c, "std")])]
                    for c in cols}, index=g.index)
pd.set_option("display.width", 250, "display.max_rows", 500)
print(out.sort_index(level=[0, 1], sort_remaining=False).to_string())
