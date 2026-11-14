"""Figures and error analysis for REPORT.md. Reads results.csv and logs/test_predictions.csv.

Usage (from improved_implementation/): python src/analysis.py"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pm.common import ROOT, log_result

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
NICE = {"logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree",
        "random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting"}
COL = {"logistic_regression": "#1f77b4", "decision_tree": "#7f7f7f",
       "random_forest": "#2ca02c", "gradient_boosting": "#ff7f0e"}
BASE, NEW = "#9aa5b1", "#1f77b4"

res = pd.read_csv(ROOT / "results.csv")


def ms(d, col="f1"):
    return d[col].mean(), d[col].std(ddof=1) if len(d) > 1 else 0.0


# 1. Why the cross-validation score overstated the test score -------------------------------
cb = res[res.experiment == "corrected_baseline"]
splits = [("cv_grouped5_shuffled_allrows_oof", "Validation: all rows\n(as in the original)"),
          ("cv_grouped5_truncated_rows_oof", "Validation: engines cut early\n(mimics the test set)"),
          ("test_all_rows", "Test set")]
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(NICE))
for i, (sp, lab) in enumerate(splits):
    vals = [ms(cb[(cb.model == m) & (cb.split == sp)]) for m in NICE]
    ax.bar(x + (i - 1) * 0.27, [v[0] for v in vals], 0.27, yerr=[v[1] for v in vals], capsize=3,
           label=lab, color=["#c9d1d9", "#6b9bd1", "#1f3b63"][i])
ax.set_xticks(x, NICE.values()); ax.set_ylim(0.5, 1.0); ax.set_ylabel("F1 (row level)")
ax.set_title("Baseline: validation only predicts the test score when it mimics the test set")
ax.legend(fontsize=8, loc="upper right", ncol=3)
fig.tight_layout(); fig.savefig(FIG / "fig1_validation_vs_test.png"); plt.close(fig)

# 2. Rolling-window ablation -----------------------------------------------------------------
wa = res[(res.experiment == "window_ablation") & (res.split == "cv_grouped5_truncated_rows_oof")].copy()
wa["window"] = wa.config.map(lambda c: json.loads(c)["window"])
fig, ax = plt.subplots(figsize=(7.5, 4))
for m in NICE:
    g = wa[wa.model == m].groupby("window").f1.agg(["mean", "std"])
    ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=3, label=NICE[m], color=COL[m])
ax.axvline(40, color="k", ls=":", lw=1); ax.text(40.5, 0.64, "chosen: 40", fontsize=9)
ax.set_xlabel("Rolling window length (cycles)"); ax.set_ylabel("Validation F1 (engines cut early)")
ax.set_title("A longer rolling window helps; logistic regression is best or tied at every length")
ax.set_xticks([5, 10, 20, 30, 40, 50]); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig2_window_ablation.png"); plt.close(fig)

# 3. Feature-set ablation at window 40 -------------------------------------------------------
fa = res[(res.experiment == "feature_set_ablation") & (res.split == "cv_grouped5_truncated_rows_oof")].copy()
fa["kinds"] = fa.config.map(lambda c: "+".join(json.loads(c)["kinds"]))
kinds = [("raw+mean+std", "raw + mean + std"), ("raw+mean", "raw + mean"), ("mean", "mean only")]
fig, ax = plt.subplots(figsize=(8, 4))
for i, (k, lab) in enumerate(kinds):
    vals = [ms(fa[(fa.model == m) & (fa.kinds == k)]) for m in NICE]
    ax.bar(x + (i - 1) * 0.27, [v[0] for v in vals], 0.27, yerr=[v[1] for v in vals], capsize=3,
           label=lab, color=["#1f3b63", "#6b9bd1", "#c9d1d9"][i])
ax.set_xticks(x, NICE.values()); ax.set_ylim(0.4, 1.0); ax.set_ylabel("Validation F1 (engines cut early)")
ax.set_title("Window 40: the rolling std (recent drift) carries much of the signal")
ax.legend(fontsize=8, ncol=3)
fig.tight_layout(); fig.savefig(FIG / "fig3_feature_sets.png"); plt.close(fig)

# 4-5. Error analysis on the test set (seed 42, threshold 0.5) --------------------------------
p = pd.read_csv(ROOT / "logs" / "test_predictions.csv")
tags = {"corrected_baseline_rescored": "Baseline (window 5)", "final_improved": "Improved (window 40)"}
buckets, labels = [-1, 10, 20, 25, 30], ["0–10", "11–20", "21–25", "26–30"]
pos = p[p.label == 1].copy()
pos["bucket"] = pd.cut(pos.RUL, buckets, labels=labels)
already = (res.experiment == "error_analysis").any()
fig, ax = plt.subplots(figsize=(7, 4))
for i, (tag, lab) in enumerate(tags.items()):
    caught = pos[f"{tag}_seed42"] >= 0.5
    rec = caught.groupby(pos.bucket, observed=False).mean()
    n = caught.groupby(pos.bucket, observed=False).size()
    ax.bar(np.arange(4) + (i - 0.5) * 0.38, rec.values, 0.38, label=lab, color=[BASE, NEW][i])
    for j, v in enumerate(rec.values):
        ax.text(j + (i - 0.5) * 0.38, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    if not already:
        for b in labels:
            log_result(phase="VI", experiment="error_analysis", task="clf", model="logistic_regression",
                       seed=42, config={"variant": tag}, split=f"test_rows_trueRUL_{b}",
                       n_rows_eval=int(n[b]), recall=float(rec[b]), threshold=0.5,
                       notes="recall among positive test rows in this true-RUL bucket")
        neg = p[p.label == 0]
        fa_rows = neg[neg[f"{tag}_seed42"] >= 0.5]
        log_result(phase="VI", experiment="error_analysis", task="clf", model="logistic_regression",
                   seed=42, config={"variant": tag}, split="test_rows_false_alarms",
                   n_rows_eval=len(fa_rows), threshold=0.5,
                   notes=f"false-alarm rows; median true RUL {fa_rows.RUL.median():.0f}, max {fa_rows.RUL.max():.0f}")
ax.set_xticks(range(4), [f"{l}\n(n={int(k)})" for l, k in zip(labels, n.values)])
ax.set_xlabel("True cycles left before failure"); ax.set_ylabel("Share of rows caught (recall)")
ax.set_ylim(0, 1.12); ax.legend(fontsize=8, loc="lower left")
ax.set_title("Test set: the gain is at the hard cases, 21–30 cycles from failure")
fig.tight_layout(); fig.savefig(FIG / "fig4_recall_by_rul.png"); plt.close(fig)

last = p.loc[p.groupby("unit").cycle.idxmax()]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (tag, lab) in zip(axes, tags.items()):
    pr = last[f"{tag}_seed42"]
    wrong = (pr >= 0.5) != (last.label == 1)
    ax.scatter(last.RUL[~wrong], pr[~wrong], s=18, color=NEW, label="correct decision")
    ax.scatter(last.RUL[wrong], pr[wrong], s=40, color="#d62728", marker="x", label="wrong decision")
    ax.axvline(30, color="k", ls=":", lw=1); ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_title(f"{lab}: {int(wrong.sum())} wrong of 100 engines"); ax.set_xlabel("True cycles left at last reading")
    print(lab, "wrong engines:", last[wrong][["unit", "RUL"]].assign(prob=pr[wrong].round(3)).to_dict("records"))
axes[0].set_ylabel("Predicted probability of failing within 30 cycles"); axes[0].legend(fontsize=8)
fig.suptitle("One decision per test engine at its last available cycle", y=1.0)
fig.tight_layout(); fig.savefig(FIG / "fig5_engine_decisions.png"); plt.close(fig)

# 6. What the final model uses ---------------------------------------------------------------
c = pd.read_csv(ROOT / "logs" / "coef_final_improved.csv", index_col=0).iloc[:, 0]
top = c.reindex(c.abs().sort_values(ascending=False).index).head(12)[::-1]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(top.index, top.values, color=["#d62728" if v > 0 else NEW for v in top.values])
ax.set_xlabel("Weight on the standardised feature (red pushes towards 'fails soon')")
ax.set_title("Final logistic regression: 12 largest weights")
fig.tight_layout(); fig.savefig(FIG / "fig6_coefficients.png"); plt.close(fig)
kind = c.index.to_series().map(lambda n: "rolling std" if "_std" in n else "rolling mean" if "_mean" in n else "raw")
print("share of |weight| by feature type:", (c.abs().groupby(kind.values).sum() / c.abs().sum()).round(2).to_dict())
print("top features:", list(top.index[::-1]))
print("figures written to", FIG)
