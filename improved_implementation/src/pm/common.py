"""Shared data loading, causal features, grouped splitting, metrics and result logging."""
import csv
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]            # improved_implementation/
RESULTS_CSV = ROOT / "results.csv"

SETTINGS = ["setting_1", "setting_2", "setting_3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit", "cycle"] + SETTINGS + SENSORS

RESULT_FIELDS = [
    "run_id", "timestamp", "phase", "experiment", "task", "model", "config", "seed", "split",
    "n_units_eval", "n_rows_eval",
    "f1", "recall", "precision", "roc_auc", "pr_auc", "threshold",
    "rmse", "mae", "nasa_score", "r2", "rmse_capped125", "frac_late",
    "train_f1", "train_rmse",
    "train_time_s", "infer_ms_single", "infer_us_per_row_batch", "peak_mem_mb", "n_params",
    "source", "notes",
]


def load_config(name="baseline.yaml"):
    with open(ROOT / "config" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_raw(cfg):
    d = (ROOT / cfg["data"]["raw_dir"]).resolve()
    sub = cfg["data"]["subset"]
    rd = lambda n: pd.read_csv(d / n, sep=r"\s+", header=None, names=COLS)
    train, test = rd(f"train_{sub}.txt"), rd(f"test_{sub}.txt")
    rul = pd.read_csv(d / f"RUL_{sub}.txt", sep=r"\s+", header=None)[0].values
    train["RUL"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    last = test.groupby("unit")["cycle"].max()
    # row i of the RUL file belongs to unit i+1; used ONLY to build evaluation ground truth
    test["RUL"] = test["unit"].map(last + rul) - test["cycle"]
    w = cfg["data"]["fail_window"]
    train["label"] = (train["RUL"] <= w).astype(int)
    test["label"] = (test["RUL"] <= w).astype(int)
    return train, test


def add_rolling(df, sensors, window):
    """Trailing (causal) per-unit rolling mean/std, identical to src/make_nb03.py:91-98.

    No fitted parameters, so computing it before the split cannot leak across units."""
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    g = df.groupby("unit")[sensors]
    means = g.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).loc[df.index]
    stds = g.rolling(window, min_periods=1).std().reset_index(level=0, drop=True).loc[df.index].fillna(0)
    new = {f"{s}_mean{window}": means[s].values for s in sensors}
    new.update({f"{s}_std{window}": stds[s].values for s in sensors})
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def feature_names(sensors, window, kinds=("raw", "mean", "std")):
    out = []
    if "raw" in kinds:
        out += list(sensors)
    if "mean" in kinds:
        out += [f"{s}_mean{window}" for s in sensors]
    if "std" in kinds:
        out += [f"{s}_std{window}" for s in sensors]
    return out


class LowStdSensorDropper(BaseEstimator, TransformerMixin):
    """In-fold version of the 'std < 0.01 -> drop' rule (src/make_nb03.py:66-68).

    Fitted on the training portion only; drops a sensor together with its derived columns."""

    def __init__(self, sensors=SENSORS, window=5, threshold=0.01, kinds=("raw", "mean", "std")):
        self.sensors, self.window, self.threshold, self.kinds = sensors, window, threshold, kinds

    def fit(self, X, y=None):
        std = X[list(self.sensors)].std()
        self.kept_sensors_ = [s for s in self.sensors if std[s] >= self.threshold]
        self.columns_ = feature_names(self.kept_sensors_, self.window, self.kinds)
        return self

    def transform(self, X):
        return X[self.columns_]


def grouped_folds(units, n_splits, seed):
    """Shuffled GroupKFold on unit ids; asserts unit-level disjointness for every fold."""
    units = np.asarray(units)
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in gkf.split(units, groups=units):
        assert set(units[tr]).isdisjoint(set(units[va]))
        yield tr, va


def truncation_index(df, k, min_cut, rng):
    """For each unit pick k random cut cycles in [min_cut, T-1]; return (row index of the cut
    row, list of row-index arrays up to the cut). Mimics the truncated test trajectories."""
    last_rows, row_sets = [], []
    for _, g in df.groupby("unit"):
        idx = g.index.values                      # sorted by cycle
        T = len(idx)
        cuts = rng.integers(min(min_cut, T - 1), T, size=k)   # cut cycle c in [min_cut, T-1]
        for c in cuts:
            last_rows.append(idx[c - 1])
            row_sets.append(idx[:c])
    return np.array(last_rows), row_sets


def clf_metrics(y, p, thr=0.5):
    y = np.asarray(y)
    pred = (p >= thr).astype(int)
    out = {"f1": f1_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
           "precision": precision_score(y, pred, zero_division=0), "threshold": thr}
    if 0 < y.sum() < len(y):
        out["roc_auc"] = roc_auc_score(y, p)
        out["pr_auc"] = average_precision_score(y, p)
    return out


def nasa_score(y_true, y_pred):
    """PHM08 score, SUM over engines. d = pred - true; late (d >= 0) penalised more."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def reg_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"rmse": float(np.sqrt(np.mean((p - y) ** 2))), "mae": float(mean_absolute_error(y, p)),
            "nasa_score": nasa_score(y, p), "r2": float(r2_score(y, p)),
            "rmse_capped125": float(np.sqrt(np.mean((p - np.minimum(y, 125)) ** 2))),
            "frac_late": float(np.mean(p > y))}


def log_result(**row):
    """Append one experiment row to results.csv (single source of truth for reported numbers)."""
    row.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    row.setdefault("source", "measured")
    if isinstance(row.get("config"), dict):
        row["config"] = json.dumps(row["config"], sort_keys=True)
    unknown = set(row) - set(RESULT_FIELDS)
    assert not unknown, unknown
    new = not RESULTS_CSV.exists()
    n = 0 if new else sum(1 for _ in open(RESULTS_CSV, encoding="utf-8")) - 1
    row["run_id"] = f"r{n + 1:05d}"
    for k, v in row.items():
        if isinstance(v, (float, np.floating)):
            row[k] = f"{v:.6g}"
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def versions():
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "platform": platform.platform()}
