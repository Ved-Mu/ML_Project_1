"""Improvement experiments (classification, scikit-learn only). From improved_implementation/:

  python src/run_phase3.py window [W ...]         # rolling-window ablation, 4 models x 5 seeds
  python src/run_phase3.py kinds <W>              # feature-set ablation at window W
  python src/run_phase3.py final <model> <W> <kinds,comma,separated>   # thresholds + test, once
"""
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pm.common import (ROOT, SENSORS, add_rolling, clf_metrics, grouped_folds, load_config, load_raw,
                       log_result, set_seeds, truncation_index)
from pm.models import make_classifier, n_params

CLF = ["logistic_regression", "decision_tree", "random_forest", "gradient_boosting"]


def prepare(cfg, windows):
    train, test = load_raw(cfg)
    for w in windows:
        train, test = add_rolling(train, SENSORS, w), add_rolling(test, SENSORS, w)
    return train, test


def variant(cfg, w, kinds):
    c = copy.deepcopy(cfg)
    c["data"]["roll_window"], c["data"]["kinds"] = w, list(kinds)
    return c


def oof(cfg, train, name, seed):
    """Out-of-fold probabilities (grouped by engine) + truncated-validation index sets."""
    set_seeds(seed)
    P, f1tr, t0 = np.full(len(train), np.nan), [], time.perf_counter()
    for tr, va in grouped_folds(train["unit"].values, cfg["cv"]["n_splits"], seed):
        m = make_classifier(name, cfg, seed).fit(train.iloc[tr], train["label"].iloc[tr])
        P[va] = m.predict_proba(train.iloc[va])[:, 1]
        f1tr.append(clf_metrics(train["label"].iloc[tr], m.predict_proba(train.iloc[tr])[:, 1])["f1"])
    lasts, rowsets = truncation_index(train, cfg["cv"]["truncations_per_unit"], cfg["cv"]["min_cut_cycle"],
                                      np.random.default_rng(seed))
    return P, lasts, np.concatenate(rowsets), float(np.mean(f1tr)), time.perf_counter() - t0


def cv_rows(cfg, train, names, experiment):
    y = train["label"].values
    for name in names:
        for seed in cfg["seeds"]:
            P, lasts, rows, f1tr, secs = oof(cfg, train, name, seed)
            common = dict(phase="III", experiment=experiment, task="clf", model=name, seed=seed,
                          config={"window": cfg["data"]["roll_window"], "kinds": cfg["data"]["kinds"],
                                  **cfg["classifiers"][name]})
            log_result(**common, split="cv_grouped5_shuffled_allrows_oof", n_units_eval=100,
                       n_rows_eval=len(y), **clf_metrics(y, P), train_f1=f1tr, train_time_s=secs)
            log_result(**common, split="cv_grouped5_truncated_rows_oof", n_units_eval=100,
                       n_rows_eval=len(rows), **clf_metrics(y[rows], P[rows]))
            log_result(**common, split="cv_grouped5_truncated_last_cycle_oof", n_units_eval=100,
                       n_rows_eval=len(lasts), **clf_metrics(y[lasts], P[lasts]))


def pick_thresholds(y, p, target_recall):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[np.argmax(f1)]), float(thr[rec[:-1] >= target_recall].max())


def run_final(cfg, icfg, name, w, kinds):
    """Thresholds from truncated out-of-fold predictions; then ONE test evaluation per seed.
    Also re-scores the corrected baseline (window 5, as-found features) for the error analysis."""
    train, test = prepare(cfg, sorted({5, w}))
    li = test.groupby("unit")["cycle"].idxmax().values
    y, yt = train["label"].values, test["label"].values
    preds = test[["unit", "cycle", "RUL", "label"]].copy()
    for tag, c in [("corrected_baseline_rescored", variant(cfg, 5, ("raw", "mean", "std"))),
                   ("final_improved", variant(cfg, w, kinds))]:
        for seed in cfg["seeds"]:
            P, lasts, rows, _, _ = oof(c, train, name, seed)
            t_f1, t_rec = pick_thresholds(y[rows], P[rows], icfg["target_recall"])
            set_seeds(seed)
            t0 = time.perf_counter()
            m = make_classifier(name, c, seed).fit(train, train["label"])
            fit_s = time.perf_counter() - t0
            p = m.predict_proba(test)[:, 1]
            preds[f"{tag}_seed{seed}"] = p
            common = dict(phase="VI", experiment=tag, task="clf", model=name, seed=seed,
                          config={"window": c["data"]["roll_window"], "kinds": c["data"]["kinds"],
                                  **c["classifiers"][name]})
            for rule, t in [("thr0.5", 0.5), ("thr_maxF1_truncCV", t_f1), ("thr_recall90_truncCV", t_rec)]:
                log_result(**common, split=f"cv_grouped5_truncated_rows_oof|{rule}", n_units_eval=100,
                           n_rows_eval=len(rows), **clf_metrics(y[rows], P[rows], t))
                log_result(**common, split=f"test_all_rows|{rule}", n_units_eval=100, n_rows_eval=len(test),
                           **clf_metrics(yt, p, t), train_time_s=fit_s, n_params=n_params(m),
                           notes="threshold chosen on truncated out-of-fold training predictions")
                log_result(**common, split=f"test_last_cycle|{rule}", n_units_eval=100, n_rows_eval=100,
                           **clf_metrics(yt[li], p[li], t))
            if seed == cfg["seeds"][0] and hasattr(m[-1], "coef_"):
                (ROOT / "logs").mkdir(exist_ok=True)
                pd.Series(m[-1].coef_[0], index=m[0].columns_).to_csv(ROOT / "logs" / f"coef_{tag}.csv")
    preds.to_csv(ROOT / "logs" / "test_predictions.csv", index=False)


if __name__ == "__main__":
    cfg, icfg = load_config(), load_config("improved.yaml")
    cmd = sys.argv[1]
    if cmd == "window":
        ws = [int(a) for a in sys.argv[2:]] or icfg["windows"]
        train, _ = prepare(cfg, ws)
        for w in ws:
            cv_rows(variant(cfg, w, ("raw", "mean", "std")), train, CLF, "window_ablation")
    elif cmd == "kinds":
        w = int(sys.argv[2])
        train, _ = prepare(cfg, [w])
        for kinds in icfg["kinds_options"]:
            cv_rows(variant(cfg, w, kinds), train, CLF, "feature_set_ablation")
    elif cmd == "final":
        run_final(cfg, icfg, sys.argv[2], int(sys.argv[3]), tuple(sys.argv[4].split(",")))
    else:
        raise SystemExit(__doc__)
