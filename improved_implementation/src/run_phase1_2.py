"""Phase I/II experiments. Usage (from improved_implementation/):

  python src/run_phase1_2.py original      # as-found protocol, re-measured + cost profile
  python src/run_phase1_2.py checks        # leakage / integrity assertions
  python src/run_phase1_2.py corrected     # corrected classification baseline, 5 seeds
  python src/run_phase1_2.py nested        # nested grouped CV (selection-bias diagnostic), seed 42
  python src/run_phase1_2.py regression    # regression transposition reference, 5 seeds
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pm.common import (SENSORS, add_rolling, clf_metrics, feature_names, grouped_folds, load_config,
                       load_raw, log_result, reg_metrics, set_seeds, truncation_index, versions)
from pm.models import make_classifier, make_regressor, n_params

CLF = ["logistic_regression", "decision_tree", "random_forest", "gradient_boosting"]
REG = ["ridge", "decision_tree", "random_forest", "gradient_boosting"]


def prepare(cfg):
    train, test = load_raw(cfg)
    w = cfg["data"]["roll_window"]
    return add_rolling(train, SENSORS, w), add_rolling(test, SENSORS, w)


def score_fn(task):
    if task == "clf":
        return lambda m, X: m.predict_proba(X)[:, 1]
    return lambda m, X: m.predict(X)


def profile(model, X, predict):
    """Single-row latency (median of 200 calls) and batch cost per row."""
    one = X.iloc[[0]]
    predict(model, one)
    ts = []
    for i in range(200):
        r = X.iloc[[i]]
        t = time.perf_counter(); predict(model, r); ts.append(time.perf_counter() - t)
    t = time.perf_counter(); predict(model, X); tb = time.perf_counter() - t
    return {"infer_ms_single": float(np.median(ts) * 1e3), "infer_us_per_row_batch": tb / len(X) * 1e6}


def last_rows(df):
    return df.groupby("unit")["cycle"].idxmax().values


def run_original(cfg, only=None):
    """Re-measure the as-found protocol exactly: unshuffled GroupKFold, all validation rows,
    global (pre-split) column drop, seed 42, test at row level and at last cycle."""
    train, test = prepare(cfg)
    seed = 42
    set_seeds(seed)
    y, groups = train["label"], train["unit"]
    for name in ([only] if only else CLF):
        vals, tvals = [], []
        for tr, va in GroupKFold(n_splits=5).split(train, y, groups):
            m = make_classifier(name, cfg, seed).fit(train.iloc[tr], y.iloc[tr])
            vals.append(clf_metrics(y.iloc[va], m.predict_proba(train.iloc[va])[:, 1]))
            tvals.append(clf_metrics(y.iloc[tr], m.predict_proba(train.iloc[tr])[:, 1])["f1"])
        cvm = pd.DataFrame(vals).mean().to_dict()
        proc = psutil.Process()
        t = time.perf_counter()
        m = make_classifier(name, cfg, seed).fit(train, y)
        fit_s = time.perf_counter() - t
        common = dict(phase="I", experiment="original_as_found", task="clf", model=name,
                      config=cfg["classifiers"][name], seed=seed)
        log_result(**common, split="cv_groupkfold5_unshuffled_allrows", n_units_eval=100,
                   n_rows_eval=len(train), train_f1=float(np.mean(tvals)), **cvm,
                   notes="mean of 5 folds, protocol as found (src/make_nb04.py:59-69)")
        p = m.predict_proba(test)[:, 1]
        prof = profile(m, test, score_fn("clf"))
        log_result(**common, split="test_all_rows", n_units_eval=100, n_rows_eval=len(test),
                   **clf_metrics(test["label"], p), train_time_s=fit_s, n_params=n_params(m),
                   peak_mem_mb=proc.memory_info().peak_wset / 2**20, **prof,
                   notes="train_time = single fit of tuned config on 100 engines; peak_mem = peak working set of a fresh process that ran 5 CV fits + final fit + test inference for this model only")
        li = last_rows(test)
        log_result(**common, split="test_last_cycle", n_units_eval=100, n_rows_eval=100,
                   **clf_metrics(test["label"].loc[li], p[li]))


def run_nbtimes(cfg):
    # notebook wall-clock from the unmodified re-execution in baseline_repro/
    for nbname, s in [("nb03_features", 15.6), ("nb04_lr_dt_grids", 204.9), ("nb05_rf_gb_grids", 378.7)]:
        log_result(phase="I", experiment="original_notebook_rerun", task="clf", model=nbname, seed=42,
                   split="n/a", train_time_s=s,
                   notes="wall-clock of unmodified notebook via nbconvert in baseline_repro/; outputs identical to committed results")


def run_checks(cfg):
    train, test = load_raw(cfg)
    w = cfg["data"]["roll_window"]
    fe = add_rolling(train, SENSORS, w)
    out = []
    # 1. rolling features never cross unit boundaries and are strictly trailing
    first = fe.groupby("unit").head(1)
    assert all(np.allclose(first[f"{s}_mean{w}"], first[s]) for s in SENSORS)
    assert (first[[f"{s}_std{w}" for s in SENSORS]] == 0).all().all()
    out.append("rolling mean at each unit's first cycle equals the raw value (no cross-unit window): OK")
    u = fe[fe.unit == 1].reset_index(drop=True)
    cut = add_rolling(train[(train.unit == 1) & (train.cycle <= 100)], SENSORS, w)
    cols = feature_names(SENSORS, w)
    assert np.allclose(u.loc[:99, cols].values, cut[cols].values)
    out.append("features of unit 1 at cycles 1..100 unchanged when cycles >100 are removed (causal): OK")
    # 2. in-fold drop rule: does the kept-sensor set ever differ from the global one?
    glob = [s for s in SENSORS if train[s].std() >= cfg["data"]["std_threshold"]]
    diffs = 0
    for seed in cfg["seeds"]:
        for tr, _ in grouped_folds(train["unit"].values, cfg["cv"]["n_splits"], seed):
            kept = [s for s in SENSORS if train.iloc[tr][s].std() >= cfg["data"]["std_threshold"]]
            diffs += kept != glob
    out.append(f"global kept sensors: {glob}")
    out.append(f"folds (25 = 5 seeds x 5) where in-fold drop rule selects a different sensor set: {diffs}")
    # 3. committed processed files equal a fresh computation
    proc = pd.read_csv((Path(__file__).resolve().parents[2] / "data/processed/train_features.csv"))
    fresh = fe[["unit", "cycle"] + feature_names(glob, w) + ["RUL", "label"]]
    assert np.allclose(proc.values, fresh.values)
    out.append("data/processed/train_features.csv equals fresh in-memory computation: OK")
    # 4. test units / train units are separate files; RUL file only used for ground truth
    out.append(f"train units {train.unit.nunique()}, test units {test.unit.nunique()}, "
               f"test rows with label=1: {int(test.label.sum())} ({test.label.mean():.3%}), "
               f"train: {train.label.mean():.3%}")
    out.append(f"versions: {versions()}")
    print("\n".join(out))
    (Path(__file__).resolve().parents[1] / "logs").mkdir(exist_ok=True)
    (Path(__file__).resolve().parents[1] / "logs" / "phase2_checks.txt").write_text("\n".join(out), encoding="utf-8")


def cv_protocol(cfg, train, task, make, names, experiment, phase):
    """Shuffled grouped 5-fold per seed; all-rows metrics (as found) and truncated metrics
    (mimic the test condition). In-fold column drop via the pipeline."""
    target = "label" if task == "clf" else "RUL"
    met = clf_metrics if task == "clf" else reg_metrics
    predict = score_fn(task)
    k, mc = cfg["cv"]["truncations_per_unit"], cfg["cv"]["min_cut_cycle"]
    for name in names:
        for seed in cfg["seeds"]:
            set_seeds(seed)
            rng = np.random.default_rng(seed)
            P = np.full(len(train), np.nan)
            tr_scores, t0 = [], time.perf_counter()
            for tr, va in grouped_folds(train["unit"].values, cfg["cv"]["n_splits"], seed):
                m = make(name, cfg, seed).fit(train.iloc[tr], train[target].iloc[tr])
                P[va] = predict(m, train.iloc[va])
                tm = met(train[target].iloc[tr], predict(m, train.iloc[tr]))
                tr_scores.append(tm["f1"] if task == "clf" else tm["rmse"])
            secs = time.perf_counter() - t0
            y = train[target].values
            lasts, rowsets = truncation_index(train, k, mc, rng)
            allrows = np.concatenate(rowsets)
            common = dict(phase=phase, experiment=experiment, task=task, model=name,
                          config=cfg["classifiers" if task == "clf" else "regressors"][name], seed=seed)
            trk = {"train_f1" if task == "clf" else "train_rmse": float(np.mean(tr_scores))}
            log_result(**common, split="cv_grouped5_shuffled_allrows_oof", n_units_eval=100,
                       n_rows_eval=len(train), **met(y, P), **trk, train_time_s=secs,
                       notes="pooled out-of-fold predictions; train_time = 5 fold fits")
            log_result(**common, split="cv_grouped5_truncated_last_cycle_oof", n_units_eval=100,
                       n_rows_eval=len(lasts), **met(y[lasts], P[lasts]),
                       notes=f"{k} random cut points per validation engine, cut cycle in [{mc}, T-1]")
            if task == "clf":
                log_result(**common, split="cv_grouped5_truncated_rows_oof", n_units_eval=100,
                           n_rows_eval=len(allrows), **met(y[allrows], P[allrows]),
                           notes="all rows up to each cut point (mimics test_all_rows)")


def test_protocol(cfg, train, test, task, make, names, experiment, phase):
    target = "label" if task == "clf" else "RUL"
    met = clf_metrics if task == "clf" else reg_metrics
    predict = score_fn(task)
    li = last_rows(test)
    for name in names:
        for seed in cfg["seeds"]:
            set_seeds(seed)
            t = time.perf_counter()
            m = make(name, cfg, seed).fit(train, train[target])
            fit_s = time.perf_counter() - t
            p = predict(m, test)
            common = dict(phase=phase, experiment=experiment, task=task, model=name,
                          config=cfg["classifiers" if task == "clf" else "regressors"][name], seed=seed)
            prof = profile(m, test, predict) if seed == cfg["seeds"][0] else {}
            log_result(**common, split="test_last_cycle", n_units_eval=100, n_rows_eval=100,
                       **met(test[target].loc[li], p[li]), train_time_s=fit_s, n_params=n_params(m), **prof,
                       notes="one prediction per test engine at its last cycle; ground truth uncapped")
            if task == "clf":
                log_result(**common, split="test_all_rows", n_units_eval=100, n_rows_eval=len(test),
                           **met(test[target], p))


def run_nested(cfg, only=None):
    """Nested grouped CV with the original grids: unbiased estimate of the tuned pipelines."""
    train, _ = prepare(cfg)
    seed = 42
    y, units = train["label"], train["unit"].values
    for name in ([only] if only else CLF):
        set_seeds(seed)
        P = np.full(len(train), np.nan)
        inner_best, chosen, t0 = [], [], time.perf_counter()
        for tr, va in grouped_folds(units, 5, seed):
            gs = GridSearchCV(make_classifier(name, cfg, seed), cfg["grids"][name],
                              cv=GroupKFold(n_splits=5), scoring="f1")
            gs.fit(train.iloc[tr], y.iloc[tr], groups=units[tr])
            P[va] = gs.predict_proba(train.iloc[va])[:, 1]
            inner_best.append(gs.best_score_); chosen.append(str(gs.best_params_))
        log_result(phase="II", experiment="nested_cv_selection_bias", task="clf", model=name,
                   config={"grid": cfg["grids"][name]}, seed=seed,
                   split="nested_grouped5x5_outer_allrows_oof", n_units_eval=100, n_rows_eval=len(train),
                   **clf_metrics(y, P), train_f1=float(np.mean(inner_best)),
                   train_time_s=time.perf_counter() - t0,
                   notes="train_f1 column holds mean INNER best CV F1 (the selection-biased number); "
                         f"distinct configs chosen across outer folds: {len(set(chosen))}")


if __name__ == "__main__":
    cfg = load_config()
    cmd = sys.argv[1]
    if cmd == "original":
        run_original(cfg, sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "nbtimes":
        run_nbtimes(cfg)
    elif cmd == "checks":
        run_checks(cfg)
    elif cmd == "corrected":
        tr, te = prepare(cfg)
        cv_protocol(cfg, tr, "clf", make_classifier, CLF, "corrected_baseline", "II")
        test_protocol(cfg, tr, te, "clf", make_classifier, CLF, "corrected_baseline", "II")
    elif cmd == "nested":
        run_nested(cfg, sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "regression":
        tr, te = prepare(cfg)
        cv_protocol(cfg, tr, "reg", make_regressor, REG, "regression_reference", "II")
        test_protocol(cfg, tr, te, "reg", make_regressor, REG, "regression_reference", "II")
    else:
        raise SystemExit(__doc__)
