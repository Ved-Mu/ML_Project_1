"""Model factories for the as-found classifiers and their regression transposition."""
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .common import SENSORS, LowStdSensorDropper


def _pipe(est, cfg, scale):
    steps = [("select", LowStdSensorDropper(SENSORS, cfg["data"]["roll_window"],
                                            cfg["data"]["std_threshold"],
                                            tuple(cfg["data"].get("kinds", ("raw", "mean", "std")))))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps + [("clf", est)])


def make_classifier(name, cfg, seed, params=None):
    p = dict(cfg["classifiers"][name] if params is None else params)
    if name == "logistic_regression":      # src/make_nb04.py:212-213 (no random_state as found)
        return _pipe(LogisticRegression(solver="liblinear", max_iter=2000, **p), cfg, True)
    if name == "decision_tree":            # src/make_nb04.py:239
        return _pipe(DecisionTreeClassifier(random_state=seed, **p), cfg, False)
    if name == "random_forest":            # src/make_nb05.py:101
        return _pipe(RandomForestClassifier(random_state=seed, n_jobs=-1, **p), cfg, False)
    if name == "gradient_boosting":        # src/make_nb05.py:136
        return _pipe(HistGradientBoostingClassifier(early_stopping=False, random_state=seed, **p), cfg, False)
    raise KeyError(name)


def make_regressor(name, cfg, seed):
    p = dict(cfg["regressors"][name])
    if name == "ridge":
        return _pipe(Ridge(**p), cfg, True)
    if name == "decision_tree":
        return _pipe(DecisionTreeRegressor(random_state=seed, **p), cfg, False)
    if name == "random_forest":
        return _pipe(RandomForestRegressor(random_state=seed, n_jobs=-1, **p), cfg, False)
    if name == "gradient_boosting":
        return _pipe(HistGradientBoostingRegressor(early_stopping=False, random_state=seed, **p), cfg, False)
    raise KeyError(name)


def n_params(pipe):
    """Learned-parameter count: weights+bias (+scaler stats) or total tree nodes."""
    est = pipe[-1]
    if hasattr(est, "coef_"):
        n = est.coef_.size + 1
        if "scale" in pipe.named_steps:
            n += 2 * pipe.named_steps["scale"].mean_.size
        return int(n)
    if hasattr(est, "tree_"):
        return int(est.tree_.node_count)
    if hasattr(est, "estimators_"):
        return int(sum(t.tree_.node_count for t in est.estimators_))
    if hasattr(est, "_predictors"):
        return int(sum(p.nodes.shape[0] for it in est._predictors for p in it))
    raise TypeError(type(est))
