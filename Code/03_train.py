"""Train and evaluate the five models on the test split.

Run:  python Code/03_train.py
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

from config import SPLIT_DIR, RESULTS_DIR, RANDOM_SEED, TARGET, SITE_COL

NON_FEATURES = ["id", SITE_COL, TARGET]

GRIDS = {
    "logistic_regression": (
        LogisticRegression(max_iter=5000, random_state=RANDOM_SEED),
        {"C": [0.01, 0.1, 1, 10, 100]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
        {"n_estimators": [300, 600],
         "max_depth": [None, 4, 8],
         "min_samples_leaf": [1, 3]},
    ),
}


def load(name):
    tr = pd.read_csv(SPLIT_DIR / f"{name}_train.csv")
    te = pd.read_csv(SPLIT_DIR / f"{name}_test.csv")
    return (tr.drop(columns=NON_FEATURES), tr[TARGET].to_numpy(),
            te.drop(columns=NON_FEATURES), te[TARGET].to_numpy(),
            te["id"].to_numpy())


def positive_proba(clf, X):
    prob = clf.predict_proba(X)
    classes = list(getattr(clf, "classes_", [0, 1]))
    return np.asarray(prob)[:, classes.index(1)]


def score(y, prob, seconds):
    pred = (prob >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "brier": brier_score_loss(y, prob),
        "seconds": seconds,
    }


def run_tabfm(X_tr, y_tr, X_te):
    from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0
    clf = TabFMClassifier(model=tabfm_v1_0_0.load(model_type="classification"))
    clf.fit(X_tr, y_tr)
    return positive_proba(clf, X_te)


def run_search(estimator, grid, X_tr, y_tr, X_te):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    search = GridSearchCV(estimator, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_tr, y_tr)
    return positive_proba(search.best_estimator_, X_te), search.best_params_


def run_xgboost(X_tr, y_tr, X_te):
    from xgboost import XGBClassifier
    grid = {"max_depth": [2, 3, 5],
            "learning_rate": [0.03, 0.1],
            "n_estimators": [200, 500],
            "subsample": [0.8, 1.0]}
    est = XGBClassifier(random_state=RANDOM_SEED, eval_metric="logloss",
                        tree_method="hist")
    return run_search(est, grid, X_tr, y_tr, X_te)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load("raw")
    prep = load("prep")

    jobs = [
        ("1_tabfm_raw", raw, lambda d: (run_tabfm(d[0], d[1], d[2]), None)),
        ("2_tabfm_preprocessed", prep, lambda d: (run_tabfm(d[0], d[1], d[2]), None)),
        ("3_logistic_regression", prep,
         lambda d: run_search(*GRIDS["logistic_regression"], d[0], d[1], d[2])),
        ("4_random_forest", prep,
         lambda d: run_search(*GRIDS["random_forest"], d[0], d[1], d[2])),
        ("5_xgboost", prep, lambda d: run_xgboost(d[0], d[1], d[2])),
    ]

    rows, preds = [], []
    for name, data, fn in jobs:
        X_tr, y_tr, X_te, y_te, ids = data
        print(f"{name} ... ", end="", flush=True)
        start = time.perf_counter()
        try:
            prob, params = fn(data)
        except ImportError as e:
            print(f"skipped ({e.name} not installed)")
            continue
        elapsed = time.perf_counter() - start
        result = {"model": name, **score(y_te, prob, elapsed)}
        rows.append(result)
        preds.append(pd.DataFrame({"model": name, "id": ids,
                                   "y_true": y_te, "prob": prob}))
        print(f"roc_auc {result['roc_auc']:.3f}  in {elapsed:.1f}s"
              + (f"  {params}" if params else ""))

    if not rows:
        print("\nNothing ran. Install the missing packages and try again.")
        return 1

    results = pd.DataFrame(rows).set_index("model").round(4)
    results.to_csv(RESULTS_DIR / "model_results.csv")
    pd.concat(preds, ignore_index=True).to_csv(
        RESULTS_DIR / "predictions.csv", index=False)
    print("\n" + results.to_string())


if __name__ == "__main__":
    sys.exit(main())
