"""Shared paths, constants and helpers used by the notebooks."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"

RANDOM_SEED = 42
TEST_SIZE = 0.20

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

SITE_FILES = {
    "cleveland":   "processed.cleveland.data",
    "hungarian":   "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va":          "processed.va.data",
}

TARGET = "target"
SITE_COL = "site"
NON_FEATURES = ["id", SITE_COL, TARGET]

MISSING_COL_THRESHOLD = 0.50
IQR_MULTIPLIER = 1.5
CORRELATION_THRESHOLD = 0.90
CONTINUOUS_MIN_UNIQUE = 10

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DASHES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
CLASS_COLOURS = ["#2a78d6", "#eb6834"]
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
       "#256abf", "#184f95", "#0d366b"]
DIV = ["#0d366b", "#256abf", "#3987e5", "#9ec5f4", "#f0efec",
       "#f0a5a5", "#e34948", "#b52d2c", "#7d1e1d"]

MODEL_LABELS = {
    "1_tabfm_raw": "TabFM (raw)",
    "2_tabfm_preprocessed": "TabFM (preprocessed)",
    "3_logistic_regression": "Logistic regression",
    "4_random_forest": "Random forest",
    "5_xgboost": "XGBoost",
}
METRIC_NAMES = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1", "mcc": "MCC",
                "accuracy": "Accuracy", "precision": "Precision",
                "recall": "Recall", "brier": "Brier", "seconds": "Seconds"}


def load_raw():
    """Read the four source files, stack them, and tag each row with its hospital."""
    import zipfile
    for z in RAW_DIR.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW_DIR)

    frames = []
    for site, fname in SITE_FILES.items():
        hits = list(RAW_DIR.rglob(fname))
        if not hits:
            raise FileNotFoundError(
                f"{fname} not found under data/raw. See data/README.md.")
        d = pd.read_csv(hits[0], header=None, names=COLUMNS,
                        na_values=["?"], skipinitialspace=True)
        d[SITE_COL] = site
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    for c in COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "id", range(len(df)))
    return df


def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURES and c != "num"]


def continuous_columns(df, feats):
    """Many distinct values means a measurement; few means a code."""
    return [c for c in feats if df[c].nunique(dropna=True) > CONTINUOUS_MIN_UNIQUE]


def stratified_test_ids(df, keys, test_size=TEST_SIZE, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    picked = []
    for _, group in df.groupby(keys, sort=True):
        ids = group["id"].to_numpy().copy()
        rng.shuffle(ids)
        picked.extend(ids[:max(1, round(len(ids) * test_size))])
    return set(picked)


def load_split(name):
    tr = pd.read_csv(PROCESSED_DIR / f"{name}_train.csv")
    te = pd.read_csv(PROCESSED_DIR / f"{name}_test.csv")
    drop = [c for c in NON_FEATURES if c in tr.columns]
    return (tr.drop(columns=drop), tr[TARGET].to_numpy(),
            te.drop(columns=drop), te[TARGET].to_numpy(), te["id"].to_numpy())


def positive_proba(clf, X):
    prob = clf.predict_proba(X)
    classes = list(getattr(clf, "classes_", [0, 1]))
    return np.asarray(prob)[:, classes.index(1)]


def score(y, prob, seconds=np.nan):
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 brier_score_loss, f1_score, matthews_corrcoef,
                                 precision_score, recall_score, roc_auc_score)
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


def apply_plot_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.size": 9,
        "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
        "axes.edgecolor": "#b8b8b3", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq", SEQ)


def div_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div", DIV)
