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

# TabFM is the model under evaluation; the conventional models are the baselines
# it is measured against.
SUBJECT = ["1_tabfm_raw", "2_tabfm_preprocessed"]
BASELINES = ["3_logistic_regression", "4_random_forest", "5_xgboost"]

# Subject models take the two strongest hues; baselines take muted greys/greens.
MODEL_COLOURS = {
    "1_tabfm_raw": "#2a78d6",
    "2_tabfm_preprocessed": "#eb6834",
    "3_logistic_regression": "#1baf7a",
    "4_random_forest": "#eda100",
    "5_xgboost": "#e87ba4",
}
MODEL_DASH = {
    "1_tabfm_raw": "-",
    "2_tabfm_preprocessed": "-",
    "3_logistic_regression": "--",
    "4_random_forest": "-.",
    "5_xgboost": ":",
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


# ---------------------------------------------------------------- data quality audit

TYPES = {"age": "continuous", "sex": "binary", "cp": "categorical",
         "trestbps": "continuous", "chol": "continuous", "fbs": "binary",
         "restecg": "categorical", "thalach": "continuous", "exang": "binary",
         "oldpeak": "continuous", "slope": "ordinal", "ca": "ordinal",
         "thal": "categorical"}
VALID = {"sex": {0, 1}, "cp": {1, 2, 3, 4}, "fbs": {0, 1}, "restecg": {0, 1, 2},
         "exang": {0, 1}, "slope": {1, 2, 3}, "ca": {0, 1, 2, 3},
         "thal": {3, 6, 7}}
PLAUSIBLE = {"age": (18, 100), "trestbps": (60, 260), "chol": (80, 700),
             "thalach": (50, 240), "oldpeak": (-3, 8)}


def build_audit(df):
    feats = [c for c in COLUMNS if c != "num"]
    rows = []
    for c in feats:
        s = df[c]
        bad = ""
        if c in PLAUSIBLE:
            lo, hi = PLAUSIBLE[c]
            m = s.notna() & ((s < lo) | (s > hi))
            if m.any():
                vals = sorted(s[m].unique())
                shown = ", ".join(f"{v:g}" for v in vals[:3])
                bad = f"{int(m.sum())}  (value {shown})"
        unexpected = ""
        if c in VALID:
            extra = set(s.dropna().unique()) - VALID[c]
            unexpected = ", ".join(f"{v:g}" for v in sorted(extra)) if extra else ""
        rows.append({
            "column": c, "type": TYPES[c], "distinct": int(s.nunique()),
            "missing": int(s.isna().sum()),
            "missing_pct": s.isna().mean() * 100,
            "impossible": bad, "unexpected": unexpected,
        })
    return pd.DataFrame(rows)


def audit_table_figure(df, audit, path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(audit)
    row_h = 0.30
    fig_h = 2.40 + n * row_h
    fig, ax = plt.subplots(figsize=(11.4, fig_h))
    ax.set_axis_off()
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1)

    X = {"column": 0.012, "type": 0.115, "distinct": 0.255, "missing": 0.345,
         "bar0": 0.375, "bar1": 0.545, "pct": 0.556,
         "impossible": 0.625, "unexpected": 0.845}

    top = 1 - (1.05 / fig_h)
    header_y = top
    body_top = top - row_h / fig_h * 1.15
    dy = row_h / fig_h

    ink, muted, bad_c = "#0b0b0b", "#52514e", "#b52d2c"

    ax.text(0, 1 - 0.18 / fig_h, "Data quality audit", fontsize=14,
            fontweight="bold", va="top", color=ink)
    complete = int(df[[c for c in COLUMNS if c != "num"]].notna().all(axis=1).sum())
    dup = int(df.duplicated(subset=[c for c in COLUMNS if c != "num"] + ["num"]).sum())
    ax.text(0, 1 - 0.52 / fig_h,
            f"UCI Heart Disease, four hospital databases   ·   "
            f"{len(df)} rows × {len(audit)} feature columns   ·   "
            f"{complete} complete rows ({complete/len(df)*100:.1f}%)   ·   "
            f"{dup} duplicate rows",
            fontsize=9.2, va="top", color=muted)

    headers = [("column", "Column", "left"), ("type", "Type", "left"),
               ("distinct", "Distinct", "right"), ("missing", "Missing", "right"),
               ("bar0", "Missing rate", "left"),
               ("impossible", "Impossible values", "left"),
               ("unexpected", "Unexpected levels", "left")]
    for key, label, align in headers:
        ax.text(X[key], header_y, label, fontsize=9, fontweight="bold",
                ha=align, va="center", color=ink)
    ax.plot([0, 1.03], [header_y - dy * 0.42] * 2, color="#b8b8b3", linewidth=1.0)

    for i, r in audit.iterrows():
        y = body_top - i * dy
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, y - dy * 0.44), 1.03, dy * 0.88,
                                   facecolor="#f6f6f4", edgecolor="none", zorder=0))
        ax.text(X["column"], y, r["column"], fontsize=9.2, va="center",
                color=ink, fontweight="medium", family="monospace")
        ax.text(X["type"], y, r["type"], fontsize=8.8, va="center", color=muted)
        ax.text(X["distinct"], y, f"{r['distinct']}", fontsize=9, ha="right",
                va="center", color=ink)
        ax.text(X["missing"], y, f"{r['missing']}", fontsize=9, ha="right",
                va="center", color=ink)

        span = X["bar1"] - X["bar0"]
        ax.add_patch(Rectangle((X["bar0"], y - dy * 0.20), span, dy * 0.40,
                               facecolor="#ececea", edgecolor="none", zorder=1))
        if r["missing_pct"] > 0:
            ax.add_patch(Rectangle((X["bar0"], y - dy * 0.20),
                                   span * r["missing_pct"] / 100, dy * 0.40,
                                   facecolor="#2a78d6", edgecolor="none", zorder=2))
        ax.text(X["pct"], y, f"{r['missing_pct']:.1f}%", fontsize=8.6,
                va="center", color=muted)

        ax.text(X["impossible"], y, r["impossible"] or "—", fontsize=8.8,
                va="center", color=bad_c if r["impossible"] else "#b8b8b3")
        ax.text(X["unexpected"], y, r["unexpected"] or "none", fontsize=8.8,
                va="center", color=bad_c if r["unexpected"] else "#b8b8b3")

    rule_y = body_top - (n - 1) * dy - dy * 0.62
    ax.plot([0, 1.03], [rule_y] * 2, color="#b8b8b3", linewidth=1.0)
    foot = rule_y - dy * 0.42
    notes = [
        "Duplicate rows: 2 exact duplicates across all 14 columns "
        "(Hungarian ids 404/405, VA ids 859/907). Retained, not dropped.",
        "Impossible values: serum cholesterol and resting blood pressure recorded "
        "as 0. All 123 Switzerland records and 49 VA records affected.",
        "Unexpected categories: none. Every coded column uses only the levels "
        "defined in the dataset documentation.",
        "Missing values are marked '?' in the source files. Missingness is "
        "structured by hospital, not random — see Figure: missing values by hospital.",
    ]
    for j, t in enumerate(notes):
        ax.text(0, foot - j * (0.19 / fig_h), t, fontsize=7.8,
                va="top", color=muted, wrap=False)

    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
