"""Exploratory analysis of the heart disease data.

Reads the minimally-processed file, so it shows the data before any rows or
columns were dropped.

Run:  python Code/01_preprocess.py
      python Code/eda.py
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from config import PREP_DIR, RESULTS_DIR, TARGET, SITE_COL, CONTINUOUS_MIN_UNIQUE

CLASS_COLOURS = ["#2a78d6", "#eb6834"]
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
       "#256abf", "#184f95", "#0d366b"]
DIV = ["#0d366b", "#256abf", "#3987e5", "#9ec5f4", "#f0efec",
       "#f0a5a5", "#e34948", "#b52d2c", "#7d1e1d"]
SEQ_MAP = LinearSegmentedColormap.from_list("seq", SEQ)
DIV_MAP = LinearSegmentedColormap.from_list("div", DIV)

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": "#b8b8b3", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def features(df):
    return [c for c in df.columns if c not in ("id", SITE_COL, TARGET)]


def correlation_heatmap(df, path):
    cols = features(df) + [TARGET]
    corr = df[cols].corr()
    n = len(cols)
    fig, ax = plt.subplots(figsize=(0.52 * n + 2.2, 0.52 * n + 1.6))
    im = ax.imshow(corr, cmap=DIV_MAP, vmin=-1, vmax=1)
    for (r, c), v in np.ndenumerate(corr.to_numpy()):
        if r == c:
            continue
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(v) > 0.55 else "#0b0b0b")
    ax.set_xticks(range(n), cols, rotation=90, fontsize=8)
    ax.set_yticks(range(n), cols, fontsize=8)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    ax.set_title("Feature correlations", loc="left", fontsize=11, pad=12)
    fig.savefig(path); plt.close(fig)


def missingness_by_site_heatmap(df, path):
    feats = features(df)
    pct = df.groupby(SITE_COL)[feats].apply(lambda g: g.isna().mean() * 100)
    pct.loc["ALL"] = df[feats].isna().mean() * 100
    fig, ax = plt.subplots(figsize=(0.52 * len(feats) + 2.6, 0.5 * len(pct) + 2))
    im = ax.imshow(pct, cmap=SEQ_MAP, vmin=0, vmax=100, aspect="auto")
    for (r, c), v in np.ndenumerate(pct.to_numpy()):
        ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                color="white" if v > 55 else "#0b0b0b")
    ax.set_xticks(range(len(feats)), feats, rotation=90, fontsize=8)
    ax.set_yticks(range(len(pct)), pct.index, fontsize=8.5)
    ax.set_xticks(np.arange(len(feats) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(pct) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="% missing")
    ax.set_title("Missing values by hospital", loc="left", fontsize=11, pad=12)
    fig.savefig(path); plt.close(fig)


def missingness_pattern_heatmap(df, path):
    feats = features(df)
    order = df[feats].isna().sum().sort_values(ascending=False).index.tolist()
    grid = df.sort_values(SITE_COL)[order].isna().astype(int)
    fig, ax = plt.subplots(figsize=(0.5 * len(order) + 2.4, 5))
    ax.imshow(grid, cmap=LinearSegmentedColormap.from_list("m", ["#eef4fd", "#184f95"]),
              aspect="auto", interpolation="nearest")
    boundary = 0
    for site, count in df[SITE_COL].value_counts().reindex(
            sorted(df[SITE_COL].unique())).items():
        boundary += count
        ax.axhline(boundary - 0.5, color="#eb6834", linewidth=1.2)
        ax.text(len(order) - 0.3, boundary - count / 2, site, fontsize=7.5,
                va="center", ha="left", color="#52514e")
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=8)
    ax.set_ylabel("Records, grouped by hospital")
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("Missing-value pattern  (dark = missing)", loc="left",
                 fontsize=11, pad=12)
    fig.savefig(path); plt.close(fig)



def distributions_figure(df, path):
    feats = features(df)
    cols = 4
    rows = int(np.ceil(len(feats) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, feats):
        continuous = df[c].nunique(dropna=True) > CONTINUOUS_MIN_UNIQUE
        if continuous:
            lo, hi = df[c].min(), df[c].max()
            bins = np.linspace(lo, hi, 20)
            for k in (0, 1):
                vals = df.loc[df[TARGET] == k, c].dropna()
                ax.hist(vals, bins=bins, density=True, histtype="stepfilled",
                        facecolor=CLASS_COLOURS[k], alpha=0.32)
                ax.hist(vals, bins=bins, density=True, histtype="step",
                        edgecolor=CLASS_COLOURS[k], linewidth=1.6)
            ax.set_yticks([])
        else:
            levels = sorted(df[c].dropna().unique())
            x = np.arange(len(levels))
            for k in (0, 1):
                sub = df.loc[df[TARGET] == k, c]
                share = [(sub == lv).sum() / max(1, sub.notna().sum()) for lv in levels]
                ax.bar(x + (k - 0.5) * 0.38, share, 0.34,
                       color=CLASS_COLOURS[k], edgecolor="white", linewidth=0.8)
            ax.set_xticks(x, [f"{lv:g}" for lv in levels], fontsize=7.5)
            ax.set_ylim(0, 1)
            ax.tick_params(axis="y", labelsize=7)
        ax.set_title(c, fontsize=9.5)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.grid(False)
    for ax in axes[len(feats):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=CLASS_COLOURS[k], linewidth=6,
                          label=lab) for k, lab in
               enumerate(["No heart disease", "Heart disease"])]
    fig.legend(handles=handles, loc="lower right", fontsize=9,
               bbox_to_anchor=(0.98, 0.02))
    fig.suptitle("Feature distributions by class", x=0.02, y=1.005, ha="left",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path); plt.close(fig)


def before_after_figure(raw, prep, path):
    sites = sorted(raw[SITE_COL].unique())
    counts = [[int((raw[SITE_COL] == s).sum()) for s in sites],
              [int((prep[SITE_COL] == s).sum()) for s in sites]]
    rates = []
    for frame in (raw, prep):
        rates.append([frame.loc[frame[SITE_COL] == s, TARGET].mean()
                      if (frame[SITE_COL] == s).any() else np.nan for s in sites])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    x = np.arange(len(sites))
    names = ["Before  (920 records)", "After  (436 records)"]

    for k in (0, 1):
        bars = ax1.bar(x + (k - 0.5) * 0.4, counts[k], 0.36,
                       color=CLASS_COLOURS[k], edgecolor="white", linewidth=1,
                       label=names[k])
        for b, v in zip(bars, counts[k]):
            ax1.text(b.get_x() + b.get_width() / 2, v + 6, str(v), ha="center",
                     fontsize=7.5, color="#52514e")
    ax1.set_xticks(x, sites, fontsize=8.5)
    ax1.set_ylabel("Records")
    ax1.set_title("Records per hospital", fontsize=10, loc="left")
    ax1.set_ylim(0, max(counts[0]) * 1.18)
    ax1.xaxis.grid(False)
    ax1.legend(fontsize=8)

    offsets = (-0.13, 0.13)
    label_dy = (-15, 10)
    for k in (0, 1):
        ok = ~np.isnan(rates[k])
        xs = x[ok] + offsets[k]
        ys = np.array(rates[k])[ok]
        ax2.scatter(xs, ys, s=70, color=CLASS_COLOURS[k], edgecolor="white",
                    linewidth=1.2, zorder=3, label=names[k])
        for xi, v in zip(xs, ys):
            ax2.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                         xytext=(0, label_dy[k]), ha="center", fontsize=7.5,
                         color="#52514e")
    for xi, s_name in enumerate(sites):
        if np.isnan(rates[1][xi]):
            ax2.annotate("removed", (xi + offsets[1], rates[0][xi]),
                         textcoords="offset points", xytext=(4, 0),
                         ha="left", va="center", fontsize=7.5, color="#b52d2c")
    ax2.set_xticks(x, sites, fontsize=8.5)
    ax2.set_ylabel("Positive rate")
    ax2.set_ylim(0, 1.12)
    ax2.set_title("Disease prevalence per hospital", fontsize=10, loc="left")
    ax2.xaxis.grid(False)
    ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle("Effect of preprocessing on the sample", x=0.02, y=1.04,
                 ha="left", fontsize=11)
    fig.savefig(path); plt.close(fig)


def summary_table(df, path):
    feats = features(df)
    rows = []
    for c in feats:
        s = df[c]
        rows.append({
            "feature": c,
            "missing": int(s.isna().sum()),
            "missing_pct": round(s.isna().mean() * 100, 1),
            "distinct": int(s.nunique()),
            "mean": round(s.mean(), 2),
            "std": round(s.std(), 2),
            "min": s.min(),
            "median": s.median(),
            "max": s.max(),
            "corr_with_target": round(df[[c, TARGET]].corr().iloc[0, 1], 3),
        })
    table = pd.DataFrame(rows).set_index("feature")
    table.to_csv(path)
    return table


def main():
    src = PREP_DIR / "heart_disease_raw.csv"
    if not src.exists():
        print("Run Code/01_preprocess.py first.")
        return 1
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)

    correlation_heatmap(df, RESULTS_DIR / "eda_correlation.png")
    distributions_figure(df, RESULTS_DIR / "eda_distributions.png")
    missingness_by_site_heatmap(df, RESULTS_DIR / "eda_missingness_by_site.png")
    missingness_pattern_heatmap(df, RESULTS_DIR / "eda_missingness_pattern.png")
    prep_path = PREP_DIR / "heart_disease_preprocessed.csv"
    if prep_path.exists():
        before_after_figure(df, pd.read_csv(prep_path),
                            RESULTS_DIR / "eda_before_after.png")
    table = summary_table(df, RESULTS_DIR / "eda_feature_summary.csv")

    print(table.to_string())
    print(f"\n{len(df)} records, {len(features(df))} features. "
          f"Figures and table in Results/")


if __name__ == "__main__":
    sys.exit(main())
