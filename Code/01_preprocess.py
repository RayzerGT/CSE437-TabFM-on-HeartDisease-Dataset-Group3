"""Clean the heart disease data and write out a modelling-ready table.

Run:  python Code/01_preprocess.py
"""
import sys
import zipfile

import numpy as np
import pandas as pd

from config import (COLUMNS, SITE_FILES, RAW_DIR, PREP_DIR, TARGET, SITE_COL,
                    MISSING_COL_THRESHOLD, IQR_MULTIPLIER, CORRELATION_THRESHOLD,
                    CONTINUOUS_MIN_UNIQUE)


def load():
    for z in RAW_DIR.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW_DIR)

    frames = []
    for site, fname in SITE_FILES.items():
        hits = list(RAW_DIR.rglob(fname))
        if not hits:
            raise FileNotFoundError(
                f"{fname} not found under Dataset/Raw. Get the archive from "
                "https://archive.ics.uci.edu/dataset/45/heart+disease")
        d = pd.read_csv(hits[0], header=None, names=COLUMNS,
                        na_values=["?"], skipinitialspace=True)
        d[SITE_COL] = site
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    for c in COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "id", range(len(df)))
    return df


def continuous_columns(df, feats):
    # Many distinct values means a measurement; few means a code.
    return [c for c in feats if df[c].nunique(dropna=True) > CONTINUOUS_MIN_UNIQUE]


def main():
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    df = load()

    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    feats = [c for c in COLUMNS if c != "num"]
    df[["id", SITE_COL] + feats + [TARGET]].to_csv(
        PREP_DIR / "heart_disease_raw.csv", index=False)

    miss = df[feats].isna().mean()
    feats = [c for c in feats if miss[c] <= MISSING_COL_THRESHOLD]

    df = df[["id", SITE_COL] + feats + [TARGET]]
    df = df.drop_duplicates(subset=feats + [TARGET], keep="first")
    df = df.dropna(subset=feats).reset_index(drop=True)

    keep = pd.Series(True, index=df.index)
    for c in continuous_columns(df, feats):
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        keep &= df[c].between(q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr)
    df = df[keep].reset_index(drop=True)

    feats = [c for c in feats if df[c].nunique() > 1]

    corr = df[feats].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    feats = [c for c in feats if not (upper[c] > CORRELATION_THRESHOLD).any()]

    cont = continuous_columns(df, feats)
    df[cont] = (df[cont] - df[cont].mean()) / df[cont].std(ddof=0)

    df[["id", SITE_COL] + feats + [TARGET]].to_csv(
        PREP_DIR / "heart_disease_preprocessed.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
