"""Split the preprocessed data into train and test sets.

Run:  python Code/02_split.py
"""
import sys

import numpy as np
import pandas as pd

from config import PREP_DIR, SPLIT_DIR, RANDOM_SEED, TEST_SIZE, TARGET, SITE_COL


def stratified_test_ids(df, keys, test_size, seed):
    rng = np.random.default_rng(seed)
    picked = []
    for _, group in df.groupby(keys, sort=True):
        ids = group["id"].to_numpy().copy()
        rng.shuffle(ids)
        picked.extend(ids[:max(1, round(len(ids) * test_size))])
    return set(picked)


def main():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    prep = pd.read_csv(PREP_DIR / "heart_disease_preprocessed.csv")
    raw = pd.read_csv(PREP_DIR / "heart_disease_raw.csv")

    test_ids = stratified_test_ids(prep, [SITE_COL, TARGET], TEST_SIZE, RANDOM_SEED)

    # Both files are split on the same patients so the two versions stay comparable.
    raw = raw[raw["id"].isin(prep["id"])]

    for name, frame in [("prep", prep), ("raw", raw)]:
        mask = frame["id"].isin(test_ids)
        frame[~mask].to_csv(SPLIT_DIR / f"{name}_train.csv", index=False)
        frame[mask].to_csv(SPLIT_DIR / f"{name}_test.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
