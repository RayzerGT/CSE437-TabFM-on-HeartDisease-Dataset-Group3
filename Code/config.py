from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "Dataset" / "Raw"
PREP_DIR = ROOT / "Dataset" / "Preprocessed"
RESULTS_DIR = ROOT / "Results"

RANDOM_SEED = 42

# The data files have no header row; names are from the dataset documentation.
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

# Cleaning thresholds. These control how much data survives, so tweak carefully.
MISSING_COL_THRESHOLD = 0.50
IQR_MULTIPLIER = 1.5
CORRELATION_THRESHOLD = 0.90
CONTINUOUS_MIN_UNIQUE = 10
