# CSE437 — Does a tabular foundation model need preprocessing?

TabFM against conventional models on multi-site heart disease data.

## Problem statement

Over the years machine learning algorithms such as XGBoost, Decision Trees (all
variants) and Logistic Regression have been at the forefront of computing and
analyzing tabular data. Although reliable and proven to work in most scenarios, the
amount of time required to preprocess large datasets has proven to be cumbersome.
Copious amounts of time have to be poured into figuring out ways to transform the
dataset so that it reaches its ideal state: a state capable of training models fast
and efficiently.

The modern-day solution to this recurrent problem has been found through the
invention of TabFM. Released by Google Research in June 2026, the foundation model
reports beating established tabular algorithms on the TabArena benchmark suite
without requiring prior modifications on datasets, providing a direct zero-shot
solution to model training.

This project asks whether the advantage survives that setting, and whether "no
feature engineering required" holds when the raw data is genuinely distorted. We fit
five models to the UCI Heart Disease dataset: TabFM on minimally processed data,
TabFM on conventionally cleaned data, and three conventional baseline models —
XGBoost, Logistic Regression and Random Forest.

## Research questions

1. Does TabFM predict heart disease more accurately than logistic regression, random
   forest, and XGBoost?
2. Does cleaning the data change how well TabFM performs?
3. Which model holds up best when tested on a hospital it has not seen before?

## Dataset

**Source:** UCI Machine Learning Repository —
<https://archive.ics.uci.edu/dataset/45/heart+disease>

The dataset includes a collection of features covering demographics, symptoms,
resting measurements and exercise stress-test results recorded beforehand, and the
label is the angiographic result — whether the patient has heart disease or not. It
is derived from an angiographic study reported by Detrano et al. (1989), donated to
UCI in 1988. The raw dataset was downloaded as four independently collected hospital
databases, which were merged by concatenation with an extra column recording the
source hospital.

| | |
|---|---|
| **Size** | 920 rows × 13 features, plus the label |
| **Time period** | Collected during the 1980s, donated 1988 |
| **Licence** | Creative Commons Attribution 4.0 International |

Patient identifiers were replaced with dummy values before release.

### Source files

| File | Hospital | Records |
|---|---|---:|
| `processed.cleveland.data` | Cleveland Clinic Foundation | 303 |
| `processed.hungarian.data` | Hungarian Institute of Cardiology, Budapest | 294 |
| `processed.switzerland.data` | University Hospitals, Zurich and Basel | 123 |
| `processed.va.data` | V.A. Medical Center, Long Beach | 200 |
| | **Total** | **920** |

### Target variable

- **Column name:** `target`
- **Type:** discrete binary

The data is present in multiclass form but for simplicity the project has been set up
for binary classification. Label value 0 is considered negative while all other
classes (1–4) are considered positive. This has been done for simplicity, as the main
goal of the study is to test and compare the state-of-the-art foundation model TabFM
to legacy machine learning methods. The strategy also mitigates the class imbalance
that would arise if it were treated as a multiclass problem.

Raw class distribution:

| `num` | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| Frequency | 411 | 265 | 109 | 107 | 28 |

Simplified class distribution:

- **[0]** — negative class, 44.6%
- **[1–4]** — positive class, 55.4%





## How to run the whole project from scratch

### 1. Requirements

- **Python 3.11 or newer.** TabFM requires it; the install fails on 3.10 or older.
  Check with `python --version`.
- Roughly **8 GB of free disk space** — 250 MB for PyTorch and 6.5 GB for the TabFM
  weights, which download on first use.
- An internet connection for the installs and the weight download.
- A GPU is optional. Everything below runs on CPU; at this dataset size the
  difference is minutes, not hours.

### 2. Clone and install

```bash
git clone https://github.com/RayzerGT/CSE437-TabFM-heartDisease.git
cd CSE437-TabFM-heartDisease
pip install -r requirements.txt
```

On PowerShell the TabFM extra needs quoting, because square brackets are wildcard
syntax there:

```powershell
pip install "tabfm[pytorch]"
```

Optionally install into a virtual environment first (`python -m venv .venv`, then
`.venv\Scripts\activate` on Windows or `source .venv/bin/activate` elsewhere) so the
7 GB of dependencies stay out of your system Python.

Verify everything imports:

```bash
python -c "import pandas, numpy, matplotlib, sklearn, xgboost, tabfm; print('ok')"
```

### 3. Get the data

Download `heart+disease.zip` from
<https://archive.ics.uci.edu/dataset/45/heart+disease> and place it in `data/raw/`.
Notebook 02 extracts it automatically on first run — no manual unzipping needed. The
archive is 129 KB and expands to about 850 KB. Only the four `processed.*.data` files
are used; the rest of the archive is ignored.

### 4. Pre-download the TabFM weights (optional but recommended)

```bash
python -c "from tabfm import tabfm_v1_0_0_pytorch as t; t.load(model_type='classification')"
```

This pulls the 6.5 GB of weights as its own step, so a network failure costs a retry
rather than a half-finished training run. They cache to `~/.cache/huggingface/hub`, so
the download happens once.

### 5. Run the notebooks in order

| # | Notebook | Produces |
|---|---|---|
| 01 | `01_data_audit_and_eda.ipynb` | data quality audit, EDA figures |
| 02 | `02_preprocessing.ipynb` | `heart_disease_raw.csv`, `heart_disease_preprocessed.csv` |
| 03 | `03_feature_engineering.ipynb` | train/test splits for both representations |
| 04 | `04_modeling_and_tuning.ipynb` | all five models, tuning traces, saved models |
| 05 | `05_evaluation_and_error_analysis.ipynb` | result tables, plots, error analysis |

Order matters — each notebook reads what the previous one wrote.

**Option A — a runner that prints progress.** `nbconvert` shows nothing until a cell
finishes, which makes the long TabFM cells look like a hang. This runs the same cells
and reports as it goes:

```bash
python src/run_notebook.py notebooks/01_data_audit_and_eda.ipynb
python src/run_notebook.py notebooks/02_preprocessing.ipynb
python src/run_notebook.py notebooks/03_feature_engineering.ipynb
python src/run_notebook.py notebooks/04_modeling_and_tuning.ipynb
python src/run_notebook.py notebooks/05_evaluation_and_error_analysis.ipynb
```

**Option B — nbconvert**, which also saves outputs back into the `.ipynb` files. The
timeout flag is required: the default kills any cell running longer than 30 seconds,
and the TabFM cells take minutes.

```bash
python -m jupyter nbconvert --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/*.ipynb
```

**Option C — interactively.** Run `python -m jupyter lab`, open each notebook in
order, then Run → Run All Cells. VS Code works the same way.


