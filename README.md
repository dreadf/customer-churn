# Customer Churn Prediction

A practice ML project that predicts whether a subscription customer will cancel (churn), and looks at which customer behaviours are linked to cancelling.

## Purpose

- **Problem:** customer retention has dropped.
- **Goal:** find the signals associated with customers cancelling their subscription.
- **Approach:** treat `Churn` as the target (1 = churned, 0 = stayed) and use every column except `CustomerID` as a feature. Train a gradient-boosted classifier (XGBoost) to predict churn and output a churn probability per customer.

## Dataset

Two CSV files in [data/](data/):

| File | Purpose |
|---|---|
| `customer_churn_dataset-training-master.csv` | Training and exploration |
| `customer_churn_dataset-testing-master.csv` | Held-out customer data |

| Column | Type | Description |
|---|---|---|
| CustomerID | id | Unique customer ID (dropped before modelling) |
| Age | numeric | Customer age |
| Gender | categorical | Male / Female |
| Tenure | numeric | Time as a customer |
| Usage Frequency | numeric | How often the service is used |
| Support Calls | numeric | Number of support calls |
| Payment Delay | numeric | Payment delay |
| Subscription Type | categorical | Basic / Standard / Premium |
| Contract Length | categorical | Monthly / Quarterly / Annual |
| Total Spend | numeric | Total amount spent |
| Last Interaction | numeric | Time since last interaction |
| Churn | target | 1 = churned, 0 = retained |

## What the notebook does

[churn.ipynb](churn.ipynb) runs top to bottom:

1. **Load data** – read the training and testing CSVs.
2. **EDA** – check types and nulls, summary statistics, churn class balance, per-feature distributions. Outliers are kept on purpose, since they may represent extreme users.
3. **Correlation** – feature-to-target correlation and a feature-to-feature Pearson heatmap. Categorical columns are label-encoded first.
4. **Preprocessing** – drop null rows, separate features from target, 80/20 train/test split (`random_state=42`).
5. **Model** – `XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)`.
6. **Prediction** – predicted class and churn probability on the held-out split.

**Status:** model evaluation (accuracy, precision/recall, ROC-AUC, feature importance) is not written yet. The separate testing CSV is loaded but not yet used for scoring.

## Setup

Requires Python 3.12 or 3.13. Newer versions may lack prebuilt wheels for some of the packages.

```bash
# 1. Clone and enter the project
cd churn-predict

# 2. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. macOS only: xgboost needs OpenMP
brew install libomp

# 4. Install dependencies (exact pinned versions)
pip install --upgrade pip
pip install -r requirements.txt

# 5. Register the environment as a Jupyter kernel
python -m ipykernel install --user --name churn-predict --display-name "churn-predict (3.12)"
```

If `pip install -r requirements.txt` fails on a non-macOS system, remove the `appnope` line (it is macOS-only).

### Dependency files

- [requirements.in](requirements.in) – the packages this project uses directly (pandas, matplotlib, seaborn, scikit-learn, xgboost, ipykernel).
- [requirements.txt](requirements.txt) – the full pinned list, including transitive dependencies, generated with `pip freeze`.

To add a package, install it, then regenerate the lock file: `pip freeze > requirements.txt`.

## Usage

1. Activate the environment: `source .venv/bin/activate`
2. Open [churn.ipynb](churn.ipynb) in VS Code or Jupyter.
3. Select the `churn-predict (3.12)` kernel.
4. Run all cells in order.

Quick check that the environment is correct, in a notebook cell:

```python
import sys, xgboost, sklearn, pandas
print(sys.executable)  # should contain .venv
```

## Project structure

```
churn-predict/
├── churn.ipynb        # EDA, preprocessing, model training
├── data/              # training and testing CSVs
├── requirements.in    # direct dependencies
├── requirements.txt   # pinned lock file
└── README.md
```

## Next steps

- Evaluate the model: confusion matrix, precision/recall, ROC-AUC (accuracy alone is misleading if classes are imbalanced).
- Plot XGBoost feature importance to answer "what drives churn?"
- Score the separate testing CSV.
- Replace label encoding with one-hot encoding for nominal columns such as Gender.
- Tune hyperparameters with cross-validation.
