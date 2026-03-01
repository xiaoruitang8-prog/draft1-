# Customer Churn Prediction — Predictive Analytics

A machine learning project that predicts which bank customers are at risk of churning (closing their accounts), enabling targeted retention campaigns within a defined budget constraint.

---

## Project Overview

| Item | Detail |
|------|--------|
| **Task** | Binary classification — predict customer churn |
| **Dataset** | `Churn_Modelling.csv` — 10,000 European bank customers |
| **Target** | `Exited` (0 = retained, 1 = churned) |
| **Class balance** | ~79.6% retained / ~20.4% churned |
| **Final model** | HistGradientBoostingClassifier |
| **Primary metric** | PR-AUC (Average Precision) |
| **Operating rule** | Flag top 20% highest-risk customers |

---

## Repository Structure

```
draft1-/
├── Predictive Analytics Notebook.ipynb   # Main notebook (all 6 tasks)
├── Churn_Modelling.csv                   # Dataset
├── requirement.txt                       # Python dependencies
└── README.md                             # This file
```

---

## Requirements

- Python 3.9+
- pip

### Dependencies (`requirement.txt`)

```
pandas
numpy
scikit-learn
matplotlib
openpyxl
jupyterlab
```

---

## Run Instructions

### 1. Clone the repository

```bash
git clone <repo-url>
cd draft1-
```

### 2. Create and activate a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

### 4. Launch JupyterLab

```bash
jupyter lab
```

Your browser will open automatically. If it does not, navigate to:
```
http://localhost:8888
```

### 5. Open and run the notebook

1. In the JupyterLab file browser, click **`Predictive Analytics Notebook.ipynb`**
2. Run all cells in order:
   - **Kernel menu → Restart Kernel and Run All Cells…** → Confirm
3. Wait for all cells to finish (a `[*]` indicator means a cell is running)

> **Note:** The notebook expects `Churn_Modelling.csv` to be in the same directory as the notebook. Do not move or rename this file.

---

## Notebook Walkthrough

The notebook is structured into 6 sequential tasks:

| Task | Title | What it does |
|------|-------|-------------|
| **1** | Problem Framing | Defines target variable, checks class balance |
| **2** | Exploratory Data Analysis | Distributions, boxplots, categorical churn rates, correlation heatmap |
| **3** | Data Preparation | 70/15/15 stratified split, preprocessing pipeline (scaling + one-hot encoding) |
| **4** | Model Exploration | Compares Dummy, Logistic Regression, Random Forest, and HistGradientBoosting |
| **5** | Fine-tuning & Evaluation | RandomizedSearchCV tuning, threshold locking, test-set evaluation, error analysis |
| **6** | Solution Presentation | Final model rationale, limitations, model card |

---

## Modeling Pipeline

```
Raw CSV
  └─► Drop identifiers (RowNumber, CustomerId, Surname)
        └─► Stratified 70 / 15 / 15 split (train / val / test)
              └─► ColumnTransformer (fit on train only)
                    ├─► Numeric:     SimpleImputer(median) → StandardScaler
                    └─► Categorical: SimpleImputer(most_frequent) → OneHotEncoder
                          └─► Model training & validation (PR-AUC primary metric)
                                └─► RandomizedSearchCV (3-fold CV, train set only)
                                      └─► Lock threshold on validation set
                                            └─► Final evaluation on test set
```

---

## Key Results

- **Best model:** HistGradientBoostingClassifier (tuned)
- **Operating threshold:** 0.3235 (80th percentile of validation predicted probabilities)
- **Evaluation outputs:** confusion matrix, PR curves, calibration curve, geographic performance slices (France / Germany / Spain)
- Germany shows the strongest PR-AUC but regional disparities are documented

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirement.txt` again inside the venv |
| `FileNotFoundError: Churn_Modelling.csv` | Ensure the CSV is in the same folder as the notebook |
| Port 8888 already in use | Run `jupyter lab --port 8889` |
| Kernel keeps dying | Upgrade scikit-learn: `pip install -U scikit-learn` |
| Cells run out of order / stale state | Use **Kernel → Restart & Run All** to reset |

---

## Dataset

- **Source:** [Kaggle — Churn Modelling Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) (CC0 Public Domain)
- **Rows:** 10,000
- **Features used:** CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography, Gender
- **Target:** `Exited` (binary)
