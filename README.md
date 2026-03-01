# Predictive Analytics — Customer Churn Notebook

MSIN0097 Individual Coursework | Candidate: RRMZ8

---

## Requirements

- Python 3.9+ (conda recommended)
- Dependencies listed in `requirement.txt`

Install all dependencies:

```bash
pip install -r requirement.txt
```

---

## Data

Place the dataset in a `data/` subdirectory before running:

```
draft1-/
├── data/
│   └── Churn_Modelling.csv   ← required
├── Predictive Analytics Notebook.ipynb
└── requirement.txt
```

The file `Churn_Modelling.csv` is also present in the repo root — move or copy it:

```bash
mkdir -p data
cp Churn_Modelling.csv data/
```

---

## How to Run

1. **Launch JupyterLab**

   ```bash
   jupyter lab
   ```

2. **Open the notebook**

   Select `Predictive Analytics Notebook.ipynb` in the file browser.

3. **Run all cells in order**

   Kernel menu → *Restart Kernel and Run All Cells*

   > All cells must be run top-to-bottom. The notebook does not support out-of-order execution.

---

## Notebook Structure

| Task | Description |
|------|-------------|
| 1 | Problem framing — target variable, metrics, assumptions |
| 2 | Exploratory data analysis (EDA) |
| 3 | Data preprocessing and train/validation/test split (70/15/15) |
| 4 | Model comparison — Dummy, Logistic Regression, Random Forest, HistGradientBoosting |
| 5 | Hyperparameter tuning (RandomizedSearchCV) and final test evaluation |
| 6 | Final model selection, limitations, and model card |

---

## Expected Output

Running all cells produces:

- EDA plots (class balance, distributions, boxplots, correlation heatmap)
- Model comparison table (PR-AUC, ROC-AUC, Recall@top20%, Precision@top20%)
- Confusion matrix, PR curve, calibration plot, and geography slice analysis
- Final test-set metrics for the tuned HistGradientBoosting model
