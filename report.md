# MSIN0097 Predictive Analytics – Individual Coursework Report

**Candidate Number: RRMZ8**

---

## 1. Problem Framing

The task is to build a binary classifier that predicts whether a bank customer will churn (close their account or become inactive). The dataset used is the Customer Churn Dataset (Kaggle, CC0 licence), comprising 10,000 customer records and 14 raw columns spanning demographic attributes (age, gender, geography), account-level features (balance, tenure, number of products), and a binary target column `Exited` (1 = churned, 0 = retained).

The positive class (`Exited = 1`) represents approximately 20.4% of records, indicating a moderate class imbalance. Selecting an appropriate evaluation metric is therefore non-trivial. Accuracy is misleading under imbalance: a naïve majority-class predictor achieves roughly 80% accuracy while identifying no churners at all. The primary metric is therefore **PR-AUC** (Area Under the Precision–Recall Curve). Davis and Goadrich (2006) demonstrate that PR curves are more informative than ROC curves under class imbalance because they condition on the positive class and suppress the inflating effect of the large number of true negatives. ROC-AUC is retained as a secondary metric for comparability with prior literature.

Two operational metrics—**Recall@top20%** and **Precision@top20%**—translate model output into a business-relevant targeting rule: identifying the 300 customers the model scores as highest-risk in each 1,500-customer test batch. This rule reflects the realistic constraint that retention teams have finite capacity and must prioritise whom to contact.

The prediction mode is **batch scoring** rather than real-time inference: the bank re-scores its customer base periodically and allocates retention resources to the top-risk tier. This framing means (a) latency is not a binding constraint, (b) interpretability matters for stakeholder trust but is not the primary selection criterion, and (c) well-calibrated probability estimates are desirable because they support proportionate prioritisation of outreach effort.

---

## 2. Exploratory Data Analysis

Exploratory analysis served two purposes: identifying predictive signals and detecting data quality or leakage risks prior to modelling.

**Class balance.** The target distribution—79.6% retained, 20.4% churned—confirms moderate imbalance, motivating stratified splitting and PR-AUC as the primary metric rather than accuracy.

**Missing values and duplicates.** A systematic null-value audit found zero missing entries across all 14 columns, and no duplicate rows were detected. The absence of missingness eliminates one common source of imputation bias (Sterne et al., 2009). A defensive median-imputation step was nonetheless retained in the preprocessing pipeline to safeguard against future data irregularities.

**Numeric distributions.** Histograms of the five continuous features reveal material heterogeneity: `Balance` follows a bimodal distribution with a substantial mass at zero and a roughly normal upper spread; `Age` is right-skewed with a long upper tail; `CreditScore` and `EstimatedSalary` are approximately uniform. These distributional properties motivated standardisation for the linear model and confirmed that scale-invariant tree-based models were natural candidates.

**Categorical churn rates.** Grouped bar charts showed substantial variation in churn rates across categories. Germany exhibits a churn rate of approximately 32.4%, roughly double that of France (16.2%) and Spain (16.7%). Female customers churn at a higher rate than male customers. Active members churn substantially less than inactive ones. These patterns indicate that geography, gender, and activity status carry genuine predictive information and should be retained as features.

**Numeric correlations.** A Pearson correlation heatmap confirmed that `Age` carries the strongest linear signal (r ≈ 0.29 with `Exited`), consistent with boxplots showing higher median age in the churned group. All other numeric correlations are weak (|r| < 0.15), which is typical in a heterogeneous customer base and does not preclude non-linear models from detecting useful interaction effects.

**Leakage assessment.** No feature records information that is causally downstream of the churn event. Identifier columns (`RowNumber`, `CustomerId`, `Surname`) carry no predictive content and were flagged for removal before modelling. No target-leakage risk was identified.

---

## 3. Data Preparation

Data preparation followed three principles: reproducibility, leakage prevention, and defensive encoding.

**Feature set.** Identifier columns (`RowNumber`, `CustomerId`, `Surname`) were removed, leaving ten predictive features: four binary or low-cardinality categoricals (`Geography`, `Gender`, `HasCrCard`, `IsActiveMember`) and six continuous or ordinal variables (`CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`). No additional features were engineered; the EDA confirmed sufficient signal in the existing feature set without identifying interactions that would justify additional derived variables.

**Train/validation/test split.** Data were partitioned into training (70%), validation (15%), and test (15%) sets using a stratified two-step `train_test_split` (scikit-learn, `random_state = 42`). Stratification preserves the approximately 20.4% churn prevalence in every split, which is important for reliable metric estimation under imbalance (Kohavi, 1995). A dedicated validation set—separate from the test set—enables model selection and threshold locking without exposing the test set, consistent with the principle that test data must remain a true holdout for final reporting (Arlot & Celisse, 2010).

**Preprocessing pipeline.** A `ColumnTransformer` containing two sub-pipelines was fitted exclusively on the training set and applied identically to validation and test sets. Numeric features were processed via `SimpleImputer` (strategy = median) followed by `StandardScaler`. Categorical features (`Geography`, `Gender`) were encoded with `OneHotEncoder` (drop = 'first' to avoid perfect multicollinearity in the linear model). Fitting the transformer only on training data prevents test statistics from leaking into the training procedure—a subtle but consequential form of contamination documented by Kaufman et al. (2012). Post-preprocessing assertions verified consistent feature matrix shapes and the expected column count after one-hot expansion.

---

## 4. Model Exploration and Shortlisting

Four models were trained and evaluated on the validation set under a single shared evaluation protocol, ensuring that all comparisons are on equal terms.

**Baseline.** A `DummyClassifier` (strategy = `most_frequent`) serves as the no-learning reference. It predicts the majority class for every observation, achieving PR-AUC ≈ 0.20 (equal to the positive-class prevalence) and ROC-AUC = 0.50. Any learned model must substantially exceed these values to be considered useful.

**Logistic Regression.** A regularised logistic regression (L2 penalty, `max_iter = 1000`) serves as the linear benchmark. It outperforms the dummy on PR-AUC but is bounded by its assumption of log-linear separability. The distributional heterogeneity observed in EDA—particularly the bimodal balance distribution and the strong geographic effect—suggested that non-linear decision boundaries would be advantageous, an expectation borne out by the validation results.

**Random Forest.** Breiman's (2001) bagging ensemble of decorrelated decision trees is well-suited to tabular data with heterogeneous feature types. Trained with 200 trees and otherwise default hyperparameters, Random Forest achieves a substantially higher PR-AUC than Logistic Regression, demonstrating the benefit of capturing non-linear feature interactions without requiring manual feature engineering.

**HistGradientBoostingClassifier (HistGBT).** Gradient boosted trees, as formalised by Friedman (2001), construct an additive model sequentially: each tree fits the residuals of its predecessors, giving the ensemble a strong inductive bias towards correcting systematic errors. Scikit-learn's histogram-based implementation adds computational efficiency via data binning and natively handles missing values. On the validation set, HistGBT achieved the highest PR-AUC and Recall@top20% among all candidates, indicating that sequential error-correction extracts patterns that bagging alone does not recover.

**Shortlist decision.** HistGBT was carried forward to fine-tuning on the basis of the highest validation PR-AUC. Random Forest was noted as a secondary benchmark but was not tuned further, keeping the experimental budget focused.

---

## 5. Fine-Tuning and Evaluation

**Tuning strategy.** Hyperparameter optimisation was performed using `RandomizedSearchCV` (Bergstra & Bengio, 2012) with `n_iter = 8` combinations and 3-fold cross-validation on the training split only, scoring by `average_precision`—the scikit-learn equivalent of PR-AUC. Random search is preferred over grid search at small iteration budgets because it samples the full joint hyperparameter space more efficiently than an exhaustive factorial grid (Bergstra & Bengio, 2012). The search ranged over `learning_rate`, `max_iter`, `max_depth`, `min_samples_leaf`, and `l2_regularization`. All choices—including the best hyperparameter configuration—were locked using validation evidence before any test-set access, consistent with principled model selection discipline.

**Threshold and operating rule.** Using the default 0.5 probability threshold is arbitrary under class imbalance, because the optimal decision boundary depends on the class prior and the relative costs of false positives and false negatives. The operating threshold was instead set to the 80th percentile of predicted probabilities on the validation set (≈ 0.3235), implementing a top-20% risk-ranking rule. This threshold was committed before the test set was accessed, preventing retrospective threshold selection.

**Test evaluation.** The locked HistGBT model attains a **test PR-AUC of 0.7344** against a validation PR-AUC of 0.7324—a difference of 0.002, indicating negligible overfitting. The near-coincident precision–recall curves on validation and test provide visual confirmation that the model generalises reliably to unseen data.

**Confusion matrix.** At the locked threshold on the 1,500-observation test set: 197 true positives, 109 false negatives, 100 false positives, and 1,094 true negatives. Because the cost of a missed churner (false negative) likely exceeds the cost of an unnecessary retention contact (false positive), the operating rule prioritises recall, though further cost-benefit analysis would be needed to set an optimal threshold for a specific business context.

**Calibration.** The reliability diagram shows that predicted probabilities track observed churn rates closely at low-to-mid probability values—the region where the majority of customers score. Slight overconfidence emerges at the upper end of the score range, a known tendency of gradient boosted ensembles documented by Niculescu-Mizil and Caruana (2005). Post-hoc calibration methods such as Platt scaling or isotonic regression could mitigate this in production without affecting ranking quality.

**Geographic slice analysis.** Disaggregated evaluation by `Geography` reveals that Germany achieves the highest ranking quality (PR-AUC = 0.8166) but the lowest Recall@top20% (0.5702). This is a structural budget effect: Germany's churn rate of approximately 30.8% means that the top-20% tier cannot contain all German churners. This finding suggests that market-specific targeting budgets warrant consideration at deployment.

---

## 6. Final Solution

The final model is a tuned **HistGradientBoostingClassifier**, deployed under a **top-20% risk-ranking rule** with a locked threshold of 0.3235. It was selected on the basis of superior PR-AUC on the held-out validation set and confirmed on the independently held test set (PR-AUC = 0.7344, Recall@top20% = 0.6548, Precision@top20% = 0.5233). The near-identical validation and test curves confirm generalisation without overfitting. Predicted probabilities are broadly calibrated in the operationally relevant range, supporting proportionate resource allocation rather than binary classification alone.

**Limitations.** Several limitations bound the scope of these conclusions. First, the dataset is cross-sectional: it captures customers at a single point in time and contains no temporal or behavioural signals such as transaction recency or engagement frequency. Models trained on snapshot data may miss the dynamic precursors to churn that longitudinal data would reveal (Neslin et al., 2006), and survival-analysis or sequence-based approaches may be more appropriate in a data-rich production setting. Second, the 10,000-row sample derives from a single institution; class composition and feature distributions may shift in deployment, requiring regular retraining schedules and distribution-shift monitoring. Third, the model has not been subject to a formal fairness audit; the demographic disparities in churn rates observed in EDA—by gender and geography—could translate into differential false-positive burdens across subgroups if not explicitly monitored (Barocas & Hardt, 2017). Fourth, the feature set is limited to summary-level account attributes; incorporating transactional, product-usage, and channel-interaction data would likely yield substantial improvements in predictive power.

---

## References

- Arlot, S. and Celisse, A. (2010). A survey of cross-validation procedures for model selection. *Statistics Surveys*, 4, pp.40–79.
- Barocas, S. and Hardt, M. (2017). *Fairness in machine learning*. NeurIPS Tutorial.
- Bergstra, J. and Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, pp.281–305.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), pp.5–32.
- Davis, J. and Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. *Proceedings of the 23rd International Conference on Machine Learning*, pp.233–240.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), pp.1189–1232.
- Kaufman, S., Rosset, S., Perlich, C. and Stitelman, O. (2012). Leakage in data mining: formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), pp.1–27.
- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *Proceedings of the 14th IJCAI*, 14(2), pp.1137–1145.
- Neslin, S. A., Gupta, S., Kamakura, W., Lu, J. and Mason, C. H. (2006). Defection detection: measuring and understanding the predictive accuracy of customer churn models. *Journal of Marketing Research*, 43(2), pp.204–211.
- Niculescu-Mizil, A. and Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd International Conference on Machine Learning*, pp.625–632.
- Sterne, J. A. C. et al. (2009). Multiple imputation for missing data in epidemiological and clinical research: potential and pitfalls. *BMJ*, 338, b2393.
