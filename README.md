# Tax Lien Redemption Probability Project

## Overview
This capstone project uses borrower application data to model default probability as a proxy for **tax lien non-redemption risk**. The goal is to test whether a probability-based risk scoring workflow can learn meaningful signal from structured financial data before moving to more tax-lien-specific datasets.

The final notebook shows a clear improvement from baseline to tuned modeling:
- **Baseline ROC-AUC:** `0.6678`
- **Improved holdout ROC-AUC:** `0.7510`
- **Improved holdout PR-AUC:** `0.2372`

From a business perspective, the strongest takeaway is that the final model is best used as a **risk-ranking and manual-review support tool**, not as a fully automated approval or rejection system.

## Research Question
**What is the probability that this tax lien will be redeemed within X months?**

Because public tax lien redemption datasets are limited, this submission uses the **Home Credit Default Risk** dataset as a proxy:
- **loan repayment** serves as a proxy for **tax lien redemption**
- **loan default** serves as a proxy for **tax lien non-redemption**

This is a methodological bridge rather than a direct substitute for true lien-level data.

## Business Understanding
The business goal is to identify which applicant- or loan-level features are associated with higher repayment/default risk so the model can support:
- risk ranking,
- manual review prioritization,
- better underwriting judgment,
- and eventually tax lien portfolio screening.

In this project, `TARGET = 1` represents default, interpreted as a proxy for **non-redemption risk**.

## Dataset
- **Dataset used:** Home Credit Default Risk competition
- **Source:** Kaggle - https://www.kaggle.com/c/home-credit-default-risk
- **Source file in this submission:** `application_train.7z`
- **Expanded file used by the notebook:** `application_train.csv`
- **Rows / columns:** `307,511` rows and `122` columns

The dataset includes financial, demographic, employment, and credit-related fields such as income, credit amount, annuity amount, contract type, housing type, and several external source risk scores.

## Why This Dataset Was Used
This submission uses a proxy repayment dataset rather than real county tax lien data because it:
1. is easier to use in a structured machine learning workflow,
2. contains a large number of observations and relevant financial variables,
3. supports the capstone goal of testing whether probability-based risk scoring is feasible.

Later project iterations can move closer to true tax lien prediction by incorporating variables such as lien amount, property value, lien-to-value ratio, delinquency duration, owner occupancy, and neighborhood conditions.

## Project Files
- `tax_lien_redemption_capstone_final_report.ipynb` – final notebook for submission
- `application_train.7z` – compressed source dataset
- `README.md` – final project summary

## Notebook Link
Open the notebook file directly from this submission package:
[Open the notebook](./notebooks/tax_lien_redemption_capstone_final_report.ipynb)

## What the Notebook Includes
The notebook contains:
- dataset inspection and target-balance review,
- missing-value and duplicate analysis,
- visualizations for numeric and categorical variables,
- outlier analysis using the IQR rule,
- engineered repayment-stress features,
- baseline logistic regression,
- ROC curve and confusion matrix review,
- improved modeling with restored `EXT_SOURCE_*` features,
- sampled model comparison for faster execution,
- logistic regression grid search on the full training set,
- final evaluation using ROC-AUC and PR-AUC.

## Key EDA Findings
- The target is **imbalanced**, with roughly **8.07%** positive cases.
- There were **no duplicate rows** in the application data.
- Many columns had substantial missingness, so the baseline workflow dropped features with more than **40%** missing values.
- Core financial variables such as `AMT_INCOME_TOTAL`, `AMT_CREDIT`, and `AMT_ANNUITY` were strongly right-skewed and contained substantial outliers.
- Default rates varied meaningfully across housing, employment, and education groups.

## Feature Engineering
The notebook includes engineered variables to capture repayment stress and improve interpretability, including:
- `CREDIT_INCOME_RATIO`
- `ANNUITY_INCOME_RATIO`
- `GOODS_CREDIT_RATIO`
- `CREDIT_ANNUITY_RATIO`
- `EXT_SOURCE_MEAN`
- `EXT_SOURCE_STD`
- `YEARS_BIRTH`
- `YEARS_EMPLOYED`
- `DAYS_EMPLOYED_ANOM`
- missingness indicators for `EXT_SOURCE_*`

These features help move the workflow closer to the kinds of inputs that would matter in a real tax lien risk model.

## Modeling Summary
### Baseline model
The baseline model is **logistic regression**, chosen because it is interpretable, computationally efficient, and a strong binary-classification benchmark.

### Evaluation metrics
The primary evaluation metric is **ROC-AUC**, with **PR-AUC** included as a companion metric.

ROC-AUC is appropriate because the problem is imbalanced and the business use case depends on ranking relative risk rather than using one arbitrary threshold. PR-AUC is also useful because the minority class matters and precision-recall behavior is important when defaults are relatively rare.

## Results
### Baseline results
- **Baseline ROC-AUC:** `0.6678`
- **Baseline PR-AUC:** `0.1519`
- **Positive-class rate:** `8.07%`

### Improved modeling results
The improved workflow restores and engineers the `EXT_SOURCE_*` variables, uses stratified cross-validation, compares multiple classifiers, and tunes logistic regression.

Key final results:
- **Cross-validated ROC-AUC (best comparison model):** `0.7457`
- **Cross-validated PR-AUC (best comparison model):** `0.2221`
- **Best tuned logistic regression CV ROC-AUC:** `0.7484`
- **Improved holdout ROC-AUC:** `0.7510`
- **Improved holdout PR-AUC:** `0.2372`

These results show a meaningful improvement over the baseline and support the idea that stronger feature retention and validation materially improve ranking performance.

## Interpretation
This model should be treated as a **decision-support and risk-ranking tool**, not as a fully automated approval or rejection system.

Even with a proxy dataset, the final workflow demonstrates that structured financial data contains enough signal to separate lower-risk and higher-risk cases better than chance.

## Limitations
This submission still has important limitations:
- the dataset is a **proxy**, not actual tax lien redemption data,
- model thresholds were not optimized for business cost tradeoffs,
- subgroup fairness analysis was not yet completed,
- more powerful boosting models were not yet incorporated in the final executed notebook.

## Next Steps
Recommended next steps include:
- compare against **XGBoost, LightGBM, or CatBoost**,
- calibrate predicted probabilities,
- optimize decision thresholds using business cost tradeoffs,
- evaluate subgroup fairness,
- add **SHAP** explainability,
- transition toward more tax-lien-specific data sources and engineered variables.

## References

- **Home Credit Default Risk dataset (Kaggle):** `https://www.kaggle.com/competitions/home-credit-default-risk` :contentReference[oaicite:0]{index=0}
- **Scikit-learn documentation**
  - `LogisticRegression`: `https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html`
  - `GridSearchCV`: `https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html`
  - `roc_auc_score`: `https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html` :contentReference[oaicite:1]{index=1}
  - `average_precision_score`: `https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html` :contentReference[oaicite:2]{index=2}
  - `RocCurveDisplay`: `https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html` :contentReference[oaicite:3]{index=3}
  - `PrecisionRecallDisplay`: `https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html` :contentReference[oaicite:4]{index=4}
- **py7zr documentation:** `https://py7zr.readthedocs.io/` and user guide `https://py7zr.readthedocs.io/en/latest/user_guide.html` :contentReference[oaicite:5]{index=5}
- **XGBoost documentation:** `https://xgboost.readthedocs.io/` and parameters guide `https://xgboost.readthedocs.io/en/stable/parameter.html` :contentReference[oaicite:6]{index=6}
- **LightGBM documentation:** `https://lightgbm.readthedocs.io/` and features/tuning docs `https://lightgbm.readthedocs.io/en/latest/Features.html` and `https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html` :contentReference[oaicite:7]{index=7}
- **CatBoost documentation:** `https://catboost.ai/docs/en/` and Python classifier reference `https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier` :contentReference[oaicite:8]{index=8}
- **SHAP documentation:** `https://shap.readthedocs.io/`, `https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html`, and overview notebook `https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html` :contentReference[oaicite:9]{index=9}
