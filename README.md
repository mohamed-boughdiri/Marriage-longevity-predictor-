# Marriage Longevity: Divorce Prediction

An end-to-end data science project predicting whether a marriage ended in divorce, using demographic, financial, and relationship-behavior data.

**Dataset:** [Marriage Longevity — What Makes Relationships Last](https://www.kaggle.com/datasets/sergionefedov/marriage-longevity-what-makes-relationships-last) (Kaggle)

> **Note:** This is a data science learning exercise on a dataset of relationship indicators — not real-life relationship advice, and none of the correlations found here imply causation.

## Overview

The notebook walks through the full pipeline: loading and inspecting the data, exploratory analysis, leakage-aware feature selection, preprocessing, training two classifiers, and comparing their performance.

- **Rows / features:** 45,000 marriages, 20 features after dropping leakage columns
- **Target:** `divorced` (0/1), fairly balanced — ~54% not divorced, ~46% divorced

## What's in the notebook

1. **Basic data checks** — shape, dtypes, missing values, target balance
2. **Exploratory analysis** — numeric summaries, categorical breakdowns, divorce rate by education and religious attendance, relationship-behavior comparisons (criticism, contempt, defensiveness, stonewalling, repair attempts), correlation heatmap
3. **Feature selection** — dropping `marriage_id` (no signal) and `years_to_divorce` / `years_married` (leakage — see below)
4. **Preprocessing** — `ColumnTransformer` with `StandardScaler` for numeric features and `OneHotEncoder` for categoricals, wrapped in a `Pipeline` to avoid leakage across the train/test split
5. **Modeling** — Logistic Regression and Random Forest, evaluated on accuracy, precision, recall, F1, and ROC AUC
6. **Model persistence** — saving the chosen model with `joblib` and running a sample inference check

## Key findings

- **Relationship behavior is the strongest signal.** Contempt has the highest correlation with divorce (0.33) of any feature, consistent with Gottman's research identifying contempt as the most corrosive of the "Four Horsemen." Every negative behavior (criticism, contempt, defensiveness, stonewalling) is higher on average among divorced couples; every positive one (repair attempt success, positive-to-negative ratio, shared activities) is lower.
- **Education level shows a strong, monotonic protective effect** — divorce rate drops from ~63% (less than high school) to ~30% (graduate degree).
- **Religious attendance barely matters here** — divorce rate sits in a tight 45–47% band across all attendance categories, despite being an intuitive variable to include.
- **`years_married` and `years_to_divorce` are perfectly correlated (1.0)** and both excluded as leakage: `years_to_divorce` is only defined for marriages that already ended, so it wouldn't exist at prediction time for an ongoing marriage.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.756 | 0.745 | 0.716 | 0.730 | 0.839 |
| Random Forest | 0.746 | 0.721 | 0.729 | 0.725 | 0.823 |

Logistic Regression slightly outperforms Random Forest across every metric. The two models' closeness suggests the underlying relationships are fairly linear/additive, so Random Forest's extra flexibility isn't buying much here. Logistic Regression was saved as the final model for its slight edge in performance plus the benefit of interpretable coefficients.

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

Download `marriage_longevity_master.csv` and `data_dictionary.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/sergionefedov/marriage-longevity-what-makes-relationships-last) and place them in the project root before running the notebook.

## Limitations

- Results are based on a single 80/20 train/test split; k-fold cross-validation would give a more robust model comparison.
- The dataset is observational — correlations (e.g. contempt and divorce) do not establish causation.
