# Obesity Estimation — ML for Classification & Regression

Analysis and optimization of machine learning techniques for **classification** and **regression**,
applied to the "Estimation of Obesity Levels Based on Eating Habits and Physical Condition" dataset.

## Project Goal

Using one dataset, this project tackles two supervised learning problems and compares
multiple ML algorithms on each, to understand their strengths, weaknesses, and best use cases:

1. **Classification** — predict a person's obesity category (7 classes, from Insufficient
   Weight to Obesity Type III) from their eating habits and physical condition.
2. **Regression** — predict a person's exact `Weight` from the same kind of features.

## Dataset

- **Source**: [UCI Machine Learning Repository — Estimation of obesity levels based on eating habits and physical condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
- **Size**: 2111 rows, 17 columns (demographics, diet, and physical activity habits)
- File included at `data/ObesityDataSet_raw_and_data_sinthetic.csv`

## Project Structure

```
.
├── data/
│   └── ObesityDataSet_raw_and_data_sinthetic.csv
├── notebooks/
│   └── AI_project.ipynb
├── requirements.txt
└── README.md
```

## Pipeline

1. **Data Cleaning** — duplicate removal, outlier capping (IQR), label encoding,
   MinMax/Z-score scaling
2. **EDA** — correlation heatmap, distribution plots
3. **Classification models** — Logistic Regression, Random Forest, XGBoost, Neural
   Network (MLP), Stacking, Voting — with GridSearchCV hyperparameter tuning
4. **Regression models** — Linear Regression (Gradient Descent from scratch), Ridge,
   Lasso, Normal Equation, plus ensemble techniques (Voting, Stacking, Bagging,
   Boosting, XGBoost, Random Forest) with 5-fold cross-validation

## Results Summary

| Task | Best model | Score |
|---|---|---|
| Classification | XGBoost | ~99% accuracy |
| Regression | Random Forest / Stacking / XGBoost | R² ~0.87-0.89 |

Linear models (Logistic Regression for classification, plain linear regression variants
for regression) consistently underperform tree-based ensembles, since the underlying
relationships in the data are non-linear.



