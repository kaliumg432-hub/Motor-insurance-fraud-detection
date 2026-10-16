"""Audit step 1: does the training data contain any fraud signal at all?

Two independent checks on the 150,000-row Kaggle-style file the project trained on:
  1. A chi-square test of every column against FraudFound (numeric columns cut into deciles).
  2. A leak-free XGBoost and logistic regression, 5-fold stratified cross-validation,
     no resampling, no engineered features that touch the label.

If the columns carry signal, the cross-validated AUC should be well above 0.5.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from audit_common import SEED, as_categories, load_kaggle, scores

df, y = load_kaggle()
print(f"Rows: {len(df):,}   fraud rate: {y.mean():.1%}")
const = [c for c in df.columns if df[c].nunique() == 1]
print(f"Constant columns (same value on every row): {const}  e.g. ClaimAmount = {df['ClaimAmount'].iloc[0]}")

# 1. Column-by-column association with the label
rows = []
for c in df.columns:
    if c in const:
        continue
    s = df[c]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 60:
        s = pd.qcut(s, 10, duplicates="drop")
    t = pd.crosstab(s, y)
    chi, p, *_ = chi2_contingency(t)
    v = np.sqrt(chi / (t.values.sum() * (min(t.shape) - 1)))
    rows.append((c, p, v))
assoc = pd.DataFrame(rows, columns=["column", "p_value", "cramers_v"]).sort_values("p_value")
print("\nStrongest column-label associations (Cramer's V: 0 = none, 0.1 = weak, 0.3 = moderate):")
print(assoc.head(6).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"Largest Cramer's V across all {len(assoc)} columns: {assoc.cramers_v.max():.3f}")
print(f"Columns with p < 0.05: {(assoc.p_value < 0.05).sum()} of {len(assoc)} "
      f"(about {0.05 * len(assoc):.1f} expected by pure chance)")

# 2. Leak-free models
X = as_categories(df.drop(columns=["PolicyNumber"] + const))
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                              enable_categorical=True, tree_method="hist", random_state=SEED)
p_xgb = cross_val_predict(xgb_model, X, y, cv=cv, method="predict_proba")[:, 1]

cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
Xo = X.copy()
Xo[cat_cols] = Xo[cat_cols].astype(str)
lr = make_pipeline(OneHotEncoder(handle_unknown="ignore", min_frequency=20), LogisticRegression(max_iter=2000))
p_lr = cross_val_predict(lr, Xo.astype(str), y, cv=cv, method="predict_proba")[:, 1]

res = pd.DataFrame({"XGBoost (5-fold CV)": scores(y, p_xgb), "Logistic regression (5-fold CV)": scores(y, p_lr)}).T
print("\nLeak-free cross-validated performance:")
print(res.round(4).to_string())
print("\nA model that always answers 'fraud' gets AUC-PR = base rate "
      f"({y.mean():.3f}) and F1 = {2 * y.mean() / (1 + y.mean()):.3f}.")
res.round(4).to_csv("audit_01_results.csv")
