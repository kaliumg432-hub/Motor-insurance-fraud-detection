"""Audit step 4: the Kenyan validation graded models against rules built from their own inputs.

The Kenyan file has no fraud label. original_Kenyan_updated.Rmd (section 6) invents one,
FraudProxy = 1 if any of four rules fires:
  - Previous_Claims_Count >= 3 and ClaimAmount >= 75th percentile
  - Vehicle_Age >= 8 and the cause is theft
  - Previous_Claims_Count >= 2 and ClaimAmount >= 80th percentile
  - ClaimAmount / Monthly_Premium_KES >= 40   (that column does not exist, so this rule never fires)
The "Kenyan XGBoost benchmark" is then trained on ClaimAmount, Previous_Claims_Count,
Vehicle_Age and Accident_Cause, the exact columns the rules use, so it simply re-learns the
rules. A near-perfect score here measures nothing about fraud.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text

from audit_common import BLUE, GREY, KENYA_CSV, ORANGE, SEED, as_categories, bar_chart, need

k = pd.read_csv(need(KENYA_CSV), keep_default_na=False, na_values=[""])
k = k.rename(columns={"Total_Claim_Amount_KES": "ClaimAmount", "Vehicle_Type": "VehicleCategory",
                      "Region": "AccidentArea", "Customer_Age": "Age", "Vehicle_Value_KES": "VehiclePrice"})
print(f"Kenyan file: {len(k):,} rows, {k.AccidentArea.nunique()} regions ({', '.join(sorted(k.AccidentArea.unique()))})")

amt = k.ClaimAmount
r1 = (k.Previous_Claims_Count >= 3) & (amt >= amt.quantile(0.75))
r2 = (k.Vehicle_Age >= 8) & k.Accident_Cause.str.lower().str.contains("theft|stolen")
r3 = (k.Previous_Claims_Count >= 2) & (amt >= amt.quantile(0.80))
y = (r1 | r2 | r3).astype(int).to_numpy()
print(f"Proxy 'fraud' rate: {y.mean():.1%}  (rule hits: {r1.sum()}, {r2.sum()}, {r3.sum()}, rule 4: 0)")

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
bench_cols = ["ClaimAmount", "VehicleCategory", "AccidentArea", "Age", "VehiclePrice",
              "Previous_Claims_Count", "Vehicle_Age", "Accident_Cause"]
model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, enable_categorical=True,
                          tree_method="hist", random_state=SEED)
p_bench = cross_val_predict(model, as_categories(k[bench_cols]), y, cv=cv, method="predict_proba")[:, 1]
auc_bench = roc_auc_score(y, p_bench)
print(f"\nBenchmark XGBoost on the rule columns (5-fold CV): AUC {auc_bench:.3f}")

others = [c for c in bench_cols if c not in ("ClaimAmount", "Previous_Claims_Count", "Vehicle_Age", "Accident_Cause")]
p_other = cross_val_predict(model, as_categories(k[others]), y, cv=cv, method="predict_proba")[:, 1]
auc_other = roc_auc_score(y, p_other)
print(f"Same model WITHOUT the four rule columns       : AUC {auc_other:.3f}")

tree = DecisionTreeClassifier(max_depth=4, random_state=SEED).fit(
    pd.get_dummies(k[["ClaimAmount", "Previous_Claims_Count", "Vehicle_Age", "Accident_Cause"]]), y)
print(f"A depth-4 decision tree reproduces the proxy label on {tree.score(pd.get_dummies(k[['ClaimAmount', 'Previous_Claims_Count', 'Vehicle_Age', 'Accident_Cause']]), y):.1%} of rows:")
print(export_text(tree, feature_names=list(pd.get_dummies(k[["ClaimAmount", "Previous_Claims_Count", "Vehicle_Age", "Accident_Cause"]]).columns), max_depth=2))

saved = pd.read_csv("saved_kenya_final_model_comparison.csv")
ens = saved[saved.Model.str.startswith("Full Ensemble")].iloc[0]
print(f"Project's own final Kenyan run (Kaggle-trained ensemble on the 400 test claims): "
      f"AUC {ens.AUC_ROC:.3f}, F1 {ens.F1:.3f}, recall {ens.Recall:.2f} at precision {ens.Precision:.2f}"
      " -> it flagged every claim as fraud.")

pd.Series({"proxy_rate": y.mean(), "benchmark_auc_rule_columns": auc_bench, "benchmark_auc_without_rule_columns": auc_other,
           "project_ensemble_auc": ens.AUC_ROC, "project_ensemble_f1": ens.F1}).round(4).to_csv(
    "audit_04_results.csv", header=["value"])

bar_chart(
    "audit_fig4_kenya.png",
    "The Kenyan 'validation' re-learned its own label",
    "AUC-ROC against a fraud label made from four rules on the same columns",
    ["Benchmark using the\nrule columns", "Benchmark without\nthe rule columns", "Project's Kaggle-trained\nensemble (saved run)"],
    [("AUC-ROC", [ORANGE, BLUE, GREY], [auc_bench, auc_other, float(ens.AUC_ROC)])],
)
