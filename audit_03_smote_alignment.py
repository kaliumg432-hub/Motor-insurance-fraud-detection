"""Audit step 3: SMOTE inflated the XGBoost score, then the row mapping scrambled it.

original_xgboost2.Rmd oversamples with smotefamily::SMOTE BEFORE splitting, so its test set
and its cross-validation folds contain synthetic claims. Step 16 then maps the out-of-fold
predictions back to the 150,000 real claims by assuming "SMOTE keeps original rows first".
smotefamily actually returns all fraud rows (real + synthetic) first, then all non-fraud rows,
so claim i receives the prediction of an unrelated row.

Part A checks this on the predictions the project saved.
Part B reproduces why SMOTE-before-split scores well on data with no signal.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from audit_common import BLUE, GREY, ORANGE, SEED, bar_chart, load_kaggle, read_rds

# ---- Part A: the saved files -------------------------------------------------------------
y = read_rds("saved_y_full.rds").astype(int)
y_bal = read_rds("saved_ensemble_target.rds").astype(int)
oof_bal = read_rds("saved_xgb_oof_predictions.rds")
oof_aligned = read_rds("saved_xgb_oof_aligned.rds")

switch = np.flatnonzero(np.diff(y_bal))
print(f"Balanced (SMOTE) target: {len(y_bal):,} rows. Class changes {len(switch)} time(s): "
      f"rows 1-{switch[0] + 1:,} are all fraud, the rest all non-fraud.")
print(f"Original labels: {len(y):,} rows, mixed order, {y.mean():.1%} fraud.")
agree = (y_bal[: len(y)] == y).mean()
print(f"Step 16 pairs claim i with balanced row i. Their labels agree on only {agree:.1%} of claims.")
auc_bal = roc_auc_score(y_bal, oof_bal)
auc_al = roc_auc_score(y, oof_aligned)
print(f"Saved XGBoost OOF AUC on the SMOTE rows : {auc_bal:.3f}")
print(f"Same predictions after Step 16 mapping  : {auc_al:.3f}   <- what the ensemble received")

# ---- Part B: reproduce SMOTE-before-split on the raw data --------------------------------
df, _ = load_kaggle()
X = pd.get_dummies(df.drop(columns=["PolicyNumber", "ClaimAmount"]), dtype=np.float32)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist", random_state=SEED)

Xs, ys = SMOTE(k_neighbors=5, random_state=SEED).fit_resample(X, y)
p_wrong = cross_val_predict(model, Xs, ys, cv=cv, method="predict_proba")[:, 1]
auc_wrong = roc_auc_score(ys, p_wrong)
is_syn = np.r_[np.zeros(len(X), bool), np.ones(len(Xs) - len(X), bool)]
real = ~is_syn
print(f"\nSMOTE before the split, CV on balanced rows : AUC {auc_wrong:.3f}")
print(f"   same predictions, real claims only          : AUC {roc_auc_score(ys[real], p_wrong[real]):.3f}")
print(f"   real fraud vs SMOTE fraud separable?         : AUC {roc_auc_score(is_syn[ys == 1], p_wrong[ys == 1]):.3f} "
      "(synthetic fraud scores higher than real fraud)")


def smote_inside_folds():
    p = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        Xt, yt = SMOTE(random_state=SEED).fit_resample(X.iloc[tr], y[tr])
        p[te] = model.fit(Xt, yt).predict_proba(X.iloc[te])[:, 1]
    return p


auc_right = roc_auc_score(y, smote_inside_folds())
print(f"SMOTE inside each training fold, real test : AUC {auc_right:.3f}")

pd.Series({"saved_oof_on_smote_rows": auc_bal, "saved_after_mapping": auc_al, "label_agreement": agree,
           "repro_smote_before_split": auc_wrong, "repro_smote_inside_folds": auc_right}).round(4).to_csv(
    "audit_03_results.csv", header=["value"])

bar_chart(
    "audit_fig3_smote.png",
    "SMOTE made noise look like signal, then the mapping erased it",
    "XGBoost AUC-ROC on the same 150,000 claims under three set-ups",
    ["SMOTE before the split\n(scored on synthetic rows too)", "Project's saved predictions\nafter the row mapping",
     "SMOTE inside training folds\n(scored on real claims only)"],
    [("AUC-ROC", [ORANGE, GREY, BLUE], [auc_wrong, auc_al, auc_right])],
)
