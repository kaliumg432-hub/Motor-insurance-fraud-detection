"""Audit step 5: the same three-learner stacked ensemble, rebuilt without leaks.

What changes compared with the original R pipeline:
  * A 20% test set is split off FIRST and touched once, at the very end.
  * No SMOTE. Class imbalance is handled with class weights, so every score is on real claims.
  * Base-learner out-of-fold predictions keep their row index, so they line up with the right claim.
  * The text learner uses label-neutral narratives (TF-IDF + logistic regression as a fast
    stand-in for the LSTM).
  * The relational learner uses the claim network the honest way: for each claim, how often
    claims sharing its agent (RepNumber), policy (PolicyNumber) or garage proxy
    (area x make x category) were fraud, computed with cross-fitting so a claim never sees
    its own label.
  * The logistic-regression meta-learner and its threshold are fitted on training OOF only.

Positive control: a pipeline that scores 0.5 might just be broken. So the script runs twice,
once on the real FraudFound label and once on a planted label with a known fraud pattern
(policy-holder fault, past claims, external agents and three "collusive" agents). If the
pipeline works, the planted run must score clearly above 0.5, and the network learner must
pick up the collusive agents.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, TargetEncoder

from audit_common import SEED, as_categories, load_kaggle, scores

df, y_real = load_kaggle()
df = df.drop(columns=["ClaimAmount"])  # constant: 20,000 on every row

text = df.astype(str).drop(columns=["PolicyNumber", "RepNumber"]).apply(
    lambda r: " ".join(f"{k}: {v} ." for k, v in r.items()), axis=1)
net = pd.DataFrame({
    "agent": df.RepNumber.astype(str),
    "policy": df.PolicyNumber.astype(str),
    "garage": df.AccidentArea + "|" + df.Make + "|" + df.VehicleCategory,
})
for c in ["agent", "policy", "garage"]:
    net[c + "_degree"] = net.groupby(c)[c].transform("size")  # label-free graph feature
tab = as_categories(df.drop(columns=["PolicyNumber"]))

# Planted label for the positive control (about the same 30% fraud rate)
rng = np.random.default_rng(SEED)
logit = (-2.0 + 0.9 * (df.Fault == "Policy Holder") + 0.25 * df.PastNumberOfClaims
         + 0.5 * (df.AgentType == "External") + 1.0 * df.RepNumber.isin([3, 7, 11]))
y_planted = (rng.random(len(df)) < 1 / (1 + np.exp(-logit))).astype(int)


def run(y, label):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pos_w = (y[tr] == 0).sum() / (y[tr] == 1).sum()
    learners = {
        "XGBoost (tabular)": (tab, xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, enable_categorical=True,
            tree_method="hist", scale_pos_weight=pos_w, random_state=SEED)),
        "Text model (LSTM stand-in)": (text, make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=5),
            LogisticRegression(max_iter=2000, class_weight="balanced"))),
        "Network model (GNN stand-in)": (net, make_pipeline(
            ColumnTransformer([
                ("rates", TargetEncoder(target_type="binary", cv=cv), ["agent", "policy", "garage"]),
                ("deg", FunctionTransformer(np.log1p), ["agent_degree", "policy_degree", "garage_degree"]),
            ]),
            LogisticRegression(max_iter=2000, class_weight="balanced"))),
    }
    print(f"\n=== {label}: train {len(tr):,} claims, test {len(te):,} claims (held out until the end) ===")
    oof, test_pred = pd.DataFrame(index=tr), pd.DataFrame(index=te)
    for name, (X, model) in learners.items():
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        oof[name] = cross_val_predict(model, Xtr, y[tr], cv=cv, method="predict_proba")[:, 1]
        test_pred[name] = model.fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]

    meta = LogisticRegression().fit(oof, y[tr])
    oof_meta = cross_val_predict(LogisticRegression(), oof, y[tr], cv=cv, method="predict_proba")[:, 1]
    grid = np.quantile(oof_meta, np.linspace(0.01, 0.99, 99))
    thr = grid[int(np.argmax([f1_score(y[tr], oof_meta >= t) for t in grid]))]
    test_pred["Stacked ensemble"] = meta.predict_proba(test_pred[list(learners)])[:, 1]
    print("Meta-learner coefficients:", {k: round(float(v), 3) for k, v in zip(learners, meta.coef_[0])})

    res = pd.DataFrame({n: scores(y[te], test_pred[n].to_numpy(), threshold=thr if n == "Stacked ensemble" else 0.5)
                        for n in test_pred.columns}).T
    print(f"Held-out test results (fraud rate {y[te].mean():.1%}; flag-everything F1 = "
          f"{2 * y[te].mean() / (1 + y[te].mean()):.3f}):")
    print(res.round(4).to_string())
    return res


real = run(y_real, "Real FraudFound label")
planted = run(y_planted, "Positive control: planted fraud pattern")
from sklearn.metrics import roc_auc_score  # noqa: E402

ceiling = roc_auc_score(y_planted, logit)
print(f"\nBest possible AUC on the planted label (scoring with the true formula): {ceiling:.3f}")
planted.loc["Best possible (true formula)"] = {"AUC_ROC": ceiling}
out = pd.concat({"real_label": real, "planted_label": planted}).round(4)
out.to_csv("audit_05_results.csv")
