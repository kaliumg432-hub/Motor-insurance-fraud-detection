"""Audit step 2: the "claim narratives" told the text model the answer.

The Kaggle file has no free text, so original_LSTM3.Rmd (step 2) generates a narrative
for every claim. It picks the template FROM THE FRAUD LABEL:
  fraud claims -> "...However the circumstances seemed unusual." /
                  "...but witness accounts were inconsistent."
  honest claims -> "...This appears to be a straightforward claim." /
                   "...All documentation is consistent."
Any text model then only has to spot those words. This script rebuilds both the leaky
narratives and the label-neutral ones from original_LSTM_Updated.Rmd, and scores a simple
text model on each with 5-fold cross-validation. (TF-IDF + logistic regression stands in
for the LSTM: the leak is in the data, so the model type does not matter.)
It then checks the LSTM predictions the project actually saved and fed to the ensemble.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from audit_common import BLUE, ORANGE, SEED, bar_chart, load_kaggle, read_rds, scores

df, y = load_kaggle()
rng = np.random.default_rng(SEED)
pick = rng.integers(0, 2, len(df))


def leaky(row, fraud, k):
    # IncidentType / IncidentTime do not exist in the data, so R pasted NA, same here
    police = "was filed" if row.PoliceReportFiled == "Yes" else "was not filed"
    wit = "present" if row.WitnessPresent == "Yes" else "absent"
    if fraud:
        return [
            f"The incident occurred NA involving a {row.VehicleCategory} The police report {police} and witnesses were {wit} "
            f"The driver who was at fault according to {row.Fault} reported damages amounting to KES {row.ClaimAmount} "
            "However the circumstances seemed unusual.",
            f"A {row.VehicleCategory} was involved in a NA at NA The claimant aged {row.Age} with {row.PastNumberOfClaims} "
            "past claims reported extensive damage but witness accounts were inconsistent.",
        ][k]
    return [
        f"The {row.VehicleCategory} was involved in a NA at NA Police report {police} with witnesses {wit} "
        f"Damage estimate KES {row.ClaimAmount} This appears to be a straightforward claim.",
        f"A routine NA involving a {row.VehicleCategory} at NA Driver was {row.Fault} with only {row.PastNumberOfClaims} "
        "past claims All documentation is consistent.",
    ][k]


NEUTRAL_FIELDS = ["Make", "VehicleCategory", "BasePolicy", "AccidentArea", "Age", "Sex", "MaritalStatus", "Fault",
                  "ClaimAmount", "VehiclePrice", "Deductible", "DriverRating", "PastNumberOfClaims", "PoliceReportFiled",
                  "WitnessPresent", "AgentType", "NumberOfSuppliments", "AddressChange-Claim", "NumberOfCars", "Month",
                  "WeekOfMonth", "DayOfWeek", "MonthClaimed", "WeekOfMonthClaimed", "Days:Policy-Accident",
                  "Days:Policy-Claim"]


text_leaky = [leaky(r, f, k) for r, f, k in zip(df.itertuples(index=False), y, pick)]
text_neutral = df[NEUTRAL_FIELDS].astype(str).apply(lambda r: " ".join(f"{k}: {v} ." for k, v in r.items()), axis=1)
print("Leaky example (fraud):  ", text_leaky[int(np.flatnonzero(y == 1)[0])])
print("Neutral example:        ", text_neutral.iloc[0][:150], "...")

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=5), LogisticRegression(max_iter=2000))
res = {}
for name, text in [("Leaky narratives (LSTM3)", text_leaky), ("Label-neutral narratives", list(text_neutral))]:
    p = cross_val_predict(model, text, y, cv=cv, method="predict_proba")[:, 1]
    res[name] = scores(y, p)
    print(f"{name:28s} AUC-ROC {res[name]['AUC_ROC']:.3f}   AUC-PR {res[name]['AUC_PR']:.3f}   F1 {res[name]['F1']:.3f}")

import pandas as pd  # noqa: E402

pd.DataFrame(res).T.round(4).to_csv("audit_02_results.csv")

saved = read_rds("saved_lstm_oof_predictions.rds")
y_saved = read_rds("saved_y_full.rds")
print(f"\nSaved LSTM out-of-fold predictions fed to the ensemble: {len(saved):,} values, "
      f"{np.isin(np.round(saved, 6), [0, 1]).mean():.1%} of them are 0 or 1 to six decimal places, "
      f"and they match the fraud label on {(np.round(saved) == y_saved).mean():.2%} of claims.")
print("Conclusion: the saved LSTM output was produced from the leaky narratives.")

bar_chart(
    "audit_fig2_text_leak.png",
    "The text model was reading the answer",
    "5-fold CV, same text model, 150,000 claims. Only the narrative wording differs.",
    ["Narratives chosen\nfrom the fraud label", "Label-neutral\nnarratives"],
    [("AUC-ROC", [ORANGE, BLUE], [res["Leaky narratives (LSTM3)"]["AUC_ROC"], res["Label-neutral narratives"]["AUC_ROC"]])],
)
