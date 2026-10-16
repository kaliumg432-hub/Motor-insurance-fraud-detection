"""Audit step 6: headline chart. What was reported vs what the files and a clean rebuild show.

"Reported" values are the AUC-ROC figures in the thesis, slides and results dashboard.
"Saved predictions" are recomputed from the project's own saved prediction files.
"Leak-free rebuild" comes from audit_05_results.csv (run audit_05_clean_pipeline.py first).
"""
import pandas as pd

from audit_common import BLUE, GREY, ORANGE, bar_chart, read_rds, scores

REPORTED = {"XGBoost": 0.92, "LSTM (text)": 0.90, "GNN (network)": 0.88, "Stacked ensemble": 0.96}

y = read_rds("saved_y_full.rds")
saved = {
    "XGBoost": scores(y, read_rds("saved_xgb_oof_aligned.rds"))["AUC_ROC"],
    "LSTM (text)": scores(y, read_rds("saved_lstm_oof_predictions.rds"))["AUC_ROC"],
    "GNN (network)": scores(y, read_rds("saved_gnn_oof_predictions.rds"))["AUC_ROC"],
}
ev = read_rds("saved_final_evaluation_results.rds")
saved["Stacked ensemble"] = float(ev.loc[ev.Model == "LogisticStack", "AUC_ROC"].iloc[0])

r = pd.read_csv("audit_05_results.csv", index_col=[0, 1]).loc["real_label"]
rebuild = {
    "XGBoost": r.loc["XGBoost (tabular)", "AUC_ROC"],
    "LSTM (text)": r.loc["Text model (LSTM stand-in)", "AUC_ROC"],
    "GNN (network)": r.loc["Network model (GNN stand-in)", "AUC_ROC"],
    "Stacked ensemble": r.loc["Stacked ensemble", "AUC_ROC"],
}
table = pd.DataFrame({"Reported": REPORTED, "Saved predictions": saved, "Leak-free rebuild": rebuild}).round(3)
print(table.to_string())
table.to_csv("audit_06_summary.csv")

labels = list(REPORTED)
bar_chart(
    "audit_fig1_summary.png",
    "Reported vs. what the code actually produced",
    "AUC-ROC on the 150,000-claim training data. Reported = thesis and dashboard.",
    labels,
    [("Reported", ORANGE, [REPORTED[k] for k in labels]),
     ("Project's saved predictions", GREY, [saved[k] for k in labels]),
     ("Leak-free rebuild (held-out test)", BLUE, [rebuild[k] for k in labels])],
)
