# Machine Learning Ensemble for Detecting Motor Insurance Fraud in Kenya

Final-year research project, BSc Actuarial Science, Jomo Kenyatta University of Agriculture and Technology (JKUAT), Group 4.

📄 **[Read the full thesis (PDF)](thesis/Motor-Insurance-Fraud-Ensemble-Thesis.pdf)**

---

## Summary

Kenya's existing systems catch fewer than 3% of estimated fraudulent motor insurance claims. This project builds a **three-model stacked ensemble** that reads each claim three different ways and combines the results:

| Model | What it reads | Signal it catches |
|---|---|---|
| **XGBoost** | Structured claim and policy data | Inflated amounts, new policies, repeat claims |
| **LSTM** | Claim narratives and claim history | Vague, exaggerated or contradictory wording |
| **Graph neural network (GCN)** | Links between claimants, garages, assessors and agents | Organised fraud rings |
| **Logistic regression meta-learner** | Out-of-fold probabilities from all three | One final fraud score |

The ensemble was trained on a 15,207-claim Kaggle dataset, balanced with SMOTE + ENN resampling, and validated out of sample on a 3,200-claim simulated Kenyan dataset.

## Headline results

On the held-out test set (3,041 claims):

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|
| Logistic regression (benchmark) | 84.12% | 61.38% | 58.24% | 59.77% | 0.8521 | 0.6934 |
| Random forest (benchmark) | 89.67% | 74.83% | 69.47% | 72.05% | 0.9041 | 0.7788 |
| XGBoost | 90.87% | 77.91% | 73.56% | 75.67% | 0.9198 | 0.8083 |
| LSTM | 86.94% | 70.44% | 78.89% | 74.46% | 0.9041 | 0.7876 |
| GNN | 84.93% | 67.82% | 75.11% | 71.29% | 0.8801 | 0.7544 |
| **Stacked ensemble** | **93.41%** | **84.27%** | **86.44%** | **85.34%** | **0.9587** | **0.9021** |

- **+9.67 percentage points F1** over the best single model (XGBoost)
- Every improvement is statistically significant (Wilcoxon signed-rank, p < 0.01)
- On the **Kenyan validation set**, the ensemble held up with F1 = 82.57% and AUC-PR = 0.8743

![Model comparison on the held-out test set](thesis/latex/fig9.png)

## Key findings

**Every model earns its place.** In the ablation study, removing any one model made the ensemble worse:

| Ensemble without… | F1 | Change |
|---|---|---|
| GNN | 81.24% | −4.10 pp |
| LSTM | 79.87% | −5.47 pp |
| XGBoost | 78.43% | −6.91 pp |
| *Full ensemble* | *85.34%* | — |

**It catches Kenyan fraud patterns.** Detection rates on the simulated Kenyan dataset, by fraud type:

| Fraud type | Cases | Detected | Rate |
|---|---|---|---|
| Staged accident | 142 | 128 | 90.1% |
| Document forgery | 98 | 87 | 88.8% |
| Damage inflation | 87 | 75 | 86.2% |
| Syndicate / ring fraud | 47 | 38 | 80.9% |
| Vehicle theft fabrication | 26 | 19 | 73.1% |
| **Total** | **400** | **347** | **86.8%** |

![Detection rate by Kenyan fraud type](thesis/latex/fig13.png)

**It helps investigators prioritise.** A confidence-scoring layer sorts claims into four tiers by how much the three models agree. Fraud is 67.8% of claims in the "Investigation Required" tier, against 13.2% in "High Confidence". Investigators can focus on the 5.4% of claims most likely to be fraud.

![Fraud prevalence by confidence tier](thesis/latex/fig15.png)

## Method

1. **Data:** Kaggle motor insurance claims (15,420 raw, 15,207 after cleaning), 14.8% fraudulent. KNN imputation, winsorised outliers, 47 engineered features reduced to 42.
2. **Class imbalance:** SMOTE oversampling followed by Edited Nearest Neighbour cleaning, applied to the training split only. The test set keeps the real 14.8% fraud rate.
3. **Base models:** each tuned with 5-fold stratified cross-validation.
   - XGBoost: max depth 6, learning rate 0.05, 400 trees
   - LSTM: two layers (128, 64 units) over GloVe 100-d embeddings, with attention to explain which words drove the score
   - GCN: two message-passing layers, with node features including degree, betweenness and PageRank
4. **Stacking:** out-of-fold probabilities feed an L2-regularised logistic regression. All three coefficients are positive and significant (XGBoost 1.247, LSTM 0.983, GNN 0.761).
5. **Validation:** a held-out Kaggle test set, then the 3,200-claim simulated Kenyan dataset with no retraining.

## Limitations

- The Kenyan validation data is **simulated**, because real insurer records weren't available. The next step is validation on real anonymised claims.
- The LSTM was trained on synthetic narratives. Production use would need retraining on real Swahili–English claim text.
- The claims network is sparse (density 0.0027), which limits the GNN on its own. It should improve as a live claims graph grows.
- Hyperparameter tuning used standard k-fold rather than fully nested cross-validation.

## Repository contents

```
├── thesis/
│   ├── Motor-Insurance-Fraud-Ensemble-Thesis.pdf   # full thesis, 86 pages
│   └── latex/
│       ├── main.tex        # LaTeX source
│       ├── sources.bib     # references (APA, apacite)
│       └── fig1–fig15.png  # all result figures
└── README.md
```

To rebuild the PDF from the `thesis/latex` folder, run `pdflatex main`, then `bibtex main`, then `pdflatex main` twice more.

## Team

Group 4, Department of Statistics and Actuarial Sciences, JKUAT

- Mwangi Gidraff Mbugua
- Obwana John Opwapo
- Wanjiru Wangui Caroline
- John Munyaka Ndinda
- Mwaura Eric Muiruri

**Supervisors:** Dr Herbert Imboga and Henry Kissinger
