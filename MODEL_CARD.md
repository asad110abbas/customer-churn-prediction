# Model card: churn classifier

| | |
|---|---|
| **Task** | Binary classification: will a subscriber cancel in the next cycle? |
| **Data** | Synthetic: 8,000 customers, 10 features (tenure, contract, charges, tickets, payments, engagement) |
| **Candidates** | Logistic Regression, Random Forest, Histogram Gradient Boosting |
| **Selection** | Highest 5-fold stratified CV ROC-AUC on the training split |
| **Evaluation** | Held-out 20% test set: ROC-AUC, PR-AUC, classification report at the chosen threshold |
| **Threshold** | Chosen to maximize retention-campaign profit (value of a saved churner vs. contact cost), not 0.5 |
| **Explainability** | Permutation importance on the test set (saved in `models/metrics.json`) |

## Intended use
Ranking customers for proactive retention outreach. Not for automated decisions that affect a customer's price or access.

## Limitations
- Trained on synthetic data with a known generating process, so real-world performance will differ.
- Assumes features are measured before the churn window (no leakage). Validate this on real data.
- Retrain and recalibrate when the product, pricing or customer mix changes.
