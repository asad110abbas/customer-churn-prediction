# 🔮 BRIHA · Customer Churn Prediction

![BRIHA](https://img.shields.io/badge/BRIHA-Genesis_of_AGI-8B5CF6?style=for-the-badge&labelColor=0B1024) ![Module](https://img.shields.io/badge/Briha-Data_Lab-00F0FF?style=for-the-badge&labelColor=0B1024)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![CI](https://github.com/asad110abbas/customer-churn-prediction/actions/workflows/ci.yml/badge.svg)

> ⚡ **BRIHA: The Genesis of AGI.** This is a **Briha Data Lab** module, part of the [BRIHA AI World](https://github.com/asad110abbas), built by **Syed Asad Abbas**.

Predict which subscribers are about to cancel, and **decide who to contact** for a retention campaign.
The model is chosen by cross-validation, and the decision threshold is set by **campaign profit**, not the default 0.5.

## Highlights
- Reproducible `scikit-learn` **Pipeline** (`ColumnTransformer` + scaler + one-hot) with no train/test leakage
- **Model selection** across Logistic Regression, Random Forest and Gradient Boosting via stratified 5-fold CV
- **Profit-optimal threshold**: picks the cut-off that maximizes (value × churners saved − cost × contacts)
- **Explainability** via permutation importance
- **Scoring CLI** that outputs risk tiers and the next best action per customer
- [Model card](MODEL_CARD.md) documenting intended use and limitations

## Quick start
```bash
pip install -r requirements.txt
python -m churn.cli train      # trains, selects, saves models/churn_model.joblib + metrics.json
python -m churn.cli score      # scores new customers -> models/scored_customers.csv
pytest -q
```

## Example output (`score`)
| customer_id | churn_probability | risk | action |
|---|---|---|---|
| C000123 | 0.91 | high | Call + retention offer |
| C000045 | 0.47 | medium | Email discount |
| C000310 | 0.08 | low | No action |

## Structure
```
churn/
  data.py      synthetic customers with an explainable churn signal
  model.py     pipelines, CV model selection, profit threshold, permutation importance
  predict.py   batch scoring + risk tiers
  cli.py       train / score commands
tests/         pytest suite
```

> The data is synthetic. The example rows above illustrate the output format.
