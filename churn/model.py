"""Training pipeline: preprocessing + model selection by cross-validated ROC-AUC."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                            n_jobs=-1, random_state=0),
    "gradient_boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, random_state=0),
}


def build(name: str) -> Pipeline:
    return Pipeline([("prep", preprocessor()), ("clf", CANDIDATES[name])])


def best_threshold(y_true, proba, retain_value: float = 3.0, contact_cost: float = 1.0) -> float:
    """Pick the probability cut-off that maximizes campaign profit.

    Each true churner we contact is worth `retain_value`; each contact costs `contact_cost`.
    """
    best_t, best_profit = 0.5, -np.inf
    for t in np.linspace(0.05, 0.95, 91):
        flag = proba >= t
        profit = retain_value * np.sum(flag & (y_true == 1)) - contact_cost * np.sum(flag)
        if profit > best_profit:
            best_t, best_profit = float(t), profit
    return round(best_t, 2)


def train(df: pd.DataFrame, out_dir: Path | None = None, cv_folds: int = 5) -> dict:
    X, y = df[NUMERIC + CATEGORICAL], df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=0)
    cv_scores = {n: float(cross_val_score(build(n), X_tr, y_tr, cv=cv, scoring="roc_auc").mean()) for n in CANDIDATES}
    best = max(cv_scores, key=cv_scores.get)
    model = build(best).fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    thr = best_threshold(y_te.to_numpy(), proba)
    imp = permutation_importance(model, X_te, y_te, n_repeats=5, random_state=0, scoring="roc_auc")
    importance = dict(sorted(zip(X.columns, imp.importances_mean.round(4)), key=lambda kv: -kv[1]))
    report = {
        "cv_roc_auc": {k: round(v, 4) for k, v in cv_scores.items()},
        "selected_model": best,
        "test_roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "test_pr_auc": round(float(average_precision_score(y_te, proba)), 4),
        "churn_rate": round(float(y.mean()), 4),
        "decision_threshold": thr,
        "classification_report": classification_report(y_te, proba >= thr, output_dict=True),
        "permutation_importance": {k: float(v) for k, v in importance.items()},
    }
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "threshold": thr}, out_dir / "churn_model.joblib")
        (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return {"model": model, "report": report}
