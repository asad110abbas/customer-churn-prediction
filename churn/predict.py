"""Score customers with a trained model and assign a retention action."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .data import CATEGORICAL, NUMERIC


def score(df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    proba = bundle["model"].predict_proba(df[NUMERIC + CATEGORICAL])[:, 1]
    out = df[["customer_id"]].copy()
    out["churn_probability"] = proba.round(4)
    high_cut = min(max(bundle["threshold"], 0.31), 0.99)  # keep bins strictly increasing
    out["risk"] = pd.cut(proba, [0, 0.3, high_cut, 1.0], labels=["low", "medium", "high"], include_lowest=True)
    out["action"] = out["risk"].map({"high": "Call + retention offer", "medium": "Email discount",
                                     "low": "No action"}).astype(str)
    return out.sort_values("churn_probability", ascending=False)
