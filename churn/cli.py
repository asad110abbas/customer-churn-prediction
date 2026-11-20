"""python -m churn.cli train | score"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import make_customers
from .model import train
from .predict import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "score"])
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--models", type=Path, default=Path("models"))
    a = ap.parse_args()
    if a.command == "train":
        r = train(make_customers(a.n), a.models)["report"]
        print(json.dumps({k: r[k] for k in ["cv_roc_auc", "selected_model", "test_roc_auc", "test_pr_auc",
                                             "decision_threshold"]}, indent=2))
    else:
        scored = score(make_customers(500, seed=99), a.models / "churn_model.joblib")
        scored.to_csv(a.models / "scored_customers.csv", index=False)
        print(scored.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
