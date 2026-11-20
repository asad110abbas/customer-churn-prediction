import numpy as np
import pytest

from churn.data import make_customers
from churn.model import best_threshold, train
from churn.predict import score


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    result = train(make_customers(1500, seed=1), out, cv_folds=3)
    return result, out


def test_data_has_reasonable_churn_rate():
    df = make_customers(3000)
    assert 0.1 < df["churned"].mean() < 0.6
    assert df["customer_id"].is_unique


def test_model_beats_random(trained):
    result, _ = trained
    assert result["report"]["test_roc_auc"] > 0.7


def test_threshold_prefers_true_churners():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    assert 0.2 < best_threshold(y, p) <= 0.8


def test_scoring_outputs_actions(trained):
    _, out = trained
    scored = score(make_customers(50, seed=5), out / "churn_model.joblib")
    assert len(scored) == 50
    assert scored["churn_probability"].between(0, 1).all()
    assert set(scored["action"]) <= {"Call + retention offer", "Email discount", "No action"}
