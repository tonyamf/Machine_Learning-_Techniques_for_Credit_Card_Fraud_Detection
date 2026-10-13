import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from conftest import FIXTURE


def test_missing_csv_prints_message_and_returns(fraud_module, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    fraud_module.main()
    assert "creditcard.csv' not found" in capsys.readouterr().out


def test_evaluate_model_reports_expected_metrics(fraud_module, capsys):
    df = pd.read_csv(FIXTURE).drop_duplicates()
    X, y = df.drop(columns="Class"), df["Class"]
    model = LogisticRegression(max_iter=1000).fit(X, y)
    fraud_module.evaluate_model(model, X, y, "LR")
    out = capsys.readouterr().out
    for needle in ("Accuracy:", "Log-Loss:", "AUC Score:", "Fraud (1)", "Normal (0)"):
        assert needle in out


def test_accuracy_misleads_on_imbalanced_labels():
    # Documents why the project reports precision/recall/AUC: a model that never
    # predicts fraud still scores ~99.8% accuracy at the dataset's 0.17% fraud rate.
    from sklearn.metrics import accuracy_score, recall_score

    y = np.zeros(10000, int)
    y[:17] = 1
    pred = np.zeros_like(y)
    assert accuracy_score(y, pred) > 0.998
    assert recall_score(y, pred) == 0.0


@pytest.fixture
def run_main(fraud_module, workdir, monkeypatch, capsys):
    """Run main() end to end on the synthetic frame, recording StandardScaler.fit calls."""
    fit_sizes = []
    real = fraud_module.StandardScaler

    class Recording(real):
        def fit(self, X, y=None, **kw):
            fit_sizes.append(len(X))
            return super().fit(X, y, **kw)

    monkeypatch.setattr(fraud_module, "StandardScaler", Recording)
    fraud_module.main()
    return capsys.readouterr().out, fit_sizes


def test_main_runs_all_three_experiments(run_main):
    out, _ = run_main
    assert "--- Analysis Complete ---" in out
    for name in ("Class-Weighted SVM", "SVM with SMOTEENN Sampling", "XGBoost Classifier"):
        assert f"--- Evaluating {name} ---" in out


def test_duplicates_dropped_and_split_is_70_30(run_main):
    out, _ = run_main
    n_unique = len(pd.read_csv(FIXTURE).drop_duplicates())
    assert n_unique == 300  # fixture has 303 rows, 3 exact duplicates
    n_train = round(n_unique * 0.7)
    assert f"Training set shape: ({n_train}, 30)" in out
    assert f"Test set shape: ({n_unique - n_train}, 30)" in out


def test_smoteenn_runs_on_train_only_and_rebalances(run_main):
    out, _ = run_main
    line = [l for l in out.splitlines() if l.startswith("Original training shape")][0]
    resampled = [l for l in out.splitlines() if l.startswith("Resampled training shape")][0]
    # Original training minority share is ~10%; after SMOTEENN the minority share is higher.
    import re
    o = {int(k): int(v) for k, v in re.findall(r"(\d): (\d+)", line)}
    r = {int(k): int(v) for k, v in re.findall(r"(\d): (\d+)", resampled)}
    assert sum(o.values()) == 210  # train rows only, not the whole frame
    assert r[1] / sum(r.values()) > o[1] / sum(o.values())


@pytest.mark.xfail(
    strict=True,
    reason="Known bug (see HubRefine review.md): StandardScaler is fit on the whole "
    "frame before train_test_split, so test-set statistics leak into the scaling.",
)
def test_scaler_is_not_fit_on_test_rows(run_main):
    _, fit_sizes = run_main
    assert max(fit_sizes) <= 210  # would be 300 if fit on all rows
