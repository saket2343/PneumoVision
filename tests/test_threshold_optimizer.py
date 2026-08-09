import numpy as np
import pytest

from src.threshold_optimizer import (
    probability_distribution_summary,
    select_best_balanced_accuracy,
    select_best_f1,
    select_best_youden_j,
    select_sensitivity_constrained,
    sweep_thresholds,
)


@pytest.fixture()
def separable_predictions():
    """20 NORMAL with low probabilities, 20 PNEUMONIA with high probabilities —
    a case where a clean threshold clearly exists."""
    rng = np.random.RandomState(0)
    y_true = np.array([0] * 20 + [1] * 20)
    y_prob = np.concatenate([rng.uniform(0.0, 0.3, 20), rng.uniform(0.7, 0.98, 20)])
    return y_true, y_prob


@pytest.fixture()
def collapsed_predictions():
    """Everything predicted PNEUMONIA regardless of threshold — the exact
    failure mode reported against the buggy pipeline (all probabilities > 0.6)."""
    rng = np.random.RandomState(1)
    y_true = np.array([0] * 15 + [1] * 25)
    y_prob = rng.uniform(0.6, 0.95, 40)  # no separation, all high
    return y_true, y_prob


def test_sweep_thresholds_shape_and_range(separable_predictions):
    y_true, y_prob = separable_predictions
    df = sweep_thresholds(y_true, y_prob, min_threshold=0.01, max_threshold=0.99, step=0.01)
    assert len(df) == 99
    assert set(["threshold", "sensitivity", "specificity", "f1", "balanced_accuracy", "youden_j"]).issubset(df.columns)
    assert (df["sensitivity"] >= 0).all() and (df["sensitivity"] <= 1).all()


def test_sweep_thresholds_extremes(separable_predictions):
    y_true, y_prob = separable_predictions
    df = sweep_thresholds(y_true, y_prob, min_threshold=0.01, max_threshold=0.99, step=0.01)
    # At a very low threshold everything is predicted positive -> sensitivity 1, specificity 0
    low = df.iloc[0]
    assert low["sensitivity"] == pytest.approx(1.0)
    # At a very high threshold everything is predicted negative -> sensitivity 0, specificity 1
    high = df.iloc[-1]
    assert high["sensitivity"] == pytest.approx(0.0)
    assert high["specificity"] == pytest.approx(1.0)


def test_sensitivity_constrained_selection_finds_a_clean_threshold(separable_predictions):
    y_true, y_prob = separable_predictions
    df = sweep_thresholds(y_true, y_prob)
    result = select_sensitivity_constrained(df, min_sensitivity=0.95)
    assert result is not None
    assert result["sensitivity"] >= 0.95
    # With clean separation we should also get near-perfect specificity at the chosen point
    assert result["specificity"] > 0.8


def test_sensitivity_constrained_selection_on_collapsed_predictions_still_finds_something(collapsed_predictions):
    """Even for the reported bug's failure mode (everything > 0.6, all predicted
    PNEUMONIA), a low-enough threshold always yields sensitivity 1.0 — so a
    result should still be returned (just with poor specificity), not None."""
    y_true, y_prob = collapsed_predictions
    df = sweep_thresholds(y_true, y_prob, min_threshold=0.01, max_threshold=0.99, step=0.01)
    result = select_sensitivity_constrained(df, min_sensitivity=0.95)
    assert result is not None
    assert result["sensitivity"] >= 0.95


def test_youden_j_best_matches_manual_argmax(separable_predictions):
    y_true, y_prob = separable_predictions
    df = sweep_thresholds(y_true, y_prob)
    result = select_best_youden_j(df)
    assert result["youden_j"] == pytest.approx(df["youden_j"].max())


def test_best_f1_and_balanced_accuracy_are_valid_rows(separable_predictions):
    y_true, y_prob = separable_predictions
    df = sweep_thresholds(y_true, y_prob)
    f1_result = select_best_f1(df)
    bal_result = select_best_balanced_accuracy(df)
    assert f1_result["f1"] == pytest.approx(df["f1"].max())
    assert bal_result["balanced_accuracy"] == pytest.approx(df["balanced_accuracy"].max())


def test_probability_distribution_summary_separates_classes(separable_predictions):
    y_true, y_prob = separable_predictions
    summary = probability_distribution_summary(y_true, y_prob, class_names=("NORMAL", "PNEUMONIA"))
    assert summary["NORMAL"]["count"] == 20
    assert summary["PNEUMONIA"]["count"] == 20
    # The whole point of this diagnostic: NORMAL's high end should sit below PNEUMONIA's low end
    assert summary["NORMAL"]["p95"] < summary["PNEUMONIA"]["p25"]


def test_probability_distribution_summary_flags_collapsed_case(collapsed_predictions):
    y_true, y_prob = collapsed_predictions
    summary = probability_distribution_summary(y_true, y_prob, class_names=("NORMAL", "PNEUMONIA"))
    # In the collapsed/bug case both classes' probabilities occupy the same high band
    assert summary["NORMAL"]["min"] >= 0.6
    assert summary["PNEUMONIA"]["min"] >= 0.6
