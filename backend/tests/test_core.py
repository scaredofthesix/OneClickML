import pandas as pd
import pytest

from core import analyze, detect_task, predict_value, prepare, usable_features


def test_detects_regression_and_finds_obvious_driver(students: pd.DataFrame):
    result = analyze(students, "exam_score")

    assert result["task"] == "regression"
    assert result["score_metric"] == "R2"
    assert result["best_feature"] == "hours_studied"
    assert result["best_score"] > 0.7


def test_detects_classification_on_binary_target(passengers: pd.DataFrame):
    result = analyze(passengers, "survived")

    assert result["task"] == "classification"
    assert result["score_metric"] == "F1 (weighted)"
    assert result["best_feature"] == "sex"


def test_reports_no_signal_on_pure_noise(noise: pd.DataFrame):
    result = analyze(noise, "target")

    assert result["best_score"] < 0.2


def test_perfect_score_on_leaked_target(leaky: pd.DataFrame):
    result = analyze(leaky, "revenue")

    assert result["best_feature"] == "total_with_vat"
    assert result["best_score"] > 0.99


def test_raises_when_target_column_is_missing(students: pd.DataFrame):
    with pytest.raises(ValueError, match="таргет"):
        analyze(students, "no_such_column")


def test_raises_when_no_usable_features():
    df = pd.DataFrame({"target": [1, 2, 3, 4]})

    with pytest.raises(ValueError, match="признаков"):
        analyze(df, "target")


def test_every_feature_gets_a_score(students: pd.DataFrame):
    result = analyze(students, "exam_score")

    assert set(result["feature_scores"]) == set(students.columns) - {"exam_score"}


def test_chart_matches_best_feature(students: pd.DataFrame):
    result = analyze(students, "exam_score")
    chart = result["chart"]

    assert chart["type"] == "scatter"
    assert chart["x_label"] == result["best_feature"]
    assert chart["y_label"] == "exam_score"
    assert len(chart["x"]) == len(chart["y"])


def test_detect_task_uses_unique_count():
    assert detect_task(pd.Series([0, 1, 1, 0, 1])) == "classification"
    assert detect_task(pd.Series(range(100))) == "regression"


def test_prepare_converts_booleans_to_int():
    df = pd.DataFrame({"flag": [True, False, True]})

    assert prepare(df)["flag"].tolist() == [1, 0, 1]


def test_usable_features_drops_constant_columns():
    df = pd.DataFrame({"useful": [1, 2, 3], "constant": ["x", "x", "x"]})

    assert "constant" not in usable_features(df).columns


def test_prediction_returns_number_for_regression(students: pd.DataFrame):
    values = {
        "hours_studied": "8",
        "sleep_hours": "7",
        "attendance_pct": "90",
        "prev_score": "70",
    }

    result = predict_value(students, "exam_score", values)

    assert result["task"] == "regression"
    assert result["target"] == "exam_score"
    assert isinstance(result["prediction"], (int, float))
