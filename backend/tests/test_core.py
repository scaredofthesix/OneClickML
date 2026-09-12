import pandas as pd
import pytest

from cleaning import clean_frame
from core import analyze, cv_folds, detect_task, predict_value, prepare, usable_features


def test_detects_classification_on_iris(iris: pd.DataFrame):
    result = analyze(iris, "Species")

    assert result["task"] == "classification"
    assert result["score_metric"] == "F1 (weighted)"
    assert result["best_feature"] == "PetalWidthCm"
    assert result["best_score"] > 0.9


def test_detects_regression_and_finds_obvious_driver(students: pd.DataFrame):
    result = analyze(students, "math score")

    assert result["task"] == "regression"
    assert result["score_metric"] == "R2"
    assert result["best_feature"] == "reading score"
    assert result["best_score"] > 0.6


def test_categorical_feature_can_win(titanic: pd.DataFrame):
    result = analyze(titanic, "Survived")

    assert result["task"] == "classification"
    assert result["best_feature"] == "Sex"


def test_raises_when_target_column_is_missing(iris: pd.DataFrame):
    with pytest.raises(ValueError, match="таргет"):
        analyze(iris, "no_such_column")


def test_raises_when_no_usable_features():
    df = pd.DataFrame({"target": [1, 2, 3, 4]})

    with pytest.raises(ValueError, match="признаков"):
        analyze(df, "target")


def test_chart_matches_best_feature(students: pd.DataFrame):
    result = analyze(students, "math score")
    chart = result["chart"]

    assert chart["type"] == "scatter"
    assert chart["x_label"] == result["best_feature"]
    assert chart["y_label"] == "math score"
    assert len(chart["x"]) == len(chart["y"])


def test_detect_task_uses_unique_count():
    assert detect_task(pd.Series([0, 1, 1, 0, 1])) == "classification"
    assert detect_task(pd.Series(range(100))) == "regression"


def test_detect_task_switches_exactly_at_the_threshold():
    assert detect_task(pd.Series(list(range(20)) * 3)) == "classification"
    assert detect_task(pd.Series(list(range(21)) * 3)) == "regression"


def test_detect_task_treats_text_target_as_classification():
    assert detect_task(pd.Series([f"class_{i}" for i in range(50)])) == "classification"


def test_prepare_converts_booleans_to_int():
    df = pd.DataFrame({"flag": [True, False, True]})

    assert prepare(df)["flag"].tolist() == [1, 0, 1]


def test_usable_features_drops_constant_columns():
    df = pd.DataFrame({"useful": [1, 2, 3], "constant": ["x", "x", "x"]})

    assert "constant" not in usable_features(df).columns


def test_prediction_returns_number_for_regression(students: pd.DataFrame):
    values = {
        "reading score": "70",
        "writing score": "70",
        "gender": "female",
        "race/ethnicity": "group B",
        "parental level of education": "some college",
        "lunch": "standard",
        "test preparation course": "none",
    }

    result = predict_value(students, "math score", values)

    assert result["task"] == "regression"
    assert result["target"] == "math score"
    assert isinstance(result["prediction"], (int, float))


# --- подготовка сырых данных с Kaggle -------------------------------------------------


def test_row_identifier_is_dropped(titanic: pd.DataFrame):
    frame, notes = clean_frame(titanic, protect="Survived")

    assert "PassengerId" not in frame.columns
    assert {"column": "PassengerId", "action": "id_column", "detail": ""} in notes


def test_index_column_from_to_csv_is_dropped(avocado: pd.DataFrame):
    frame, _ = clean_frame(avocado, protect="AveragePrice")

    assert not [col for col in frame.columns if col.lower().startswith("unnamed")]


def test_date_column_becomes_numeric_parts(avocado: pd.DataFrame):
    frame, notes = clean_frame(avocado, protect="AveragePrice")

    assert "Date" not in frame.columns
    assert "Date_year" in frame.columns
    assert "Date_month" in frame.columns
    assert pd.api.types.is_numeric_dtype(frame["Date_year"])
    assert any(n["column"] == "Date" and n["action"] == "date_split" for n in notes)


def test_text_date_written_in_words_is_parsed(netflix: pd.DataFrame):
    frame, _ = clean_frame(netflix, protect="type")

    assert "date_added_year" in frame.columns
    assert frame["date_added_year"].dropna().between(2000, 2030).all()


def test_number_stored_as_text_is_recovered(telco: pd.DataFrame):
    frame, notes = clean_frame(telco, protect="Churn")

    assert pd.api.types.is_numeric_dtype(frame["TotalCharges"])
    assert any(n["column"] == "TotalCharges" and n["action"] == "text_number" for n in notes)


def test_units_are_stripped_from_numbers(cars: pd.DataFrame):
    frame, _ = clean_frame(cars, protect="selling_price")

    assert pd.api.types.is_numeric_dtype(frame["mileage"])
    assert pd.api.types.is_numeric_dtype(frame["engine"])
    assert frame["engine"].max() > 600


def test_codes_are_not_mistaken_for_numbers(titanic: pd.DataFrame):
    frame, _ = clean_frame(titanic, protect="Survived")

    assert not pd.api.types.is_numeric_dtype(frame["Ticket"])


def test_target_column_is_never_touched(telco: pd.DataFrame):
    frame, _ = clean_frame(telco, protect="TotalCharges")

    assert not pd.api.types.is_numeric_dtype(frame["TotalCharges"])


def test_analysis_survives_missing_target_values(titanic: pd.DataFrame):
    result = analyze(titanic, "Age")

    assert result["task"] == "regression"
    gaps = [n for n in result["cleanup"] if n["action"] == "target_gaps"]
    assert gaps and gaps[0]["column"] == "Age" and int(gaps[0]["detail"]) == 177


def test_analysis_survives_rare_classes_and_gaps(netflix: pd.DataFrame):
    result = analyze(netflix, "rating")

    assert result["task"] == "classification"
    assert result["best_feature"] in result["feature_scores"]


def test_cleanup_report_is_returned_to_the_client(cars: pd.DataFrame):
    result = analyze(cars, "selling_price")

    assert any(n["column"] == "mileage" and n["action"] == "text_number" for n in result["cleanup"])
    assert result["best_feature"] == "max_power"


def test_folds_shrink_to_the_smallest_class():
    y = pd.Series(["a"] * 50 + ["b"] * 3)

    assert cv_folds(y, "classification") == 3
    assert cv_folds(y, "regression") == 5
