"""
core.py - ML-ядро сервиса OneClickML.

Тут только функции, никакого кода на верхнем уровне (кроме самотеста).
Веб-слой (app.py) импортирует analyze() и вызывает её по запросу.

Точка входа: analyze(df, target) -> dict, готовый к отдаче в JSON.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Числовой таргет с бОльшим числом уникальных значений считаем регрессией.
REGRESSION_UNIQUE_THRESHOLD = 20
CV_FOLDS = 5


def detect_task(y: pd.Series) -> str:
    """Регрессия, если таргет числовой и значений много (непрерывная величина).
    Иначе классификация (текстовые метки или мало уникальных значений)."""
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > REGRESSION_UNIQUE_THRESHOLD:
        return "regression"
    return "classification"


def split_features(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Делим колонки на числовые и категориальные (их обрабатывают по-разному)."""
    numerical = x.select_dtypes(include="number").columns.tolist()
    categorical = [col for col in x.columns if col not in numerical]
    return numerical, categorical


def build_preprocessor(numerical: list[str], categorical: list[str]) -> ColumnTransformer:
    """Препроцессор внутри Pipeline: imputer/scaler учатся только на train,
    значит нет утечки данных. Числа -> медиана + масштаб, строки -> мода + one-hot."""
    numerical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numerical_pipe, numerical),
        ("cat", categorical_pipe, categorical),
    ])


def find_best_feature(
    x: pd.DataFrame,
    y: pd.Series,
    task: str,
    numerical: list[str],
    categorical: list[str],
) -> tuple[str, dict[str, float]]:
    """Для каждого признака по отдельности обучаем простую модель и меряем
    качество кросс-валидацией. Победитель = самый предсказательный признак."""
    if task == "regression":
        model = LinearRegression()
        scoring = "r2"
    else:
        model = LogisticRegression(max_iter=1000)
        scoring = "f1_weighted"

    scores: dict[str, float] = {}
    for feature in x.columns:
        if feature in numerical:
            prep = build_preprocessor([feature], [])
        else:
            prep = build_preprocessor([], [feature])
        pipe = Pipeline([("prep", prep), ("model", model)])
        scores[feature] = float(cross_val_score(pipe, x, y, cv=CV_FOLDS, scoring=scoring).mean())

    best = max(scores, key=scores.get)
    return best, scores


def _chart_data(df: pd.DataFrame, feature: str, target: str, task: str) -> dict:
    """Данные для графика связи лучшего признака с таргетом."""
    sub = df[[feature, target]].dropna()
    if pd.api.types.is_numeric_dtype(sub[feature]):
        # Числовой признак -> точечный график (scatter).
        return {
            "type": "scatter",
            "x_label": feature,
            "y_label": target,
            "x": sub[feature].tolist(),
            "y": sub[target].tolist(),
        }
    # Категориальный признак -> столбики (среднее таргета / счётчик по категориям).
    grouped = sub.groupby(feature)[target]
    agg = grouped.mean() if task == "regression" else grouped.count()
    return {
        "type": "bar",
        "x_label": feature,
        "y_label": target if task == "regression" else "count",
        "labels": agg.index.astype(str).tolist(),
        "values": agg.values.tolist(),
    }


def analyze(df: pd.DataFrame, target: str) -> dict:
    """Главная функция: таблица + имя таргета -> результат (готов к JSON)."""
    if target not in df.columns:
        raise ValueError(f"Колонка-таргет {target!r} не найдена в таблице")

    x = df.drop(columns=[target])
    y = df[target]
    if x.shape[1] == 0:
        raise ValueError("В таблице нет признаков, кроме таргета")

    task = detect_task(y)
    numerical, categorical = split_features(x)
    best_feature, feature_scores = find_best_feature(x, y, task, numerical, categorical)

    return {
        "task": task,
        "target": target,
        "best_feature": best_feature,
        "best_score": round(feature_scores[best_feature], 4),
        "score_metric": "R2" if task == "regression" else "F1 (weighted)",
        "feature_scores": dict(sorted(feature_scores.items(), key=lambda kv: kv[1], reverse=True)),
        "chart": _chart_data(df, best_feature, target, task),
    }


if __name__ == "__main__":
    import json
    import os

    sample = "sample_data.csv"
    if os.path.exists(sample):
        demo = pd.read_csv(sample)
        target_col = demo.columns[-1]
        print(json.dumps(analyze(demo, target_col), ensure_ascii=False, indent=2)[:600])
    else:
        print(f"Нет {sample} для самотеста")
