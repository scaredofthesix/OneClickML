# ML-ядро сервиса. Тут только функции, ничего не выполняется при импорте
# (кроме блока самопроверки внизу). Веб-слой app.py импортирует отсюда
# analyze и predict_value.
# Главная точка входа: analyze(df, target) возвращает готовый для JSON словарь.

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
# Категориальный признак с бОльшим числом уникальных значений считаем мусором
# (id, имена, время): в one-hot он раздулся бы на тысячи колонок.
MAX_CATEGORY_UNIQUE = 50


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # Приводим типы к тем, что понимает sklearn: bool превращаем в 0 и 1.
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    return df


def usable_features(x: pd.DataFrame) -> pd.DataFrame:
    # Отсеиваем непригодные колонки: константы и категориальные
    # идентификаторы с огромным числом значений.
    keep = []
    for col in x.columns:
        nunique = x[col].nunique(dropna=True)
        if nunique <= 1:
            continue  # константа, предсказывать по ней нечего
        if not pd.api.types.is_numeric_dtype(x[col]) and nunique > MAX_CATEGORY_UNIQUE:
            continue  # id, текст или дата, раздуло бы one-hot
        keep.append(col)
    return x[keep]


def detect_task(y: pd.Series) -> str:
    # Регрессия, если таргет числовой и значений много, то есть величина непрерывная.
    # Всё остальное (текстовые метки, мало уникальных значений) считаем классификацией.
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > REGRESSION_UNIQUE_THRESHOLD:
        return "regression"
    return "classification"


def split_features(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    # Разделяем колонки на числовые и категориальные, обрабатываются они по-разному.
    numerical = x.select_dtypes(include="number").columns.tolist()
    categorical = [col for col in x.columns if col not in numerical]
    return numerical, categorical


def build_preprocessor(numerical: list[str], categorical: list[str]) -> ColumnTransformer:
    # Препроцессор живёт внутри Pipeline, поэтому imputer и scaler учатся
    # только на train-части и утечки данных нет.
    # Числа: заполняем медианой и масштабируем. Строки: заполняем модой и делаем one-hot.
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
    # Каждый признак обучаем по отдельности и меряем качество кросс-валидацией.
    # Победитель и есть самый предсказательный признак.
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
    # Готовим данные для графика связи лучшего признака с таргетом.
    sub = df[[feature, target]].dropna()
    if pd.api.types.is_numeric_dtype(sub[feature]):
        # Числовой признак рисуем точками.
        return {
            "type": "scatter",
            "x_label": feature,
            "y_label": target,
            "x": sub[feature].tolist(),
            "y": sub[target].tolist(),
        }
    # Категориальный признак рисуем столбиками: для регрессии берём среднее
    # таргета по категории, для классификации количество записей.
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
    # Главная функция: таблица и имя таргета на вход, весь результат разбора на выход.
    if target not in df.columns:
        raise ValueError(f"Колонка-таргет {target!r} не найдена в таблице")

    df = prepare(df)
    y = df[target]
    x = usable_features(df.drop(columns=[target]))
    if x.shape[1] == 0:
        raise ValueError("В таблице нет пригодных признаков для анализа")

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
        "features": feature_spec(x),
    }


def feature_spec(x: pd.DataFrame) -> list[dict]:
    # Описание признаков для формы предсказания на фронте: числовой станет
    # полем ввода, категориальный выпадающим списком значений.
    spec = []
    for col in x.columns:
        if pd.api.types.is_numeric_dtype(x[col]):
            spec.append({"name": col, "type": "number"})
        else:
            options = sorted(x[col].dropna().astype(str).unique().tolist())
            spec.append({"name": col, "type": "category", "options": options})
    return spec


def train_model(df: pd.DataFrame, target: str):
    # Обучаем модель уже на всех признаках сразу, это нужно для предсказания новых значений.
    # Возвращаем обученный пайплайн, тип задачи и список колонок-признаков.
    df = prepare(df)
    y = df[target]
    x = usable_features(df.drop(columns=[target]))
    task = detect_task(y)
    numerical, categorical = split_features(x)
    prep = build_preprocessor(numerical, categorical)
    model = LinearRegression() if task == "regression" else LogisticRegression(max_iter=1000)
    pipe = Pipeline([("prep", prep), ("model", model)])
    pipe.fit(x, y)
    return pipe, task, x.columns.tolist()


def predict_value(df: pd.DataFrame, target: str, values: dict) -> dict:
    # Обучаем модель и предсказываем таргет по значениям, введённым в форме.
    df = prepare(df)
    pipe, task, columns = train_model(df, target)
    row = {}
    for col in columns:
        # собираем одну строку в том же порядке колонок, что был при обучении
        raw = values.get(col)
        if pd.api.types.is_numeric_dtype(df[col]):
            row[col] = float(raw) if raw not in (None, "") else None
        else:
            row[col] = raw
    pred = pipe.predict(pd.DataFrame([row], columns=columns))[0]
    prediction = round(float(pred), 4) if task == "regression" else str(pred)
    return {"task": task, "target": target, "prediction": prediction}


if __name__ == "__main__":
    # Самопроверка: запустить файл напрямую и посмотреть разбор на демо-датасете.
    import json
    import os

    sample = "sample_data.csv"
    if os.path.exists(sample):
        demo = pd.read_csv(sample)
        target_col = demo.columns[-1]
        print(json.dumps(analyze(demo, target_col), ensure_ascii=False, indent=2)[:600])
    else:
        print(f"Нет {sample} для самотеста")
