from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REGRESSION_UNIQUE_THRESHOLD = 20
CV_FOLDS = 5
MAX_CATEGORY_UNIQUE = 50


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    return df


def usable_features(x: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for col in x.columns:
        nunique = x[col].nunique(dropna=True)
        if nunique <= 1:
            continue
        if not pd.api.types.is_numeric_dtype(x[col]) and nunique > MAX_CATEGORY_UNIQUE:
            continue
        keep.append(col)
    return x[keep]


def detect_task(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > REGRESSION_UNIQUE_THRESHOLD:
        return "regression"
    return "classification"


def split_features(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    numerical = x.select_dtypes(include="number").columns.tolist()
    categorical = [col for col in x.columns if col not in numerical]
    return numerical, categorical


def build_preprocessor(numerical: list[str], categorical: list[str]) -> ColumnTransformer:
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
    sub = df[[feature, target]].dropna()
    if pd.api.types.is_numeric_dtype(sub[feature]):
        return {
            "type": "scatter",
            "x_label": feature,
            "y_label": target,
            "x": sub[feature].tolist(),
            "y": sub[target].tolist(),
        }
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
    spec = []
    for col in x.columns:
        if pd.api.types.is_numeric_dtype(x[col]):
            spec.append({"name": col, "type": "number"})
        else:
            options = sorted(x[col].dropna().astype(str).unique().tolist())
            spec.append({"name": col, "type": "category", "options": options})
    return spec


def train_model(df: pd.DataFrame, target: str):
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
    df = prepare(df)
    pipe, task, columns = train_model(df, target)
    row = {}
    for col in columns:
        raw = values.get(col)
        if pd.api.types.is_numeric_dtype(df[col]):
            row[col] = float(raw) if raw not in (None, "") else None
        else:
            row[col] = raw
    pred = pipe.predict(pd.DataFrame([row], columns=columns))[0]
    prediction = round(float(pred), 4) if task == "regression" else str(pred)
    return {"task": task, "target": target, "prediction": prediction}


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
