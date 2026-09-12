from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cleaning import clean_frame, note

REGRESSION_UNIQUE_THRESHOLD = 20
CV_FOLDS = 5
MAX_CATEGORY_UNIQUE = 50
# Класс, который встречается реже, кросс-валидация всё равно не разложит по фолдам
MIN_CLASS_ROWS = 2


def prepare(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    frame, _ = prepare_with_notes(df, target)
    return frame


def prepare_with_notes(df: pd.DataFrame, target: str | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Чистит сырую таблицу и рассказывает, что именно пришлось поменять."""
    frame, notes = clean_frame(df, protect=target)
    for col in frame.columns:
        if frame[col].dtype == bool:
            frame[col] = frame[col].astype(int)
    return frame, notes


def cv_folds(y: pd.Series, task: str) -> int:
    """Фолдов не может быть больше, чем объектов в самом редком классе."""
    if task == "regression":
        return CV_FOLDS
    return max(2, min(CV_FOLDS, int(y.value_counts().min())))


def drop_rare_classes(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    """Убирает классы-одиночки: на них обучение падает, а пользы от них нет."""
    counts = df[target].value_counts()
    rare = counts[counts < MIN_CLASS_ROWS].index.tolist()
    if not rare:
        return df, []
    return df[~df[target].isin(rare)], rare


def _dataset(df: pd.DataFrame, target: str) -> dict:
    """Единая подготовка данных для анализа, обучения и предсказания."""
    if target not in df.columns:
        raise ValueError(f"Колонка-таргет {target!r} не найдена в таблице")

    frame, notes = prepare_with_notes(df, target)

    empty = int(frame[target].isna().sum())
    if empty:
        frame = frame[frame[target].notna()]
        notes.append(note(target, "target_gaps", str(empty)))
    if frame.empty:
        raise ValueError(f"В колонке {target!r} не осталось ни одного значения")

    if frame[target].dtype == object:
        # sklearn не умеет сравнивать строки с пропусками-числами в одном столбце
        frame[target] = frame[target].astype(str)

    x = usable_features(frame.drop(columns=[target]))
    if x.shape[1] == 0:
        raise ValueError("В таблице нет пригодных признаков для анализа")

    task = detect_task(frame[target])
    if task == "classification":
        frame, rare = drop_rare_classes(frame, target)
        if rare:
            notes.append(note(target, "rare_classes", ", ".join(map(str, rare[:5]))))
        if frame[target].nunique() < 2:
            raise ValueError(f"В колонке {target!r} остался один класс, предсказывать нечего")
        x = x.loc[frame.index]

    numerical, categorical = split_features(x)
    return {
        "frame": frame,
        "x": x,
        "y": frame[target],
        "task": task,
        "numerical": numerical,
        "categorical": categorical,
        "folds": cv_folds(frame[target], task),
        "notes": notes,
    }


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
    folds: int = CV_FOLDS,
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
        scores[feature] = float(cross_val_score(pipe, x, y, cv=folds, scoring=scoring).mean())

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
    data = _dataset(df, target)
    frame, x, y, task = data["frame"], data["x"], data["y"], data["task"]
    best_feature, feature_scores = find_best_feature(
        x, y, task, data["numerical"], data["categorical"], data["folds"]
    )

    return {
        "task": task,
        "target": target,
        "best_feature": best_feature,
        "best_score": round(feature_scores[best_feature], 4),
        "score_metric": "R2" if task == "regression" else "F1 (weighted)",
        "feature_scores": dict(sorted(feature_scores.items(), key=lambda kv: kv[1], reverse=True)),
        "chart": _chart_data(frame, best_feature, target, task),
        "features": feature_spec(x),
        "cleanup": data["notes"],
    }


def feature_spec(x: pd.DataFrame) -> list[dict]:
    spec = []
    for col in x.columns:
        if pd.api.types.is_numeric_dtype(x[col]):
            spec.append({"name": col, "type": "number"})
        else:
            options = sorted(x[col].dropna().astype(str).unique().tolist(), key=str)
            spec.append({"name": col, "type": "category", "options": options})
    return spec


def train_model(df: pd.DataFrame, target: str):
    data = _dataset(df, target)
    prep = build_preprocessor(data["numerical"], data["categorical"])
    task = data["task"]
    model = LinearRegression() if task == "regression" else LogisticRegression(max_iter=1000)
    pipe = Pipeline([("prep", prep), ("model", model)])
    pipe.fit(data["x"], data["y"])
    return pipe, task, data["x"].columns.tolist(), data["x"]


def predict_value(df: pd.DataFrame, target: str, values: dict) -> dict:
    pipe, task, columns, x = train_model(df, target)
    row = {}
    for col in columns:
        raw = values.get(col)
        if pd.api.types.is_numeric_dtype(x[col]):
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
