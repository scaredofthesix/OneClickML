import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

df = pd.read_csv("50_Startups_dataset.csv")
target = "Profit"
X = df.drop(target, axis=1)
y = df[target]

def detect_task(y):
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        return "regression"
    return "classification"

print(detect_task(df["State"]))

def divide(x):
    numerical = x.select_dtypes(include="number").columns
    categorical = x.select_dtypes(include="object").columns
    return numerical.tolist(), categorical.tolist()

numerical, categorical = divide(X)

def preprocessing(numerical, categorical):
    numerical_pipe = Pipeline(steps = [("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])
    categorical_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", numerical_pipe, numerical),("cat", categorical_pipe, categorical)])
    return preprocessor

preprocessor = preprocessing(numerical, categorical)
task = detect_task(y)

def get_models(task):
    models_reg = {"lasso": {"model": Lasso(),
                        "params": {"model__alpha": np.logspace(-3, 1, 50)}},
                  "tree": {"model": DecisionTreeRegressor(random_state=42),
                       "params": {"model__max_depth": [3, 5, 10, 15, None],
                                  "model__min_samples_leaf": [1, 5, 10, 20, 30]}}}
    models_clas = {"logreg": {"model": LogisticRegression(max_iter=1000),
                        "params": {"model__C": np.logspace(-3, 2, 50)}},
                  "tree": {"model": DecisionTreeClassifier(random_state=42),
                       "params": {"model__max_depth": [3, 5, 10, 15, None],
                                  "model__min_samples_leaf": [1, 5, 10, 20, 30]}}}
    if task == "regression":
        return models_reg
    if task == "classification":
        return models_clas
    raise ValueError(f"Invalid: {task!r}")
models = get_models(task)

def train(X, y, models,preprocessor, task):
    if task == "regression":
        scoring = "r2"
    else:
        scoring = "f1_weighted"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=67)
    results = {}
    for name, model_info in models.items():
        pipe = Pipeline(steps = [("prep", preprocessor), ("model", model_info["model"])])
        search = RandomizedSearchCV(pipe, model_info["params"], n_iter=20, cv=5, random_state=42,scoring = scoring)
        search.fit(X_train, y_train)
        results[name] = search
    best_score = -np.inf
    best_model = None
    best_name = None
    for name, search in results.items():
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = search
            best_name = name
    test_score = best_model.score(X_test, y_test)
    return best_name, best_model.best_estimator_, test_score

def find_best_feature(X, y, task, numerical , categorical):
    results = {}
    if task == "regression":
        model = LinearRegression()
        scoring = "r2"
    else:
        model = LogisticRegression(max_iter=1000)
        scoring = "f1_weighted"
    for feature in X.columns:
        if feature in numerical:
            prep = preprocessing([feature], [])
        else:
            prep = preprocessing([],[feature])
        steps = [("prep",prep),("model",model)]
        pipe = Pipeline(steps)
        results[feature] = cross_val_score(pipe,X, y, cv = 5, scoring = scoring).mean()
    return max(results, key=results.get), results


def predict_new(model, new_data, X):
    new_df = pd.DataFrame([new_data], columns=X.columns)
    prediction = model.predict(new_df)
    return prediction[0]

