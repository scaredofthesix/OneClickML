import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
SEED_DIR = BACKEND_DIR.parent / "database" / "seed"

sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("SEED_DIR", str(SEED_DIR))
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_oneclickml.db")

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def seed_dir() -> Path:
    return SEED_DIR


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(SEED_DIR / name)


@pytest.fixture(scope="session")
def iris() -> pd.DataFrame:
    return _load("iris.csv")


@pytest.fixture(scope="session")
def students() -> pd.DataFrame:
    return _load("students_performance.csv")


@pytest.fixture(scope="session")
def titanic() -> pd.DataFrame:
    return _load("titanic.csv")


@pytest.fixture(scope="session")
def telco() -> pd.DataFrame:
    return _load("telco_churn.csv")


@pytest.fixture(scope="session")
def cars() -> pd.DataFrame:
    return _load("car_details.csv")


@pytest.fixture(scope="session")
def avocado() -> pd.DataFrame:
    return _load("avocado.csv")


@pytest.fixture(scope="session")
def netflix() -> pd.DataFrame:
    return _load("netflix_titles.csv")


@pytest.fixture(scope="session")
def wine() -> pd.DataFrame:
    return _load("wine_quality.csv")


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    import app as app_module

    with TestClient(app_module.app) as test_client:
        test_client.delete("/api/history")
        yield test_client
