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


@pytest.fixture(scope="session")
def students(seed_dir: Path) -> pd.DataFrame:
    return pd.read_csv(seed_dir / "students.csv")


@pytest.fixture(scope="session")
def noise(seed_dir: Path) -> pd.DataFrame:
    return pd.read_csv(seed_dir / "noise.csv")


@pytest.fixture(scope="session")
def leaky(seed_dir: Path) -> pd.DataFrame:
    return pd.read_csv(seed_dir / "leaky_sales.csv")


@pytest.fixture(scope="session")
def passengers(seed_dir: Path) -> pd.DataFrame:
    return pd.read_csv(seed_dir / "passengers.csv")


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    import app as app_module

    with TestClient(app_module.app) as test_client:
        test_client.delete("/api/history")
        yield test_client
