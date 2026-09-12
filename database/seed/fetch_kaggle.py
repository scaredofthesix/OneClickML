"""Скачивает тестовые датасеты с Kaggle и раскладывает их в этой папке.

Все файлы каталога взяты с Kaggle и лежат в репозитории в том виде, в каком
их отдаёт Kaggle: колонки, пропуски и форматы не правились руками — иначе
пропал бы весь смысл проверки (сервис должен сам разбираться с датами,
единицами измерения и мусорными столбцами).

Единственное изменение — размер: файлы крупнее MAX_ROWS строк прорежены
случайной выборкой с фиксированным random_state, чтобы репозиторий и ответы
API не распухали. Порядок строк при этом сохраняется.

Запуск (интернет нужен, ключ Kaggle — нет, датасеты публичные):

    python database/seed/fetch_kaggle.py
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

SEED_DIR = Path(__file__).resolve().parent
DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/{ref}"
MAX_ROWS = 1500
RANDOM_STATE = 42

# ref -- владелец/датасет на kaggle.com, member -- файл внутри архива
SOURCES = [
    {"slug": "iris", "ref": "uciml/iris", "member": "Iris.csv", "out": "iris.csv"},
    {
        "slug": "students",
        "ref": "spscientist/students-performance-in-exams",
        "member": "StudentsPerformance.csv",
        "out": "students_performance.csv",
    },
    {
        "slug": "titanic",
        "ref": "yasserh/titanic-dataset",
        "member": "Titanic-Dataset.csv",
        "out": "titanic.csv",
    },
    {"slug": "insurance", "ref": "mirichoi0218/insurance", "member": "insurance.csv", "out": "insurance.csv"},
    {
        "slug": "telco",
        "ref": "blastchar/telco-customer-churn",
        "member": "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "out": "telco_churn.csv",
    },
    {
        "slug": "housing",
        "ref": "camnugent/california-housing-prices",
        "member": "housing.csv",
        "out": "california_housing.csv",
    },
    {
        "slug": "wine",
        "ref": "uciml/red-wine-quality-cortez-et-al-2009",
        "member": "winequality-red.csv",
        "out": "wine_quality.csv",
    },
    {"slug": "avocado", "ref": "neuromusic/avocado-prices", "member": "avocado.csv", "out": "avocado.csv"},
    {
        "slug": "netflix",
        "ref": "shivamb/netflix-shows",
        "member": "netflix_titles.csv",
        "out": "netflix_titles.csv",
        # в описаниях фильмов много текста, поэтому строк берём меньше
        "rows": 700,
    },
    {
        "slug": "cars",
        "ref": "nehalbirla/vehicle-dataset-from-cardekho",
        "member": "Car details v3.csv",
        "out": "car_details.csv",
    },
]


def download(ref: str) -> bytes:
    request = urllib.request.Request(DOWNLOAD_URL.format(ref=ref), headers={"User-Agent": "oneclickml"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - адрес фиксированный
        return response.read()


def extract(archive: bytes, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        with zf.open(member) as raw:
            return pd.read_csv(raw)


def shrink(df: pd.DataFrame, limit: int = MAX_ROWS) -> pd.DataFrame:
    if len(df) <= limit:
        return df
    return df.sample(limit, random_state=RANDOM_STATE).sort_index()


def main() -> int:
    for source in SOURCES:
        try:
            df = shrink(extract(download(source["ref"]), source["member"]), source.get("rows", MAX_ROWS))
        except Exception as exc:  # noqa: BLE001 - скрипт запускают руками, нужен понятный вывод
            print(f"[fail] {source['ref']}: {exc}")
            return 1
        target_path = SEED_DIR / source["out"]
        df.to_csv(target_path, index=False)
        print(f"[ok] {source['ref']:55} -> {source['out']:24} {df.shape[0]}x{df.shape[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
