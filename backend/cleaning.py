"""Приведение сырых таблиц к виду, с которым умеет работать пайплайн.

Датасеты с Kaggle редко бывают в том виде, который ждёт sklearn: даты лежат
строками, числа приезжают с единицами измерения ("23.4 kmpl", "1248 CC"),
пропуски записаны пробелом, а первая колонка часто оказывается порядковым
номером строки. Без этой подготовки такие колонки либо молча выбрасывались
как "категория с 5000 значений", либо роняли обучение.

Модуль ничего не знает про модели: на вход таблица, на выходе таблица и
список того, что было сделано. Заметки отдаются структурой ``{column, action,
detail}``, а не готовой фразой: интерфейс двуязычный, текст собирает фронтенд.
"""

from __future__ import annotations

import re
import warnings

import pandas as pd

# Какую долю непустых значений нужно распознать, чтобы поменять тип всей колонки
RECOGNITION_RATIO = 0.8
# Колонка считается идентификатором, если почти в каждой строке своё значение
ID_UNIQUE_RATIO = 0.95
# Столбец без имени или "Unnamed: 0" -- это индекс, дописанный pandas при to_csv
INDEX_NAME_PATTERN = re.compile(r"^(unnamed.*|№)?$", re.IGNORECASE)
# Первое число в строке: "23.4 kmpl" -> 23.4, "190Nm@ 2000rpm" -> 190
NUMBER_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)")
# Разделитель тысяч внутри числа: 1,234.56 -> 1234.56
THOUSANDS_PATTERN = re.compile(r"(?<=\d),(?=\d{3}(\D|$))")
# Текстовые заглушки, которыми в выгрузках обозначают пропуск
MISSING_TOKENS = {"", "-", "--", "?", "n/a", "na", "nan", "none", "null", "unknown"}
# Дата должна выглядеть как дата, а не как одно число: нужен разделитель или буквы
DATE_HINT_PATTERN = re.compile(r"[-/. ]|[a-zA-Zа-яА-Я]")


def _as_text(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    return text.mask(text.str.lower().isin(MISSING_TOKENS))


def _is_index_column(name: str) -> bool:
    return bool(INDEX_NAME_PATTERN.match(name.strip()))


def _looks_like_id(series: pd.Series, name: str) -> bool:
    """Ключ строки: имя заканчивается на id и значение своё почти в каждой строке."""
    plain = re.sub(r"[^a-zа-я]", "", name.lower())
    if not plain.endswith("id"):
        return False
    rows = len(series)
    return rows > 0 and series.nunique(dropna=True) / rows >= ID_UNIQUE_RATIO


def _numbers_from_text(series: pd.Series) -> pd.Series | None:
    """Числа, спрятанные в тексте: единицы измерения, валюта, проценты, пробелы."""
    text = _as_text(series)
    filled = text.notna().sum()
    if filled == 0:
        return None

    stripped = text.str.replace(THOUSANDS_PATTERN, "", regex=True)
    numbers = pd.to_numeric(stripped.str.extract(NUMBER_PATTERN.pattern, expand=False), errors="coerce")
    if numbers.notna().sum() / filled < RECOGNITION_RATIO:
        return None
    if numbers.nunique(dropna=True) <= 1:
        return None
    if not _has_common_shape(stripped):
        return None
    return numbers


def _has_common_shape(text: pd.Series) -> bool:
    """Отделяет число с единицей измерения от кода вроде номера билета.

    У "23.4 kmpl" и "1248 CC" после вычёркивания цифр остаётся одна и та же
    подпись, у "A/5 21171" и "STON/O2. 3101282" -- каждый раз своя.
    """
    shapes = text.dropna().str.replace(NUMBER_PATTERN.pattern, "", regex=True).str.strip()
    if shapes.empty:
        return False
    return shapes.value_counts(normalize=True).iloc[0] >= RECOGNITION_RATIO


def _dates_from_text(series: pd.Series) -> pd.Series | None:
    """Даты в любом читаемом формате: 2015-12-27, 25/09/2021, September 25, 2021."""
    text = _as_text(series)
    filled = text.notna().sum()
    if filled == 0:
        return None
    if not text.dropna().str.contains(DATE_HINT_PATTERN).all():
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=False)
        except (ValueError, TypeError):
            return None

    if parsed.notna().sum() / filled < RECOGNITION_RATIO:
        return None
    return parsed


def _date_parts(parsed: pd.Series, name: str) -> dict[str, pd.Series]:
    """Из даты получаются три обычных числовых признака."""
    parts = {
        f"{name}_year": parsed.dt.year,
        f"{name}_month": parsed.dt.month,
        f"{name}_dayofweek": parsed.dt.dayofweek,
    }
    return {key: values for key, values in parts.items() if values.nunique(dropna=True) > 1}


def note(column: str, action: str, detail: str = "") -> dict[str, str]:
    return {"column": column, "action": action, "detail": detail}


def clean_frame(df: pd.DataFrame, protect: str | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Готовит сырую таблицу к анализу и возвращает её вместе со списком правок.

    Колонка ``protect`` (таргет) остаётся нетронутой: её тип пользователь выбрал сам.
    """
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    notes: list[dict] = []

    for col in list(df.columns):
        if col == protect:
            continue

        if df[col].isna().all():
            df = df.drop(columns=[col])
            notes.append(note(col, "empty_column"))
            continue

        if _is_index_column(col):
            df = df.drop(columns=[col])
            notes.append(note(col, "index_column"))
            continue

        if _looks_like_id(df[col], col):
            df = df.drop(columns=[col])
            notes.append(note(col, "id_column"))
            continue

        if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype == bool:
            continue

        parsed = _dates_from_text(df[col])
        if parsed is not None:
            parts = _date_parts(parsed, col)
            df = df.drop(columns=[col])
            for name, values in parts.items():
                df[name] = values
            if parts:
                notes.append(note(col, "date_split", ", ".join(parts)))
            else:
                notes.append(note(col, "empty_column"))
            continue

        numbers = _numbers_from_text(df[col])
        if numbers is not None:
            df[col] = numbers
            notes.append(note(col, "text_number"))

    return df, notes
