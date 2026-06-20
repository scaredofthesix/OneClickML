"""
app.py - веб-слой OneClickML на FastAPI.

Принимает CSV + имя таргета, прогоняет через ML-ядро (core.analyze)
и возвращает JSON. Раздаёт статический фронтенд из папки static/.
"""

import io
import json

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core import analyze, predict_value

app = FastAPI(title="OneClickML")


def _read_csv(raw: bytes) -> pd.DataFrame:
    """Читаем CSV устойчиво: пробуем разные кодировки, авто-определяем
    разделитель (запятая, точка с запятой, табуляция), снимаем BOM."""
    if not raw.strip():
        raise HTTPException(status_code=400, detail="Файл пустой")

    last_error = None
    for encoding in ("utf-8-sig", "cp1251", "latin-1"):
        try:
            # sep=None + engine="python" -> pandas сам определяет разделитель
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=encoding)
            if df.shape[1] == 0:
                raise ValueError("в файле не найдено колонок")
            # убираем мусорный авто-индекс, если он попал первой колонкой
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")
    raise HTTPException(status_code=400, detail=f"Не удалось определить кодировку файла: {last_error}")


@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...), target: str = Form(...)) -> JSONResponse:
    """Загруженный CSV + имя таргета -> результат анализа в JSON."""
    df = _read_csv(await file.read())
    try:
        result = analyze(df, target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {exc}")
    return JSONResponse(result)


@app.post("/api/predict")
async def api_predict(
    file: UploadFile = File(...),
    target: str = Form(...),
    values: str = Form(...),
) -> JSONResponse:
    """CSV + таргет + значения признаков (JSON) -> предсказание таргета."""
    df = _read_csv(await file.read())
    try:
        parsed = json.loads(values)
        result = predict_value(df, target, parsed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {exc}")
    return JSONResponse(result)


@app.get("/")
def index() -> FileResponse:
    """Главная страница."""
    return FileResponse("static/index.html")


# Статика (CSS/JS/видео). StaticFiles поддерживает range-запросы -> видео перематывается.
app.mount("/static", StaticFiles(directory="static"), name="static")
