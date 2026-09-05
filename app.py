# Веб-слой OneClickML на FastAPI.
# Принимает CSV и имя целевой колонки, гоняет через ML-ядро из core.py,
# отдаёт результат в JSON. Отсюда же раздаётся фронтенд из папки static.

import io
import json

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core import analyze, predict_value

app = FastAPI(title="OneClickML")


def _read_csv(raw: bytes) -> pd.DataFrame:
    # Читаем CSV с запасом прочности: перебираем кодировки, разделитель
    # определяем автоматически, BOM снимаем.
    if not raw.strip():
        raise HTTPException(status_code=400, detail="Файл пустой")

    last_error = None
    for encoding in ("utf-8-sig", "cp1251", "latin-1"):
        try:
            # sep=None вместе с engine="python" заставляет pandas угадать разделитель
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=encoding)
            if df.shape[1] == 0:
                raise ValueError("в файле не найдено колонок")
            # чистим названия колонок от лишних пробелов
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except UnicodeDecodeError as exc:
            # кодировка не подошла, пробуем следующую
            last_error = exc
            continue
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")
    raise HTTPException(status_code=400, detail=f"Не удалось определить кодировку файла: {last_error}")


@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...), target: str = Form(...)) -> JSONResponse:
    # Основной эндпоинт: CSV плюс имя таргета на вход, разбор в JSON на выход.
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
    # Тот же CSV плюс значения признаков из формы, в ответе предсказанный таргет.
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
    # Главная страница, она же весь интерфейс.
    return FileResponse("static/index.html")


# Статика: CSS, JS, картинки. StaticFiles умеет range-запросы, поэтому видео перематывается.
app.mount("/static", StaticFiles(directory="static"), name="static")
