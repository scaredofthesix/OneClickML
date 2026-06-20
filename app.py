"""
app.py - веб-слой OneClickML на FastAPI.

Принимает CSV + имя таргета, прогоняет через ML-ядро (core.analyze)
и возвращает JSON. Раздаёт статический фронтенд из папки static/.
"""

import io

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core import analyze

app = FastAPI(title="OneClickML")


@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...), target: str = Form(...)) -> JSONResponse:
    """Загруженный CSV + имя таргета -> результат анализа в JSON."""
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    try:
        result = analyze(df, target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {exc}")

    return JSONResponse(result)


@app.get("/")
def index() -> FileResponse:
    """Главная страница."""
    return FileResponse("static/index.html")


# Статика (CSS/JS/видео). StaticFiles поддерживает range-запросы -> видео перематывается.
app.mount("/static", StaticFiles(directory="static"), name="static")
