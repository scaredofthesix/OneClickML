import io
import json
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import db
from core import analyze, predict_value

STATIC_DIR = Path(os.getenv("STATIC_DIR", "../frontend/static"))

app = FastAPI(title="OneClickML")


@app.on_event("startup")
def on_startup() -> None:
    db.init_db()


def _read_csv(raw: bytes) -> pd.DataFrame:
    if not raw.strip():
        raise HTTPException(status_code=400, detail="Файл пустой")

    last_error = None
    for encoding in ("utf-8-sig", "cp1251", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=encoding)
            if df.shape[1] == 0:
                raise ValueError("в файле не найдено колонок")
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")
    raise HTTPException(status_code=400, detail=f"Не удалось определить кодировку файла: {last_error}")


@app.get("/api/health")
def health() -> dict:
    try:
        return {"status": "ok", "database": "ok", "datasets": db.count_datasets()}
    except Exception as exc:
        return {"status": "ok", "database": f"error: {exc}", "datasets": 0}


@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...), target: str = Form(...)) -> JSONResponse:
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
    df = _read_csv(await file.read())
    try:
        parsed = json.loads(values)
        result = predict_value(df, target, parsed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {exc}")
    return JSONResponse(result)


@app.get("/api/datasets")
def api_datasets() -> JSONResponse:
    return JSONResponse({"datasets": db.list_datasets()})


@app.get("/api/datasets/{slug}")
def api_dataset(slug: str) -> JSONResponse:
    data = db.get_dataset(slug)
    if data is None:
        raise HTTPException(status_code=404, detail="Датасет не найден")
    return JSONResponse(data)


if STATIC_DIR.is_dir():

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
