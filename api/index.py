"""
FastAPI Backend for HW2 Multiple Linear Regression Dashboard.
Endpoints:
- GET  /api/health
- POST /api/dataset/preview
- POST /api/train
Supports both application/json and multipart/form-data.
"""
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.ml_engine import (
    validate_and_fetch_url,
    parse_csv_dataframe,
    preview_dataset,
    train_regression_model,
)

app = FastAPI(
    title="Anna ML Studio API",
    description="Multiple Linear Regression & Automated Feature Selection Engine",
    version="2.0.0"
)

# Enable CORS for local dev and Vercel domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "Anna ML Studio API",
        "version": "2.0.0",
        "engine": "scikit-learn"
    }

@app.post("/api/dataset/preview")
async def api_preview_dataset(request: Request):
    try:
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" in content_type:
            form = await request.form()
            file = form.get("file")
            url = form.get("url")
            csv_text = form.get("csv_text")

            if file and hasattr(file, "read"):
                content = await file.read()
                if len(content) > 12 * 1024 * 1024:
                    raise HTTPException(status_code=400, detail="上傳檔案超過 12MB 上限")
                df = parse_csv_dataframe(content)
            elif url:
                raw_text = validate_and_fetch_url(str(url))
                df = parse_csv_dataframe(raw_text)
            elif csv_text:
                df = parse_csv_dataframe(str(csv_text))
            else:
                raise HTTPException(status_code=400, detail="請提供 HTTPS CSV 網址、CSV 內容或上傳檔案")
        else:
            body = await request.json()
            url = body.get("url")
            csv_text = body.get("csv_text")
            if url:
                raw_text = validate_and_fetch_url(url)
                df = parse_csv_dataframe(raw_text)
            elif csv_text:
                df = parse_csv_dataframe(csv_text)
            else:
                raise HTTPException(status_code=400, detail="請提供 HTTPS CSV 網址或 CSV 內容")

        preview = preview_dataset(df)
        return preview
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"預覽資料集發生錯誤: {str(e)}")

@app.post("/api/train")
async def api_train_model(request: Request):
    try:
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" in content_type:
            form = await request.form()
            file = form.get("file")
            url = form.get("url")
            csv_text = form.get("csv_text")
            target_col = form.get("target")
            omit_raw = form.get("omit", "")
            if isinstance(omit_raw, str):
                try:
                    omit_list = json.loads(omit_raw) if omit_raw.startswith("[") else [c.strip() for c in omit_raw.split(",") if c.strip()]
                except Exception:
                    omit_list = [c.strip() for c in omit_raw.split(",") if c.strip()]
            else:
                omit_list = []
            folds = int(form.get("folds", 3))
            quick_val = form.get("quick", "true")
            quick = str(quick_val).lower() in ("true", "1", "yes")

            if file and hasattr(file, "read"):
                content = await file.read()
                if len(content) > 12 * 1024 * 1024:
                    raise HTTPException(status_code=400, detail="上傳檔案超過 12MB 上限")
                df = parse_csv_dataframe(content)
            elif url:
                raw_text = validate_and_fetch_url(str(url))
                df = parse_csv_dataframe(raw_text)
            elif csv_text:
                df = parse_csv_dataframe(str(csv_text))
            else:
                raise HTTPException(status_code=400, detail="請提供 HTTPS CSV 網址、CSV 內容或上傳檔案")
        else:
            body = await request.json()
            url = body.get("url")
            csv_text = body.get("csv_text")
            target_col = body.get("target")
            omit_list = body.get("omit", [])
            folds = int(body.get("folds", 3))
            quick = bool(body.get("quick", True))

            if url:
                raw_text = validate_and_fetch_url(url)
                df = parse_csv_dataframe(raw_text)
            elif csv_text:
                df = parse_csv_dataframe(csv_text)
            else:
                raise HTTPException(status_code=400, detail="請提供 HTTPS CSV 網址或 CSV 內容")

        if not target_col:
            raise HTTPException(status_code=400, detail="必須指定 Target 目標欄位名稱")

        result = train_regression_model(
            df=df,
            target_col=str(target_col),
            omit_list=omit_list,
            folds=folds,
            quick=quick
        )
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型訓練失敗: {str(e)}")

# Mount static frontend for single-command local dev (e.g. uvicorn api.index:app)
import os
from fastapi.staticfiles import StaticFiles

_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists(os.path.join(_root_dir, "index.html")):
    app.mount("/", StaticFiles(directory=_root_dir, html=True), name="static")

