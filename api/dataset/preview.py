"""
Vercel Serverless Function: Dataset Preview Endpoint
Route: /api/dataset/preview
"""
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from api.ml_engine import (
    validate_and_fetch_url,
    parse_csv_dataframe,
    preview_dataset,
)

app = FastAPI(title="Preview Dataset API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/dataset/preview")
@app.post("/dataset/preview")
@app.post("/api/preview")
@app.post("/preview")
@app.post("/")
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
