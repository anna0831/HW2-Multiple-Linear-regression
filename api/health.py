"""
Vercel Serverless Function: Health Check Endpoint
Route: /api/health
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Health API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
@app.get("/health")
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "Anna ML Studio API",
        "version": "2.0.0",
        "engine": "scikit-learn"
    }
