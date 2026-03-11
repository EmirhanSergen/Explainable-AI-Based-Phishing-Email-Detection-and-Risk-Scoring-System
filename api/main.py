"""
FastAPI uygulama giriş noktası.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import router

app = FastAPI(
    title="Phishing Email Detection API",
    description="Explainable AI-Based Phishing Email Detection and Risk Scoring",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="", tags=["analysis"])

# Web arayüzü: http://localhost:8000/web/
web_dir = Path(__file__).resolve().parent.parent / "web"
if web_dir.exists():
    app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")


@app.get("/")
def root():
    """API root."""
    return {"message": "Phishing Email Detection API", "docs": "/docs", "ui": "/web/"}
