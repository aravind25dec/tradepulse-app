"""
signalforge/main.py
FastAPI server for SignalForge — locally-trained stock predictor fused with
live Claude Code CLI context. See ../trade-app.md for the original spec this
grew out of, and README.md in this folder for the full design.

Run: uvicorn main:app --reload --port 8011
"""
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

sys.path.insert(0, str(Path(__file__).parent / "train"))

import pipeline
import registry  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="SignalForge", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def ui():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "current_model_version": registry.get_current_version()}


@app.get("/api/predict")
async def predict(ticker: str = Query(..., description="Stock ticker symbol, e.g. NVDA")):
    ticker = ticker.upper().strip()
    if not ticker or len(ticker) > 10 or not ticker.replace("-", "").isalnum():
        return JSONResponse(status_code=400, content={"error": f"Invalid ticker symbol: '{ticker}'"})
    if registry.get_current_version() is None:
        return JSONResponse(
            status_code=503,
            content={"error": "No trained model yet — run train/train_classifier.py first."},
        )
    try:
        result = await pipeline.predict(ticker)
        return result
    except Exception as exc:
        log.error("Prediction failed for %s: %s", ticker, exc, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/api/models")
def list_models():
    """Every trained model version with its real backtested metrics — the
    transparency endpoint behind the 'no hardcoded accuracy claims' design goal."""
    return {
        "current_version": registry.get_current_version(),
        "versions": registry.list_versions(),
    }


@app.post("/api/models/{version}/promote")
def promote_model(version: str):
    """Roll back/forward to a specific already-trained version without retraining."""
    try:
        registry.set_current(version)
        return {"current_version": registry.get_current_version()}
    except ValueError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8011)
