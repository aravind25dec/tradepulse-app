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

import agent_engine
import pipeline
import registry  # type: ignore
import robinhood_positions
import storage

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


def _valid_ticker(ticker: str) -> bool:
    return bool(ticker) and len(ticker) <= 10 and ticker.replace("-", "").isalnum()


@app.get("/api/predict")
async def predict(
    ticker: str = Query(..., description="Stock ticker symbol, e.g. NVDA"),
    force_refresh: bool = Query(False, description="Bypass the shared cache and re-run the full pipeline"),
):
    """
    Cached by ticker (storage.py) so the Favorites section and the Robinhood
    section share exactly one prediction per ticker — whichever renders first
    triggers the (slow, Claude-CLI-bound) pipeline; the other reads the same
    cached entry. Pass force_refresh=true (the tile's refresh icon) to bypass
    the cache and recompute; the result then updates the shared cache entry,
    so BOTH sections reflect the refresh, not just whichever tile triggered it.
    """
    ticker = ticker.upper().strip()
    if not _valid_ticker(ticker):
        return JSONResponse(status_code=400, content={"error": f"Invalid ticker symbol: '{ticker}'"})

    if not force_refresh:
        cached = storage.get_cached_prediction(ticker)
        if cached is not None:
            return {**cached, "from_cache": True}

    if registry.get_current_version() is None:
        return JSONResponse(
            status_code=503,
            content={"error": "No trained model yet — run train/train_classifier.py first."},
        )
    try:
        result = await pipeline.predict(ticker)
        stamped = storage.set_cached_prediction(ticker, result)
        return {**stamped, "from_cache": False}
    except Exception as exc:
        log.error("Prediction failed for %s: %s", ticker, exc, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ── Favorites ──────────────────────────────────────────────────────────────
# Deliberately decoupled from /api/predict — the frontend's search action calls
# POST /api/favorites/{ticker} itself rather than /api/predict auto-adding, so
# that tickers viewed only via the Robinhood section (not searched) don't get
# silently added to Favorites too.

@app.get("/api/favorites")
def get_favorites():
    return {"favorites": storage.get_favorites()}


@app.post("/api/favorites/{ticker}")
def add_favorite(ticker: str):
    ticker = ticker.upper().strip()
    if not _valid_ticker(ticker):
        return JSONResponse(status_code=400, content={"error": f"Invalid ticker symbol: '{ticker}'"})
    return {"favorites": storage.add_favorite(ticker)}


@app.delete("/api/favorites/{ticker}")
def remove_favorite(ticker: str):
    return {"favorites": storage.remove_favorite(ticker.upper().strip())}


# ── Robinhood holdings ───────────────────────────────────────────────────────

@app.get("/api/robinhood/status")
def robinhood_status():
    cli_ok = agent_engine.claude_available()
    mcp_ok = agent_engine.check_robinhood_mcp() if cli_ok else False
    return {"claude_available": cli_ok, "robinhood_mcp": mcp_ok}


@app.get("/api/robinhood/positions")
async def robinhood_positions_endpoint():
    if not agent_engine.claude_available():
        return JSONResponse(status_code=503, content={"error": "Claude CLI not found."})
    if not agent_engine.check_robinhood_mcp():
        return JSONResponse(
            status_code=503,
            content={"error": "Robinhood MCP not configured. Run: claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading"},
        )
    try:
        positions = await robinhood_positions.fetch_distinct_tickers()
        return {"positions": positions}
    except Exception as exc:
        log.error("Robinhood positions fetch failed: %s", exc, exc_info=True)
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
