"""
autotrader/main.py
FastAPI server for the AutoTrader screening + Alpaca execution agent.
Run: uvicorn main:app --reload --port 8010
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Body, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# Load env from sibling backend/.env (APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, ...)
load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

import engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(title="TradePulse AutoTrader", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory cache ───────────────────────────────────────────────────────────
_cache: dict = {
    "result":       None,
    "running":      False,
    "last_run":     None,
    "auto_execute": False,   # OFF by default — scans never place live orders unless toggled on
}


async def _run_and_cache(tickers: list[str] | None):
    if _cache["running"]:
        return
    _cache["running"] = True
    scope = f"{len(tickers)} tickers" if tickers else "auto-discovering top movers + core watchlist"
    log.info("AutoTrader scan starting — %s (auto_execute=%s)", scope, _cache["auto_execute"])
    try:
        loop = asyncio.get_event_loop()
        # yfinance + pandas_ta are blocking — run off the event loop thread pool
        result = await loop.run_in_executor(
            None, engine.run_screen_and_trade, tickers, _cache["auto_execute"]
        )
        _cache["result"]   = result
        _cache["last_run"] = datetime.utcnow().isoformat()
        log.info("AutoTrader scan done — %d tickers evaluated.", len(result))
    except Exception as exc:
        log.error("AutoTrader run failed: %s", exc)
    finally:
        _cache["running"] = False


@app.on_event("startup")
async def startup():
    async def _delayed():
        await asyncio.sleep(15)
        await _run_and_cache(None)  # None -> auto-discover today's top movers + core watchlist
    asyncio.create_task(_delayed())


# ── UI ────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def ui():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


# ── API ───────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "running": _cache["running"]}


@app.get("/api/autotrader/status")
def status():
    return {
        "running":      _cache["running"],
        "last_run":     _cache["last_run"],
        "auto_execute": _cache["auto_execute"],
        "alpaca":       engine.get_account_snapshot(),
    }


@app.get("/api/autotrader/movers")
async def movers():
    """Preview today's auto-discovered universe (core watchlist + Yahoo's
    day_gainers/day_losers/most_actives) without running a full screen."""
    loop = asyncio.get_event_loop()
    tickers, sources = await loop.run_in_executor(None, engine.build_scan_universe)
    return {"tickers": tickers, "sources": sources, "count": len(tickers)}


@app.get("/api/autotrader/results")
def get_results():
    if _cache["result"] is None:
        if _cache["running"]:
            return JSONResponse(status_code=202, content={"status": "running", "message": "Scan in progress — check back shortly."})
        return JSONResponse(status_code=503, content={"status": "no_data", "message": "No results yet. Starting scan..."})

    return {
        "results":      _cache["result"],
        "running":      _cache["running"],
        "last_run":     _cache["last_run"],
        "auto_execute": _cache["auto_execute"],
    }


@app.post("/api/autotrader/run")
async def trigger_run(background_tasks: BackgroundTasks, tickers: str = Query(None)):
    if _cache["running"]:
        return {"status": "already_running"}
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None
    background_tasks.add_task(_run_and_cache, ticker_list)
    return {"status": "started", "tickers": ticker_list or "auto-discover"}


@app.post("/api/autotrader/auto-execute")
def set_auto_execute(payload: dict = Body(...)):
    """Toggle whether future scans automatically submit bracket orders for
    HIGH-confidence BUY signals. Defaults OFF — this is a real order-placement
    switch (paper trading unless APCA_API_BASE_URL points at the live endpoint)."""
    _cache["auto_execute"] = bool(payload.get("enabled", False))
    log.info("auto_execute set to %s", _cache["auto_execute"])
    return {"auto_execute": _cache["auto_execute"]}


@app.post("/api/autotrader/execute")
async def execute_trade(payload: dict = Body(...)):
    """Manually fire a single bracket order for a ticker (the 'Execute Trade' button)."""
    ticker      = str(payload.get("ticker", "")).upper().strip()
    entry_price = payload.get("entry_price")
    if not ticker or not entry_price:
        return JSONResponse(status_code=400, content={"error": "ticker and entry_price are required"})
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, engine.execute_bracket_order, ticker, float(entry_price))
        return JSONResponse(status_code=200 if result.get("executed") else 422, content=result)
    except Exception as exc:
        log.error("Manual execute failed for %s: %s", ticker, exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/api/autotrader/orders/{order_id}")
def order_status(order_id: str):
    return engine.check_order_status(order_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
