"""
smartpicks/main.py
FastAPI server for the Smart Picks fundamental catalyst scanner.
Run: uvicorn main:app --reload --port 8004
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

from engine import SMART_UNIVERSE, run_smart_picks, analyze_single_ticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(title="TradePulse Smart Picks", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: dict = {
    "result":   None,
    "running":  False,
    "last_run": None,
}


async def _run_and_cache():
    if _cache["running"]:
        return
    _cache["running"] = True
    log.info("Smart Picks scan starting — %d tickers", len(SMART_UNIVERSE))
    try:
        result             = await run_smart_picks()
        _cache["result"]   = result
        _cache["last_run"] = datetime.utcnow().isoformat()
        log.info(
            "Smart Picks done. Scanned %d tickers, %d top picks.",
            result.get("total_scanned", 0),
            len(result.get("top_picks", [])),
        )
    except Exception as exc:
        log.error("Smart Picks run failed: %s", exc)
    finally:
        _cache["running"] = False


@app.on_event("startup")
async def startup():
    async def _delayed():
        await asyncio.sleep(20)
        await _run_and_cache()
    asyncio.create_task(_delayed())


@app.get("/", response_class=HTMLResponse)
async def ui():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "scanning": _cache["running"]}


@app.get("/api/smartpicks")
async def get_smartpicks(
    background_tasks: BackgroundTasks,
    refresh: bool = Query(False),
):
    if refresh and not _cache["running"]:
        background_tasks.add_task(_run_and_cache)

    if _cache["result"] is None:
        if _cache["running"]:
            return JSONResponse(
                status_code=202,
                content={"status": "running", "message": "Scanning market universe — check back in ~90s"},
            )
        return JSONResponse(
            status_code=503,
            content={"status": "no_data", "message": "No results yet. Starting scan..."},
        )

    age = None
    if _cache["last_run"]:
        age = int(
            (datetime.utcnow() - datetime.fromisoformat(_cache["last_run"])).total_seconds()
        )

    return {
        **_cache["result"],
        "running":           _cache["running"],
        "last_run":          _cache["last_run"],
        "cache_age_seconds": age,
    }


@app.get("/api/smartpicks/status")
def get_status():
    return {
        "running":  _cache["running"],
        "last_run": _cache["last_run"],
        "picks":    len(_cache["result"].get("top_picks", [])) if _cache["result"] else 0,
    }


@app.post("/api/smartpicks/run")
async def trigger_run(background_tasks: BackgroundTasks):
    if _cache["running"]:
        return {"status": "already_running"}
    background_tasks.add_task(_run_and_cache)
    return {"status": "started", "message": f"Scanning {len(SMART_UNIVERSE)} tickers..."}


@app.get("/api/smartpicks/analyze")
async def analyze_custom(ticker: str = Query(..., description="Stock symbol, e.g. NVDA")):
    ticker = ticker.upper().strip()
    if not ticker or len(ticker) > 10 or not ticker.replace("-", "").replace(".", "").isalnum():
        return JSONResponse(status_code=400, content={"error": f"Invalid ticker: '{ticker}'"})
    log.info("Custom analysis requested: %s", ticker)
    try:
        result = await analyze_single_ticker(ticker)
        if "error" in result:
            return JSONResponse(status_code=404, content=result)
        return result
    except Exception as exc:
        log.error("Custom analysis failed for %s: %s", ticker, exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
