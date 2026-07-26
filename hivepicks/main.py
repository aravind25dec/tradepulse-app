"""
hivepicks/main.py
FastAPI server for the Hive Picks multi-agent day-trading system.
Run: uvicorn main:app --reload --port 8005
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

from hive_engine import HIVE_UNIVERSE, run_hive, analyze_single_hive

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(title="TradePulse Hive Picks", version="1.0.0")

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
    log.info("Hive scan starting — %d tickers in universe", len(HIVE_UNIVERSE))
    try:
        result             = await run_hive()
        _cache["result"]   = result
        _cache["last_run"] = datetime.utcnow().isoformat()
        log.info(
            "Hive done. %d strong buys, %d buys, %d watches.",
            len(result.get("strong_buys", [])),
            len(result.get("buys", [])),
            len(result.get("watches", [])),
        )
    except Exception as exc:
        log.error("Hive run failed: %s", exc, exc_info=True)
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


@app.get("/api/hivepicks")
async def get_hivepicks(
    background_tasks: BackgroundTasks,
    refresh: bool = Query(False),
):
    if refresh and not _cache["running"]:
        background_tasks.add_task(_run_and_cache)

    if _cache["result"] is None:
        if _cache["running"]:
            return JSONResponse(
                status_code=202,
                content={"status": "running", "message": "Hive is deliberating — agents are analyzing candidates (~3–5 min)"},
            )
        return JSONResponse(
            status_code=503,
            content={"status": "no_data", "message": "No results yet. Trigger /api/hivepicks/run to start."},
        )

    age = None
    if _cache["last_run"]:
        age = int((datetime.utcnow() - datetime.fromisoformat(_cache["last_run"])).total_seconds())

    return {
        **_cache["result"],
        "running":           _cache["running"],
        "last_run":          _cache["last_run"],
        "cache_age_seconds": age,
    }


@app.get("/api/hivepicks/status")
def get_status():
    r = _cache["result"]
    return {
        "running":     _cache["running"],
        "last_run":    _cache["last_run"],
        "strong_buys": len(r.get("strong_buys", [])) if r else 0,
        "buys":        len(r.get("buys", []))        if r else 0,
        "watches":     len(r.get("watches", []))     if r else 0,
    }


@app.post("/api/hivepicks/run")
async def trigger_run(background_tasks: BackgroundTasks):
    if _cache["running"]:
        return {"status": "already_running", "message": "Hive is already deliberating…"}
    background_tasks.add_task(_run_and_cache)
    return {
        "status":  "started",
        "message": f"Hive convened — analyzing {len(HIVE_UNIVERSE)} tickers with 5 agents",
    }


@app.get("/api/hivepicks/analyze")
async def analyze_custom(ticker: str = Query(..., description="Stock ticker e.g. NVDA")):
    ticker = ticker.upper().strip()
    if not ticker or len(ticker) > 10 or not ticker.replace("-", "").replace(".", "").isalnum():
        return JSONResponse(status_code=400, content={"error": f"Invalid ticker: '{ticker}'"})
    log.info("On-demand hive analysis: %s", ticker)
    try:
        result = await analyze_single_hive(ticker)
        if "error" in result:
            return JSONResponse(status_code=404, content=result)
        return result
    except Exception as exc:
        log.error("Hive analysis failed for %s: %s", ticker, exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
