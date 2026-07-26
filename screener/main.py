"""
screener/main.py
FastAPI server exposing the day-trading screener.
Run: uvicorn main:app --reload --port 8001
"""
import os
import asyncio
import logging
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load env from parent backend/.env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

from screener_engine import run_screener, DEFAULT_UNIVERSE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(title="TradePulse Screener", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory cache ───────────────────────────────────────────────────────────
_cache: dict = {
    "result": None,
    "running": False,
    "last_run": None,
}

FMP_API_KEY = os.getenv("FMP_API_KEY", "")


async def _run_and_cache(universe: list[str]):
    if _cache["running"]:
        return
    _cache["running"] = True
    log.info("Starting screener run...")
    try:
        result = await run_screener(universe=universe, fmp_api_key=FMP_API_KEY)
        _cache["result"] = result
        _cache["last_run"] = datetime.utcnow().isoformat()
        log.info(f"Screener done. Screened {result.get('total_screened', 0)} tickers.")
    except Exception as e:
        log.error(f"Screener run failed: {e}")
    finally:
        _cache["running"] = False


@app.on_event("startup")
async def startup():
    """Kick off first screener run 30s after startup to avoid racing the backend's yfinance load."""
    async def _delayed():
        await asyncio.sleep(30)
        await _run_and_cache(DEFAULT_UNIVERSE)
    asyncio.create_task(_delayed())


@app.get("/health")
def health():
    return {"status": "ok", "screener": "running" if _cache["running"] else "idle"}


@app.get("/api/screener")
async def get_screener_results(
    background_tasks: BackgroundTasks,
    refresh: bool = Query(False, description="Force a new screen run in background"),
):
    """
    Returns the latest screener results.
    Pass ?refresh=true to trigger a fresh scan in the background.
    """
    if refresh and not _cache["running"]:
        background_tasks.add_task(_run_and_cache, DEFAULT_UNIVERSE)

    if _cache["result"] is None:
        if _cache["running"]:
            return JSONResponse(
                status_code=202,
                content={"status": "running", "message": "Screener is warming up, check back in ~60s"},
            )
        return JSONResponse(
            status_code=503,
            content={"status": "no_data", "message": "No results yet. Try again shortly."},
        )

    return {
        **_cache["result"],
        "running": _cache["running"],
        "last_run": _cache["last_run"],
        "cache_age_seconds": int(
            (datetime.utcnow() - datetime.fromisoformat(_cache["last_run"])).total_seconds()
        ) if _cache["last_run"] else None,
    }


@app.get("/api/screener/status")
def get_status():
    return {
        "running": _cache["running"],
        "last_run": _cache["last_run"],
        "total_screened": _cache["result"]["total_screened"] if _cache["result"] else 0,
    }


@app.post("/api/screener/run")
async def trigger_run(background_tasks: BackgroundTasks):
    """Manually trigger a fresh screener run."""
    if _cache["running"]:
        return {"status": "already_running"}
    background_tasks.add_task(_run_and_cache, DEFAULT_UNIVERSE)
    return {"status": "started", "message": f"Screening {len(DEFAULT_UNIVERSE)} tickers..."}
