"""
localai/main.py
Local AI investment analyst — Ollama + LangChain, zero cloud token cost.
Runs 100% offline after one-time model download.

Port: 8008
"""

import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
import engine
import usage_tracker
import memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(title="TradePulse Local AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_HTML = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
async def ui():
    return _HTML


@app.get("/api/localai/status")
async def status():
    ollama = await engine.check_ollama()
    portfolio = await engine.fetch_portfolio()
    return {
        "ollama_available": ollama["available"],
        "models":           ollama["models"],
        "portfolio_count":  len(portfolio),
        "portfolio_source": "hotpicks-cache" if portfolio else "none",
    }


@app.get("/api/localai/portfolio")
async def portfolio():
    positions = await engine.fetch_portfolio()
    return {
        "positions": positions,
        "summary":   engine._format_portfolio(positions),
        "count":     len(positions),
    }


class AnalysisRequest(BaseModel):
    question: str
    model: str = engine.DEFAULT_MODEL
    portfolio: list = []   # optional manual override


@app.post("/api/localai/analyze")
async def analyze(req: AnalysisRequest):
    q = req.question.strip()
    if not q:
        return JSONResponse({"error": "Question required."}, status_code=400)

    positions = req.portfolio or await engine.fetch_portfolio()
    memories  = memory.retrieve(q, top_k=3)
    log.info(
        "Local AI request: %r  model=%s  positions=%d  memories=%d",
        q[:80], req.model, len(positions), len(memories),
    )

    async def event_stream():
        chunks: list[str] = []
        try:
            async for chunk in engine.stream_analysis(q, positions, req.model, memories):
                chunks.append(chunk)
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        except Exception as exc:
            log.exception("Local AI stream error")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            if chunks:
                tickers = [p.get("ticker", "") for p in positions]
                memory.save(q, "".join(chunks), req.model, tickers)
                log.info("Saved to memory (total: %d)", memory.count())
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Memory endpoints ──────────────────────────────────────────────────────────
@app.get("/api/localai/memory")
def memory_stats():
    return {"count": memory.count(), "entries": memory.all_entries()[-10:]}

@app.delete("/api/localai/memory")
def memory_clear():
    memory.clear()
    return {"success": True}


# ── Usage proxy (reads shared usage_stats.json) ───────────────────────────────
@app.get("/api/usage")
def usage_get():
    return usage_tracker.get_stats()

@app.post("/api/usage/reset")
def usage_reset():
    usage_tracker.reset_stats()
    return {"success": True}

@app.post("/api/usage/plan")
async def usage_plan(payload: dict = Body(...)):
    usage_tracker.set_plan(payload.get("plan", "Pro"))
    return {"success": True}
