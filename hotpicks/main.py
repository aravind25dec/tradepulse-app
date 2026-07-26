"""
hotpicks/main.py
FastAPI server for the Hot Picks day-trading signal engine.
Run: uvicorn main:app --reload --port 8003
"""

import asyncio
import json as _json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# Load env from sibling backend/.env
load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

import agent_engine
import debate_engine

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
import usage_tracker
import whatsapp_notifier
from engine import DEFAULT_UNIVERSE, run_hot_picks, analyze_single_ticker
from portfolio_engine import (
    robinhood_login, robinhood_logout, is_logged_in,
    get_robinhood_positions, get_all_accounts, place_order,
    analyze_portfolio, ROBINHOOD_AVAILABLE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(title="TradePulse Hot Picks", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory cache ───────────────────────────────────────────────────────────
_cache: dict = {
    "result":   None,
    "running":  False,
    "last_run": None,
}


async def _run_and_cache():
    if _cache["running"]:
        return
    _cache["running"] = True
    log.info("Hot Picks scan starting — %d tickers", len(DEFAULT_UNIVERSE))
    try:
        result = await run_hot_picks()
        _cache["result"]   = result
        _cache["last_run"] = datetime.utcnow().isoformat()
        log.info(
            "Hot Picks done. Scanned %d tickers, %d hot picks found.",
            result.get("total_scanned", 0),
            len(result.get("hot_picks", [])),
        )
    except Exception as exc:
        log.error("Hot Picks run failed: %s", exc)
    finally:
        _cache["running"] = False


# ── Startup: kick off first scan after 15 s ───────────────────────────────────
@app.on_event("startup")
async def startup():
    async def _delayed():
        await asyncio.sleep(15)
        await _run_and_cache()
    asyncio.create_task(_delayed())


# ── UI ────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def ui():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


# ── API ───────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "scanning": _cache["running"]}


@app.get("/api/hotpicks")
async def get_hotpicks(
    background_tasks: BackgroundTasks,
    refresh: bool = Query(False, description="Trigger a fresh scan in background"),
):
    """
    Returns the latest Hot Picks results.
    Pass ?refresh=true to kick off a new scan while immediately returning cached data.
    """
    if refresh and not _cache["running"]:
        background_tasks.add_task(_run_and_cache)

    if _cache["result"] is None:
        if _cache["running"]:
            return JSONResponse(
                status_code=202,
                content={"status": "running", "message": "Scanning market universe — check back in ~60 s"},
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


@app.get("/api/hotpicks/status")
def get_status():
    return {
        "running":  _cache["running"],
        "last_run": _cache["last_run"],
        "picks":    len(_cache["result"].get("hot_picks", [])) if _cache["result"] else 0,
    }


@app.post("/api/hotpicks/run")
async def trigger_run(background_tasks: BackgroundTasks):
    if _cache["running"]:
        return {"status": "already_running"}
    background_tasks.add_task(_run_and_cache)
    return {"status": "started", "message": f"Scanning {len(DEFAULT_UNIVERSE)} tickers..."}


@app.get("/api/hotpicks/analyze")
async def analyze_custom(ticker: str = Query(..., description="Stock ticker symbol, e.g. NVDA")):
    """On-demand full analysis for any ticker — runs complete signal pipeline + AI trade plan."""
    ticker = ticker.upper().strip()
    if not ticker or len(ticker) > 10 or not ticker.isalpha():
        return JSONResponse(status_code=400, content={"error": f"Invalid ticker symbol: '{ticker}'"})
    log.info("Custom analysis requested: %s", ticker)
    try:
        result = await analyze_single_ticker(ticker)
        if "error" in result:
            return JSONResponse(status_code=404, content=result)
        return result
    except Exception as exc:
        log.error("Custom analysis failed for %s: %s", ticker, exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ── Portfolio page ────────────────────────────────────────────────────────────
@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio_ui():
    return (Path(__file__).parent / "portfolio.html").read_text(encoding="utf-8")


# ── Portfolio API ─────────────────────────────────────────────────────────────
from fastapi import Body

@app.get("/api/portfolio/status")
def portfolio_status():
    return {"logged_in": is_logged_in(), "robinhood_available": ROBINHOOD_AVAILABLE}


@app.get("/api/portfolio/debug/accounts")
def debug_accounts():
    """Raw dump of everything Robinhood returns for /accounts/ — diagnose missing accounts."""
    if not is_logged_in():
        return JSONResponse(status_code=401, content={"error": "Not logged in"})
    try:
        import robin_stocks.robinhood as _rh
        import json
        raw = _rh.account.get_all_accounts() or []
        # Convert every value to string so nothing breaks JSON serialization
        safe = json.loads(json.dumps(raw, default=str))
        log.info("DEBUG /accounts/ returned %d item(s): %s", len(safe), safe)
        return JSONResponse(content={"count": len(safe), "raw": safe})
    except Exception as exc:
        log.error("debug_accounts error: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/api/portfolio/login")
async def portfolio_login(payload: dict = Body(...)):
    username = payload.get("username", "").strip()
    password = payload.get("password", "")
    mfa_code = (payload.get("mfa_code") or "").strip() or None
    if not username or not password:
        return JSONResponse(status_code=400, content={"error": "Username and password required"})
    result = robinhood_login(username, password, mfa_code)
    return result


@app.post("/api/portfolio/logout")
def portfolio_logout():
    robinhood_logout()
    return {"success": True}


@app.get("/api/portfolio/positions")
def portfolio_positions():
    if not is_logged_in():
        return JSONResponse(status_code=401, content={"error": "Not logged in to Robinhood"})
    try:
        positions = get_robinhood_positions()
        return {"positions": positions}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/api/portfolio/accounts")
def portfolio_accounts():
    """Return all Robinhood accounts with type and buying power."""
    if not is_logged_in():
        return JSONResponse(status_code=401, content={"error": "Not logged in to Robinhood"})
    try:
        accounts = get_all_accounts()
        return {"accounts": accounts}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/api/portfolio/order")
async def portfolio_order(payload: dict = Body(...)):
    """Place a buy or sell order. Body: {ticker, side, order_type, quantity, limit_price?, account_number?}."""
    if not is_logged_in():
        return JSONResponse(status_code=401, content={"error": "Not logged in to Robinhood"})
    ticker         = str(payload.get("ticker",       "")).upper().strip()
    side           = str(payload.get("side",          "")).lower().strip()
    order_type     = str(payload.get("order_type",   "market")).lower().strip()
    quantity       = float(payload.get("quantity",    0) or 0)
    raw_limit      = payload.get("limit_price")
    limit_price    = float(raw_limit) if raw_limit else None
    account_number = str(payload.get("account_number", "") or "").strip() or None
    time_in_force  = str(payload.get("time_in_force", "gtc") or "gtc").lower().strip()

    if not ticker or not side or quantity <= 0:
        return JSONResponse(status_code=400, content={"error": "ticker, side, and quantity > 0 are required"})

    result = place_order(ticker, side, order_type, quantity, limit_price, account_number, time_in_force)
    if not result.get("success"):
        return JSONResponse(status_code=422, content=result)
    return result


@app.post("/api/portfolio/analyze")
async def portfolio_analyze(payload: dict = Body(...)):
    """Analyze a list of positions. Each item: {ticker, quantity, avg_cost}."""
    positions = payload.get("positions", [])
    if not positions:
        return JSONResponse(status_code=400, content={"error": "No positions provided"})
    # Validate
    clean = []
    for p in positions:
        t = str(p.get("ticker", "")).upper().strip()
        if not t:
            continue
        clean.append({
            "ticker":   t,
            "quantity": float(p.get("quantity", 0) or 0),
            "avg_cost": float(p.get("avg_cost",  0) or 0),
        })
    if not clean:
        return JSONResponse(status_code=400, content={"error": "No valid positions"})
    try:
        result = await analyze_portfolio(clean)
        return result
    except Exception as exc:
        log.error("Portfolio analysis failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ── Agent page ────────────────────────────────────────────────────────────────
@app.get("/agent", response_class=HTMLResponse)
async def agent_ui():
    return (Path(__file__).parent / "agent.html").read_text(encoding="utf-8")


@app.get("/api/agent/status")
def agent_status():
    """Return Claude CLI availability + Robinhood MCP registration."""
    cli_ok = agent_engine.claude_available()
    mcp_ok = agent_engine.check_robinhood_mcp() if cli_ok else False
    return {"claude_available": cli_ok, "robinhood_mcp": mcp_ok}


# ── Snip tool-call/status noise out of the raw agent stream ──────────────────
_SNIP_PATTERNS = [
    r"\n*_🧠 optimized query:.*?_\n*",
    r"\n*_🔧 calling `[^`]*`…_\n*",
    r"\n*⚠ Claude error:\n```[\s\S]*?```",
    r"\n*⚠ \*\*No response for[\s\S]*?\*\*[^\n]*",
    r"\n*⚠ \*\*Claude Code CLI not found\.\*\*[\s\S]*",
]


def _snip_agent_logs(text: str) -> str:
    cleaned = text
    for pat in _SNIP_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


OLLAMA_BASE = "http://localhost:11434"
OLLAMA_SUMMARY_MODEL = "gemma3:4b"

_SUMMARY_PROMPT = """\
Summarize the assistant's answer below into a short WhatsApp-friendly brief.
Keep any specific numbers, tickers, and figures. 2-4 sentences max. No preamble.

ANSWER:
{answer}

Summary:"""


async def _ollama_summarize(text: str) -> str:
    """Condense text via a local Ollama model — no Claude/API usage. Falls back to
    the original (truncated) text if Ollama is unreachable."""
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": OLLAMA_SUMMARY_MODEL,
                    "messages": [{"role": "user", "content": _SUMMARY_PROMPT.format(answer=text)}],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.2, "num_predict": 150},
                },
            )
        if resp.status_code == 200:
            summary = resp.json().get("message", {}).get("content", "").strip()
            if summary:
                return summary
    except Exception as exc:
        log.warning("Ollama summarize failed, falling back to raw text: %s: %s", type(exc).__name__, exc)
    return text[:1000]


def _format_agent_whatsapp_brief(question: str, summary: str) -> str:
    ts = datetime.utcnow().strftime("%b %d, %I:%M %p UTC")
    return f"*TradePulse Agent* — {ts}\n\n*Q:* {question}\n\n{summary}"


async def _send_agent_whatsapp_brief(question: str, answer: str):
    if not whatsapp_notifier.configured():
        return
    cleaned = _snip_agent_logs(answer)
    if not cleaned:
        return
    try:
        summary = await _ollama_summarize(cleaned)
        outcome = await whatsapp_notifier.send_whatsapp_message(
            _format_agent_whatsapp_brief(question, summary)
        )
        if outcome.get("success"):
            log.info("Agent WhatsApp brief sent.")
        else:
            log.warning("Agent WhatsApp brief failed: %s", outcome.get("error"))
    except Exception as exc:
        log.error("Agent WhatsApp brief error: %s", exc)


@app.post("/api/agent/chat")
async def agent_chat(payload: dict = Body(...)):
    messages = payload.get("messages", [])
    include_historicals = bool(payload.get("historicals", False))
    optimize_query = bool(payload.get("optimize_query", False))
    if not messages:
        return JSONResponse(status_code=400, content={"error": "No messages provided"})

    question = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
    )

    async def event_gen():
        chunks: list[str] = []
        try:
            async for text in agent_engine.stream_chat(
                messages, include_historicals=include_historicals, optimize=optimize_query
            ):
                chunks.append(text)
                yield f"data: {_json.dumps({'text': text})}\n\n"
        except Exception as exc:
            log.error("Agent chat stream error: %s", exc, exc_info=True)
            yield f"data: {_json.dumps({'error': str(exc)})}\n\n"
        finally:
            asyncio.create_task(_send_agent_whatsapp_brief(question, "".join(chunks)))
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/api/agent/reset")
def agent_reset():
    agent_engine.reset_session()
    return {"success": True}


# ── My Picks page (portfolio positions as hot-pick tiles via MCP) ─────────────
@app.get("/my-picks", response_class=HTMLResponse)
async def my_picks_ui():
    return (Path(__file__).parent / "my_picks.html").read_text(encoding="utf-8")


@app.get("/api/my-picks/positions")
async def my_picks_positions():
    """Fetch the user's Robinhood positions through Claude + Robinhood MCP."""
    if not agent_engine.claude_available():
        return JSONResponse(
            status_code=503,
            content={"error": "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"},
        )
    if not agent_engine.check_robinhood_mcp():
        return JSONResponse(
            status_code=503,
            content={"error": "Robinhood MCP not configured. Run: claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading"},
        )
    try:
        positions = await agent_engine.fetch_positions_via_mcp()
        return {"positions": positions}
    except Exception as exc:
        log.error("MCP positions fetch failed: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ── Consensus / Debate page ───────────────────────────────────────────────────
@app.get("/debate", response_class=HTMLResponse)
async def debate_ui():
    return (Path(__file__).parent / "debate.html").read_text(encoding="utf-8")


@app.get("/api/debate/portfolio")
async def debate_portfolio():
    """Return the cached portfolio (or a fresh fetch if cache is cold)."""
    if not agent_engine.check_robinhood_mcp():
        return JSONResponse(status_code=503, content={"error": "Robinhood MCP not connected"})
    try:
        positions, text, was_cached = await debate_engine.get_portfolio()
        return {
            "positions": positions,
            "summary":   text.strip(),
            "cached":    was_cached,
            "age_min":   int(debate_engine.portfolio_cache_age() // 60),
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/api/debate/portfolio/refresh")
async def debate_portfolio_refresh():
    """Force-invalidate the portfolio cache and re-fetch from Robinhood MCP."""
    if not agent_engine.check_robinhood_mcp():
        return JSONResponse(status_code=503, content={"error": "Robinhood MCP not connected"})
    try:
        debate_engine.invalidate_portfolio_cache()
        positions, text, _ = await debate_engine.get_portfolio(force_refresh=True)
        return {"positions": positions, "summary": text.strip(), "count": len(positions)}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/api/debate/status")
def debate_status():
    """Claude CLI + Robinhood MCP availability for the consensus panel."""
    cli_ok = agent_engine.claude_available()
    mcp_ok = agent_engine.check_robinhood_mcp() if cli_ok else False
    return {"claude_available": cli_ok, "robinhood_mcp": mcp_ok}


class DebateRequest(BaseModel):
    question: str


@app.post("/api/debate/chat")
async def debate_chat(req: DebateRequest):
    """
    Stream the multi-agent consensus debate as SSE.

    Event types: status, portfolio_ready, phase_start,
                 agent_start, chunk, agent_done,
                 synthesis_chunk, done, error
    """
    q = req.question.strip()
    if not q:
        return JSONResponse({"error": "Question is required."}, status_code=400)

    mcp = agent_engine.check_robinhood_mcp()
    log.info("Consensus request: %r  |  mcp=%s", q[:80], mcp)

    async def event_stream():
        try:
            async for event in debate_engine.run_consensus(q, mcp_available=mcp):
                yield f"data: {_json.dumps(event)}\n\n"
        except Exception as exc:
            log.exception("Consensus stream error")
            yield f"data: {_json.dumps({'type':'error','message':str(exc)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ── Usage tracking endpoints (shared across all apps via usage_stats.json) ────
from pydantic import BaseModel as _BM

class _PlanReq(_BM):
    plan: str

@app.get("/api/usage")
def usage_get():
    return usage_tracker.get_stats()

@app.post("/api/usage/reset")
def usage_reset():
    usage_tracker.reset_stats()
    return {"success": True}

@app.post("/api/usage/plan")
def usage_plan(req: _PlanReq):
    usage_tracker.set_plan(req.plan)
    return {"success": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
