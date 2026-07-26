"""
analyst/main.py
Single-call investment analyst — ONE Claude CLI call covers all 5 analysis perspectives.
4-5x cheaper than the debate panel (1 call vs 5).

The trick: the prompt is built as a single line (no newlines) so Windows cmd.exe
does not split it. Claude uses Robinhood MCP internally to fetch portfolio data,
then returns a structured 5-section analysis in one streaming response.

Port: 8007
"""

import json
import logging
import sys
from pathlib import Path

# Reuse the working agent_engine from hotpicks — same claude binary, same Pro subscription
sys.path.insert(0, str(Path(__file__).parent.parent / "hotpicks"))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi import Body
from pydantic import BaseModel

load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

import agent_engine
import usage_tracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(title="TradePulse Analyst", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_HTML = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

# ── Single-call structured prompt ─────────────────────────────────────────────
# Intentionally NO \n characters — Windows cmd.exe treats \n as a command
# separator when shell=True, which truncates the prompt. Claude parses this fine.

ANALYSIS_PREFIX = (
    "Use the Robinhood MCP tools to fetch my current portfolio across ALL accounts first. "
    "Then answer my question with exactly these 5 labeled sections (keep each under 80 words): "
    "## 📈 Fundamentals — earnings quality, P/E vs peers, revenue growth, competitive moat. "
    "## 📊 Technical — current price action, trend direction, RSI/MACD signal, key support and resistance. "
    "## 🛡️ Risk — my portfolio concentration, downside scenario, suggested stop-loss levels. "
    "## ⚡ Catalysts — specific upcoming events in next 4 weeks with dates and expected impact. "
    "## ✅ Decision — start with BUY/SELL/HOLD/WAIT in bold, "
    "then one plain-English sentence, "
    "then: 1. [action with ticker and specific price], 2. [stop-loss at $X], 3. [target $X by date]. "
    "Then: Risk: [one sentence]. Confidence: 🟢 High / 🟡 Medium / 🔴 Low. "
    "Do NOT introduce yourself. Start immediately with ## 📈 Fundamentals. "
    "My question: "
)


@app.get("/", response_class=HTMLResponse)
async def ui():
    return _HTML


@app.get("/api/analyst/status")
def status():
    cli_ok = agent_engine.claude_available()
    mcp_ok = agent_engine.check_robinhood_mcp() if cli_ok else False
    return {"claude_available": cli_ok, "robinhood_mcp": mcp_ok}


class ChatRequest(BaseModel):
    question: str
    history: list = []   # [{role, content}] for follow-up questions


@app.post("/api/analyst/chat")
async def chat(req: ChatRequest):
    q = req.question.strip()
    if not q:
        return JSONResponse({"error": "Question required."}, status_code=400)

    # Build message list — prefix only the FIRST message in a new session
    history = list(req.history)
    if not history:
        # Fresh question: inject the full structured-analysis prefix (single line, no \n)
        full_q = ANALYSIS_PREFIX + q
    else:
        # Follow-up in an existing session: Claude already knows the format
        full_q = q

    history.append({"role": "user", "content": full_q})

    log.info("Analyst request: %r (history=%d)", q[:80], len(history) - 1)

    async def event_stream():
        try:
            async for chunk in agent_engine.stream_chat(history):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        except Exception as exc:
            log.exception("Analyst stream error")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/analyst/reset")
def reset():
    agent_engine.reset_session()
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
