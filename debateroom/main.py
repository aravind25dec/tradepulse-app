"""
debateroom/main.py
FastAPI server for the DebateRoom multi-agent investment debate app.
Run: uvicorn main:app --reload --port 8006
"""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

from debate_engine import claude_available, check_robinhood_mcp, run_debate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(title="TradePulse DebateRoom", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_HTML = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML


@app.get("/api/debate/status")
async def status():
    return JSONResponse({
        "claude_available": claude_available(),
        "robinhood_mcp":    check_robinhood_mcp(),
    })


class DebateRequest(BaseModel):
    question: str


@app.post("/api/debate/chat")
async def debate_chat(req: DebateRequest):
    """
    Stream the full multi-agent debate as SSE events.

    If the Robinhood MCP is connected, the debate is preceded by a live
    portfolio fetch so every analyst can reference the user's actual holdings.

    Event types:
      status          — informational message (e.g. "fetching portfolio")
      portfolio_ready — portfolio context fetched (includes summary text)
      phase_start     — new debate round beginning
      agent_start     — one analyst starts responding
      chunk           — streaming text token from an analyst
      agent_done      — analyst finished (includes full_text)
      done            — entire debate complete
    """
    q = req.question.strip()
    if not q:
        return JSONResponse({"error": "Question is required."}, status_code=400)

    mcp = check_robinhood_mcp()
    log.info("Debate request: %r  |  mcp=%s", q[:80], mcp)

    async def event_stream():
        try:
            async for event in run_debate(q, mcp_available=mcp):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            log.exception("Debate stream error")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
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
