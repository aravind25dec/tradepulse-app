"""
hotpicks/debate_engine.py
Multi-agent consensus engine.

Fetches live Robinhood portfolio via MCP (reusing agent_engine infrastructure),
runs 4 specialist analysts in parallel with that shared context, then asks a
CIO judge to synthesise a consensus recommendation.
"""
import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
from typing import AsyncGenerator

from agent_engine import _find_claude, fetch_positions_via_mcp, _record_usage

log = logging.getLogger(__name__)

# ── Agent definitions ─────────────────────────────────────────────────────────

AGENTS = [
    {
        "id": "fundamental",
        "name": "Fundamental Analyst",
        "emoji": "📈",
        "color": "#3fb950",
        "bg": "rgba(63,185,80,0.08)",
        "focus": (
            "earnings quality, revenue growth, P/E vs peers, "
            "balance-sheet health, and competitive moat"
        ),
        "verdict": "BUY / HOLD / SELL with a 4-week price target",
    },
    {
        "id": "technical",
        "name": "Technical Analyst",
        "emoji": "📊",
        "color": "#58a6ff",
        "bg": "rgba(88,166,255,0.08)",
        "focus": (
            "price action, trend direction, momentum (RSI/MACD), "
            "key support & resistance levels, and chart patterns"
        ),
        "verdict": "BUY NOW / WAIT FOR DIP / SELL with specific entry and stop-loss prices",
    },
    {
        "id": "risk",
        "name": "Risk Analyst",
        "emoji": "🛡️",
        "color": "#d29922",
        "bg": "rgba(210,153,34,0.08)",
        "focus": (
            "portfolio concentration, sector overlap, downside scenarios, "
            "volatility, and position sizing"
        ),
        "verdict": "LOW / MODERATE / HIGH risk rating with specific stop-loss levels",
    },
    {
        "id": "catalyst",
        "name": "Catalyst Hunter",
        "emoji": "⚡",
        "color": "#bc8cff",
        "bg": "rgba(188,140,255,0.08)",
        "focus": (
            "upcoming earnings dates, product launches, FDA decisions, "
            "macro events (Fed/CPI), and unusual options activity"
        ),
        "verdict": "STRONG / WEAK / NO CATALYST with specific dates and expected move size",
    },
]

JUDGE = {
    "id": "judge",
    "name": "Investment Decision",
    "emoji": "⚖️",
    "color": "#f0883e",
    "bg": "rgba(240,136,62,0.10)",
    "system": (
        "You are a plain-English investment advisor for a regular person who is NOT a finance expert.\n"
        "You have four expert analyses. Distil them into ONE simple decision a layman can act on today.\n\n"
        "Use this EXACT format and keep it SHORT (under 200 words total):\n\n"
        "## ✅ DECISION: [BUY / SELL / HOLD / WAIT]\n"
        "**[One sentence in plain English. What to do and why — no jargon.]**\n\n"
        "### Do this right now:\n"
        "1. [Specific action — e.g. 'Buy 5 shares of NVDA at around $127']\n"
        "2. [Stop-loss — e.g. 'Sell if it drops below $119 (that means something went wrong)']\n"
        "3. [Target — e.g. 'Aim to sell at $145, likely within 1-2 weeks']\n\n"
        "### The one risk:\n"
        "[One sentence. The single biggest thing that could go wrong, explained simply.]\n\n"
        "### What the experts said:\n"
        "- 📈 [Fundamental in one plain line]\n"
        "- 📊 [Technical in one plain line]\n"
        "- 🛡️ [Risk in one plain line]\n"
        "- ⚡ [Catalyst in one plain line]\n\n"
        "**Confidence: [🟢 High / 🟡 Medium / 🔴 Low]**\n\n"
        "Hard rules:\n"
        "- No finance jargon. If you use a term, explain it in plain words.\n"
        "- Always name the specific stock ticker and a specific price number.\n"
        "- If experts disagree, side with the majority and say 'most experts agree…'.\n"
        "- Total response must be under 200 words."
    ),
}


# ── Portfolio formatting ──────────────────────────────────────────────────────

def _portfolio_block(positions: list[dict]) -> str:
    """Format positions into a readable block for injection into agent prompts."""
    if not positions:
        return ""
    by_acct: dict[str, list] = {}
    for p in positions:
        acct = p.get("account_label", "Brokerage")
        by_acct.setdefault(acct, []).append(p)
    lines: list[str] = []
    for acct, pos_list in by_acct.items():
        lines.append(f"{acct}:")
        for p in pos_list:
            lines.append(
                f"  {p['ticker']:6s}  {p['quantity']} shares  |  avg cost ${p['avg_cost']}"
            )
    return "\n".join(lines)


# ── Portfolio cache (15-min TTL) ─────────────────────────────────────────────

_CACHE_TTL = 15 * 60   # seconds

_portfolio_cache: dict = {
    "positions": [],
    "text":      "",
    "fetched_at": 0.0,
}


def portfolio_cache_age() -> float:
    """Seconds since the portfolio was last fetched. Returns inf if never fetched."""
    fetched = _portfolio_cache["fetched_at"]
    return (time.time() - fetched) if fetched else float("inf")


def invalidate_portfolio_cache() -> None:
    _portfolio_cache["fetched_at"] = 0.0
    _portfolio_cache["positions"]  = []
    _portfolio_cache["text"]       = ""


async def get_portfolio(force_refresh: bool = False) -> tuple[list[dict], str, bool]:
    """
    Return (positions, formatted_text, was_cached).
    Uses cached data when < 15 min old unless force_refresh=True.
    """
    age = portfolio_cache_age()
    if not force_refresh and age < _CACHE_TTL and _portfolio_cache["positions"]:
        return _portfolio_cache["positions"], _portfolio_cache["text"], True

    positions = await fetch_positions_via_mcp()
    text = _portfolio_block(positions)
    _portfolio_cache["positions"]  = positions
    _portfolio_cache["text"]       = text
    _portfolio_cache["fetched_at"] = time.time()
    return positions, text, False


# ── Claude subprocess streaming ───────────────────────────────────────────────

async def _run_claude_stream(
    prompt: str,
    agent_id: str,
    queue: asyncio.Queue,
    timeout: float = 150.0,
) -> None:
    """
    Spawn a Claude CLI subprocess for one agent.
    Pushes agent_start / chunk / agent_done events to the shared queue.
    """
    exe = _find_claude()
    if not exe:
        await queue.put({"type": "error", "agent_id": agent_id, "message": "Claude CLI not found"})
        await queue.put({"type": "agent_done", "agent_id": agent_id, "full_text": ""})
        return

    cmd = [
        exe, "--print", "--verbose",
        "--output-format", "stream-json",
        "--dangerously-skip-permissions",
        prompt,
    ]
    clean_env = os.environ.copy()
    clean_env.pop("ANTHROPIC_API_KEY", None)   # use Pro subscription, not API credits

    loop = asyncio.get_event_loop()
    local_q: asyncio.Queue = asyncio.Queue()

    def _worker():
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=clean_env,
                shell=(sys.platform == "win32"),
            )
            for raw in proc.stdout:
                loop.call_soon_threadsafe(local_q.put_nowait, ("line", raw))
            proc.wait()
            if proc.returncode not in (0, None):
                err = proc.stderr.read().decode("utf-8", errors="replace").strip()
                if err:
                    loop.call_soon_threadsafe(local_q.put_nowait, ("err", err))
        except Exception as exc:
            loop.call_soon_threadsafe(local_q.put_nowait, ("err", str(exc)))
        finally:
            loop.call_soon_threadsafe(local_q.put_nowait, ("done", None))

    threading.Thread(target=_worker, daemon=True).start()
    await queue.put({"type": "agent_start", "agent_id": agent_id})

    full_text = ""
    last_len = 0

    try:
        while True:
            tag, val = await asyncio.wait_for(local_q.get(), timeout=timeout)
            if tag == "done":
                break
            if tag == "err":
                log.error("Agent %s error: %s", agent_id, val)
                break
            line = val.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                if line:
                    full_text += line + "\n"
                    await queue.put({"type": "chunk", "agent_id": agent_id, "text": line + "\n"})
                continue

            etype = evt.get("type", "")
            if etype == "result":
                _record_usage(evt, "debate")
            if etype == "assistant":
                for block in evt.get("message", {}).get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if len(text) > last_len:
                            delta = text[last_len:]
                            full_text += delta
                            last_len = len(text)
                            await queue.put({"type": "chunk", "agent_id": agent_id, "text": delta})

    except asyncio.TimeoutError:
        log.warning("Agent %s timed out after %.0fs", agent_id, timeout)
        await queue.put({
            "type": "error", "agent_id": agent_id,
            "message": f"Timed out after {timeout:.0f}s",
        })

    await queue.put({"type": "agent_done", "agent_id": agent_id, "full_text": full_text})


# ── Public entry point ────────────────────────────────────────────────────────

async def run_consensus(
    question: str,
    mcp_available: bool = False,
) -> AsyncGenerator[dict, None]:
    """
    Async generator driving the full multi-agent consensus flow:
      Phase 0 — fetch live portfolio via MCP (when available)
      Phase 1 — 4 specialist analysts in parallel
      Phase 2 — CIO synthesis
    """
    portfolio_ctx = ""
    positions: list[dict] = []

    # ── Phase 0: portfolio (cached 15 min) ───────────────────────────────────
    if mcp_available:
        age = portfolio_cache_age()
        cached = age < _CACHE_TTL and bool(_portfolio_cache["positions"])
        if not cached:
            yield {"type": "status", "message": "Fetching your live Robinhood portfolio via MCP…"}
        try:
            positions, portfolio_ctx, was_cached = await get_portfolio()
            age_min = int(portfolio_cache_age() // 60)
            yield {
                "type":      "portfolio_ready",
                "positions": positions,
                "summary":   portfolio_ctx.strip(),
                "cached":    was_cached,
                "age_min":   age_min,
            }
        except Exception as exc:
            log.warning("Portfolio fetch failed: %s", exc)
            yield {
                "type":    "status",
                "message": f"Portfolio unavailable ({exc}). Running general analysis.",
            }

    # ── Phase 1: parallel expert analysis ────────────────────────────────────
    yield {"type": "phase_start", "phase": "analysis", "label": "Expert Analysis"}

    shared_q: asyncio.Queue = asyncio.Queue()
    full_texts: dict[str, str] = {}

    # Build portfolio lines for inline injection (plain text, no box-drawing chars)
    port_lines = ""
    if portfolio_ctx:
        port_lines = (
            "\n=== CLIENT PORTFOLIO (live from Robinhood) ===\n"
            + portfolio_ctx.strip().replace("━", "=").replace("\n\n", "\n")
            + "\n=== END PORTFOLIO ===\n"
        )

    for agent in AGENTS:
        prompt = (
            f"You are a {agent['name']}. "
            f"Analyze the investment question below using your expertise in "
            f"{agent['focus']}.\n\n"
            f"Do NOT introduce yourself. Do NOT ask for clarification. "
            f"Start your analysis immediately with specific findings.\n"
            f"{port_lines}\n"
            f"=== QUESTION TO ANALYZE ===\n{question}\n=== END QUESTION ===\n\n"
            f"End your analysis with a clear verdict: {agent['verdict']}.\n\n"
            f"Your {agent['name'].lower()} analysis:"
        )
        asyncio.create_task(_run_claude_stream(prompt, agent["id"], shared_q))

    done_count = 0
    while done_count < len(AGENTS):
        event = await shared_q.get()
        yield event
        if event.get("type") == "agent_done":
            full_texts[event["agent_id"]] = event.get("full_text", "")
            done_count += 1

    # ── Phase 2: CIO synthesis ────────────────────────────────────────────────
    yield {"type": "phase_start", "phase": "synthesis", "label": "Building Consensus"}

    analyses_block = "\n\n".join(
        f"=== {a['name'].upper()} ===\n{full_texts.get(a['id'], '(no response)')}"
        for a in AGENTS
    )
    judge_prompt = (
        f"{JUDGE['system']}\n\n"
        f"Do NOT introduce yourself. Write your recommendation immediately.\n"
        f"{port_lines}\n"
        f"=== CLIENT QUESTION ===\n{question}\n=== END QUESTION ===\n\n"
        f"=== FOUR EXPERT ANALYSES ===\n{analyses_block}\n=== END ANALYSES ===\n\n"
        f"Your consensus recommendation:"
    )

    judge_q: asyncio.Queue = asyncio.Queue()
    asyncio.create_task(_run_claude_stream(judge_prompt, JUDGE["id"], judge_q))

    while True:
        event = await judge_q.get()
        etype = event.get("type")
        if etype == "chunk":
            yield {"type": "synthesis_chunk", "text": event.get("text", "")}
        elif etype == "agent_done":
            yield {"type": "done", "consensus": event.get("full_text", "")}
            break
        elif etype in ("agent_start", "phase_start"):
            pass  # swallow internal judge events
        else:
            yield event
