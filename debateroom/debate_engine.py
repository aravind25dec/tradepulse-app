"""
debateroom/debate_engine.py

Multi-agent investment debate engine — powered by your Robinhood portfolio.

Flow:
  0. Pre-flight: fetch live portfolio via Robinhood MCP (if connected)
  1. Round 1 — Opening Arguments  (all 4 analysts in parallel, with portfolio context)
  2. Round 2 — Cross-Examination  (all 4 in parallel, with Round 1 context)
  3. Verdict  — The Arbiter synthesizes → final call with entry/sizing/horizon

Analyst roster:
  🐂 MaxBull    — Bullish analyst, finds buying opportunities in your portfolio
  🐻 BearMark   — Risk analyst, identifies risks specific to your holdings
  📊 DataDave   — Quant / technical analyst
  🌍 MacroMike  — Macro economist, reads the environment for your positions
  ⚖️  The Arbiter — Senior PM delivering the final personalised verdict
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import AsyncGenerator

log = logging.getLogger(__name__)

# ── Agent definitions ─────────────────────────────────────────────────────────

AGENTS = [
    {
        "id":    "bull",
        "name":  "MaxBull",
        "emoji": "🐂",
        "color": "#3fb950",
        "bg":    "#1a3a24",
        "title": "Bullish Analyst",
        "system": (
            "You are MaxBull, a passionate bullish equity analyst at a top hedge fund. "
            "You champion growth opportunities and always find strong reasons to buy great companies. "
            "When portfolio data is available, reference the user's actual holdings, costs, and P&L "
            "to make your argument personal and actionable. "
            "Be sharp and specific — cite catalysts, growth metrics, and position-level insights. "
            "4-6 sentences. "
            "End with exactly: STANCE: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]"
        ),
    },
    {
        "id":    "bear",
        "name":  "BearMark",
        "emoji": "🐻",
        "color": "#f85149",
        "bg":    "#3a1a1a",
        "title": "Risk Analyst",
        "system": (
            "You are BearMark, a sharp risk-focused analyst whose first duty is capital preservation. "
            "You identify overvaluation, concentration risk, macro headwinds, and hidden dangers. "
            "When portfolio data is available, call out position-specific risks — "
            "are they overweight in one sector? Underwater on a position? Overleveraged? "
            "Be precise — cite specific P/E, debt levels, or portfolio concentration numbers. "
            "4-6 sentences. "
            "End with exactly: STANCE: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]"
        ),
    },
    {
        "id":    "quant",
        "name":  "DataDave",
        "emoji": "📊",
        "color": "#58a6ff",
        "bg":    "#0d1f3c",
        "title": "Quantitative Analyst",
        "system": (
            "You are DataDave, a quantitative analyst who lives by charts, data, and statistics. "
            "Focus on price action, technical indicators (RSI, MACD, moving averages, Bollinger Bands), "
            "support/resistance levels, and volume patterns. "
            "When portfolio data is available, reference the user's average cost vs current price "
            "and identify key technical levels relevant to their specific entry points. "
            "Cite specific price levels or indicator readings. "
            "4-6 sentences. "
            "End with exactly: STANCE: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]"
        ),
    },
    {
        "id":    "macro",
        "name":  "MacroMike",
        "emoji": "🌍",
        "color": "#d29922",
        "bg":    "#3a2e10",
        "title": "Macro Economist",
        "system": (
            "You are MacroMike, a macro economist who evaluates everything through the lens of "
            "Fed policy, interest rates, dollar strength, sector rotation, and economic cycles. "
            "When portfolio data is available, assess whether the user's overall portfolio mix "
            "is well-positioned for the current macro regime — or if rebalancing is warranted. "
            "4-6 sentences. "
            "End with exactly: STANCE: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]"
        ),
    },
]

JUDGE = {
    "id":    "judge",
    "name":  "The Arbiter",
    "emoji": "⚖️",
    "color": "#bc8cff",
    "bg":    "#1e1a2e",
    "title": "Senior Portfolio Manager",
    "system": (
        "You are The Arbiter, a senior portfolio manager who has just presided over "
        "a heated debate between four specialist analysts — with access to the client's "
        "live Robinhood portfolio data. Your verdict must be personalised to their "
        "actual holdings, not generic market advice.\n\n"
        "Format your verdict EXACTLY as follows:\n\n"
        "## ⚖️ FINAL VERDICT: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]\n\n"
        "**Consensus:** [unanimous / majority / split / divided]\n\n"
        "**Why this call:** 2-3 sentences synthesising the key factors — reference the client's "
        "specific positions where relevant.\n\n"
        "**The bull case:** One sentence — strongest reason to be long.\n\n"
        "**The bear case:** One sentence — biggest risk to this call.\n\n"
        "**Recommended action:** Specific and personal — entry zone, position sizing relative to "
        "their portfolio, time horizon, and any existing positions they should adjust."
    ),
}


# ── Claude CLI discovery ──────────────────────────────────────────────────────

def _find_claude() -> str | None:
    for name in ("claude", "claude.cmd", "claude.exe"):
        p = shutil.which(name)
        if p:
            return p

    if sys.platform != "win32":
        return None

    candidates: list[Path] = []
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        candidates.append(Path(appdata) / "npm" / "claude.cmd")
    candidates.append(Path.home() / "AppData" / "Roaming" / "npm" / "claude.cmd")
    try:
        r = subprocess.run(
            "npm config get prefix", shell=True,
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            candidates.append(Path(r.stdout.strip()) / "claude.cmd")
    except Exception:
        pass

    for p in candidates:
        if p.is_file():
            log.info("claude found at: %s", p)
            return str(p)
    return None


def claude_available() -> bool:
    return _find_claude() is not None


def check_robinhood_mcp() -> bool:
    """Return True if the Robinhood MCP server is registered in Claude Code."""
    exe = _find_claude()
    if not exe:
        return False
    try:
        r = subprocess.run(
            [exe, "mcp", "list"],
            capture_output=True, text=True,
            shell=(sys.platform == "win32"),
            timeout=8,
        )
        return "robinhood" in r.stdout.lower()
    except Exception:
        return False


# ── Non-streaming helper: collect full response from one Claude call ──────────

async def _collect_response(prompt: str, timeout: float = 90.0) -> str:
    """
    Run a single prompt through Claude CLI (with MCP) and return the full text.
    Used for the portfolio pre-fetch step before the debate starts.
    """
    exe = _find_claude()
    if not exe:
        raise RuntimeError("Claude CLI not found")

    cmd = [
        exe, "--print", "--verbose", "--output-format", "stream-json",
        "--dangerously-skip-permissions", prompt,
    ]

    loop  = asyncio.get_event_loop()
    line_q: asyncio.Queue = asyncio.Queue()
    clean_env = os.environ.copy()
    clean_env.pop("ANTHROPIC_API_KEY", None)

    def _worker():
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=clean_env, shell=(sys.platform == "win32"),
            )
            for raw in proc.stdout:
                loop.call_soon_threadsafe(line_q.put_nowait, ("line", raw))
            proc.wait()
        except Exception as exc:
            loop.call_soon_threadsafe(line_q.put_nowait, ("err", str(exc)))
        finally:
            loop.call_soon_threadsafe(line_q.put_nowait, ("done", None))

    threading.Thread(target=_worker, daemon=True).start()

    full_text = ""
    last_len  = 0

    try:
        while True:
            tag, val = await asyncio.wait_for(line_q.get(), timeout=timeout)
            if tag == "done":
                break
            if tag == "err":
                log.warning("_collect_response error: %s", val)
                break
            line = val.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if evt.get("type") == "assistant":
                for block in evt.get("message", {}).get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        full = block.get("text", "")
                        if len(full) > last_len:
                            full_text += full[last_len:]
                            last_len = len(full)
    except asyncio.TimeoutError:
        log.warning("_collect_response timed out after %.0fs", timeout)

    return full_text


async def fetch_portfolio_context() -> str:
    """
    Use Claude CLI + Robinhood MCP to fetch the user's live portfolio.
    Returns a formatted text summary for injection into agent prompts.
    """
    prompt = (
        "Use the robinhood-trading MCP tools to fetch a complete snapshot of my Robinhood portfolio. "
        "I need the following for a live investment debate among my AI analysts:\n\n"
        "1. **Account overview**: total portfolio value, available buying power, "
        "   each account type (Brokerage, Traditional IRA, Roth IRA)\n"
        "2. **All current positions** across all accounts: ticker, number of shares, "
        "   average cost per share, current price, unrealized P&L ($ and %), account\n"
        "3. **Top winners and losers** by unrealized P&L %\n"
        "4. **Recent orders** (last 5): ticker, direction, status\n\n"
        "Format the output as a clean markdown summary. Use tables where helpful. "
        "Be precise with numbers. Do not omit any account or position."
    )
    log.info("Fetching Robinhood portfolio via MCP…")
    try:
        text = await _collect_response(prompt, timeout=90.0)
        if text.strip():
            log.info("Portfolio context fetched (%d chars)", len(text))
            return text.strip()
        return ""
    except Exception as exc:
        log.warning("Portfolio fetch failed: %s", exc)
        return ""


# ── Single-agent streaming ────────────────────────────────────────────────────

async def _stream_agent(
    agent: dict,
    prompt: str,
    queue: asyncio.Queue,
    phase: str,
    timeout: float = 120.0,
) -> None:
    """
    Launch one analyst's Claude CLI call in a background thread and push
    streaming events to the shared queue.

    Events pushed:
      {"type": "agent_start",  phase, agent_id, agent_name, emoji, color, bg, title}
      {"type": "chunk",        phase, agent_id, chunk}
      {"type": "agent_done",   phase, agent_id, full_text}
    """
    base = {
        "phase":      phase,
        "agent_id":   agent["id"],
        "agent_name": agent["name"],
        "emoji":      agent["emoji"],
        "color":      agent["color"],
        "bg":         agent["bg"],
        "title":      agent["title"],
    }

    await queue.put({**base, "type": "agent_start"})

    exe = _find_claude()
    if not exe:
        await queue.put({
            **base, "type": "agent_done",
            "full_text": f"⚠ Claude CLI not found — {agent['name']} unavailable.",
        })
        return

    full_prompt = f"{agent['system']}\n\n{prompt}"
    cmd = [
        exe, "--print", "--verbose", "--output-format", "stream-json",
        "--dangerously-skip-permissions", full_prompt,
    ]

    loop      = asyncio.get_event_loop()
    line_q: asyncio.Queue = asyncio.Queue()
    clean_env = os.environ.copy()
    clean_env.pop("ANTHROPIC_API_KEY", None)

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
                loop.call_soon_threadsafe(line_q.put_nowait, ("line", raw))
            proc.wait()
            if proc.returncode not in (0, None):
                err = proc.stderr.read().decode("utf-8", errors="replace").strip()
                if err:
                    loop.call_soon_threadsafe(line_q.put_nowait, ("err", err))
        except Exception as exc:
            loop.call_soon_threadsafe(line_q.put_nowait, ("err", str(exc)))
        finally:
            loop.call_soon_threadsafe(line_q.put_nowait, ("done", None))

    threading.Thread(target=_worker, daemon=True).start()

    full_text = ""
    last_len  = 0

    try:
        while True:
            tag, val = await asyncio.wait_for(line_q.get(), timeout=timeout)
            if tag == "done":
                break
            if tag == "err":
                log.warning("%s stderr: %s", agent["name"], val)
                if not full_text:
                    full_text = f"⚠ {agent['name']} error: {val}"
                break
            line = val.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if evt.get("type") == "assistant":
                for block in evt.get("message", {}).get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        full = block.get("text", "")
                        if len(full) > last_len:
                            chunk = full[last_len:]
                            full_text += chunk
                            last_len = len(full)
                            await queue.put({**base, "type": "chunk", "chunk": chunk})
    except asyncio.TimeoutError:
        log.warning("%s timed out after %.0fs", agent["name"], timeout)
        if not full_text:
            full_text = f"⚠ {agent['name']} did not respond within {timeout:.0f}s."

    await queue.put({**base, "type": "agent_done", "full_text": full_text})


# ── Portfolio context block builder ───────────────────────────────────────────

def _portfolio_block(portfolio_ctx: str) -> str:
    """Format the portfolio context for injection into agent prompts."""
    if not portfolio_ctx:
        return ""
    return (
        "═══════════════════════════════════════════════════\n"
        "📊 CLIENT'S LIVE ROBINHOOD PORTFOLIO (via MCP)\n"
        "═══════════════════════════════════════════════════\n\n"
        + portfolio_ctx
        + "\n\n═══════════════════════════════════════════════════\n\n"
        "Use the above REAL portfolio data to personalise your analysis. "
        "Reference the client's actual positions, costs, and P&L in your argument.\n\n"
    )


# ── Debate orchestrator ───────────────────────────────────────────────────────

async def run_debate(question: str, mcp_available: bool = False) -> AsyncGenerator[dict, None]:
    """
    Three-phase debate, optionally enriched with live Robinhood portfolio data.

    Phase 0 (if MCP)  — fetch live portfolio from Robinhood MCP
    Phase 1           — 4 analysts give opening arguments in parallel
    Phase 2           — 4 analysts cross-examine each other in parallel
    Phase 3           — Judge synthesises → final personalised verdict

    Yields SSE-ready event dicts.
    """

    # ── Phase 0: Fetch portfolio via MCP ─────────────────────────────────────
    portfolio_ctx = ""
    if mcp_available:
        yield {
            "type":    "status",
            "message": "📊 Fetching your live Robinhood portfolio via MCP…",
        }
        portfolio_ctx = await fetch_portfolio_context()
        if portfolio_ctx:
            yield {
                "type":    "portfolio_ready",
                "summary": portfolio_ctx,
            }
        else:
            yield {
                "type":    "status",
                "message": "⚠ Could not fetch portfolio — proceeding with general analysis.",
            }

    port_block = _portfolio_block(portfolio_ctx)

    # ── Round 1: Opening Arguments ────────────────────────────────────────────
    yield {
        "type":        "phase_start",
        "phase":       "opening",
        "label":       "Round 1 — Opening Arguments",
        "description": "Each analyst gives their independent take",
    }

    opening_prompt = (
        port_block
        + f'The investment question under debate: "{question}"\n\n'
        "Give your opening argument from your specialist perspective. "
        "Be specific and personal — reference the client's actual holdings if portfolio data is available. "
        "Cite numbers, catalysts, or technical levels relevant to your role. 4-6 sentences.\n\n"
        "End with: STANCE: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]"
    )

    q1: asyncio.Queue = asyncio.Queue()
    tasks1 = [
        asyncio.create_task(_stream_agent(agent, opening_prompt, q1, "opening"))
        for agent in AGENTS
    ]

    round1_texts: dict[str, str] = {}
    done1 = 0
    while done1 < len(AGENTS):
        event = await q1.get()
        yield event
        if event.get("type") == "agent_done":
            round1_texts[event["agent_id"]] = event.get("full_text", "")
            done1 += 1

    await asyncio.gather(*tasks1, return_exceptions=True)

    # ── Round 2: Cross-Examination ────────────────────────────────────────────
    yield {
        "type":        "phase_start",
        "phase":       "rebuttal",
        "label":       "Round 2 — Cross-Examination",
        "description": "Analysts challenge each other's positions",
    }

    round1_ctx = "\n\n".join(
        f"{a['emoji']} **{a['name']}** ({a['title']}):\n{round1_texts.get(a['id'], '[no response]')}"
        for a in AGENTS
    )

    rebuttal_prompt = (
        port_block
        + f'Investment question: "{question}"\n\n'
        f"=== Round 1 — What the other analysts said ===\n\n{round1_ctx}\n\n"
        "=== Your task for Round 2 ===\n"
        "1. Identify the argument most opposed to yours and challenge it directly (2-3 sentences). "
        "   Where relevant, refer to the client's actual portfolio positions.\n"
        "2. Strengthen your own key point with one additional piece of evidence.\n"
        "3. If another analyst made a genuinely valid point, briefly acknowledge it.\n\n"
        "Stay in character. Be direct and sharp. 4-6 sentences total.\n"
        "End with: REVISED STANCE: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]"
    )

    q2: asyncio.Queue = asyncio.Queue()
    tasks2 = [
        asyncio.create_task(_stream_agent(agent, rebuttal_prompt, q2, "rebuttal"))
        for agent in AGENTS
    ]

    round2_texts: dict[str, str] = {}
    done2 = 0
    while done2 < len(AGENTS):
        event = await q2.get()
        yield event
        if event.get("type") == "agent_done":
            round2_texts[event["agent_id"]] = event.get("full_text", "")
            done2 += 1

    await asyncio.gather(*tasks2, return_exceptions=True)

    # ── Verdict: The Arbiter ──────────────────────────────────────────────────
    yield {
        "type":        "phase_start",
        "phase":       "verdict",
        "label":       "Final Verdict",
        "description": "The Arbiter synthesizes the debate",
    }

    transcript = (
        f"QUESTION: {question}\n\n"
        "═══ ROUND 1 — OPENING ARGUMENTS ═══\n\n"
        + "\n\n".join(
            f"{a['emoji']} {a['name']} ({a['title']}):\n{round1_texts.get(a['id'], '[no response]')}"
            for a in AGENTS
        )
        + "\n\n═══ ROUND 2 — CROSS-EXAMINATION ═══\n\n"
        + "\n\n".join(
            f"{a['emoji']} {a['name']}:\n{round2_texts.get(a['id'], '[no response]')}"
            for a in AGENTS
        )
    )

    verdict_prompt = (
        port_block
        + "Here is the complete debate transcript:\n\n"
        + transcript
        + "\n\nDeliver your final personalised verdict. "
        "Reference the client's actual portfolio where it adds value to the call. "
        "Be decisive and specific."
    )

    q3: asyncio.Queue = asyncio.Queue()
    task3 = asyncio.create_task(_stream_agent(JUDGE, verdict_prompt, q3, "verdict"))

    while True:
        event = await q3.get()
        yield event
        if event.get("type") == "agent_done":
            break

    await task3
    yield {"type": "done"}
