"""
localai/engine.py
Offline investment analysis using a local Ollama LLM via LangChain.
Zero cloud API cost — runs entirely on your machine.

Setup (one-time):
  1. Install Ollama: https://ollama.ai  (or: winget install Ollama.Ollama)
  2. Pull a model:   ollama pull llama3.1:8b    (~4.7 GB, best quality)
                  or ollama pull phi3:mini       (~2.3 GB, faster)
                  or ollama pull mistral:7b      (~4.1 GB, good reasoning)
  3. Ollama auto-starts on boot. Verify: http://localhost:11434
"""

import asyncio
import logging
from typing import AsyncGenerator

import httpx

log = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:4b"

# ── Ollama health check ───────────────────────────────────────────────────────

async def check_ollama() -> dict:
    """Return {available, models} — models is a list of pulled model names."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{OLLAMA_BASE}/api/tags")
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                return {"available": True, "models": models}
    except Exception:
        pass
    return {"available": False, "models": []}


# ── Portfolio fetch from hotpicks cache ───────────────────────────────────────

async def fetch_portfolio() -> list[dict]:
    """
    Try to pull cached portfolio from hotpicks (port 8003).
    Returns [] if hotpicks is not running or MCP is not connected.
    """
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get("http://localhost:8003/api/debate/portfolio")
            if r.status_code == 200:
                return r.json().get("positions", [])
    except Exception:
        pass
    return []


async def fetch_current_prices(tickers: list[str]) -> dict[str, dict]:
    """
    Fetch live prices from Yahoo Finance v8 chart API — free, no key needed.
    Fires all ticker requests in parallel; completes in ~2-3s for 40 tickers.
    """
    if not tickers:
        return {}

    async with httpx.AsyncClient(
        timeout=8,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
    ) as c:
        async def _one(ticker: str):
            try:
                r = await c.get(
                    f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
                    f"?interval=1d&range=1d"
                )
                if r.status_code == 200:
                    meta = r.json()["chart"]["result"][0]["meta"]
                    return ticker, {
                        "price":   meta.get("regularMarketPrice", 0),
                        "chg_pct": meta.get("regularMarketChangePercent", 0),
                    }
            except Exception:
                pass
            return ticker, None

        results = await asyncio.gather(*[_one(t) for t in tickers])
        return {t: d for t, d in results if d is not None}


def _format_portfolio(positions: list[dict], prices: dict | None = None) -> str:
    if not positions:
        return "No portfolio data available."
    by_acct: dict[str, list] = {}
    for p in positions:
        acct = p.get("account_label", "Brokerage")
        by_acct.setdefault(acct, []).append(p)
    lines = []
    for acct, pos_list in by_acct.items():
        lines.append(f"{acct}:")
        for p in pos_list:
            ticker  = p["ticker"]
            avg     = float(p["avg_cost"])
            qty     = float(p["quantity"])
            line    = f"  {ticker:6s}  {qty:.4g} shares  |  avg cost ${avg:.2f}"
            if prices and ticker in prices:
                cur  = prices[ticker]["price"]
                chg  = prices[ticker]["chg_pct"]
                pl   = (cur - avg) / avg * 100 if avg else 0
                sign = "+" if pl >= 0 else ""
                line += f"  |  now ${cur:.2f} ({sign}{pl:.1f}% vs cost, {chg:+.1f}% today)"
            lines.append(line)
    return "\n".join(lines)


# ── Analysis prompt ───────────────────────────────────────────────────────────

ANALYSIS_TEMPLATE = """\
You are an expert investment analyst. Analyze the portfolio and question below.

PORTFOLIO:
{portfolio}
{past_context}
QUESTION: {question}

Provide your analysis in exactly 5 sections:

1. FUNDAMENTALS: Analyze earnings quality, revenue growth, P/E valuation, balance sheet.
2. TECHNICAL: Analyze price trend, momentum, key support and resistance levels.
3. RISK: Identify portfolio concentration issues, downside scenarios, stop-loss levels.
4. CATALYSTS: List specific upcoming events (earnings dates, macro events) in next 4 weeks.
5. DECISION: State BUY / SELL / HOLD / WAIT. Give one plain-English reason. List 3 action steps with specific prices.

Keep each section under 80 words. Be specific with stock tickers and price numbers.
Do not introduce yourself. Start immediately with "1. FUNDAMENTALS:"

Your analysis:"""


# ── Streaming inference ───────────────────────────────────────────────────────

async def stream_analysis(
    question: str,
    portfolio: list[dict],
    model: str = DEFAULT_MODEL,
    memories: list[dict] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream investment analysis directly from Ollama's /api/generate endpoint.
    Uses httpx streaming — no LangChain buffering, works with every Ollama model.
    """
    import json as _json

    # Fetch live prices so the model knows current P&L, not just avg cost
    tickers = [p["ticker"] for p in portfolio if p.get("ticker")]
    prices  = await fetch_current_prices(tickers)
    if prices:
        log.info("Live prices fetched for %d/%d tickers", len(prices), len(tickers))
    else:
        log.warning("Price fetch returned nothing — model will work from avg cost only")

    portfolio_text = _format_portfolio(portfolio, prices)

    # Compress portfolio if large — LLMLingua removes filler words while keeping
    # tickers, prices, and percentages intact
    try:
        from compression import compress
        portfolio_text = compress(portfolio_text, question=question, rate=0.65)
    except Exception:
        pass

    past_context = ""
    if memories:
        lines = ["RELEVANT PAST ANALYSES (use as reference, not as final answer):"]
        for m in memories:
            snippet = m["a"][:400].replace("\n", " ").strip()
            # Compress each memory snippet — keeps key analysis, drops filler
            try:
                from compression import compress
                snippet = compress(snippet, question=question, rate=0.6)
            except Exception:
                pass
            lines.append(f'Q: {m["q"]}\nA: {snippet}')
            lines.append("---")
        past_context = "\n" + "\n".join(lines) + "\n"

    full_prompt = ANALYSIS_TEMPLATE.format(
        portfolio=portfolio_text,
        past_context=past_context,
        question=question,
    )

    payload = {
        "model":   model,
        "messages": [{"role": "user", "content": full_prompt}],
        "stream":  True,
        "think":   False,   # disable thinking mode for qwen3 / deepseek-r1 etc.
        "options": {"temperature": 0.1, "num_predict": 700},
    }

    try:
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream(
                "POST",
                f"{OLLAMA_BASE}/api/chat",
                json=payload,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield f"⚠ Ollama error {resp.status_code}: {body.decode()[:200]}"
                    return

                in_think = False
                think_buf = ""
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        ev = _json.loads(line)
                    except Exception:
                        continue

                    token = ev.get("message", {}).get("content", "")
                    if token:
                        # Strip <think>…</think> blocks as a safety net
                        if in_think:
                            think_buf += token
                            if "</think>" in think_buf:
                                in_think = False
                                after = think_buf.split("</think>", 1)[1]
                                think_buf = ""
                                if after.strip():
                                    yield after
                        else:
                            if "<think>" in token:
                                in_think = True
                                before = token.split("<think>", 1)[0]
                                think_buf = token.split("<think>", 1)[1]
                                if before.strip():
                                    yield before
                                if "</think>" in think_buf:
                                    in_think = False
                                    after = think_buf.split("</think>", 1)[1]
                                    think_buf = ""
                                    if after.strip():
                                        yield after
                            else:
                                yield token

                    if ev.get("done"):
                        return
    except Exception as exc:
        log.error("Ollama stream error: %s", exc)
        yield f"\n\n⚠ Error: {exc}"
