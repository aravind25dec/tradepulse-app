"""
signalforge/live_context.py
Uses the Claude Code CLI (agent_engine.oneshot_claude) to pull a qualitative,
right-now read on a ticker: current price, recent news/sentiment, and a plain-
English candle/technical read.

This is a cross-check and narrative input only — the actual RSI/MACD/etc.
numbers fed to the model always come from features/build_features.py computed
on freshly fetched OHLCV, not from Claude's prose, so the numbers the model
sees stay consistent between training and prediction.
"""
import asyncio
import json
import logging
import re

import agent_engine

log = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """Give me a brief, current, real-world read on {ticker} stock right now.

Respond with ONLY a raw JSON object (no markdown fences, no explanation) in exactly this shape:
{{"ticker": "{ticker}", "current_price": <number or null>, "sentiment": "<bullish|bearish|neutral>", "sentiment_summary": "<1-2 sentence plain-English summary of recent news/sentiment>", "candle_read": "<1 sentence plain-English read of the most recent price action/candle shape>"}}

If you are unsure of the exact current price, use your best available knowledge and note the uncertainty inside sentiment_summary, but ticker/sentiment/sentiment_summary/candle_read must always be present."""

_FALLBACK = {
    "sentiment": "neutral",
    "sentiment_summary": "Live context unavailable — Claude Code CLI could not be reached.",
    "candle_read": "N/A",
    "current_price": None,
}


def _parse_json_object(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object found in Claude response: {text[:200]}")
    return json.loads(cleaned[start : end + 1])


async def get_live_snapshot(ticker: str) -> dict:
    """Best-effort — never raises. Falls back to a neutral placeholder snapshot
    (clearly labeled as unavailable) if the Claude CLI isn't installed, times out,
    or returns something unparseable, so a live-context hiccup never breaks
    /api/predict's actual numeric prediction."""
    ticker = ticker.upper().strip()
    if not agent_engine.claude_available():
        log.warning("Claude CLI not found — returning fallback live context for %s", ticker)
        return {"ticker": ticker, "available": False, **_FALLBACK}

    try:
        raw = await agent_engine.oneshot_claude(_PROMPT_TEMPLATE.format(ticker=ticker), timeout=45.0)
        parsed = _parse_json_object(raw)
        return {
            "ticker": ticker,
            "available": True,
            "current_price": parsed.get("current_price"),
            "sentiment": parsed.get("sentiment", "neutral"),
            "sentiment_summary": parsed.get("sentiment_summary", ""),
            "candle_read": parsed.get("candle_read", ""),
        }
    except Exception as exc:
        log.warning("Live context fetch failed for %s: %s", ticker, exc)
        return {"ticker": ticker, "available": False, "error": str(exc), **_FALLBACK}


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(json.dumps(asyncio.run(get_live_snapshot(t)), indent=2))
