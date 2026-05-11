"""
signals/sentiment.py
Uses Claude to score news sentiment for a ticker.
Returns structured JSON: sentiment direction, score, catalysts, risks.
"""
import os
import json
import logging

import anthropic

log = logging.getLogger(__name__)
_client = None


def get_client():
    global _client
    if _client is None:
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            return None
        _client = anthropic.Anthropic(api_key=key)
    return _client


def score_news_sentiment(ticker: str, articles: list[dict], events: list[dict]) -> dict:
    """
    Call Claude to produce a structured sentiment analysis.
    Falls back to neutral if no API key.
    """
    client = get_client()
    if not client:
        log.warning("No Anthropic key — returning neutral sentiment.")
        return _neutral(ticker)

    # Build context string
    headlines = [a.get("title", "") for a in articles[:8] if a.get("title")]
    event_descs = [e.get("description", "") for e in events if e.get("ticker") == ticker]

    if not headlines and not event_descs:
        return _neutral(ticker)

    context_lines = "\n".join(
        [f"- {h}" for h in headlines] + [f"[EVENT] {d}" for d in event_descs[:3]]
    )

    prompt = f"""You are a professional day trader's AI analyst. Analyze the following recent news and events for {ticker}.

News & Events (last 12 hours):
{context_lines}

Respond ONLY with a JSON object — no preamble, no markdown, no explanation:
{{
  "sentiment": "bullish" | "bearish" | "neutral",
  "score": <float between -1.0 (very bearish) and 1.0 (very bullish)>,
  "catalysts": ["<short bullet>", ...],
  "risk_factors": ["<short bullet>", ...],
  "urgency": "high" | "medium" | "low",
  "summary": "<one sentence trader-friendly summary>"
}}"""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        # Strip any accidental markdown fences
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        result["ticker"] = ticker
        return result
    except Exception as e:
        log.warning(f"Claude sentiment error for {ticker}: {e}")
        return _neutral(ticker)


def _neutral(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "sentiment": "neutral",
        "score": 0.0,
        "catalysts": [],
        "risk_factors": [],
        "urgency": "low",
        "summary": "Insufficient news data for sentiment analysis.",
    }
