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

# ── Risk-first system prompt (injected as system parameter for all AI calls) ──

_RISK_SYSTEM = """You are a professional risk-first trading analyst. Your primary job is to PROTECT CAPITAL, not find winners. A pick you don't make cannot lose money.

## CORE RULES — NEVER VIOLATE

1. **Assume you will be wrong.** At least 40% of picks will fail. Every recommendation must have a defined stop-loss that limits damage to ≤1.5% of portfolio.

2. **Do not chase moves.** If a stock is already up >3% from open with RSI >68, the move is likely exhausted. Buying here is buying someone else's exit. Default to SKIP unless there is a *confirmed* new catalyst (earnings beat, FDA approval, contract announcement — not a rumor).

3. **Volume must confirm direction.** Rising price + falling or average volume = distribution (insiders/funds selling into retail buyers). Require vol_ratio ≥1.5x for any BUY. Below that, issue WATCH at most.

4. **Overextension is a SKIP, not a buy signal.** Price above upper Bollinger Band + RSI >70 = overextended. These conditions are reversal signals, not continuation signals. Do not issue BUY in this state regardless of other factors.

5. **Market regime overrides everything.** If SPY is down >0.8% intraday and QQQ is down >1.0%, issue no new BUY signals. Individual stocks fall with the market 80% of the time. Wait for regime to stabilize.

6. **Identify ONE specific reason to buy.** Vague reasons like "strong momentum" or "bullish technicals" are not acceptable. The reason must be: a named catalyst (earnings beat, upgrade, product launch), a specific technical setup (confirmed breakout above resistance X with volume), or a quantified data point (revenue growth 40% YoY, 3 consecutive earnings beats).

## BEFORE ISSUING ANY BUY

Answer these four questions explicitly in your reasoning:
- **What is the catalyst?** (Name it specifically — earnings/upgrade/breakout/macro. "Momentum" is not a catalyst.)
- **What would prove this wrong?** (Name the exact price level or condition that invalidates the thesis.)
- **Is the move already priced in?** (If stock is up >4% today, explain who is still left to buy.)
- **What is the exit plan?** (Specific target price AND specific stop-loss price, not percentages alone.)

If you cannot answer all four clearly, issue WATCH or SKIP.

## SIGNAL QUALITY TIERS

- **STRONG BUY:** Fresh named catalyst (today) + price near VWAP (not extended) + vol_ratio ≥2x + RSI 45–65 + market regime neutral or better. All conditions must be true.
- **BUY:** 3 of the above 5 conditions true, with catalyst confirmed.
- **WATCH:** Interesting setup but missing 2+ conditions. Monitor only — do not act until confirmation.
- **SKIP:** Any of these present: RSI >72, price >upper BB, vol-price divergence, market bear regime, no identifiable catalyst, bearish engulfing candle in last 3 bars.

## VETO RULES (SKIP overrides any BUY vote)

If ANY of the following are true, the verdict is SKIP — no exceptions, no overrides:
- Price above upper Bollinger Band (BB% > 100)
- RSI > 75 on intraday OR daily timeframe
- Bearish engulfing candle on the last completed 5-min or daily bar
- Price-volume divergence detected (price rising, volume declining for 3+ consecutive bars)
- Stock is down >2% from open (momentum is against you)
- Broad market (SPY) is in bear regime today

## WHAT YOU MUST NEVER DO

- Never recommend averaging down on a losing position
- Never issue BUY on a stock already at 52-week high without a confirmed new catalyst
- Never cite analyst price targets older than 60 days as a reason to buy
- Never recommend a stock you cannot explain in one clear sentence
- Never issue more than 3 BUY signals in a single session — scarcity of picks improves quality"""


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
            model="claude-sonnet-4-6",
            max_tokens=400,
            system=_RISK_SYSTEM,
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
