"""
signals/engine.py
Combines technical indicators + AI sentiment + earnings proximity
into a final trade signal: LONG / SHORT / HOLD with confidence %.
"""
import logging
from datetime import date

from db.store import get_price_df, get_news, get_events, get_earnings_calendar, save_signal
from signals.technical import compute_signals
from signals.sentiment import score_news_sentiment

log = logging.getLogger(__name__)


def earnings_proximity_days(ticker: str) -> int | None:
    """Returns days until next earnings report, or None if not found."""
    calendar = get_earnings_calendar()
    for entry in calendar:
        if entry["ticker"] == ticker and entry.get("report_date"):
            try:
                report = date.fromisoformat(entry["report_date"])
                diff = (report - date.today()).days
                if 0 <= diff <= 30:
                    return diff
            except ValueError:
                pass
    return None


def generate_signal(ticker: str) -> dict:
    """
    Main signal generation function.
    Returns a complete signal dict ready to broadcast and store.
    """
    # ── 1. Technical analysis ────────────────────────────────────────────────
    df = get_price_df(ticker)
    tech = compute_signals(df)

    if "error" in tech:
        log.warning(f"Skipping signal for {ticker}: {tech['error']}")
        return {"ticker": ticker, "signal": "HOLD", "confidence": 0,
                "reason": "Insufficient price data", "tech": tech}

    # ── 2. Sentiment analysis ────────────────────────────────────────────────
    articles = get_news(ticker)
    events = get_events()
    sent = score_news_sentiment(ticker, articles, events)

    # ── 3. Earnings proximity risk adjustment ────────────────────────────────
    earnings_days = earnings_proximity_days(ticker)
    earnings_caution = earnings_days is not None and earnings_days <= 3

    # ── 4. Composite scoring ─────────────────────────────────────────────────
    # tech_score: -3.0 to +3.0
    # sent score: -1.0 to +1.0, weighted 1.5x
    total = tech["tech_score"] + (sent["score"] * 1.5)

    # Dampen near earnings (high volatility, unpredictable direction)
    if earnings_caution:
        total *= 0.5
        log.info(f"{ticker}: earnings in {earnings_days}d — signal dampened")

    # ── 5. Direction & confidence ─────────────────────────────────────────────
    # Max possible total = 3 + 1.5 = 4.5
    if total >= 1.8:
        signal = "LONG"
    elif total <= -1.8:
        signal = "SHORT"
    else:
        signal = "HOLD"

    # Confidence: map total abs value to 0–100%
    confidence = min(100, int(abs(total) / 4.5 * 100))

    # ── 6. Human-readable reason ─────────────────────────────────────────────
    reasons = []
    if tech["macd_cross_up"]:
        reasons.append("MACD bullish crossover")
    elif not tech["macd_bullish"]:
        reasons.append("MACD bearish")
    if tech["rsi"] < 30:
        reasons.append(f"RSI oversold ({tech['rsi']})")
    elif tech["rsi"] > 70:
        reasons.append(f"RSI overbought ({tech['rsi']})")
    if tech["price_above_vwap"]:
        reasons.append("Price above VWAP")
    else:
        reasons.append("Price below VWAP")
    if sent["sentiment"] != "neutral":
        reasons.append(f"News sentiment: {sent['sentiment']} ({sent['urgency']} urgency)")
    if earnings_caution:
        reasons.append(f"⚠ Earnings in {earnings_days}d — position sized down")

    result = {
        "ticker": ticker,
        "signal": signal,
        "confidence": confidence,
        "total_score": round(total, 2),
        "current_price": tech.get("current_price"),
        "tech": tech,
        "sentiment": sent,
        "earnings_days": earnings_days,
        "earnings_caution": earnings_caution,
        "reasons": reasons,
        "summary": sent.get("summary", ""),
        "catalysts": sent.get("catalysts", []),
        "risk_factors": sent.get("risk_factors", []),
    }

    save_signal(result)
    log.info(f"Signal for {ticker}: {signal} (conf={confidence}%, score={total:.2f})")
    return result
