"""
screener_engine.py
Screens a universe of stocks/crypto using:
  - yfinance for historical OHLCV data
  - Technical analysis (RSI, MACD, VWAP, Bollinger, EMA)
  - Claude sentiment analysis on recent news
  - FMP MCP data when available (analyst ratings, price targets)
Returns ranked list with top 10 best and bottom 10 worst day-trade candidates.
"""
import os
import sys
import json
import logging
import asyncio
from datetime import datetime, timedelta

import httpx
import numpy as np
import pandas as pd
import anthropic

log = logging.getLogger(__name__)

# ── Default screener universe ────────────────────────────────────────────────
DEFAULT_UNIVERSE = [
    # Large-cap US stocks
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "NFLX", "CRM", "ORCL", "INTC", "QCOM", "AVGO", "MU", "NOW",
    "UBER", "SHOP", "SNOW", "PLTR", "COIN", "HOOD", "RIVN", "LCID",
    # High-beta / volatile names popular for day trading
    "SOFI", "MARA", "RIOT", "GME", "AMC", "BBBY", "SPCE", "RBLX",
    "DKNG", "PENN", "CHPT", "BLNK", "PLUG", "FCEL", "BE",
    # ETFs
    "SPY", "QQQ", "IWM", "SOXS", "SOXL", "TQQQ", "SQQQ",
    # Crypto proxies
    "BTC-USD", "ETH-USD", "SOL-USD",
]

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

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

# ── Direct Yahoo Finance API (bypasses yfinance session issues) ───────────────

YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 TradePulse/1.0",
    "Accept": "application/json",
}

async def _fetch_ohlcv(ticker: str, client: httpx.AsyncClient) -> "pd.DataFrame | None":
    """Fetch 60d daily OHLCV from Yahoo Chart API without using yfinance."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": "60d", "interval": "1d", "includePrePost": "false", "events": "div,splits"}
    try:
        r = await client.get(url, params=params, headers=YAHOO_HEADERS, timeout=15.0)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        log.debug(f"Yahoo chart fetch failed for {ticker}: {e}")
        return None

    chart = payload.get("chart", {})
    if chart.get("error"):
        return None
    results = chart.get("result") or []
    if not results:
        return None

    item = results[0]
    timestamps = item.get("timestamp") or []
    indicators = item.get("indicators", {})
    quote = (indicators.get("quote") or [{}])[0]
    adj = (indicators.get("adjclose") or [{}])[0].get("adjclose") or []

    closes = adj if adj else (quote.get("close") or [])
    volumes = quote.get("volume") or []

    rows = []
    for i, ts in enumerate(timestamps):
        if i >= len(closes) or closes[i] is None:
            continue
        rows.append({
            "Close": float(closes[i]),
            "Volume": float(volumes[i]) if i < len(volumes) and volumes[i] is not None else 0.0,
        })
    if len(rows) < 26:
        return None

    return pd.DataFrame(rows)


# ── Technical analysis ────────────────────────────────────────────────────────

def compute_technicals(df: pd.DataFrame) -> dict:
    """Compute RSI, MACD, VWAP, Bollinger, EMA from OHLCV DataFrame."""
    if df is None or df.empty or len(df) < 26:
        return {"error": "insufficient data", "tech_score": 0}

    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_val = float(macd_line.iloc[-1])
    macd_hist = float((macd_line - signal_line).iloc[-1])
    macd_bullish = bool(macd_line.iloc[-1] > signal_line.iloc[-1])
    macd_cross_up = bool(
        macd_line.iloc[-1] > signal_line.iloc[-1]
        and macd_line.iloc[-2] <= signal_line.iloc[-2]
    ) if len(macd_line) >= 2 else False
    macd_cross_down = bool(
        macd_line.iloc[-1] < signal_line.iloc[-1]
        and macd_line.iloc[-2] >= signal_line.iloc[-2]
    ) if len(macd_line) >= 2 else False

    # VWAP
    if volume.sum() > 0:
        vwap = float((close * volume).cumsum().iloc[-1] / volume.cumsum().iloc[-1])
    else:
        vwap = float(close.mean())
    price_above_vwap = bool(close.iloc[-1] > vwap)

    # Bollinger Bands (20, 2)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = float((sma20 + 2 * std20).iloc[-1])
    bb_lower = float((sma20 - 2 * std20).iloc[-1])
    bb_mid = float(sma20.iloc[-1])
    current_price = float(close.iloc[-1])
    prev_price = float(close.iloc[-2]) if len(close) >= 2 else current_price
    bb_range = (bb_upper - bb_lower) or 1e-9
    bb_pct = round((current_price - bb_lower) / bb_range * 100, 1)

    # EMA 9/21
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema_bullish = bool(ema9.iloc[-1] > ema21.iloc[-1])

    # 1-day & 5-day price change
    change_1d = round((current_price - prev_price) / (prev_price or 1) * 100, 2)
    price_5d_ago = float(close.iloc[-5]) if len(close) >= 5 else prev_price
    change_5d = round((current_price - price_5d_ago) / (price_5d_ago or 1) * 100, 2)

    # Average volume (10d)
    avg_vol_10d = float(volume.rolling(10).mean().iloc[-1]) if len(volume) >= 10 else float(volume.mean())
    today_vol = float(volume.iloc[-1])
    vol_ratio = round(today_vol / avg_vol_10d, 2) if avg_vol_10d > 0 else 1.0

    # Composite technical score (-5 to +5)
    score = 0.0
    if rsi < 25:
        score += 2.0   # deeply oversold
    elif rsi < 35:
        score += 1.0
    elif rsi < 45:
        score += 0.5
    elif rsi > 75:
        score -= 2.0   # deeply overbought
    elif rsi > 65:
        score -= 1.0
    elif rsi > 55:
        score -= 0.5

    if macd_cross_up:
        score += 1.5
    elif macd_bullish:
        score += 0.5
    elif macd_cross_down:
        score -= 1.5
    else:
        score -= 0.5

    if price_above_vwap:
        score += 0.5
    else:
        score -= 0.5

    if bb_pct < 5:
        score += 1.0   # near lower band = potential bounce
    elif bb_pct < 15:
        score += 0.3
    elif bb_pct > 95:
        score -= 1.0   # near upper band = extended
    elif bb_pct > 85:
        score -= 0.3

    if ema_bullish:
        score += 0.5
    else:
        score -= 0.5

    # Volume confirmation
    if vol_ratio > 2.0 and macd_bullish:
        score += 0.5   # high volume with bullish MACD
    elif vol_ratio > 2.0 and not macd_bullish:
        score -= 0.5   # high volume with bearish MACD (distribution)

    return {
        "rsi": round(rsi, 1),
        "macd": round(macd_val, 4),
        "macd_hist": round(macd_hist, 4),
        "macd_bullish": macd_bullish,
        "macd_cross_up": macd_cross_up,
        "macd_cross_down": macd_cross_down,
        "vwap": round(vwap, 2),
        "price_above_vwap": price_above_vwap,
        "bb_upper": round(bb_upper, 2),
        "bb_lower": round(bb_lower, 2),
        "bb_mid": round(bb_mid, 2),
        "bb_pct": bb_pct,
        "ema_bullish": ema_bullish,
        "current_price": round(current_price, 4),
        "change_1d": change_1d,
        "change_5d": change_5d,
        "avg_vol_10d": int(avg_vol_10d),
        "today_vol": int(today_vol),
        "vol_ratio": vol_ratio,
        "tech_score": round(score, 2),
    }


# ── News & sentiment ──────────────────────────────────────────────────────────

async def fetch_news(ticker: str, client: httpx.AsyncClient) -> list[str]:
    """Fetch recent headlines for a ticker."""
    if not NEWS_API_KEY:
        return []
    base = ticker.split("-")[0]
    try:
        r = await client.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": base,
                "sortBy": "publishedAt",
                "apiKey": NEWS_API_KEY,
                "pageSize": 5,
                "language": "en",
                "from": (datetime.utcnow() - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S"),
            },
            timeout=8,
        )
        articles = r.json().get("articles", [])
        return [a["title"] for a in articles if a.get("title") and "[Removed]" not in a["title"]][:5]
    except Exception as e:
        log.debug(f"News fetch error for {ticker}: {e}")
        return []


def score_sentiment(ticker: str, headlines: list[str]) -> dict:
    """Use Claude to score sentiment. Returns score dict."""
    if not ANTHROPIC_API_KEY or not headlines:
        return {"sentiment": "neutral", "score": 0.0, "summary": "No news data", "urgency": "low"}

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    context = "\n".join(f"- {h}" for h in headlines)

    prompt = f"""You are a professional day trader's AI analyst. Analyze these recent headlines for {ticker}.

Headlines:
{context}

Respond ONLY with a JSON object — no preamble, no markdown:
{{
  "sentiment": "bullish" | "bearish" | "neutral",
  "score": <float -1.0 to 1.0>,
  "summary": "<one sentence trader-friendly summary>",
  "urgency": "high" | "medium" | "low",
  "catalysts": ["<bullet>"],
  "risk_factors": ["<bullet>"]
}}"""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=_RISK_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        log.debug(f"Sentiment error for {ticker}: {e}")
        return {"sentiment": "neutral", "score": 0.0, "summary": "Analysis unavailable", "urgency": "low"}


# ── FMP integration (when MCP connected, replace with MCP tool calls) ─────────

async def fetch_fmp_data(ticker: str, fmp_api_key: str, client: httpx.AsyncClient) -> dict:
    """
    Fetch analyst price target and recommendation from FMP.
    When the FMP MCP is connected, replace these HTTP calls with:
      mcp.call("analyst", {"symbol": ticker})
    """
    if not fmp_api_key:
        return {}
    try:
        # Analyst recommendations
        r = await client.get(
            f"https://financialmodelingprep.com/api/v3/analyst-stock-recommendations/{ticker}",
            params={"apikey": fmp_api_key, "limit": 1},
            timeout=8,
        )
        rec_data = r.json()
        analyst_rec = rec_data[0].get("analystRatingsbuy", 0) if rec_data else 0

        # Price target
        pt = await client.get(
            f"https://financialmodelingprep.com/api/v4/price-target-consensus",
            params={"symbol": ticker, "apikey": fmp_api_key},
            timeout=8,
        )
        pt_data = pt.json()
        target_price = pt_data[0].get("targetConsensus") if pt_data else None

        return {
            "analyst_buy_count": analyst_rec,
            "price_target": target_price,
        }
    except Exception:
        return {}


# ── Core screening ────────────────────────────────────────────────────────────

async def screen_ticker(
    ticker: str,
    http_client: httpx.AsyncClient,
    fmp_api_key: str = "",
) -> dict | None:
    """Download data and score a single ticker. Returns None on failure."""
    try:
        # Use direct Yahoo Chart API (same approach as backend/ingestion/prices.py)
        # to avoid yfinance session / cookie failures.
        df = await _fetch_ohlcv(ticker, http_client)
        if df is None:
            return None

        tech = compute_technicals(df)
        if "error" in tech:
            return None

        # News sentiment
        headlines = await fetch_news(ticker, http_client)
        sent = score_sentiment(ticker, headlines)

        # FMP extras (optional)
        fmp = await fetch_fmp_data(ticker, fmp_api_key, http_client) if fmp_api_key else {}

        # Composite score: tech (-5 to +5) + sentiment (-1.5 to +1.5)
        sent_contribution = sent.get("score", 0.0) * 1.5
        total_score = tech["tech_score"] + sent_contribution

        # FMP analyst boost (+0.5 if strong buy consensus)
        if fmp.get("analyst_buy_count", 0) > 5:
            total_score += 0.3

        # Signal classification
        if total_score >= 2.0:
            signal = "STRONG BUY"
        elif total_score >= 1.0:
            signal = "BUY"
        elif total_score <= -2.0:
            signal = "STRONG AVOID"
        elif total_score <= -1.0:
            signal = "AVOID"
        else:
            signal = "NEUTRAL"

        confidence = min(100, int(abs(total_score) / 6.5 * 100))

        return {
            "ticker": ticker,
            "signal": signal,
            "confidence": confidence,
            "total_score": round(total_score, 2),
            "tech": tech,
            "sentiment": sent,
            "fmp": fmp,
            "headlines": headlines[:3],
            "screened_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        log.warning(f"screen_ticker failed for {ticker}: {e}")
        return None


async def run_screener(universe: list[str] | None = None, fmp_api_key: str = "") -> dict:
    """
    Screen the full universe. Returns:
      { "top10": [...], "bottom10": [...], "all": [...], "screened_at": "..." }
    """
    tickers = universe or DEFAULT_UNIVERSE
    log.info(f"Screening {len(tickers)} tickers...")

    # Semaphore limits yfinance to 1 concurrent download to avoid rate-limit bans.
    dl_sem = asyncio.Semaphore(1)

    async def _screen_with_sem(ticker, client):
        async with dl_sem:
            result = await screen_ticker(ticker, client, fmp_api_key)
            await asyncio.sleep(0.4)  # small gap between each download
            return result

    async with httpx.AsyncClient() as client:
        tasks = [_screen_with_sem(t, client) for t in tickers]
        all_results = await asyncio.gather(*tasks, return_exceptions=False)
        results = [r for r in all_results if r is not None]

    if not results:
        return {"top10": [], "bottom10": [], "all": [], "total_screened": 0, "screened_at": datetime.utcnow().isoformat()}

    # Sort by total_score descending
    ranked = sorted(results, key=lambda x: x["total_score"], reverse=True)

    return {
        "top10": ranked[:10],
        "bottom10": ranked[-10:][::-1],   # worst 10, worst first
        "all": ranked,
        "total_screened": len(ranked),
        "screened_at": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(run_screener())
    print(f"Screened {result['total_screened']} tickers")
    print("\n=== TOP 10 ===")
    for r in result["top10"]:
        print(f"  {r['ticker']:12s} {r['signal']:12s}  score={r['total_score']:+.2f}  conf={r['confidence']}%")
    print("\n=== BOTTOM 10 ===")
    for r in result["bottom10"]:
        print(f"  {r['ticker']:12s} {r['signal']:12s}  score={r['total_score']:+.2f}  conf={r['confidence']}%")
