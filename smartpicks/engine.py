"""
smartpicks/engine.py
Fundamental catalyst scanner — finds reliable swing trade opportunities based on
earnings beats, revenue growth, M&A activity, analyst consensus, and technical
confirmation. Swing horizon: days to months (vs Hot Picks' minutes to hours).
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta

import anthropic
import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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

# ── Universe ──────────────────────────────────────────────────────────────────
SMART_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Growth / fintech
    "PLTR", "SOFI", "COIN", "HOOD", "UPST", "MSTR",
    # Semis
    "AMD", "INTC", "MU", "ASML", "TSM", "ARM", "SMCI",
    # Healthcare / pharma
    "LLY", "ABBV", "MRNA", "BNTX", "PFE",
    # Finance
    "JPM", "GS", "BAC", "V", "MA",
    # Energy
    "XOM", "CVX", "OXY",
    # Consumer / retail
    "SBUX", "NKE", "HD", "WMT",
    # Aerospace / defense / space
    "LMT", "RTX", "RKLB", "JOBY",
    # EV / transport
    "RIVN", "F", "GM",
    # Cloud / SaaS
    "CRM", "SNOW", "NET",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

_MA_KEYWORDS = [
    "acqui", "merger", "buyout", "takeover", "acquisition",
    "purchase agreement", "offer to buy", "buying out", "combine with",
    "deal to buy", "bid for",
]

# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_cap(market_cap):
    if not market_cap:
        return "N/A"
    if market_cap >= 1e12:
        return f"${market_cap / 1e12:.1f}T"
    if market_cap >= 1e9:
        return f"${market_cap / 1e9:.1f}B"
    if market_cap >= 1e6:
        return f"${market_cap / 1e6:.1f}M"
    return f"${market_cap:.0f}"


def _cap_tier(market_cap):
    if not market_cap:
        return "unknown"
    if market_cap >= 200e9:
        return "mega"
    if market_cap >= 10e9:
        return "large"
    if market_cap >= 2e9:
        return "mid"
    return "small"


# ── Data fetching ─────────────────────────────────────────────────────────────

async def fetch_fundamentals(ticker: str, client: httpx.AsyncClient) -> dict | None:
    """Fetch fundamental data from Yahoo Finance quoteSummary API."""
    try:
        url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
        params = {
            "modules": (
                "summaryDetail,defaultKeyStatistics,"
                "financialData,calendarEvents,earningsHistory,assetProfile"
            )
        }
        resp = await client.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        result = (data.get("quoteSummary") or {}).get("result")
        if not result:
            return None
        r = result[0]

        def safe(module, key, default=None):
            m = r.get(module) or {}
            v = m.get(key) or {}
            if isinstance(v, dict):
                return v.get("raw", default)
            return v if v is not None else default

        # Last 4 quarters EPS history
        earnings_hist = []
        for h in ((r.get("earningsHistory") or {}).get("history") or []):
            eps_act = ((h.get("epsActual") or {}).get("raw"))
            eps_est = ((h.get("epsEstimate") or {}).get("raw"))
            surp    = ((h.get("surprisePercent") or {}).get("raw"))
            qtr     = ((h.get("quarter") or {}).get("fmt", ""))
            if eps_act is not None and eps_est is not None:
                earnings_hist.append({
                    "quarter":      qtr,
                    "eps_actual":   round(float(eps_act), 2),
                    "eps_estimate": round(float(eps_est), 2),
                    "surprise_pct": round(float(surp or 0) * 100, 1),
                    "beat":         float(eps_act) >= float(eps_est),
                })

        # Next earnings date (Unix timestamp)
        cal = (r.get("calendarEvents") or {}).get("earnings") or {}
        earnings_dates = cal.get("earningsDate") or []
        next_earnings_ts = None
        if earnings_dates:
            raw_ts = (earnings_dates[0] or {}).get("raw")
            if raw_ts:
                next_earnings_ts = int(raw_ts)

        # Earnings estimate for next quarter
        earnings_avg = ((cal.get("earningsAverage") or {}).get("raw"))
        earnings_low = ((cal.get("earningsLow") or {}).get("raw"))
        earnings_high = ((cal.get("earningsHigh") or {}).get("raw"))

        return {
            "company_name":        (r.get("assetProfile") or {}).get("longName") or ticker,
            "sector":              safe("assetProfile", "sector") or "Unknown",
            "industry":            safe("assetProfile", "industry") or "Unknown",
            "market_cap":          safe("summaryDetail", "marketCap"),
            "trailing_pe":         safe("summaryDetail", "trailingPE"),
            "forward_pe":          safe("summaryDetail", "forwardPE"),
            "fifty_two_wk_high":   safe("summaryDetail", "fiftyTwoWeekHigh"),
            "fifty_two_wk_low":    safe("summaryDetail", "fiftyTwoWeekLow"),
            "avg_volume":          safe("summaryDetail", "averageVolume"),
            "revenue_growth":      safe("financialData", "revenueGrowth"),  # decimal
            "earnings_qtr_growth": safe("defaultKeyStatistics", "earningsQuarterlyGrowth"),
            "gross_margins":       safe("financialData", "grossMargins"),
            "operating_margins":   safe("financialData", "operatingMargins"),
            "rec_mean":            safe("financialData", "recommendationMean"),
            "rec_key":             (r.get("financialData") or {}).get("recommendationKey") or "none",
            "target_mean_price":   safe("financialData", "targetMeanPrice"),
            "target_high_price":   safe("financialData", "targetHighPrice"),
            "target_low_price":    safe("financialData", "targetLowPrice"),
            "analyst_count":       safe("financialData", "numberOfAnalystOpinions") or 0,
            "next_earnings_ts":    next_earnings_ts,
            "next_earnings_avg":   round(float(earnings_avg), 2) if earnings_avg else None,
            "next_earnings_low":   round(float(earnings_low), 2) if earnings_low else None,
            "next_earnings_high":  round(float(earnings_high), 2) if earnings_high else None,
            "earnings_history":    earnings_hist,
        }
    except Exception as exc:
        logger.debug("Fundamentals fetch failed %s: %s", ticker, exc)
        return None


async def fetch_daily(ticker: str, client: httpx.AsyncClient) -> pd.DataFrame | None:
    """Fetch 3-month daily OHLCV from Yahoo Finance chart API."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"interval": "1d", "range": "3mo", "includePrePost": "false"}
        resp = await client.get(url, params=params, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        result = (data.get("chart") or {}).get("result")
        if not result:
            return None
        r = result[0]
        q = r["indicators"]["quote"][0]
        df = pd.DataFrame({
            "ts":     r.get("timestamp", []),
            "open":   q.get("open", []),
            "high":   q.get("high", []),
            "low":    q.get("low", []),
            "close":  q.get("close", []),
            "volume": q.get("volume", []),
        }).dropna()
        df["ts"] = pd.to_datetime(df["ts"], unit="s")
        return df if len(df) >= 10 else None
    except Exception as exc:
        logger.debug("Daily fetch failed %s: %s", ticker, exc)
        return None


async def fetch_news(ticker: str, api_key: str, client: httpx.AsyncClient) -> list[str]:
    if not api_key:
        return []
    try:
        since = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%S")
        resp = await client.get(
            "https://newsapi.org/v2/everything",
            params={"q": ticker, "apiKey": api_key, "pageSize": 8,
                    "sortBy": "publishedAt", "from": since},
            timeout=8,
        )
        articles = resp.json().get("articles", [])
        return [a.get("title", "") for a in articles if a.get("title")]
    except Exception:
        return []


async def fetch_market_context(client: httpx.AsyncClient) -> dict:
    """SPY + QQQ intraday to determine current market regime."""
    try:
        async def _intraday(sym):
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
            r = await client.get(url, params={"interval": "5m", "range": "1d"}, timeout=10)
            r.raise_for_status()
            d = r.json()
            res = (d.get("chart") or {}).get("result")
            if not res:
                return None
            q = res[0]["indicators"]["quote"][0]
            df = pd.DataFrame({
                "open": q.get("open", []), "close": q.get("close", []),
                "high": q.get("high", []), "low": q.get("low", []),
                "volume": q.get("volume", []),
            }).dropna()
            return df if len(df) >= 3 else None

        spy_df, qqq_df = await asyncio.gather(_intraday("SPY"), _intraday("QQQ"))
        spy_chg = qqq_chg = 0.0
        if spy_df is not None:
            sc, so = spy_df["close"].to_numpy()[-1], spy_df["open"].to_numpy()[0]
            spy_chg = (sc - so) / so * 100 if so > 0 else 0.0
        if qqq_df is not None:
            qc, qo = qqq_df["close"].to_numpy()[-1], qqq_df["open"].to_numpy()[0]
            qqq_chg = (qc - qo) / qo * 100 if qo > 0 else 0.0
        avg = (spy_chg + qqq_chg) / 2.0
        regime = "bull" if avg >= 0.5 else "bear" if avg <= -0.75 else "caution" if avg <= -0.25 else "neutral"
        return {"regime": regime, "spy_chg": round(spy_chg, 2), "qqq_chg": round(qqq_chg, 2)}
    except Exception as exc:
        logger.warning("Market context failed: %s", exc)
        return {"regime": "neutral", "spy_chg": 0.0, "qqq_chg": 0.0}


async def fetch_fmp_dynamic_tickers(client: httpx.AsyncClient) -> list[str]:
    """Augment the scan universe with fundamental catalyst tickers.

    Sources (all attempted, no API key required for Yahoo sources):
      1. Yahoo Finance trending tickers (free, always runs)
      2. Yahoo Finance upcoming_earnings screener (free, always runs)
      3. FMP earning_calendar next 14 days (free FMP tier, runs if FMP_API_KEY set)
      4. FMP upgrades-downgrades recent (paid FMP tier, silently skips on 403)

    Returns tickers NOT already in SMART_UNIVERSE, capped at 20 additions.
    """
    candidates: set[str] = set()

    def _valid_sym(sym: str) -> bool:
        return bool(sym) and sym.isalpha() and 1 <= len(sym) <= 5

    # 1 ── Yahoo Finance trending tickers (no key needed) ──────────────────────
    try:
        resp = await client.get(
            "https://query1.finance.yahoo.com/v1/finance/trending/US",
            params={"count": 25, "useQuotes": "true"},
            timeout=10,
        )
        resp.raise_for_status()
        for item in (resp.json().get("finance", {})
                               .get("result", [{}])[0]
                               .get("quotes", [])):
            sym = item.get("symbol", "")
            if _valid_sym(sym):
                candidates.add(sym.upper())
    except Exception as exc:
        logger.debug("Yahoo trending failed: %s", exc)

    # 2 ── Yahoo Finance upcoming earnings screener (no key needed) ────────────
    try:
        resp = await client.get(
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved",
            params={"scrIds": "upcoming_earnings", "count": 25,
                    "formatted": "false", "lang": "en-US", "region": "US"},
            timeout=10,
        )
        if resp.status_code == 200:
            for item in (resp.json().get("finance", {})
                                    .get("result", [{}])[0]
                                    .get("quotes", [])):
                sym = item.get("symbol", "")
                if _valid_sym(sym):
                    candidates.add(sym.upper())
    except Exception as exc:
        logger.debug("Yahoo upcoming_earnings screener failed: %s", exc)

    # 3 & 4 ── FMP (optional enhancement, requires key) ────────────────────────
    api_key = os.getenv("FMP_API_KEY", "")
    if api_key:
        base_url = "https://financialmodelingprep.com/api/v3"

        # Earning calendar (available on free FMP plan)
        try:
            from_date = datetime.utcnow().strftime("%Y-%m-%d")
            to_date   = (datetime.utcnow() + timedelta(days=14)).strftime("%Y-%m-%d")
            resp = await client.get(
                f"{base_url}/earning_calendar",
                params={"from": from_date, "to": to_date, "apikey": api_key},
                timeout=12,
            )
            if resp.status_code == 200:
                for item in (resp.json() or []):
                    sym = item.get("symbol", "")
                    if _valid_sym(sym) and item.get("epsEstimated") is not None:
                        candidates.add(sym.upper())
        except Exception as exc:
            logger.debug("FMP earning_calendar failed: %s", exc)

        # Analyst upgrades (paid FMP plan — silently skips on 403)
        try:
            for page in (0, 1):
                resp = await client.get(
                    f"{base_url}/upgrades-downgrades",
                    params={"page": page, "apikey": api_key},
                    timeout=10,
                )
                if resp.status_code == 200:
                    for item in (resp.json() or []):
                        sym    = item.get("symbol", "")
                        action = (item.get("action") or "").lower()
                        if _valid_sym(sym) and "upgrade" in action:
                            candidates.add(sym.upper())
        except Exception as exc:
            logger.debug("FMP upgrades-downgrades failed: %s", exc)

    new_tickers = [t for t in candidates if t not in set(SMART_UNIVERSE)]
    if new_tickers:
        logger.info(
            "Dynamic universe: +%d tickers added (trending/earnings/upgrades) — %s",
            len(new_tickers), new_tickers[:10],
        )
    return new_tickers[:20]


# ── Technical indicator helpers (daily timeframe) ─────────────────────────────

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    k = 2.0 / (period + 1)
    out = np.empty(len(arr), dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    n = min(period, len(gains))
    ag = float(np.mean(gains[-n:]))
    al = float(np.mean(losses[-n:]))
    if al == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def _macd_daily(closes: np.ndarray) -> dict:
    empty = {"macd_bullish": True, "macd_trend": "flat", "macd": 0.0, "macd_hist": 0.0}
    if len(closes) < 12:
        return empty
    fast = _ema(closes, min(12, len(closes)))
    slow = _ema(closes, min(26, len(closes)))
    line = fast - slow
    sig  = _ema(line, min(9, len(line)))
    hist = line - sig
    trend = "flat"
    if len(hist) >= 3:
        h = hist[-3:]
        if h[-1] > h[-2] > h[-3]:
            trend = "rising"
        elif h[-1] < h[-2] < h[-3]:
            trend = "falling"
    return {
        "macd_bullish": bool(line[-1] > sig[-1]),
        "macd_trend":   trend,
        "macd":         round(float(line[-1]), 4),
        "macd_hist":    round(float(hist[-1]), 4),
    }


# ── Signal computation ────────────────────────────────────────────────────────

def compute_signals(
    fund: dict | None,
    daily: pd.DataFrame,
    headlines: list[str],
) -> dict:
    closes  = daily["close"].to_numpy(dtype=float)
    volumes = daily["volume"].to_numpy(dtype=float)
    price   = closes[-1]

    # Price performance
    price_chg_1m = (price / closes[-21] - 1) * 100 if len(closes) >= 21 else (price / closes[0] - 1) * 100
    price_chg_3m = (price / closes[0] - 1) * 100

    # Daily technicals
    rsi    = _rsi(closes, 14)
    macd_d = _macd_daily(closes)
    ema9   = float(_ema(closes, min(9, len(closes)))[-1])
    ema21  = float(_ema(closes, min(21, len(closes)))[-1])
    ema50  = float(_ema(closes, min(50, len(closes)))[-1])
    vol_avg  = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
    vol_ratio = float(volumes[-1]) / vol_avg if vol_avg > 0 else 1.0

    # M&A detection from headlines
    ma_detected = False
    for h in headlines:
        if any(k in h.lower() for k in _MA_KEYWORDS):
            ma_detected = True
            break

    # Fundamentals
    market_cap         = fund.get("market_cap") if fund else None
    revenue_growth_raw = fund.get("revenue_growth") if fund else None
    revenue_growth     = round(revenue_growth_raw * 100, 1) if revenue_growth_raw is not None else None
    rec_mean           = fund.get("rec_mean") if fund else None
    analyst_count      = int(fund.get("analyst_count") or 0) if fund else 0
    target_mean        = fund.get("target_mean_price") if fund else None
    target_upside      = round((target_mean - price) / price * 100, 1) if (target_mean and price) else None
    fifty_two_high     = fund.get("fifty_two_wk_high") if fund else None
    fifty_two_low      = fund.get("fifty_two_wk_low") if fund else None
    fifty_two_pct      = None
    if fifty_two_high and fifty_two_low and fifty_two_high > fifty_two_low:
        fifty_two_pct = round((price - fifty_two_low) / (fifty_two_high - fifty_two_low) * 100, 1)

    next_ts   = fund.get("next_earnings_ts") if fund else None
    earn_days = None
    if next_ts:
        d = (next_ts - time.time()) / 86400.0
        earn_days = round(d, 1) if d > 0 else None

    return {
        "price":            round(price, 2),
        "price_chg_1m":     round(price_chg_1m, 2),
        "price_chg_3m":     round(price_chg_3m, 2),
        "market_cap":       market_cap,
        "market_cap_fmt":   _fmt_cap(market_cap),
        "market_cap_tier":  _cap_tier(market_cap),
        "trailing_pe":      fund.get("trailing_pe") if fund else None,
        "forward_pe":       fund.get("forward_pe") if fund else None,
        "revenue_growth":   revenue_growth,
        "gross_margins":    round((fund.get("gross_margins") or 0) * 100, 1) if fund else None,
        "operating_margins":round((fund.get("operating_margins") or 0) * 100, 1) if fund else None,
        "rec_mean":         round(rec_mean, 2) if rec_mean else None,
        "rec_key":          fund.get("rec_key", "none") if fund else "none",
        "analyst_count":    analyst_count,
        "target_mean_price":round(target_mean, 2) if target_mean else None,
        "target_high_price":fund.get("target_high_price") if fund else None,
        "target_low_price": fund.get("target_low_price") if fund else None,
        "target_upside_pct":target_upside,
        "fifty_two_wk_high":fifty_two_high,
        "fifty_two_wk_low": fifty_two_low,
        "fifty_two_wk_pct": fifty_two_pct,
        "next_earnings_ts":   next_ts,
        "next_earnings_days": earn_days,
        "next_earnings_avg":  fund.get("next_earnings_avg") if fund else None,
        "next_earnings_low":  fund.get("next_earnings_low") if fund else None,
        "next_earnings_high": fund.get("next_earnings_high") if fund else None,
        "earnings_history":   fund.get("earnings_history", []) if fund else [],
        "ma_detected":      ma_detected,
        "sector":           fund.get("sector", "Unknown") if fund else "Unknown",
        "company_name":     fund.get("company_name", "") if fund else "",
        "rsi":              round(rsi, 1),
        "ema9":             round(ema9, 2),
        "ema21":            round(ema21, 2),
        "ema50":            round(ema50, 2),
        "ema_bullish":      bool(ema9 > ema21),
        "above_50ema":      bool(price > ema50),
        "vol_ratio":        round(vol_ratio, 2),
        **macd_d,
    }


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_setup(sig: dict, market_ctx: dict) -> tuple[float, str, list[str]]:
    score   = 0.0
    reasons = []

    # 1. Latest earnings surprise
    earnings_hist = sig.get("earnings_history") or []
    if earnings_hist:
        latest = earnings_hist[-1]
        surp   = latest["surprise_pct"]
        qtr    = latest.get("quarter", "")
        if surp >= 15:
            score += 3.0
            reasons.append(f"Blowout earnings {qtr}: +{surp:.1f}% above EPS estimate")
        elif surp >= 8:
            score += 2.0
            reasons.append(f"Strong earnings beat {qtr}: +{surp:.1f}% above estimate")
        elif surp >= 3:
            score += 1.0
            reasons.append(f"Earnings beat {qtr}: +{surp:.1f}% above estimate")
        elif surp >= -3:
            reasons.append(f"Earnings in-line {qtr}: {surp:+.1f}%")
        elif surp >= -8:
            score -= 1.5
            reasons.append(f"Earnings miss {qtr}: {surp:.1f}% — guidance watch needed")
        else:
            score -= 3.0
            reasons.append(f"Major earnings miss {qtr}: {surp:.1f}% — fundamental concern")

        # Consecutive beats bonus
        if len(earnings_hist) >= 3 and all(e["beat"] for e in earnings_hist[-3:]):
            score += 0.75
            reasons.append("3 consecutive earnings beats — consistent outperformer")
        if len(earnings_hist) >= 4 and all(e["beat"] for e in earnings_hist[-4:]):
            score += 0.5
            reasons.append("4-for-4 earnings beats last year — top-tier execution")

    # 2. Upcoming earnings catalyst
    earn_days = sig.get("next_earnings_days")
    if earn_days is not None and earn_days > 0:
        if earn_days <= 7:
            score += 1.5
            reasons.append(f"Earnings catalyst in {earn_days:.0f} days — pre-announcement window")
        elif earn_days <= 21:
            score += 0.75
            reasons.append(f"Earnings in {earn_days:.0f} days — pre-earnings momentum window")

    # 3. Revenue growth
    rev = sig.get("revenue_growth")
    if rev is not None:
        if rev >= 30:
            score += 2.5
            reasons.append(f"Explosive revenue growth: +{rev:.1f}% YoY — hyper-growth mode")
        elif rev >= 15:
            score += 1.5
            reasons.append(f"Strong revenue growth: +{rev:.1f}% YoY")
        elif rev >= 5:
            score += 0.75
            reasons.append(f"Solid revenue growth: +{rev:.1f}% YoY")
        elif rev >= -5:
            pass
        elif rev >= -15:
            score -= 1.0
            reasons.append(f"Revenue declining: {rev:.1f}% YoY — business headwinds")
        else:
            score -= 2.5
            reasons.append(f"Revenue contraction: {rev:.1f}% YoY — major fundamental weakness")

    # 4. Analyst consensus
    rec        = sig.get("rec_mean")
    n_analysts = sig.get("analyst_count", 0) or 0
    if rec is not None and n_analysts >= 3:
        if rec <= 1.5:
            score += 2.0
            reasons.append(f"Analyst consensus: STRONG BUY ({n_analysts} analysts, mean {rec:.1f})")
        elif rec <= 2.0:
            score += 1.5
            reasons.append(f"Analyst consensus: BUY ({n_analysts} analysts, mean {rec:.1f})")
        elif rec <= 2.5:
            score += 0.75
            reasons.append(f"Analyst consensus: MODERATE BUY ({n_analysts} analysts)")
        elif rec <= 3.0:
            pass
        elif rec <= 3.5:
            score -= 0.5
            reasons.append(f"Analyst consensus: HOLD ({rec:.1f})")
        elif rec <= 4.0:
            score -= 1.5
            reasons.append(f"Analyst consensus: UNDERPERFORM ({n_analysts} analysts, mean {rec:.1f})")
        else:
            score -= 2.0
            reasons.append(f"Analyst consensus: SELL ({n_analysts} analysts) — institutional bearish")

    # 5. Analyst price target upside
    upside = sig.get("target_upside_pct")
    target = sig.get("target_mean_price")
    if upside is not None:
        if upside >= 30:
            score += 2.0
            reasons.append(f"Major analyst upside: {upside:.1f}% to mean target ${target}")
        elif upside >= 15:
            score += 1.0
            reasons.append(f"Strong analyst upside: {upside:.1f}% to consensus target ${target}")
        elif upside >= 5:
            score += 0.5
            reasons.append(f"Analyst upside: {upside:.1f}% to mean price target ${target}")
        elif upside <= -15:
            score -= 1.5
            reasons.append(f"Trading {-upside:.1f}% above analyst mean target — consensus downside risk")
        elif upside <= -5:
            score -= 0.75
            reasons.append(f"Near/above analyst target — limited upside per consensus")

    # 6. M&A activity
    if sig.get("ma_detected"):
        score += 2.0
        reasons.append("M&A activity detected in recent news — potential premium catalyst")

    # 7. 52-week price position
    wk_pct = sig.get("fifty_two_wk_pct")
    if wk_pct is not None:
        if wk_pct >= 85:
            score += 0.75
            reasons.append(f"Near 52-week high ({wk_pct:.0f}% of annual range) — strong price leadership")
        elif wk_pct >= 65:
            score += 0.25
            reasons.append(f"Upper half of annual range ({wk_pct:.0f}%) — positive trend intact")
        elif wk_pct <= 20:
            score -= 0.5
            reasons.append(f"Near 52-week low ({wk_pct:.0f}% of range) — falling knife risk")

    # 8. Daily RSI
    rsi = sig.get("rsi", 50.0)
    if 45 <= rsi <= 65:
        score += 0.75
        reasons.append(f"Daily RSI {rsi:.0f} — healthy swing momentum zone")
    elif rsi > 75:
        score -= 1.0
        reasons.append(f"Daily RSI overbought ({rsi:.0f}) — short-term pullback risk")
    elif rsi < 35:
        score -= 0.5
        reasons.append(f"Daily RSI oversold ({rsi:.0f}) — continued selling pressure")

    # 9. Daily MACD
    if sig.get("macd_bullish") and sig.get("macd_trend") == "rising":
        score += 1.0
        reasons.append("Daily MACD bullish & histogram rising — momentum building")
    elif sig.get("macd_bullish"):
        score += 0.5
        reasons.append("Daily MACD above signal line — bullish daily bias")
    elif not sig.get("macd_bullish") and sig.get("macd_trend") == "falling":
        score -= 1.0
        reasons.append("Daily MACD bearish & falling — momentum deteriorating")
    else:
        score -= 0.5
        reasons.append("Daily MACD below signal — bearish daily bias")

    # 10. EMA structure
    if sig.get("ema_bullish"):
        score += 0.5
        reasons.append("EMA 9 > 21 on daily — short-term trend intact")
    else:
        score -= 0.25
        reasons.append("EMA 9 < 21 on daily — short-term bearish structure")

    if sig.get("above_50ema"):
        score += 0.5
        reasons.append("Price above 50-day EMA — medium-term uptrend confirmed")
    else:
        score -= 0.25
        reasons.append("Below 50-day EMA — medium-term trend not supportive")

    # 11. Market regime
    regime  = market_ctx.get("regime", "neutral")
    spy_chg = market_ctx.get("spy_chg", 0.0)
    qqq_chg = market_ctx.get("qqq_chg", 0.0)
    if regime == "bull":
        score += 0.75
        reasons.append(f"Bull market tailwind: SPY {spy_chg:+.1f}%, QQQ {qqq_chg:+.1f}%")
    elif regime == "bear":
        score -= 1.5
        reasons.append(f"Bear market headwind — macro risk to swing positions")
    elif regime == "caution":
        score -= 0.5
        reasons.append("Market caution mode — consider reduced position sizing")

    # Grade
    if score >= 7.0:
        grade = "TOP PICK"
    elif score >= 5.0:
        grade = "STRONG BUY"
    elif score >= 3.0:
        grade = "BUY"
    elif score >= 1.5:
        grade = "WATCH"
    elif score >= -1.0:
        grade = "HOLD"
    else:
        grade = "AVOID"

    return round(score, 2), grade, reasons


def _primary_catalyst(sig: dict) -> str:
    if sig.get("ma_detected"):
        return "M&A ACTIVITY"
    hist = sig.get("earnings_history") or []
    if hist:
        surp = hist[-1]["surprise_pct"]
        if surp >= 8:
            return "EARNINGS BEAT"
        if surp <= -8:
            return "EARNINGS MISS"
    earn_days = sig.get("next_earnings_days")
    if earn_days is not None and earn_days <= 14:
        return "UPCOMING EARNINGS"
    rec = sig.get("rec_mean")
    if rec is not None and rec <= 1.5:
        return "STRONG ANALYST BUY"
    rev = sig.get("revenue_growth")
    if rev is not None and rev >= 20:
        return "HIGH REVENUE GROWTH"
    upside = sig.get("target_upside_pct")
    if upside is not None and upside >= 20:
        return "ANALYST UPSIDE"
    if sig.get("macd_bullish") and sig.get("macd_trend") == "rising" and sig.get("ema_bullish"):
        return "TECHNICAL SETUP"
    return "VALUE PLAY"


# ── Mechanical thesis (fallback) ─────────────────────────────────────────────

def _mechanical_thesis(sig: dict, score: float, grade: str) -> dict:
    price  = sig["price"]
    target = sig.get("target_mean_price")
    rev    = sig.get("revenue_growth")
    rec    = sig.get("rec_key", "hold")

    if grade in ("TOP PICK", "STRONG BUY", "BUY"):
        entry      = round(price * 1.002, 2)
        tgt_price  = round(target * 0.97, 2) if target else round(price * 1.15, 2)
        stop_price = round(price * 0.91, 2)
        horizon    = "4-12 weeks" if grade == "TOP PICK" else "2-8 weeks" if grade == "STRONG BUY" else "2-6 weeks"
        conviction = "high" if grade in ("TOP PICK", "STRONG BUY") else "medium"
        rev_str    = f"+{rev:.1f}% YoY revenue growth" if rev else "solid fundamentals"
        analysis   = (
            f"Fundamental catalyst play driven by {rev_str}. "
            f"Analyst consensus: {rec.upper()} with mean target ${sig.get('target_mean_price') or 'N/A'}. "
            f"Daily technicals {'supportive' if sig.get('ema_bullish') else 'mixed'} "
            f"(RSI {sig.get('rsi', 50):.0f}, MACD {'bullish' if sig.get('macd_bullish') else 'bearish'})."
        )
        key_risk  = "Earnings miss or sector rotation could invalidate setup"
        key_cat   = sig.get("catalyst", "fundamental setup")
    elif grade == "WATCH":
        entry      = round(price, 2)
        tgt_price  = round(price * 1.07, 2)
        stop_price = round(price * 0.95, 2)
        horizon    = "1-4 weeks — monitor for catalyst"
        conviction = "low"
        analysis   = "Marginal setup. Await earnings or analyst catalyst before entering full position."
        key_risk   = "No near-term catalyst confirmed"
        key_cat    = "monitoring"
    else:
        entry = tgt_price = stop_price = price
        horizon    = "Skip"
        conviction = "none"
        analysis   = "No actionable setup. Insufficient fundamentals or unfavorable trend."
        key_risk   = "N/A"
        key_cat    = "avoid"

    risk   = max(entry - stop_price, 0.01)
    reward = max(tgt_price - entry, 0.01)
    return {
        "entry":              entry,
        "target_price":       tgt_price,
        "stop_loss":          stop_price,
        "rr":                 f"1:{round(reward / risk, 1)}",
        "investment_horizon": horizon,
        "conviction":         conviction,
        "thesis":             analysis,
        "key_catalyst":       key_cat,
        "key_risk":           key_risk,
        "ai_generated":       False,
    }


# ── AI thesis via Claude API ─────────────────────────────────────────────────

async def ai_investment_thesis(
    ticker: str,
    sig: dict,
    score: float,
    grade: str,
    reasons: list[str],
    market_ctx: dict,
    headlines: list[str],
) -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _mechanical_thesis(sig, score, grade)

    try:
        hist = sig.get("earnings_history") or []
        earn_lines = "\n".join(
            f"  {e['quarter']}: EPS ${e['eps_actual']} vs est ${e['eps_estimate']} "
            f"({'BEAT' if e['beat'] else 'MISS'} {e['surprise_pct']:+.1f}%)"
            for e in hist[-4:]
        ) or "  No earnings history available"

        news_block = "\n".join(f"  - {h}" for h in headlines[:5]) or "  No recent news"
        earn_days  = sig.get("next_earnings_days")
        next_earn_str = f"In {earn_days:.0f} days" if earn_days else "Not scheduled"

        prompt = f"""You are an expert swing trader and fundamental analyst. Analyze this investment opportunity.

TICKER: {ticker} | COMPANY: {sig.get("company_name", ticker)} | SECTOR: {sig.get("sector", "?")}
GRADE: {grade} (Score: {score:.1f}) | Market Cap: {sig.get("market_cap_fmt", "N/A")}

=== FUNDAMENTALS ===
Revenue Growth (YoY): {f"+{sig.get('revenue_growth'):.1f}%" if sig.get('revenue_growth') is not None else "N/A"}
Forward P/E: {sig.get("forward_pe") or "N/A"} | Trailing P/E: {sig.get("trailing_pe") or "N/A"}
Gross Margins: {f"{sig.get('gross_margins'):.1f}%" if sig.get('gross_margins') else "N/A"}

=== ANALYST CONSENSUS ({sig.get("analyst_count", 0)} analysts) ===
Rating: {sig.get("rec_key", "N/A").upper()} (mean: {sig.get("rec_mean") or "N/A"})
Mean Target: ${sig.get("target_mean_price") or "N/A"} | Upside: {f"{sig.get('target_upside_pct'):+.1f}%" if sig.get("target_upside_pct") is not None else "N/A"}
52-Week Position: {f"{sig.get('fifty_two_wk_pct'):.0f}% of range" if sig.get("fifty_two_wk_pct") else "N/A"}

=== EARNINGS HISTORY ===
{earn_lines}
Next Earnings: {next_earn_str}{f" (est EPS ${sig.get('next_earnings_avg')})" if sig.get("next_earnings_avg") else ""}
M&A Activity Detected: {"YES" if sig.get("ma_detected") else "No"}

=== DAILY TECHNICALS ===
RSI: {sig.get("rsi", 50):.1f} | MACD: {"Bullish" if sig.get("macd_bullish") else "Bearish"} ({sig.get("macd_trend", "flat")})
EMA: {"9>21 bullish" if sig.get("ema_bullish") else "9<21 bearish"} | 50-EMA: {"Above" if sig.get("above_50ema") else "Below"}
1-Month Trend: {f"{sig.get('price_chg_1m'):+.1f}%" if sig.get("price_chg_1m") is not None else "N/A"}

=== KEY SIGNALS ===
{chr(10).join(f"• {r}" for r in reasons[:6])}

=== MARKET & NEWS ===
Regime: {market_ctx.get("regime", "neutral").upper()} | SPY {market_ctx.get("spy_chg", 0):+.1f}%
{news_block}

Provide a swing trade thesis (2-8 week horizon). Focus on the fundamental edge and most reliable catalyst.

Respond in valid JSON only (no markdown fences):
{{"thesis": "2-3 specific sentences on investment thesis and fundamental edge", "entry_strategy": "when and how to enter", "target_price": <number based on analyst target or technical resistance>, "stop_loss": <number — 8-12% below current price {round(sig["price"] * 0.91, 2)}>, "investment_horizon": "X-Y weeks", "key_catalyst": "single most important upcoming catalyst", "conviction": "high|medium|low", "key_risk": "single most specific risk factor"}}"""

        def _call():
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                system=_RISK_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text

        raw     = await asyncio.to_thread(_call)
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"```", "", cleaned).strip()
        s       = cleaned.find("{")
        e       = cleaned.rfind("}")
        if s < 0 or e <= s:
            raise ValueError(f"No JSON in response: {raw[:120]}")
        thesis             = json.loads(cleaned[s: e + 1])
        thesis["ai_generated"] = True
        return thesis

    except Exception as exc:
        logger.warning("AI thesis failed for %s: %s", ticker, exc)
        return _mechanical_thesis(sig, score, grade)


# ── Price history ─────────────────────────────────────────────────────────────

def _price_history(daily: pd.DataFrame | None) -> list[dict]:
    if daily is None or len(daily) < 2:
        return []
    tail = daily.tail(min(60, len(daily)))
    out  = []
    for _, row in tail.iterrows():
        ts = row["ts"]
        out.append({
            "date":   ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10],
            "close":  round(float(row["close"]), 2),
            "high":   round(float(row["high"]), 2),
            "low":    round(float(row["low"]), 2),
            "volume": int(row["volume"]),
        })
    return out


# ── Per-ticker scan ───────────────────────────────────────────────────────────

async def _scan_ticker(
    ticker: str,
    client: httpx.AsyncClient,
    news_api_key: str,
    sem: asyncio.Semaphore,
    market_ctx: dict,
) -> dict | None:
    async with sem:
        try:
            fund, daily, headlines = await asyncio.gather(
                fetch_fundamentals(ticker, client),
                fetch_daily(ticker, client),
                fetch_news(ticker, news_api_key, client),
            )

            if daily is None or len(daily) < 10:
                return None

            sig            = compute_signals(fund, daily, headlines)
            score, grade, reasons = score_setup(sig, market_ctx)
            catalyst       = _primary_catalyst(sig)
            sig["catalyst"] = catalyst

            return {
                "ticker":        ticker,
                "grade":         grade,
                "score":         score,
                "catalyst":      catalyst,
                "signals":       sig,
                "reasons":       reasons,
                "thesis":        _mechanical_thesis(sig, score, grade),
                "headlines":     headlines[:4],
                "price_history": _price_history(daily),
                "scanned_at":    datetime.utcnow().isoformat(),
            }
        except Exception as exc:
            logger.error("Error scanning %s: %s", ticker, exc)
            return None
        finally:
            await asyncio.sleep(0.5)


async def _apply_ai_thesis(result: dict, market_ctx: dict) -> None:
    thesis = await ai_investment_thesis(
        result["ticker"],
        result["signals"],
        result["score"],
        result["grade"],
        result["reasons"],
        market_ctx,
        result["headlines"],
    )
    result["thesis"] = thesis
    logger.info(
        "AI thesis for %s: conviction=%s, horizon=%s",
        result["ticker"],
        thesis.get("conviction", "—"),
        thesis.get("investment_horizon", "—"),
    )


# ── Main entry ────────────────────────────────────────────────────────────────

async def analyze_single_ticker(ticker: str) -> dict:
    ticker       = ticker.upper().strip()
    news_api_key = os.getenv("NEWS_API_KEY", "")
    sem          = asyncio.Semaphore(1)

    async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
        market_ctx = await fetch_market_context(client)
        result     = await _scan_ticker(ticker, client, news_api_key, sem, market_ctx)

    if result is None:
        return {"error": f"Could not fetch data for '{ticker}'. Check the symbol and try again."}

    await _apply_ai_thesis(result, market_ctx)
    result["market_ctx"] = market_ctx
    return result


async def run_smart_picks(custom_tickers: list[str] | None = None) -> dict:
    base_tickers = list(custom_tickers or SMART_UNIVERSE)
    news_api_key = os.getenv("NEWS_API_KEY", "")
    sem          = asyncio.Semaphore(3)

    async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
        # Fetch market context and FMP dynamic universe in parallel (zero extra latency)
        market_ctx, dynamic = await asyncio.gather(
            fetch_market_context(client),
            fetch_fmp_dynamic_tickers(client),
        )

        # Merge dynamic tickers without duplicates
        seen    = set(base_tickers)
        tickers = list(base_tickers)
        for t in dynamic:
            if t not in seen:
                seen.add(t)
                tickers.append(t)

        logger.info(
            "Market regime: %s | SPY %+.2f%% | QQQ %+.2f%% | Universe: %d (%d base + %d dynamic)",
            market_ctx["regime"].upper(),
            market_ctx["spy_chg"],
            market_ctx["qqq_chg"],
            len(tickers),
            len(base_tickers),
            len(tickers) - len(base_tickers),
        )

        # Phase 1: parallel scan (no AI)
        logger.info("Smart Picks phase 1: scanning %d tickers...", len(tickers))
        tasks  = [_scan_ticker(t, client, news_api_key, sem, market_ctx) for t in tickers]
        raw    = await asyncio.gather(*tasks, return_exceptions=True)

    results = [r for r in raw if r and not isinstance(r, Exception)]
    results.sort(key=lambda x: x["score"], reverse=True)

    top_picks = [r for r in results if r["grade"] in ("TOP PICK", "STRONG BUY", "BUY")][:6]
    on_watch  = [r for r in results if r["grade"] == "WATCH"][:6]
    avoid     = [r for r in results if r["grade"] == "AVOID"][-5:]

    # Phase 2: AI thesis for top picks only
    if top_picks:
        logger.info("Phase 2: AI thesis for %d top picks...", len(top_picks))
        await asyncio.gather(*[_apply_ai_thesis(p, market_ctx) for p in top_picks], return_exceptions=True)

    # Earnings calendar: upcoming earnings within 30 days across all scanned
    cal_items = []
    for r in results:
        sig = r["signals"]
        d   = sig.get("next_earnings_days")
        if d is not None and 0 < d <= 30:
            cal_items.append({
                "ticker":         r["ticker"],
                "grade":          r["grade"],
                "score":          r["score"],
                "next_earn_days": d,
                "next_earn_avg":  sig.get("next_earnings_avg"),
                "last_surprise":  (sig.get("earnings_history") or [{}])[-1].get("surprise_pct"),
                "price":          sig.get("price"),
            })
    cal_items.sort(key=lambda x: x["next_earn_days"])

    return {
        "run_at":           datetime.utcnow().isoformat(),
        "total_scanned":    len(results),
        "universe_size":    len(tickers),
        "dynamic_tickers":  dynamic,
        "market_ctx":       market_ctx,
        "top_picks":        top_picks,
        "on_watch":         on_watch,
        "avoid":            avoid,
        "earnings_calendar":cal_items[:10],
        "all_results":      results,
    }
