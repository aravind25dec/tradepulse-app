"""
hotpicks/engine.py
Intraday momentum scanner — finds what to BUY RIGHT NOW for day-trading profit.

v2 — multi-factor risk analysis to catch falls before entry:
  - Market regime (SPY + QQQ): if market is falling, most picks will fall too
  - Bollinger Bands (10-period intraday): price near upper band = reversal risk
  - MACD (12,26,9 on daily): momentum direction and divergence detection
  - ADX (14 on daily): trend strength — filters choppy/sideways markets
  - Relative strength vs SPY: stock must lead the market, not lag it
  - Bearish candle patterns: shooting star, doji, bearish engulfing at highs
  - Volume-price divergence: price rising on shrinking volume = weak/fake move
  - ATR-based stops: volatility-adjusted risk management
  - Danger counter: 4+ red flags → hard SKIP override regardless of score
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Risk-first system prompt (prepended to AI trade plan calls) ───────────────

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
- Never issue more than 3 BUY signals in a single session — scarcity of picks improves quality

IMPORTANT: When your specific task below provides its own JSON response schema, output ONLY that JSON — the risk rules above govern your *reasoning and decisions*, not the output format."""

# ── Universe ──────────────────────────────────────────────────────────────────
DEFAULT_UNIVERSE = [
    "AAPL", "NVDA", "TSLA", "META", "AMZN", "GOOGL", "MSFT", "AMD",
    "PLTR", "SOFI", "COIN", "HOOD", "MSTR", "RIVN",
    "SPY", "QQQ", "SQQQ",
    "XOM", "CVX", "JPM", "GS", "BAC",
    "SMCI", "ARM", "ASML", "TSM", "INTC", "MU",
    "MRNA", "BNTX",
    "UPST", "RKLB", "JOBY",
]

# ── Data fetching ─────────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


async def _fetch_yahoo(ticker: str, interval: str, range_: str, client: httpx.AsyncClient):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval": interval, "range": range_, "includePrePost": "false"}
    resp = await client.get(url, params=params, timeout=12)
    resp.raise_for_status()
    data = resp.json()
    result = data.get("chart", {}).get("result")
    if not result:
        return None
    return result[0]


async def fetch_intraday(ticker: str, client: httpx.AsyncClient) -> pd.DataFrame | None:
    try:
        r = await _fetch_yahoo(ticker, "5m", "1d", client)
        if not r:
            return None
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
        return df if len(df) >= 5 else None
    except Exception as exc:
        logger.debug("Intraday fetch failed %s: %s", ticker, exc)
        return None


async def fetch_daily(ticker: str, client: httpx.AsyncClient) -> pd.DataFrame | None:
    try:
        r = await _fetch_yahoo(ticker, "1d", "3mo", client)
        if not r:
            return None
        q = r["indicators"]["quote"][0]
        df = pd.DataFrame({
            "ts":     r.get("timestamp", []),
            "close":  q.get("close", []),
            "volume": q.get("volume", []),
            "open":   q.get("open", []),
            "high":   q.get("high", []),
            "low":    q.get("low", []),
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
        since = (datetime.utcnow() - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%S")
        resp = await client.get(
            "https://newsapi.org/v2/everything",
            params={"q": ticker, "apiKey": api_key, "pageSize": 5,
                    "sortBy": "publishedAt", "from": since},
            timeout=8,
        )
        articles = resp.json().get("articles", [])
        return [a.get("title", "") for a in articles if a.get("title")]
    except Exception:
        return []


async def fetch_earnings_data(ticker: str, client: httpx.AsyncClient) -> dict:
    """Fetch next earnings date, EPS estimates, and last 4 quarters history.
    Returns enrichment dict; empty dict on any failure (non-critical data).
    """
    try:
        url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
        resp = await client.get(url, params={"modules": "calendarEvents,earningsHistory"}, timeout=10)
        resp.raise_for_status()
        r = ((resp.json().get("quoteSummary") or {}).get("result") or [None])[0]
        if not r:
            return {}

        earnings_hist = []
        for h in ((r.get("earningsHistory") or {}).get("history") or []):
            eps_act = (h.get("epsActual") or {}).get("raw")
            eps_est = (h.get("epsEstimate") or {}).get("raw")
            surp    = (h.get("surprisePercent") or {}).get("raw")
            qtr     = (h.get("quarter") or {}).get("fmt", "")
            if eps_act is not None and eps_est is not None:
                earnings_hist.append({
                    "quarter":      qtr,
                    "eps_actual":   round(float(eps_act), 2),
                    "eps_estimate": round(float(eps_est), 2),
                    "surprise_pct": round(float(surp or 0) * 100, 1),
                    "beat":         float(eps_act) >= float(eps_est),
                })

        cal       = (r.get("calendarEvents") or {}).get("earnings") or {}
        earn_dates = cal.get("earningsDate") or []
        next_ts   = None
        if earn_dates:
            raw_ts = (earn_dates[0] or {}).get("raw")
            if raw_ts:
                next_ts = int(raw_ts)

        earn_avg  = (cal.get("earningsAverage") or {}).get("raw")
        earn_low  = (cal.get("earningsLow") or {}).get("raw")
        earn_high = (cal.get("earningsHigh") or {}).get("raw")

        earn_days = None
        if next_ts:
            d = (next_ts - time.time()) / 86400.0
            earn_days = round(d, 1) if d > 0 else None

        return {
            "next_earnings_ts":   next_ts,
            "next_earnings_days": earn_days,
            "next_earnings_avg":  round(float(earn_avg), 2) if earn_avg else None,
            "next_earnings_low":  round(float(earn_low), 2) if earn_low else None,
            "next_earnings_high": round(float(earn_high), 2) if earn_high else None,
            "earnings_history":   earnings_hist,
        }
    except Exception as exc:
        logger.debug("Earnings fetch failed %s: %s", ticker, exc)
        return {}


async def fetch_market_context(client: httpx.AsyncClient) -> dict:
    """Fetch SPY and QQQ to determine broad market regime before scanning any picks."""
    try:
        spy_intra, qqq_intra = await asyncio.gather(
            fetch_intraday("SPY", client),
            fetch_intraday("QQQ", client),
        )

        spy_chg = 0.0
        qqq_chg = 0.0
        spy_above_vwap = True
        qqq_above_vwap = True

        if spy_intra is not None and len(spy_intra) >= 3:
            sc = spy_intra["close"].to_numpy(dtype=float)
            so = spy_intra["open"].to_numpy(dtype=float)
            sh = spy_intra["high"].to_numpy(dtype=float)
            sl = spy_intra["low"].to_numpy(dtype=float)
            sv = spy_intra["volume"].to_numpy(dtype=float)
            spy_chg = float((sc[-1] - so[0]) / so[0] * 100)
            tp = (sh + sl + sc) / 3.0
            spy_vwap = float(np.sum(tp * sv) / np.sum(sv)) if np.sum(sv) > 0 else sc[-1]
            spy_above_vwap = bool(sc[-1] > spy_vwap)

        if qqq_intra is not None and len(qqq_intra) >= 3:
            qc = qqq_intra["close"].to_numpy(dtype=float)
            qo = qqq_intra["open"].to_numpy(dtype=float)
            qh = qqq_intra["high"].to_numpy(dtype=float)
            ql = qqq_intra["low"].to_numpy(dtype=float)
            qv = qqq_intra["volume"].to_numpy(dtype=float)
            qqq_chg = float((qc[-1] - qo[0]) / qo[0] * 100)
            tp = (qh + ql + qc) / 3.0
            qqq_vwap = float(np.sum(tp * qv) / np.sum(qv)) if np.sum(qv) > 0 else qc[-1]
            qqq_above_vwap = bool(qc[-1] > qqq_vwap)

        avg_chg = (spy_chg + qqq_chg) / 2.0
        if avg_chg >= 0.5:
            regime = "bull"
        elif avg_chg <= -0.75:
            regime = "bear"
        elif avg_chg <= -0.25:
            regime = "caution"
        else:
            regime = "neutral"

        return {
            "regime":         regime,
            "spy_chg":        round(spy_chg, 2),
            "qqq_chg":        round(qqq_chg, 2),
            "spy_above_vwap": spy_above_vwap,
            "qqq_above_vwap": qqq_above_vwap,
        }
    except Exception as exc:
        logger.warning("Market context fetch failed: %s", exc)
        return {"regime": "neutral", "spy_chg": 0.0, "qqq_chg": 0.0,
                "spy_above_vwap": True, "qqq_above_vwap": True}


async def fetch_dynamic_tickers(client: httpx.AsyncClient) -> list[str]:
    """Augment the scan universe with today's top movers from Yahoo Finance screener.

    Fetches day_gainers and most_actives — no API key required.
    Returns only tickers NOT already in DEFAULT_UNIVERSE, capped at 15 additions.
    """
    candidates: set[str] = set()
    base = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"

    def _valid(item: dict) -> bool:
        sym = item.get("symbol", "")
        return bool(sym) and sym.isalpha() and 1 <= len(sym) <= 5

    for scr_id in ("day_gainers", "most_actives"):
        try:
            resp = await client.get(
                base,
                params={"scrIds": scr_id, "count": 25, "formatted": "false",
                        "lang": "en-US", "region": "US"},
                timeout=10,
            )
            resp.raise_for_status()
            quotes = (resp.json()
                      .get("finance", {})
                      .get("result", [{}])[0]
                      .get("quotes", []))
            for item in quotes[:20]:
                if _valid(item):
                    candidates.add(item["symbol"].upper())
        except Exception as exc:
            logger.debug("Yahoo screener %s failed: %s", scr_id, exc)

    # Also try FMP if key is present (paid plan has gainers/actives)
    api_key = os.getenv("FMP_API_KEY", "")
    if api_key:
        fmp_base = "https://financialmodelingprep.com/api/v3"
        for endpoint in ("stock_market/gainers", "stock_market/actives"):
            try:
                resp = await client.get(
                    f"{fmp_base}/{endpoint}",
                    params={"apikey": api_key},
                    timeout=10,
                )
                if resp.status_code == 200:
                    for item in (resp.json() or [])[:15]:
                        sym = item.get("symbol", "")
                        if sym and sym.isalpha() and len(sym) <= 5:
                            candidates.add(sym.upper())
            except Exception as exc:
                logger.debug("FMP %s failed: %s", endpoint, exc)

    new_tickers = [t for t in candidates if t not in set(DEFAULT_UNIVERSE)]
    if new_tickers:
        logger.info("Dynamic universe: +%d tickers added — %s", len(new_tickers), new_tickers)
    return new_tickers[:15]


# ── Technical indicator helpers ───────────────────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    k = 2.0 / (period + 1)
    out = np.empty(len(arr), dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def _bollinger(closes: np.ndarray, period: int = 10, num_std: float = 2.0) -> dict:
    period = min(period, len(closes))
    if period < 3:
        return {"bb_upper": float(closes[-1]), "bb_lower": float(closes[-1]),
                "bb_mid": float(closes[-1]), "bb_pct": 0.5, "bb_squeeze": False}
    window = closes[-period:]
    mid    = float(np.mean(window))
    std    = float(np.std(window))
    upper  = mid + num_std * std
    lower  = mid - num_std * std
    price  = float(closes[-1])
    bb_range = upper - lower
    bb_pct   = float((price - lower) / bb_range) if bb_range > 0 else 0.5
    bb_squeeze = bool((bb_range / mid) < 0.008) if mid > 0 else False
    return {
        "bb_upper":   round(upper, 2),
        "bb_lower":   round(lower, 2),
        "bb_mid":     round(mid, 2),
        "bb_pct":     round(bb_pct, 3),
        "bb_squeeze": bb_squeeze,
    }


def _macd(daily: pd.DataFrame | None) -> dict:
    empty = {"macd": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
             "macd_bullish": True, "macd_trend": "flat"}
    if daily is None or len(daily) < 12:
        return empty
    closes    = daily["close"].to_numpy(dtype=float)
    fast_p    = min(12, len(closes))
    slow_p    = min(26, len(closes))
    sig_p     = min(9,  len(closes))
    ema_fast  = _ema(closes, fast_p)
    ema_slow  = _ema(closes, slow_p)
    macd_line = ema_fast - ema_slow
    sig_line  = _ema(macd_line, sig_p)
    histogram = macd_line - sig_line
    if len(histogram) >= 3:
        h = histogram[-3:]
        if h[-1] > h[-2] > h[-3]:
            trend = "rising"
        elif h[-1] < h[-2] < h[-3]:
            trend = "falling"
        else:
            trend = "flat"
    else:
        trend = "flat"
    return {
        "macd":        round(float(macd_line[-1]), 4),
        "macd_signal": round(float(sig_line[-1]),  4),
        "macd_hist":   round(float(histogram[-1]), 4),
        "macd_bullish": bool(macd_line[-1] > sig_line[-1]),
        "macd_trend":   trend,
    }


def _adx(daily: pd.DataFrame | None, period: int = 14) -> dict:
    empty = {"adx": 20.0, "plus_di": 50.0, "minus_di": 50.0, "trending": False}
    if daily is None or len(daily) < period + 2:
        return empty
    highs  = daily["high"].to_numpy(dtype=float)
    lows   = daily["low"].to_numpy(dtype=float)
    closes = daily["close"].to_numpy(dtype=float)
    n = len(closes)
    trs, pdms, mdms = [], [], []
    for i in range(1, n):
        tr   = max(highs[i] - lows[i],
                   abs(highs[i] - closes[i - 1]),
                   abs(lows[i]  - closes[i - 1]))
        up   = highs[i]  - highs[i - 1]
        down = lows[i - 1] - lows[i]
        pdms.append(up   if up > down and up > 0 else 0.0)
        mdms.append(down if down > up and down > 0 else 0.0)
        trs.append(tr)
    p   = min(period, len(trs))
    atr = float(np.mean(trs[-p:]))
    if atr == 0:
        return empty
    plus_di  = 100.0 * float(np.mean(pdms[-p:])) / atr
    minus_di = 100.0 * float(np.mean(mdms[-p:])) / atr
    dxs = []
    for i in range(p):
        pd_ = 100.0 * pdms[-(p - i)] / atr
        md_ = 100.0 * mdms[-(p - i)] / atr
        s   = pd_ + md_
        dxs.append(100.0 * abs(pd_ - md_) / s if s > 0 else 0.0)
    adx = float(np.mean(dxs)) if dxs else 20.0
    return {
        "adx":      round(adx, 1),
        "plus_di":  round(plus_di, 1),
        "minus_di": round(minus_di, 1),
        "trending": bool(adx > 22),
    }


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 10) -> float:
    if len(closes) < 2:
        return float(highs[-1] - lows[-1])
    trs = [
        max(highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]))
        for i in range(1, len(closes))
    ]
    return float(np.mean(trs[-min(period, len(trs)):]))


def _bearish_candle(opens: np.ndarray, highs: np.ndarray,
                    lows: np.ndarray, closes: np.ndarray) -> tuple[bool, str]:
    if len(closes) < 1:
        return False, "none"
    o, h, l, c = float(opens[-1]), float(highs[-1]), float(lows[-1]), float(closes[-1])
    body        = abs(c - o)
    upper_wick  = h - max(c, o)
    total_range = h - l

    # Shooting star: small body + large upper wick (rejection at top)
    if total_range > 0 and body > 0 and upper_wick >= 2.0 * body and upper_wick >= total_range * 0.55:
        return True, "shooting_star"

    # Doji: body is tiny relative to range — indecision, danger when at highs
    if total_range > 0 and body <= total_range * 0.1:
        return True, "doji"

    # Bearish engulfing: prev candle bullish, current candle bearish and fully engulfs it
    if len(closes) >= 2:
        po, pc = float(opens[-2]), float(closes[-2])
        if pc > po and c < o and o >= pc and c <= po:
            return True, "bearish_engulfing"

    return False, "none"


def _volume_divergence(closes: np.ndarray, volumes: np.ndarray, lookback: int = 4) -> bool:
    """Price making higher highs on falling volume = weak/fake move."""
    if len(closes) < lookback * 2:
        return False
    vol_early  = float(np.mean(volumes[-lookback * 2:-lookback]))
    vol_recent = float(np.mean(volumes[-lookback:]))
    price_up   = bool(closes[-1] > closes[-lookback * 2])
    vol_down   = bool(vol_early > 0 and vol_recent < vol_early * 0.72)
    return price_up and vol_down


# ── Historical data helpers (for modal detail view) ──────────────────────────

def _price_history(daily: pd.DataFrame | None) -> list[dict]:
    """Last 30 daily OHLCV rows for the modal sparkline chart."""
    if daily is None or len(daily) < 2:
        return []
    tail = daily.tail(min(30, len(daily)))
    out = []
    for _, row in tail.iterrows():
        ts = row["ts"]
        out.append({
            "date":   ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10],
            "close":  round(float(row["close"]), 2),
            "high":   round(float(row["high"]),  2),
            "low":    round(float(row["low"]),   2),
            "volume": int(row["volume"]),
        })
    return out


def _historical_patterns(daily: pd.DataFrame | None, sig: dict) -> list[dict]:
    """Scan past daily bars for RSI + volume + EMA conditions similar to today.
    Returns up to 4 historical matches with what the stock did over the next 5 and 10 days.
    """
    if daily is None or len(daily) < 25:
        return []

    df = daily.copy().reset_index(drop=True)

    # Precompute indicators with pandas (O(n) vs O(n²) loop)
    delta   = df["close"].diff()
    gain    = delta.where(delta > 0, 0.0).rolling(14, min_periods=3).mean()
    loss    = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=3).mean()
    rs      = gain / loss.replace(0, np.nan)
    df["_rsi"]   = (100 - 100 / (1 + rs)).fillna(50.0)
    df["_vr"]    = (df["volume"] / df["volume"].rolling(20, min_periods=5).mean()).fillna(1.0)
    df["_ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["_ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    target_rsi  = sig.get("rsi", 50.0)
    target_vr   = sig.get("vol_ratio", 1.0)
    target_bull = sig.get("ema_bullish", True)

    closes = df["close"].to_numpy(dtype=float)
    matches = []

    for i in range(10, len(df) - 5):
        row       = df.iloc[i]
        hist_rsi  = float(row["_rsi"])
        hist_vr   = float(row["_vr"])
        hist_bull = bool(row["_ema9"] > row["_ema21"])

        rsi_ok = abs(hist_rsi - target_rsi) <= 8
        vol_ok = (hist_vr > 1.3) == (target_vr > 1.3)
        ema_ok = hist_bull == target_bull

        if rsi_ok and vol_ok and ema_ok:
            price   = float(closes[i])
            ret_5   = float((closes[i + 5] - price) / price * 100)
            ret_10  = float((closes[min(i + 10, len(closes) - 1)] - price) / price * 100)
            ts      = row["ts"]
            matches.append({
                "date":      ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10],
                "price":     round(price, 2),
                "rsi":       round(hist_rsi, 1),
                "vol_ratio": round(hist_vr, 1),
                "ret_5d":    round(ret_5, 2),
                "ret_10d":   round(ret_10, 2),
                "bullish":   bool(ret_5 > 0),
            })

    return matches[-4:] if matches else []


# ── Signal computation ────────────────────────────────────────────────────────

def compute_signals(intra: pd.DataFrame, daily: pd.DataFrame | None) -> dict:
    closes  = intra["close"].to_numpy(dtype=float)
    highs   = intra["high"].to_numpy(dtype=float)
    lows    = intra["low"].to_numpy(dtype=float)
    volumes = intra["volume"].to_numpy(dtype=float)
    opens   = intra["open"].to_numpy(dtype=float)

    price    = closes[-1]
    day_open = opens[0]

    # Price velocity
    chg_open = (price - day_open) / day_open * 100
    chg_15m  = (closes[-1] - closes[-3]) / closes[-3] * 100 if len(closes) >= 3 else 0.0
    chg_30m  = (closes[-1] - closes[-6]) / closes[-6] * 100 if len(closes) >= 6 else 0.0

    # Volume surge vs today's avg
    avg_vol   = float(np.mean(volumes[:-1])) if len(volumes) > 1 else float(volumes[-1])
    vol_ratio = float(volumes[-1]) / avg_vol if avg_vol > 0 else 1.0

    # VWAP
    tp   = (highs + lows + closes) / 3.0
    vwap = float(np.sum(tp * volumes) / np.sum(volumes)) if np.sum(volumes) > 0 else price

    # RSI (up to 14 intraday bars)
    rsi = 50.0
    if len(closes) >= 3:
        d   = np.diff(closes)
        g   = np.where(d > 0, d, 0.0)
        l_  = np.where(d < 0, -d, 0.0)
        n   = min(14, len(g))
        ag, al = float(np.mean(g[-n:])), float(np.mean(l_[-n:]))
        rsi = 100.0 - 100.0 / (1.0 + ag / al) if al > 0 else 100.0

    # EMA cross
    n9  = min(9,  len(closes))
    n21 = min(21, len(closes))
    ema9  = float(_ema(closes, n9)[-1])
    ema21 = float(_ema(closes, n21)[-1])
    ema_bullish = bool(ema9 > ema21)

    # Gap from previous close + projected volume vs 20-day avg
    gap_pct      = 0.0
    prev_close   = None
    hist_avg_vol = None
    if daily is not None and len(daily) >= 2:
        prev_close   = float(daily["close"].iloc[-2])
        gap_pct      = float((day_open - prev_close) / prev_close * 100)
        hist_avg_vol = float(daily["volume"].rolling(20).mean().iloc[-1])

    if hist_avg_vol and hist_avg_vol > 0:
        today_total_vol = float(np.sum(volumes))
        bars_elapsed    = len(volumes)
        projected_vol   = today_total_vol / bars_elapsed * 78 if bars_elapsed > 0 else today_total_vol
        vol_ratio       = float(projected_vol / hist_avg_vol)

    # ATR (intraday, 10-period)
    atr_val = _atr(highs, lows, closes, period=10)
    atr_pct = atr_val / price * 100 if price > 0 else 0.0

    # Bollinger Bands (intraday 10-period, 2σ)
    bb = _bollinger(closes, period=10)

    # MACD on daily data (momentum direction)
    macd_data = _macd(daily)

    # ADX on daily data (trend strength)
    adx_data = _adx(daily)

    # Bearish candle on last bar
    bearish_candle, candle_pattern = _bearish_candle(opens, highs, lows, closes)

    # Volume-price divergence
    vol_divergence = _volume_divergence(closes, volumes)

    return {
        "price":          round(float(price), 2),
        "day_open":       round(float(day_open), 2),
        "prev_close":     round(float(prev_close), 2) if prev_close is not None else None,
        "chg_open":       round(float(chg_open), 2),
        "chg_15m":        round(float(chg_15m), 2),
        "chg_30m":        round(float(chg_30m), 2),
        "vol_ratio":      round(float(vol_ratio), 2),
        "vwap":           round(float(vwap), 2),
        "above_vwap":     bool(price > vwap),
        "vwap_dist":      round(float((price - vwap) / vwap * 100), 2),
        "rsi":            round(float(rsi), 1),
        "ema9":           round(float(ema9), 2),
        "ema21":          round(float(ema21), 2),
        "ema_bullish":    ema_bullish,
        "gap_pct":        round(float(gap_pct), 2),
        "bars":           int(len(closes)),
        "atr":            round(float(atr_val), 3),
        "atr_pct":        round(float(atr_pct), 2),
        **bb,
        **macd_data,
        **adx_data,
        "bearish_candle": bearish_candle,
        "candle_pattern": candle_pattern,
        "vol_divergence": vol_divergence,
    }


# ── Setup scoring ─────────────────────────────────────────────────────────────

def score_setup(sig: dict, market_ctx: dict) -> tuple[float, str, list[str], int]:
    """Return (score, grade, reasons, danger_count).
    danger_count >= 4 triggers a hard SKIP override.
    """
    score   = 0.0
    reasons = []
    danger  = 0

    # 1. Price velocity from open
    chg = sig["chg_open"]
    if chg >= 2.5:
        score += 2.5; reasons.append(f"Strong thrust: +{chg:.1f}% from open")
    elif chg >= 1.2:
        score += 1.5; reasons.append(f"Positive momentum: +{chg:.1f}% from open")
    elif chg >= 0.4:
        score += 0.75; reasons.append(f"Upward drift: +{chg:.1f}% from open")
    elif chg <= -2.0:
        score -= 2.0; danger += 1; reasons.append(f"Selling hard: {chg:.1f}% from open")
    elif chg <= -0.8:
        score -= 1.0; reasons.append(f"Under pressure: {chg:.1f}% from open")

    # 2. 15-minute acceleration
    v = sig["chg_15m"]
    if v >= 1.0:
        score += 1.5; reasons.append(f"Accelerating: +{v:.1f}% in 15 min")
    elif v >= 0.4:
        score += 0.75; reasons.append(f"Building: +{v:.1f}% in 15 min")
    elif v <= -0.6:
        score -= 1.0; reasons.append(f"Fading 15 min: {v:.1f}%")

    # 3. Volume confirmation
    vr = sig["vol_ratio"]
    if vr >= 3.0:
        score += 2.0; reasons.append(f"Massive volume: {vr:.1f}x avg — strong catalyst")
    elif vr >= 2.0:
        score += 1.5; reasons.append(f"High volume: {vr:.1f}x avg")
    elif vr >= 1.4:
        score += 0.75; reasons.append(f"Above-avg volume: {vr:.1f}x")
    elif vr < 0.6:
        score -= 0.75; danger += 1; reasons.append(f"Thin volume: {vr:.1f}x — weak conviction")

    # 4. VWAP position
    if sig["above_vwap"]:
        d = sig["vwap_dist"]
        if 0 < d <= 0.5:
            score += 1.25; reasons.append(f"Just above VWAP (+{d:.2f}%) — ideal entry zone")
        elif d <= 1.5:
            score += 0.75; reasons.append(f"Trading above VWAP (+{d:.2f}%)")
        else:
            score += 0.25; reasons.append(f"Extended above VWAP (+{d:.2f}%) — chase risk")
    else:
        d = sig["vwap_dist"]
        score -= 0.75; danger += 1; reasons.append(f"Below VWAP ({d:.2f}%) — bearish intraday bias")

    # 5. RSI momentum zone
    rsi = sig["rsi"]
    if 52 <= rsi <= 68:
        score += 0.75; reasons.append(f"RSI in momentum zone ({rsi:.0f})")
    elif rsi > 75:
        score -= 1.5; danger += 1; reasons.append(f"RSI overbought ({rsi:.0f}) — high reversal risk")
    elif rsi > 70:
        score -= 0.75; reasons.append(f"RSI elevated ({rsi:.0f}) — watch for exhaustion")
    elif rsi < 35:
        score -= 0.5; reasons.append(f"RSI oversold ({rsi:.0f}) — no upward momentum yet")

    # 6. EMA cross structure
    if sig["ema_bullish"]:
        score += 0.5; reasons.append("EMA 9 > EMA 21 — bullish intraday structure")
    else:
        score -= 0.25; reasons.append("EMA 9 < EMA 21 — bearish short-term structure")

    # 7. Gap catalyst
    gap = sig["gap_pct"]
    if gap >= 3.0:
        score += 1.5; reasons.append(f"Strong gap-up: +{gap:.1f}% from yesterday")
    elif gap >= 1.0:
        score += 0.75; reasons.append(f"Gap-up: +{gap:.1f}% — possible catalyst")
    elif gap <= -2.0:
        score -= 1.0; danger += 1; reasons.append(f"Gap-down: {gap:.1f}% — overnight negative pressure")

    # 8. Bollinger Bands — most important new signal
    bb_pct = sig.get("bb_pct", 0.5)
    if bb_pct > 1.0:
        score -= 2.0; danger += 1
        reasons.append(f"Price ABOVE upper Bollinger Band ({bb_pct:.2f}) — extreme overextension, expect reversal")
    elif bb_pct > 0.90:
        score -= 1.5; danger += 1
        reasons.append(f"Price near upper BB ({bb_pct:.0%}) — high reversal risk, avoid chasing")
    elif bb_pct > 0.75:
        score -= 0.5
        reasons.append(f"Approaching upper BB ({bb_pct:.0%}) — getting extended")
    elif 0.35 <= bb_pct <= 0.65:
        score += 0.5
        reasons.append(f"BB midzone ({bb_pct:.0%}) — room to run toward upper band")
    elif bb_pct < 0.15:
        score -= 0.5; danger += 1
        reasons.append(f"Near lower BB ({bb_pct:.0%}) — sustained downtrend pressure")
    if sig.get("bb_squeeze"):
        reasons.append("BB squeeze — breakout imminent but direction unconfirmed")

    # 9. MACD (daily momentum direction)
    macd_bullish = sig.get("macd_bullish", True)
    macd_trend   = sig.get("macd_trend", "flat")
    if macd_bullish and macd_trend == "rising":
        score += 1.25; reasons.append("MACD bullish & histogram rising — daily momentum confirmed")
    elif macd_bullish:
        score += 0.5;  reasons.append("MACD above signal — daily bias bullish")
    elif not macd_bullish and macd_trend == "falling":
        score -= 1.5; danger += 1
        reasons.append("MACD bearish & histogram falling — daily momentum deteriorating")
    else:
        score -= 0.75; reasons.append("MACD below signal — daily momentum bearish")

    # 10. ADX trend strength
    adx      = sig.get("adx", 20.0)
    plus_di  = sig.get("plus_di", 50.0)
    minus_di = sig.get("minus_di", 50.0)
    if adx > 30 and plus_di > minus_di:
        score += 1.0; reasons.append(f"ADX {adx:.0f} with +DI > -DI — strong bullish trend")
    elif adx > 22:
        score += 0.5; reasons.append(f"ADX {adx:.0f} — trending market, momentum plays viable")
    elif adx < 18:
        score -= 1.0; danger += 1
        reasons.append(f"ADX {adx:.0f} — choppy/sideways market, momentum plays unreliable")
    if adx > 22 and minus_di > plus_di:
        score -= 0.75; reasons.append(f"ADX trending but -DI({minus_di:.0f}) > +DI({plus_di:.0f}) — downtrend structure")

    # 11. Market regime — SPY + QQQ
    regime  = market_ctx.get("regime", "neutral")
    spy_chg = market_ctx.get("spy_chg", 0.0)
    qqq_chg = market_ctx.get("qqq_chg", 0.0)
    if regime == "bull":
        score += 1.0; reasons.append(f"Market tailwind: SPY {spy_chg:+.1f}%, QQQ {qqq_chg:+.1f}%")
    elif regime == "caution":
        score -= 0.75; reasons.append(f"Market caution: SPY {spy_chg:+.1f}%, QQQ {qqq_chg:+.1f}% — trade smaller")
    elif regime == "bear":
        score -= 2.5; danger += 2
        reasons.append(f"BEARISH MARKET: SPY {spy_chg:+.1f}%, QQQ {qqq_chg:+.1f}% — most picks will fall with market")
    else:
        reasons.append(f"Market neutral: SPY {spy_chg:+.1f}%, QQQ {qqq_chg:+.1f}%")

    # 12. Relative strength vs SPY
    rs = sig["chg_open"] - spy_chg
    if rs >= 1.5:
        score += 1.25; reasons.append(f"Strong RS vs SPY: outperforming by {rs:+.1f}% — clear market leader")
    elif rs >= 0.3:
        score += 0.5;  reasons.append(f"RS vs SPY: outperforming by {rs:+.1f}%")
    elif rs <= -1.5:
        score -= 1.5; danger += 1
        reasons.append(f"Weak RS vs SPY: underperforming by {rs:.1f}% — market laggard, skip")
    elif rs <= -0.5:
        score -= 0.75; reasons.append(f"RS vs SPY: underperforming by {rs:.1f}%")

    # 13. Bearish candle patterns (last bar)
    if sig.get("bearish_candle"):
        pattern = sig.get("candle_pattern", "none")
        if pattern == "bearish_engulfing":
            score -= 2.5; danger += 2
            reasons.append("BEARISH ENGULFING candle — strong reversal confirmation, do not buy")
        elif pattern == "shooting_star":
            score -= 1.5; danger += 1
            reasons.append("Shooting star candle — rejection at highs, reversal risk")
        elif pattern == "doji":
            score -= 0.75
            reasons.append("Doji candle — indecision at current level, wait for direction")

    # 14. Volume-price divergence
    if sig.get("vol_divergence"):
        score -= 1.25; danger += 1
        reasons.append("Vol-price divergence: price rising on shrinking volume — move likely fake")

    # Hard kill switch: 4+ danger signals = override to SKIP
    hard_skip = danger >= 4

    if hard_skip:
        reasons.append(f"HARD SKIP: {danger} danger signals — risk too high to trade")
        grade = "SKIP"
    elif score >= 6.0:
        grade = "STRONG BUY"
    elif score >= 3.5:
        grade = "BUY"
    elif score >= 1.5:
        grade = "WATCH"
    elif score >= -0.5:
        grade = "NEUTRAL"
    else:
        grade = "SKIP"

    return round(score, 2), grade, reasons, int(danger)


# ── Trade plan ────────────────────────────────────────────────────────────────

def _mechanical_plan(sig: dict, score: float, grade: str) -> dict:
    price = sig["price"]
    vwap  = sig["vwap"]
    atr   = sig.get("atr", price * 0.005)

    if grade in ("STRONG BUY", "BUY"):
        entry  = round(max(price, vwap * 1.001), 2)
        target = round(entry + 2.5 * atr, 2)
        stop   = round(entry - 1.2 * atr, 2)
        risk   = max(entry - stop, 0.01)
        reward = target - entry
        rr     = f"1:{round(reward / risk, 1)}"
        conv   = "high" if grade == "STRONG BUY" else "medium"
        analysis = (
            f"Momentum setup with {sig['vol_ratio']:.1f}x volume surge. "
            f"{'Above' if sig['above_vwap'] else 'Near'} VWAP at ${vwap}. "
            f"RSI {sig['rsi']:.0f}. ATR-based stop ${stop} ({sig.get('atr_pct', 0.5):.2f}% risk)."
        )
        key_risk = "Volume dry-up or market-wide reversal"
        horizon  = "30–90 minutes"
    elif grade == "WATCH":
        entry  = round(price, 2)
        target = round(price + 1.5 * atr, 2)
        stop   = round(price - 0.8 * atr, 2)
        risk   = max(entry - stop, 0.01)
        reward = target - entry
        rr     = f"1:{round(reward / risk, 1)}"
        conv   = "low"
        analysis = "Marginal setup. Wait for volume confirmation before entry."
        key_risk = "Setup may not follow through — size small"
        horizon  = "15–30 minutes — quick scalp only"
    else:
        entry = target = stop = price
        rr = "N/A"; conv = "none"; horizon = "Skip"
        analysis = "No actionable setup. Stand aside and protect capital."
        key_risk = "N/A"

    return {
        "entry":       entry,
        "target":      target,
        "stop":        stop,
        "rr":          rr,
        "horizon":     horizon,
        "conviction":  conv,
        "analysis":    analysis,
        "key_risk":    key_risk,
        "ai_generated": False,
    }


async def ai_trade_plan(
    ticker: str,
    sig: dict,
    headlines: list[str],
    score: float,
    grade: str,
    reasons: list[str],
    market_ctx: dict,
) -> dict:
    try:
        import agent_engine

        news_block = "\n".join(f"- {h}" for h in headlines[:5]) or "No recent news found."
        spy_chg    = market_ctx.get("spy_chg", 0.0)
        rs_vs_spy  = sig["chg_open"] - spy_chg

        prompt = f"""You are an expert intraday day trader with strict risk management. Analyze this live multi-factor setup.

TICKER: {ticker}
PRICE: {sig['price']} | OPEN: {sig['day_open']} | PREV CLOSE: {sig.get('prev_close') or 'N/A'}

=== PRICE ACTION ===
Change from open: {sig['chg_open']:+.2f}%
15-min velocity:  {sig['chg_15m']:+.2f}%
Gap from prev close: {sig['gap_pct']:+.2f}%
Relative strength vs SPY: {rs_vs_spy:+.2f}%

=== VOLUME & VWAP ===
Volume ratio:     {sig['vol_ratio']:.2f}x avg
VWAP:             {sig['vwap']} (price is {'ABOVE' if sig['above_vwap'] else 'BELOW'}, dist={sig['vwap_dist']:+.2f}%)
Vol-price divergence: {'YES - rising price on falling volume' if sig.get('vol_divergence') else 'No'}

=== MOMENTUM INDICATORS ===
RSI:    {sig['rsi']:.1f}
EMA:    {'Bullish (9>21)' if sig['ema_bullish'] else 'Bearish (9<21)'}
MACD:   {'Bullish' if sig.get('macd_bullish') else 'Bearish'} | Histogram {sig.get('macd_trend', 'flat')}
ADX:    {sig.get('adx', 20):.1f} (+DI:{sig.get('plus_di', 50):.1f}, -DI:{sig.get('minus_di', 50):.1f})

=== BOLLINGER BANDS ===
BB position: {sig.get('bb_pct', 0.5):.0%} of band width (0%=lower, 50%=mid, 100%=upper, >100%=blown out)
Upper BB: {sig.get('bb_upper', 0):.2f} | Lower BB: {sig.get('bb_lower', 0):.2f}
BB squeeze: {'YES' if sig.get('bb_squeeze') else 'No'}

=== RISK FACTORS ===
ATR: {sig.get('atr', 0):.3f} ({sig.get('atr_pct', 0):.2f}% of price)
Last candle pattern: {sig.get('candle_pattern', 'none').upper()}
Market regime: {market_ctx.get('regime', 'neutral').upper()} (SPY {spy_chg:+.1f}%, QQQ {market_ctx.get('qqq_chg', 0):+.1f}%)

=== OVERALL SCORE ===
Score: {score:.2f} — {grade}
Key signals: {'; '.join(reasons[:5])}

RECENT NEWS:
{news_block}

CRITICAL: If multiple risk factors are present (RSI overbought, price near/above upper BB, bearish market,
bearish candle pattern, MACD deteriorating), explicitly recommend avoiding this trade.

Respond in valid JSON only (no markdown fences):
{{"analysis": "2-3 specific sentences including what could make this trade FAIL", "entry": <number>, "target": <number based on 2-2.5x ATR of {sig.get('atr', sig['price'] * 0.005):.3f}>, "stop": <number exactly 1.2x ATR below entry>, "rr": "<ratio like 1:2.5>", "horizon": "<timeframe>", "conviction": "high|medium|low|none", "key_risk": "<single most specific risk>"}}"""

        raw = await agent_engine._oneshot_claude(f"{_RISK_SYSTEM}\n\n---\n\n{prompt}")

        # Strip markdown fences Claude may add
        import re as _re
        cleaned = _re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = _re.sub(r"```", "", cleaned).strip()

        # Find the JSON object
        s = cleaned.find("{")
        e = cleaned.rfind("}")
        if s < 0 or e <= s:
            raise ValueError(f"No JSON object in response: {raw[:200]}")

        plan = json.loads(cleaned[s : e + 1])
        plan["ai_generated"] = True
        return plan

    except Exception as exc:
        logger.warning("AI trade plan failed for %s: %s", ticker, exc)
        return _mechanical_plan(sig, score, grade)


# ── Phase 1: fast technical scan (no AI) ─────────────────────────────────────

async def _scan_ticker(
    ticker: str,
    client: httpx.AsyncClient,
    news_api_key: str,
    sem: asyncio.Semaphore,
    market_ctx: dict,
) -> dict | None:
    """Fetch data, compute all technical signals, score the setup.
    Always uses the mechanical trade plan — AI is NOT called here.
    Only confirmed hot picks (BUY / STRONG BUY) proceed to Phase 2.
    """
    async with sem:
        try:
            intra, daily, headlines, earnings = await asyncio.gather(
                fetch_intraday(ticker, client),
                fetch_daily(ticker, client),
                fetch_news(ticker, news_api_key, client),
                fetch_earnings_data(ticker, client),
            )

            if intra is None or len(intra) < 5:
                return None

            sig = compute_signals(intra, daily)
            if earnings:
                sig.update(earnings)
            score, grade, reasons, danger = score_setup(sig, market_ctx)

            return {
                "ticker":          ticker,
                "grade":           grade,
                "score":           score,
                "danger":          danger,
                "signals":         sig,
                "reasons":         reasons,
                "trade_plan":      _mechanical_plan(sig, score, grade),
                "headlines":       headlines[:3],
                "price_history":   _price_history(daily),
                "pattern_history": _historical_patterns(daily, sig),
                "scanned_at":      datetime.utcnow().isoformat(),
            }
        except Exception as exc:
            logger.error("Error scanning %s: %s", ticker, exc)
            return None
        finally:
            await asyncio.sleep(0.35)


# ── Phase 2: AI enrichment — only for confirmed hot picks ─────────────────────

async def _apply_ai_plan(result: dict, market_ctx: dict) -> None:
    """Replace the mechanical plan with an AI-generated plan via Claude CLI (in-place)."""
    plan = await ai_trade_plan(
        result["ticker"],
        result["signals"],
        result["headlines"],
        result["score"],
        result["grade"],
        result["reasons"],
        market_ctx,
    )
    result["trade_plan"] = plan
    logger.info(
        "AI plan for %s: conviction=%s, entry=%s, target=%s, stop=%s",
        result["ticker"],
        plan.get("conviction", "—"),
        plan.get("entry",  "—"),
        plan.get("target", "—"),
        plan.get("stop",   "—"),
    )


# ── Main entry ────────────────────────────────────────────────────────────────

async def analyze_single_ticker(ticker: str) -> dict:
    """Full pipeline for one custom ticker — always runs AI if key is available.
    Used by the /api/hotpicks/analyze endpoint for on-demand analysis.
    """
    ticker        = ticker.upper().strip()
    news_api_key  = os.getenv("NEWS_API_KEY", "")
    sem           = asyncio.Semaphore(1)

    async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
        market_ctx = await fetch_market_context(client)
        result     = await _scan_ticker(ticker, client, news_api_key, sem, market_ctx)

    if result is None:
        return {"error": f"Could not fetch data for '{ticker}'. Check the symbol and try again."}

    await _apply_ai_plan(result, market_ctx)

    result["market_ctx"] = market_ctx
    return result


async def run_hot_picks(custom_tickers: list[str] | None = None) -> dict:
    base_tickers  = custom_tickers or DEFAULT_UNIVERSE
    news_api_key  = os.getenv("NEWS_API_KEY", "")
    sem           = asyncio.Semaphore(4)

    async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
        # ── Market context + FMP dynamic tickers fetched in parallel ──────────
        market_ctx, dynamic = await asyncio.gather(
            fetch_market_context(client),
            fetch_dynamic_tickers(client),
        )
        logger.info(
            "Market regime: %s | SPY %+.2f%% | QQQ %+.2f%%",
            market_ctx["regime"].upper(),
            market_ctx["spy_chg"],
            market_ctx["qqq_chg"],
        )

        # Merge base universe with FMP movers (order-preserving, deduplicated)
        seen    = set(base_tickers)
        tickers = list(base_tickers)
        for t in dynamic:
            if t not in seen:
                seen.add(t)
                tickers.append(t)

        # ── Phase 1: technical scan — ALL tickers, NO AI ──────────────────────
        logger.info(
            "Phase 1: scanning %d tickers (%d base + %d FMP dynamic)...",
            len(tickers), len(base_tickers), len(tickers) - len(base_tickers),
        )
        tasks = [_scan_ticker(t, client, news_api_key, sem, market_ctx) for t in tickers]
        raw   = await asyncio.gather(*tasks, return_exceptions=True)

    results = [r for r in raw if r and not isinstance(r, Exception)]
    results.sort(key=lambda x: x["score"], reverse=True)

    hot_picks = [r for r in results if r["grade"] in ("STRONG BUY", "BUY")][:5]
    on_watch  = [r for r in results if r["grade"] == "WATCH"][:5]
    avoid     = [r for r in results if r["grade"] == "SKIP"][-5:]

    # ── Phase 2: AI analysis — ONLY confirmed hot picks (via Claude CLI) ────────
    if hot_picks:
        logger.info(
            "Phase 2: running AI analysis on %d confirmed hot picks: %s",
            len(hot_picks),
            [p["ticker"] for p in hot_picks],
        )
        ai_tasks = [_apply_ai_plan(pick, market_ctx) for pick in hot_picks]
        await asyncio.gather(*ai_tasks, return_exceptions=True)
    else:
        logger.info("Phase 2 skipped — no hot picks.")

    return {
        "run_at":          datetime.utcnow().isoformat(),
        "total_scanned":   len(results),
        "universe_size":   len(tickers),
        "dynamic_tickers": dynamic,
        "market_ctx":      market_ctx,
        "hot_picks":       hot_picks,
        "on_watch":        on_watch,
        "avoid":           avoid,
        "all_results":     results,
    }
