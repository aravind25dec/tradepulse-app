"""
autotrader/engine.py
Automated stock screening + trading agent (see ../trade-app.md for the spec).

1. Data Engine      — yf.download() pulls 2y of daily OHLCV per ticker (free, no API key).
2. Analytics Engine  — pandas_ta computes RSI, SMA50/200, MACD, Bollinger Bands, OBV.
3. Decision Matrix   — evaluates the most recent daily candle against a 4-category
                        confirmation matrix (momentum / trend / volume / volatility)
                        and classifies BUY / SELL / HOLD with a confidence score.
4. Execution Engine  — when a BUY signal is HIGH confidence, sizes a position at 2%
                        portfolio risk and submits a bracket order (limit entry +
                        3% stop-loss + 9% take-profit) via the Alpaca Trade API.
"""

import json
import logging
import os
from datetime import datetime, timezone

import numpy as np

# pandas_ta 0.3.14b0 (last PyPI release) imports `numpy.NaN`, an alias removed
# in numpy>=2.0 — shim it back before pandas_ta is imported.
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas as pd
import yfinance as yf
import pandas_ta as ta  # noqa: F401 — importing registers the df.ta accessor

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import APIError
except ImportError:  # SDK not installed — screener still works, trading is disabled
    tradeapi = None
    APIError = Exception

log = logging.getLogger(__name__)

DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA"]

RISK_PER_TRADE_PCT = 0.02   # 2% of portfolio value risked per trade
STOP_LOSS_PCT      = 0.03   # stop-loss set 3% below entry
TAKE_PROFIT_PCT    = 0.09   # take-profit set 9% above entry

# ── Dynamic universe discovery ──────────────────────────────────────────
# Yahoo's day_gainers / day_losers / most_actives screeners (via yfinance's
# screen() API) surface tickers outside the fixed core watchlist so the scan
# reacts to whatever is actually moving today, not just 5 hardcoded names.
MOVER_SCREENS      = ["day_gainers", "day_losers", "most_actives"]
MIN_MOVER_PRICE    = 3.0        # skip sub-$3 penny stocks — noisy, hard to size/execute
MIN_MOVER_VOLUME   = 300_000    # skip illiquid names
MAX_PER_SCREEN     = 15
MAX_UNIVERSE_SIZE  = 40         # cap total scan size — each ticker is a separate yfinance fetch


# ═══════════════════════════════════════════════════════════════════════
# 1. Data Engine
# ═══════════════════════════════════════════════════════════════════════

def fetch_history(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    """Pull daily OHLCV via yfinance. 2y of daily bars (~500 rows) comfortably
    covers the 300-day minimum needed for an accurate 200-day SMA.
    Delisted tickers, bad symbols, and network errors return None instead of raising.
    """
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty or len(df) < 210:
            log.warning("Insufficient history for %s (%d rows)", ticker, 0 if df is None else len(df))
            return None
        # Single-ticker requests can still come back with a MultiIndex column set
        # depending on yfinance version — flatten to plain column names.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.rename(columns=str.lower)
    except Exception as exc:
        log.error("yfinance fetch failed for %s: %s", ticker, exc)
        return None


def discover_top_movers(count_per_screen: int = MAX_PER_SCREEN) -> dict[str, str]:
    """Query Yahoo's day_gainers / day_losers / most_actives screeners and return
    {ticker: source_label} for liquid, real-priced equities. Best-effort — a
    failed or empty screener is skipped rather than raised, so a Yahoo outage
    just shrinks the discovered universe instead of breaking the scan.
    """
    discovered: dict[str, str] = {}
    for screen_name in MOVER_SCREENS:
        try:
            res = yf.screen(screen_name, count=count_per_screen)
        except Exception as exc:
            log.warning("Screener '%s' failed: %s", screen_name, exc)
            continue
        for q in (res or {}).get("quotes", []):
            symbol = q.get("symbol")
            if not symbol or q.get("quoteType") != "EQUITY":
                continue
            price  = q.get("regularMarketPrice") or 0
            volume = q.get("regularMarketVolume") or 0
            if price < MIN_MOVER_PRICE or volume < MIN_MOVER_VOLUME:
                continue
            discovered.setdefault(symbol, screen_name)
    return discovered


def build_scan_universe(max_size: int = MAX_UNIVERSE_SIZE) -> tuple[list[str], dict[str, str]]:
    """Core watchlist + freshly discovered top movers, deduped and capped.
    Returns (tickers, {ticker: source}) so callers/UI can show why each
    ticker is in today's scan (core / day_gainers / day_losers / most_actives).
    """
    sources = {t: "core" for t in DEFAULT_WATCHLIST}
    for symbol, screen_name in discover_top_movers().items():
        if symbol not in sources and len(sources) < max_size:
            sources[symbol] = screen_name
    return list(sources.keys()), sources


# ═══════════════════════════════════════════════════════════════════════
# 2. Analytics Engine — pandas_ta indicator matrix
# ═══════════════════════════════════════════════════════════════════════

def _find_col(df: pd.DataFrame, prefix: str) -> str:
    """pandas_ta names Bollinger columns with a version-dependent std suffix
    (e.g. '2.0' vs '2'), so resolve by prefix rather than a hardcoded name."""
    matches = [c for c in df.columns if c.startswith(prefix)]
    if not matches:
        raise KeyError(f"pandas_ta did not produce a column starting with '{prefix}'")
    return matches[0]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the full indicator matrix using pandas_ta's DataFrame accessor
    (`df.ta.*`). Each call computes the indicator over the whole series and
    appends it as new column(s) directly onto `df` (append=True), so the
    decision matrix below just reads the last row — no manual merge needed.
    """
    # Momentum — 14-period RSI, the standard daily-swing lookback
    df.ta.rsi(length=14, append=True)                        # -> RSI_14

    # Trend — 50/200-day SMAs for structural bias, MACD(12,26,9) for cross timing
    df.ta.sma(length=50,  append=True)                        # -> SMA_50
    df.ta.sma(length=200, append=True)                        # -> SMA_200
    df.ta.macd(fast=12, slow=26, signal=9, append=True)       # -> MACD_12_26_9 / MACDh_12_26_9 / MACDs_12_26_9

    # Volatility — 20-period Bollinger Bands, 2 standard deviations
    df.ta.bbands(length=20, std=2, append=True)               # -> BBL/BBM/BBU/BBB/BBP_20_*

    # Volume — On-Balance Volume for accumulation/distribution direction
    df.ta.obv(append=True)                                    # -> OBV

    # 20-day average volume, for the "current vs average" intraday volume check
    df["VOL_SMA_20"] = df["volume"].rolling(20).mean()

    # VWAP is inherently intraday; the daily-bar substitute is the rolling
    # 20-day Volume-Weighted Average Close.
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["VWAC_20"] = (
        (typical_price * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    )

    # Normalize the Bollinger columns to stable names for the rest of the module
    df["BB_LOWER"] = df[_find_col(df, "BBL_20")]
    df["BB_UPPER"] = df[_find_col(df, "BBU_20")]
    df["BB_WIDTH"] = df[_find_col(df, "BBB_20")]

    return df


# ═══════════════════════════════════════════════════════════════════════
# 3. Decision Matrix
# ═══════════════════════════════════════════════════════════════════════

def _macd_bull_cross_recent(df: pd.DataFrame, lookback: int = 3) -> bool:
    hist = (df["MACD_12_26_9"] - df["MACDs_12_26_9"]).tail(lookback + 1)
    return bool(((hist.shift(1) <= 0) & (hist > 0)).any())


def _macd_bear_cross_recent(df: pd.DataFrame, lookback: int = 3) -> bool:
    hist = (df["MACD_12_26_9"] - df["MACDs_12_26_9"]).tail(lookback + 1)
    return bool(((hist.shift(1) >= 0) & (hist < 0)).any())


def _rsi_bounce_from_oversold(df: pd.DataFrame, lookback: int = 3) -> bool:
    rsi = df["RSI_14"].tail(lookback + 1)
    return bool(((rsi.shift(1) <= 30) & (rsi > 30)).any())


def _rsi_drop_from_overbought(df: pd.DataFrame, lookback: int = 3) -> bool:
    rsi = df["RSI_14"].tail(lookback + 1)
    return bool(((rsi.shift(1) >= 70) & (rsi < 70)).any())


def _build_indicator_details(
    *, rsi: float, rsi_bounce_up: bool, rsi_drop_down: bool,
    price: float, sma50: float, sma200: float, above_rising_200sma: bool, below_50sma: bool,
    macd_cross_up: bool, macd_cross_down: bool,
    vol_today: float, vol_avg20: float, bullish_volume: bool, bearish_volume: bool,
    bb_lower: float, bb_upper: float, bounce_off_lower: bool, squeeze_expansion: bool,
    near_upper_band: bool, rejection: bool,
) -> list[dict]:
    """Turn the same booleans/values the decision matrix already computed into
    plain-English, per-category explanations for the UI's click-to-expand detail panel.
    Each entry: category, title, verdict (bullish/bearish/neutral), a ticker-specific
    explanation, and a one-time "what does this indicator mean" primer.
    """
    vol_ratio = vol_today / vol_avg20 if vol_avg20 else None
    details = []

    # ── Momentum (RSI) ──
    if rsi_bounce_up:
        verdict, text = "bullish", (
            f"RSI just bounced up from oversold territory (below 30) to {rsi:.1f} — a classic sign that "
            f"selling pressure is fading and buyers are stepping back in."
        )
    elif rsi_drop_down:
        verdict, text = "bearish", (
            f"RSI just dropped out of overbought territory (above 70) to {rsi:.1f} — often an early warning "
            f"that a rally is running out of steam."
        )
    elif rsi > 55:
        verdict, text = "bullish", f"RSI is at {rsi:.1f}, comfortably above the neutral 50 line — buyers currently have the upper hand."
    elif rsi < 45:
        verdict, text = "bearish", f"RSI is at {rsi:.1f}, below the neutral 50 line — sellers currently have the upper hand."
    else:
        verdict, text = "neutral", f"RSI is at {rsi:.1f}, right around the neutral 50 line — no clear momentum edge either way."
    details.append({
        "category": "momentum", "title": "Momentum (RSI)", "verdict": verdict, "explanation": text,
        "plain": "RSI (Relative Strength Index) measures how fast and how far the price has moved recently, "
                 "on a 0-100 scale. Above 70 usually means 'overbought' (due for a pullback); below 30 usually "
                 "means 'oversold' (due for a bounce).",
    })

    # ── Trend (SMA + MACD) ──
    bits = []
    if above_rising_200sma:
        bits.append(f"price (${price:.2f}) is above a rising 200-day average (${sma200:.2f}), so the long-term trend is up")
    elif below_50sma:
        bits.append(f"price (${price:.2f}) has broken below its 50-day average (${sma50:.2f}), so short-term momentum is weakening")
    if macd_cross_up:
        bits.append("the MACD lines just crossed bullish, a fresh signal that upward momentum is building")
    elif macd_cross_down:
        bits.append("the MACD lines just crossed bearish, a fresh signal that downward momentum is building")
    if above_rising_200sma and macd_cross_up:
        verdict = "bullish"
    elif below_50sma or macd_cross_down:
        verdict = "bearish"
    else:
        verdict = "neutral"
    text = ("; ".join(bits).capitalize() + ".") if bits else (
        f"Price (${price:.2f}) is drifting between its short and long-term averages "
        f"(50d ${sma50:.2f} / 200d ${sma200:.2f}) with no fresh MACD cross — no clean trend confirmation right now."
    )
    details.append({
        "category": "trend", "title": "Trend (Moving Averages + MACD)", "verdict": verdict, "explanation": text,
        "plain": "Moving averages smooth out daily noise to reveal the underlying trend. MACD compares a fast "
                 "and slow average to catch trend changes early — a 'bullish cross' often marks the start of "
                 "upward momentum, a 'bearish cross' the start of downward momentum.",
    })

    # ── Volume + OBV ──
    ratio_txt = f"{vol_ratio:.2f}x" if vol_ratio is not None else "an unclear multiple of"
    if bullish_volume:
        verdict, text = "bullish", (
            f"Today's volume is {ratio_txt} the 20-day average, and On-Balance Volume has been rising — real "
            f"buying pressure is showing up behind this move, not just a quiet drift."
        )
    elif bearish_volume:
        verdict, text = "bearish", (
            f"Today's volume is {ratio_txt} the 20-day average on a down day — that's distribution, real "
            f"sellers stepping in rather than just noise."
        )
    else:
        verdict, text = "neutral", (
            f"Volume is unremarkable right now ({ratio_txt} the 20-day average) — not enough conviction "
            f"behind the move to read much into it."
        )
    details.append({
        "category": "volume", "title": "Volume & OBV", "verdict": verdict, "explanation": text,
        "plain": "Volume is how many shares are trading — a price move on unusually high volume means more "
                 "traders are 'voting' with real money, so it's more meaningful. On-Balance Volume (OBV) tracks "
                 "whether that volume has been flowing in on up-days (accumulation) or down-days (distribution).",
    })

    # ── Volatility (Bollinger Bands) ──
    if bounce_off_lower:
        verdict, text = "bullish", (
            f"Price (${price:.2f}) is bouncing off the lower Bollinger Band (${bb_lower:.2f}) — an area that's "
            f"often oversold and due for a reflex bounce."
        )
    elif squeeze_expansion:
        verdict, text = "bullish", (
            "The Bollinger Bands are expanding out of a tight squeeze — a classic setup right before a bigger "
            "move gets underway."
        )
    elif near_upper_band and rejection:
        verdict, text = "bearish", (
            f"Price (${price:.2f}) pushed up near the upper Bollinger Band (${bb_upper:.2f}) and just reversed "
            f"lower — a classic exhaustion/rejection signal."
        )
    else:
        verdict, text = "neutral", (
            f"Price (${price:.2f}) is trading in the middle of its normal range (${bb_lower:.2f}-${bb_upper:.2f}) "
            f"— no volatility extreme to read into right now."
        )
    details.append({
        "category": "volatility", "title": "Volatility (Bollinger Bands)", "verdict": verdict, "explanation": text,
        "plain": "Bollinger Bands draw a 'normal trading range' around the price (2 standard deviations above/below "
                 "a 20-day average). Bouncing off the lower band can signal an oversold bounce; stretching to the "
                 "upper band and reversing can signal exhaustion; a tight squeeze followed by expansion often "
                 "precedes a bigger move.",
    })

    return details


def evaluate_ticker(ticker: str, df: pd.DataFrame) -> dict:
    """Evaluate the most current daily candle against the BUY/SELL/HOLD confirmation matrix.
    A signal only fires when ALL FOUR categories (momentum, trend, volume, volatility)
    confirm the same direction; otherwise the ticker is HOLD.
    """
    last   = df.iloc[-1]
    ago10  = df.iloc[-11] if len(df) > 10 else df.iloc[0]
    ago5_w = df["BB_WIDTH"].iloc[-6] if len(df) > 6 else float(last["BB_WIDTH"])
    ago5_s = df["SMA_200"].iloc[-6]  if len(df) > 6 else float(last["SMA_200"])

    price      = float(last["close"])
    prev_close = float(df["close"].iloc[-2])
    rsi        = float(last["RSI_14"])
    sma50      = float(last["SMA_50"])
    sma200     = float(last["SMA_200"])
    bb_lower   = float(last["BB_LOWER"])
    bb_upper   = float(last["BB_UPPER"])
    bb_width   = float(last["BB_WIDTH"])
    obv_now    = float(last["OBV"])
    obv_10ago  = float(ago10["OBV"])
    vol_today  = float(last["volume"])
    vol_avg20  = float(last["VOL_SMA_20"])

    # ── Momentum: RSI bouncing out of oversold OR holding steady above 50 ──
    rsi_bounce_up      = _rsi_bounce_from_oversold(df)
    bullish_momentum    = rsi_bounce_up or rsi > 50
    rsi_drop_down       = _rsi_drop_from_overbought(df)
    bearish_momentum    = rsi_drop_down or rsi < 50

    # ── Trend: price above a rising 200-SMA + fresh bullish MACD cross ──
    above_rising_200sma = price > sma200 and sma200 > float(ago5_s)
    macd_cross_up        = _macd_bull_cross_recent(df)
    bullish_trend         = above_rising_200sma and macd_cross_up

    below_50sma           = price < sma50
    macd_cross_down       = _macd_bear_cross_recent(df)
    bearish_trend         = below_50sma or macd_cross_down

    # ── Volume: today's volume >= 1.2x the 20-day average, OBV confirms direction ──
    high_volume  = vol_avg20 > 0 and vol_today >= 1.2 * vol_avg20
    obv_positive = obv_now > obv_10ago
    bullish_volume = high_volume and obv_positive
    price_dropping = price < prev_close
    bearish_volume  = high_volume and price_dropping

    # ── Volatility: bouncing off the lower band, or squeeze expanding; rejection near the upper band ──
    bounce_off_lower  = price <= bb_lower * 1.02
    squeeze_expansion = bb_width > float(ago5_w) * 1.15
    bullish_volatility = bounce_off_lower or squeeze_expansion

    near_upper_band = price >= bb_upper * 0.98
    rejection       = price_dropping and price < float(last["high"]) * 0.995
    bearish_volatility = near_upper_band and rejection

    buy_categories  = [bullish_momentum, bullish_trend, bullish_volume, bullish_volatility]
    sell_categories = [bearish_momentum, bearish_trend, bearish_volume, bearish_volatility]
    buy_count  = sum(buy_categories)
    sell_count = sum(sell_categories)

    metrics = {
        "rsi":                round(rsi, 1),
        "above_200sma":       bool(price > sma200),
        "macd_crossover":     bool(macd_cross_up),
        "high_volume":        bool(high_volume),
        "vol_ratio":          round(vol_today / vol_avg20, 2) if vol_avg20 else None,
        "bb_pct":             round((price - bb_lower) / (bb_upper - bb_lower), 3) if bb_upper > bb_lower else None,
        "obv_trend_positive": bool(obv_positive),
    }

    if all(buy_categories):
        signal     = "BUY"
        confidence = "HIGH" if vol_avg20 and vol_today >= 1.5 * vol_avg20 else "MEDIUM"
        reasoning = (
            f"RSI at {rsi:.1f} confirms "
            f"{'a fresh bounce out of oversold' if rsi_bounce_up else 'bullish momentum above the 50 centerline'}, "
            f"while price holds above a rising 200-day SMA with a bullish MACD cross within the last 3 candles. "
            f"Volume is {vol_today / vol_avg20:.2f}x the 20-day average with accumulating OBV, and price is "
            f"{'bouncing off the lower Bollinger Band' if bounce_off_lower else 'expanding out of a volatility squeeze'}."
        )
    elif all(sell_categories):
        signal     = "SELL"
        confidence = "HIGH" if high_volume and abs(price / prev_close - 1) > 0.02 else "MEDIUM"
        reasoning = (
            f"RSI at {rsi:.1f} confirms "
            f"{'a drop out of overbought' if rsi_drop_down else 'bearish momentum below the 50 centerline'}, "
            f"while price {'breaks below the 50-day SMA' if below_50sma else 'shows a bearish MACD cross'}. "
            f"Volume is {vol_today / vol_avg20:.2f}x the 20-day average on a down day, with price rejecting near "
            f"the upper Bollinger Band."
        )
    else:
        signal     = "HOLD"
        confidence = "MEDIUM" if max(buy_count, sell_count) >= 2 else "LOW"
        reasoning = (
            f"Mixed signals — {buy_count}/4 bullish and {sell_count}/4 bearish confirmation categories met. "
            f"No collective confirmation across momentum, trend, volume, and volatility, so this stays on the sidelines."
        )

    if signal == "BUY":
        verdict_summary = (
            f"All 4 signals agree — momentum, trend, volume, and volatility are all pointing up. "
            f"That agreement is why this is a BUY with {confidence} confidence."
        )
    elif signal == "SELL":
        verdict_summary = (
            f"All 4 signals agree — momentum, trend, volume, and volatility are all pointing down. "
            f"That agreement is why this is a SELL with {confidence} confidence."
        )
    else:
        verdict_summary = (
            f"Only {buy_count} of 4 checks lean bullish and {sell_count} of 4 lean bearish — since they don't all "
            f"agree, the system stays on the sidelines (HOLD) rather than guessing."
        )

    indicator_details = _build_indicator_details(
        rsi=rsi, rsi_bounce_up=rsi_bounce_up, rsi_drop_down=rsi_drop_down,
        price=price, sma50=sma50, sma200=sma200,
        above_rising_200sma=above_rising_200sma, below_50sma=below_50sma,
        macd_cross_up=macd_cross_up, macd_cross_down=macd_cross_down,
        vol_today=vol_today, vol_avg20=vol_avg20, bullish_volume=bullish_volume, bearish_volume=bearish_volume,
        bb_lower=bb_lower, bb_upper=bb_upper, bounce_off_lower=bounce_off_lower, squeeze_expansion=squeeze_expansion,
        near_upper_band=near_upper_band, rejection=rejection,
    )

    return {
        "ticker":             ticker,
        "signal":             signal,
        "confidence_score":   confidence,
        "metrics":            metrics,
        "reasoning":          reasoning,
        "verdict_summary":    verdict_summary,
        "indicator_details":  indicator_details,
        "price":              round(price, 2),
        "evaluated_at":       datetime.now(timezone.utc).isoformat(),
    }


def screen_ticker(ticker: str) -> dict | None:
    """Full pipeline for one ticker: fetch -> indicators -> decision. None on failure."""
    try:
        df = fetch_history(ticker)
        if df is None:
            return None
        df = compute_indicators(df)
        return evaluate_ticker(ticker, df)
    except Exception as exc:
        log.error("screen_ticker failed for %s: %s", ticker, exc)
        return None


def run_screen(tickers: list[str] | None = None, sources: dict[str, str] | None = None) -> list[dict]:
    """4. Main Orchestrator — screens the universe and returns the JSON-ready result array.
    When `tickers` is None, auto-discovers today's universe (core watchlist +
    top gainers/losers/most-actives) via build_scan_universe() instead of just
    the 5-name DEFAULT_WATCHLIST.
    """
    if tickers is None:
        tickers, sources = build_scan_universe()
    sources = sources or {}
    results = []
    for t in tickers:
        r = screen_ticker(t)
        if r is None:
            results.append({
                "ticker":           t,
                "signal":           "HOLD",
                "confidence_score": "LOW",
                "metrics":          {},
                "reasoning":        "Could not fetch sufficient market data for this ticker.",
                "error":            True,
                "source":           sources.get(t, "custom"),
            })
        else:
            r["source"] = sources.get(t, "custom")
            results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════
# 4. Alpaca execution engine
# ═══════════════════════════════════════════════════════════════════════

def _get_alpaca_client():
    """Load Alpaca credentials from the environment (os.environ) and build a client.
    Returns None — never raises — when the SDK or keys are missing, so the
    screener keeps running in signal-only mode without Alpaca configured.
    """
    if tradeapi is None:
        log.warning("alpaca-trade-api not installed — trading disabled, signal-only mode")
        return None

    # Prefer the alpaca-trade-api SDK's standard APCA_* names, but fall back to
    # ALPACA_*  — the names the rest of this repo's backend already uses in
    # backend/.env — so credentials don't need to be duplicated under both.
    key    = os.environ.get("APCA_API_KEY_ID")     or os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    base   = os.environ.get("APCA_API_BASE_URL")   or os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not key or not secret:
        log.warning("APCA_API_KEY_ID / APCA_API_SECRET_KEY not set — trading disabled, signal-only mode")
        return None

    return tradeapi.REST(key, secret, base, api_version="v2")


def calculate_position_size(portfolio_value: float, entry_price: float, stop_price: float,
                             risk_pct: float = RISK_PER_TRADE_PCT) -> int:
    """Size the position so a full stop-out loses exactly risk_pct of the portfolio."""
    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0 or portfolio_value <= 0:
        return 0
    return int((portfolio_value * risk_pct) // risk_per_share)


def execute_bracket_order(ticker: str, entry_price: float, api=None) -> dict:
    """
    Check the account's available cash (api.get_account()), size the position at
    2% portfolio risk, and submit a bracket order: a limit entry with a nested
    stop-loss (3% below entry) and a nested take-profit (9% above entry).

    Every failure mode — no credentials, no cash, a rejected order, a dropped
    connection, insufficient buying power — is caught and returned as a
    structured dict instead of raised, so one bad order never crashes a caller.
    """
    api = api or _get_alpaca_client()
    if api is None:
        return {"executed": False, "reason": "Alpaca API not configured (missing keys or SDK)."}

    stop_price  = round(entry_price * (1 - STOP_LOSS_PCT), 2)
    take_profit = round(entry_price * (1 + TAKE_PROFIT_PCT), 2)

    try:
        account = api.get_account()

        if getattr(account, "trading_blocked", False):
            return {"executed": False, "reason": "Account is currently restricted from trading."}

        cash            = float(account.cash)
        portfolio_value = float(account.portfolio_value)

        if cash <= 0:
            return {"executed": False, "reason": "No available cash balance."}

        qty = calculate_position_size(portfolio_value, entry_price, stop_price)
        if qty < 1:
            return {"executed": False, "reason": "2% portfolio risk sizes to less than 1 share at this stop distance."}

        # Cap to what cash can actually cover — guards against a partial-fill /
        # buying-power rejection when other open positions have consumed cash.
        max_affordable = int(cash // entry_price)
        if max_affordable < 1:
            return {"executed": False, "reason": f"Insufficient buying power: ${cash:.2f} available, need ${entry_price:.2f}/share."}
        qty = min(qty, max_affordable)

        order = api.submit_order(
            symbol=ticker,
            qty=qty,
            side="buy",
            type="limit",
            limit_price=round(entry_price, 2),
            time_in_force="gtc",
            order_class="bracket",
            stop_loss={"stop_price": stop_price},
            take_profit={"limit_price": take_profit},
        )

        return {
            "executed":    True,
            "order_id":    str(order.id),
            "status":      order.status,   # e.g. "new" — bracket legs activate once the entry fills
            "qty":         qty,
            "entry":       round(entry_price, 2),
            "stop_loss":   stop_price,
            "take_profit": take_profit,
        }

    except APIError as exc:
        msg = str(exc)
        if "insufficient" in msg.lower() and "buying power" in msg.lower():
            return {"executed": False, "reason": f"Insufficient buying power: {msg}"}
        return {"executed": False, "reason": f"Alpaca rejected the order: {msg}"}
    except (ConnectionError, TimeoutError) as exc:
        return {"executed": False, "reason": f"Connectivity dropped while placing the order: {exc}"}
    except Exception as exc:
        log.error("Unexpected error submitting bracket order for %s: %s", ticker, exc)
        return {"executed": False, "reason": f"Unexpected error: {exc}"}


def check_order_status(order_id: str, api=None) -> dict:
    """Poll an order's fill state — surfaces partial fills explicitly for the UI."""
    api = api or _get_alpaca_client()
    if api is None:
        return {"error": "Alpaca API not configured."}
    try:
        order = api.get_order(order_id)
        return {
            "order_id":         order_id,
            "status":           order.status,  # filled / partially_filled / canceled / ...
            "filled_qty":       order.filled_qty,
            "qty":              order.qty,
            "filled_avg_price": order.filled_avg_price,
        }
    except APIError as exc:
        return {"error": f"Alpaca error: {exc}"}
    except Exception as exc:
        return {"error": f"Unexpected error: {exc}"}


def get_account_snapshot(api=None) -> dict:
    api = api or _get_alpaca_client()
    if api is None:
        return {"connected": False}
    try:
        account = api.get_account()
        return {
            "connected":           True,
            "cash":                float(account.cash),
            "portfolio_value":     float(account.portfolio_value),
            "buying_power":        float(account.buying_power),
            "trading_blocked":     bool(getattr(account, "trading_blocked", False)),
            "pattern_day_trader":  bool(getattr(account, "pattern_day_trader", False)),
        }
    except Exception as exc:
        return {"connected": False, "error": str(exc)}


def run_screen_and_trade(tickers: list[str] | None = None, auto_execute: bool = False,
                          sources: dict[str, str] | None = None) -> list[dict]:
    """
    Runs the full screen and — only when auto_execute=True — submits a bracket
    order for every HIGH-confidence BUY signal. Defaults to False so a scan
    never places live orders unless a caller (or the UI toggle) opts in explicitly.
    """
    results = run_screen(tickers, sources)
    if not auto_execute:
        return results

    api = _get_alpaca_client()
    for r in results:
        if r.get("signal") == "BUY" and r.get("confidence_score") == "HIGH" and not r.get("error"):
            r["trade_execution"] = execute_bracket_order(r["ticker"], r["price"], api=api)
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    watchlist = sys.argv[1:] or DEFAULT_WATCHLIST
    print(json.dumps(run_screen(watchlist), indent=2))
