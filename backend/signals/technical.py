"""
signals/technical.py
Computes RSI, MACD, VWAP, Bollinger Bands, and EMA from price history.
Returns a dict of indicator values and a composite technical score.
"""
import numpy as np
import pandas as pd


def compute_signals(df: pd.DataFrame) -> dict:
    """
    Expects a DataFrame with columns: close, volume, ts
    Returns dict of indicator values + a score from -3 to +3
    """
    if df.empty or len(df) < 26:
        return {"error": "insufficient data", "tech_score": 0}

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    # ── RSI (14) ─────────────────────────────────────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    # ── MACD (12, 26, 9) ──────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_val = round(float(macd_line.iloc[-1]), 4)
    macd_hist = round(float((macd_line - signal_line).iloc[-1]), 4)
    macd_bullish = bool(macd_line.iloc[-1] > signal_line.iloc[-1])
    macd_cross_up = bool(
        macd_line.iloc[-1] > signal_line.iloc[-1]
        and macd_line.iloc[-2] <= signal_line.iloc[-2]
    )

    # ── VWAP ─────────────────────────────────────────────────────────────────
    vwap = float((close * volume).cumsum().iloc[-1] / volume.cumsum().iloc[-1]) if volume.sum() > 0 else float(close.mean())
    price_above_vwap = bool(close.iloc[-1] > vwap)

    # ── Bollinger Bands (20, 2) ───────────────────────────────────────────────
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = float((sma20 + 2 * std20).iloc[-1])
    bb_lower = float((sma20 - 2 * std20).iloc[-1])
    bb_mid = float(sma20.iloc[-1])
    current_price = float(close.iloc[-1])
    bb_pct = round((current_price - bb_lower) / (bb_upper - bb_lower + 1e-9) * 100, 1)

    # ── EMA 9 / 21 cross ─────────────────────────────────────────────────────
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema_bullish = bool(ema9.iloc[-1] > ema21.iloc[-1])

    # ── Composite score ──────────────────────────────────────────────────────
    score = 0.0
    if rsi < 30:
        score += 1.5   # oversold
    elif rsi < 40:
        score += 0.5
    elif rsi > 70:
        score -= 1.5   # overbought
    elif rsi > 60:
        score -= 0.5

    if macd_cross_up:
        score += 1.0
    elif macd_bullish:
        score += 0.5
    else:
        score -= 0.5

    if price_above_vwap:
        score += 0.5
    else:
        score -= 0.5

    if bb_pct < 10:
        score += 0.5   # near lower band = potential bounce
    elif bb_pct > 90:
        score -= 0.5   # near upper band = extended

    if ema_bullish:
        score += 0.5
    else:
        score -= 0.5

    return {
        "rsi": round(float(rsi), 2) if not np.isnan(rsi) else 50.0,
        "macd": macd_val,
        "macd_hist": macd_hist,
        "macd_bullish": macd_bullish,
        "macd_cross_up": macd_cross_up,
        "vwap": round(vwap, 4),
        "price_above_vwap": price_above_vwap,
        "bb_upper": round(bb_upper, 4),
        "bb_lower": round(bb_lower, 4),
        "bb_mid": round(bb_mid, 4),
        "bb_pct": bb_pct,
        "ema_bullish": ema_bullish,
        "tech_score": round(score, 2),
        "current_price": round(current_price, 4),
    }
