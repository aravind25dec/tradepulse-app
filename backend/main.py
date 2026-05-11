"""
main.py
FastAPI application entry point.
- REST endpoints for signals, news, earnings, events, price history
- WebSocket endpoint that broadcasts live updates to all connected clients
- Background tasks: price polling, news polling, earnings polling, signal refresh
"""
import asyncio
import logging
import os
import math
from contextlib import asynccontextmanager
from typing import Any

import httpx
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from db.store import (
    get_all_signals, get_signal, get_news, get_events,
    get_earnings_calendar, get_price_df
)
from ingestion.prices import price_poll_loop, ALL_TICKERS, STOCK_TICKERS, CRYPTO_TICKERS, load_historical
from ingestion.news import news_poll_loop
from ingestion.earnings import earnings_poll_loop
from signals.engine import generate_signal
from signals.technical import compute_signals

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

# ── WebSocket connection manager ─────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.append(ws)
        log.info(f"WS client connected. Total: {len(self._clients)}")

    def disconnect(self, ws: WebSocket):
        self._clients.remove(ws)
        log.info(f"WS client disconnected. Total: {len(self._clients)}")

    async def broadcast(self, data: Any):
        dead = []
        for ws in self._clients:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.remove(ws)


manager = ConnectionManager()


async def broadcast(data: dict):
    await manager.broadcast(data)


def _fetch_yahoo_screen(scr_id: str, count: int = 50) -> list[dict]:
    url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
    params = {"scrIds": scr_id, "count": count, "formatted": "false"}
    headers = {"User-Agent": "Mozilla/5.0 TradePulse/1.0", "Accept": "application/json"}
    with httpx.Client(timeout=15.0, headers=headers) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()
    return (
        payload.get("finance", {})
        .get("result", [{}])[0]
        .get("quotes", [])
    )


def _fetch_yahoo_chart_closes(ticker: str, period: str = "1y", interval: str = "1d") -> list[float]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": period, "interval": interval, "includePrePost": "false", "events": "div,splits"}
    headers = {"User-Agent": "Mozilla/5.0 TradePulse/1.0", "Accept": "application/json"}
    with httpx.Client(timeout=15.0, headers=headers) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()

    results = payload.get("chart", {}).get("result") or []
    if not results:
        return []
    quote = (results[0].get("indicators", {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    return [float(v) for v in closes if v is not None]


def _rank_watchlist_movers(direction: str) -> list[dict]:
    candidates = []
    for ticker in STOCK_TICKERS:
        df = get_price_df(ticker)
        if df.empty or len(df) < 30:
            continue
        tech = compute_signals(df)
        if "error" in tech:
            continue

        close = df["close"].astype(float)
        current_price = float(close.iloc[-1])
        recent_12 = float(close.iloc[-12]) if len(close) >= 12 else float(close.iloc[0])
        recent_24 = float(close.iloc[-24]) if len(close) >= 24 else float(close.iloc[0])
        ret_1h = ((current_price - recent_12) / max(recent_12, 1e-9)) * 100
        ret_2h = ((current_price - recent_24) / max(recent_24, 1e-9)) * 100
        rsi = float(tech.get("rsi", 50))

        if direction == "up":
            if ret_1h <= 0 or ret_2h <= 0:
                continue
            if not tech.get("ema_bullish", False) or not tech.get("macd_bullish", False) or not tech.get("price_above_vwap", False):
                continue
            if rsi < 52 or rsi > 78:
                continue
            score = (ret_1h * 1.5) + (ret_2h * 1.0) + (float(tech.get("tech_score", 0)) * 6) + (2 if tech.get("macd_cross_up") else 0)
        else:
            if ret_1h >= 0 or ret_2h >= 0:
                continue
            if tech.get("ema_bullish", False) or tech.get("macd_bullish", False) or tech.get("price_above_vwap", False):
                continue
            if rsi < 22 or rsi > 48:
                continue
            score = (abs(ret_1h) * 1.5) + (abs(ret_2h) * 1.0) + (abs(float(tech.get("tech_score", 0))) * 6) + (2 if not tech.get("macd_cross_up") else 0)

        candidates.append({
            "ticker": ticker,
            "price": round(current_price, 4),
            "ret_1h_pct": round(ret_1h, 2),
            "ret_2h_pct": round(ret_2h, 2),
            "rsi": round(rsi, 2),
            "tech_score": round(float(tech.get("tech_score", 0)), 2),
            "momentum_score": round(score, 2),
            "source": "watchlist_technical",
        })
    candidates.sort(key=lambda x: x["momentum_score"], reverse=True)
    return candidates


async def signal_refresh_loop(tickers: list[str], interval: int = 120):
    """Regenerate signals every 2 minutes and broadcast."""
    await asyncio.sleep(30)  # wait for price data to load first
    while True:
        for ticker in tickers:
            try:
                signal = generate_signal(ticker)
                await broadcast({"type": "signal", "data": signal})
            except Exception as e:
                log.warning(f"Signal refresh error for {ticker}: {e}")
        await asyncio.sleep(interval)


# ── App lifespan (start background tasks) ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting background tasks...")
    tasks = [
        asyncio.create_task(price_poll_loop(broadcast, interval_seconds=60)),
        asyncio.create_task(news_poll_loop(ALL_TICKERS, broadcast, interval_seconds=600)),
        asyncio.create_task(earnings_poll_loop(STOCK_TICKERS, broadcast, interval_seconds=3600)),
        asyncio.create_task(signal_refresh_loop(ALL_TICKERS, interval=120)),
    ]
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="Day Trading Prediction API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── WebSocket ────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # Send current state on connect
        await ws.send_json({"type": "init", "data": {
            "signals": get_all_signals(),
            "events": get_events(20),
            "earnings": get_earnings_calendar(),
            "tickers": ALL_TICKERS,
            "stock_tickers": STOCK_TICKERS,
            "crypto_tickers": CRYPTO_TICKERS,
        }})
        while True:
            await ws.receive_text()  # keep alive
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ── REST endpoints ───────────────────────────────────────────────────────────
@app.get("/api/tickers")
def list_tickers():
    return {"stocks": STOCK_TICKERS, "crypto": CRYPTO_TICKERS}


@app.get("/api/watchlists")
def watchlists():
    return {
        "all": ALL_TICKERS,
        "stocks": STOCK_TICKERS,
        "crypto": CRYPTO_TICKERS,
    }


@app.post("/api/watchlist/add")
async def add_watchlist_ticker(payload: dict[str, str]):
    """
    Add a user ticker to runtime watchlist and backfill historical data.
    The ticker is kept in-memory for the current backend session.
    """
    raw = (payload.get("ticker") or "").strip().upper()
    if not raw:
        raise HTTPException(status_code=400, detail="ticker is required")

    ticker = raw
    if ticker in ALL_TICKERS:
        return {"ok": True, "ticker": ticker, "already_exists": True}

    is_crypto = ticker.endswith("-USD")
    if is_crypto:
        CRYPTO_TICKERS.append(ticker)
    else:
        STOCK_TICKERS.append(ticker)
    ALL_TICKERS.append(ticker)

    # Kick off historical load so signals/charts become available quickly.
    asyncio.create_task(load_historical(ticker))
    return {
        "ok": True,
        "ticker": ticker,
        "category": "crypto" if is_crypto else "stocks",
        "already_exists": False,
    }


@app.get("/api/movers/stocks")
def top_stock_movers(limit: int = 10):
    """
    Rank stocks that are elevated and moving up, based on:
    - positive short-term momentum
    - bullish technical structure
    - no overbought extreme
    """
    candidates = _rank_watchlist_movers("up")

    candidates.sort(key=lambda x: x["momentum_score"], reverse=True)
    limit = max(1, min(limit, 50))
    if len(candidates) >= limit:
        return candidates[:limit]

    # Fallback: pull broader market movers from Yahoo "day_gainers" API.
    try:
        quotes = _fetch_yahoo_screen("day_gainers", 50)
        existing = {c["ticker"] for c in candidates}
        for q in quotes:
            symbol = (q.get("symbol") or "").upper()
            if not symbol or symbol in existing:
                continue
            # Keep equities only (skip crypto/forex/funds-style symbols)
            if "-" in symbol or "=" in symbol or "^" in symbol:
                continue

            price = q.get("regularMarketPrice")
            day_pct = q.get("regularMarketChangePercent")
            volume = q.get("regularMarketVolume") or 0
            if price is None or day_pct is None:
                continue
            # Basic quality floor for actionable movers
            if float(price) < 3 or float(volume) < 300_000:
                continue

            candidates.append({
                "ticker": symbol,
                "price": round(float(price), 4),
                "ret_1h_pct": round(float(day_pct), 2),  # shown as momentum column in UI
                "ret_2h_pct": round(float(day_pct), 2),
                "rsi": None,
                "tech_score": None,
                "momentum_score": round(float(day_pct), 2),
                "source": "yahoo_day_gainers",
            })
            existing.add(symbol)
            if len(candidates) >= limit:
                break
    except Exception as e:
        log.warning(f"Unable to fetch Yahoo day gainers fallback: {e}")

    candidates.sort(key=lambda x: x.get("momentum_score", 0), reverse=True)
    return candidates[:limit]


@app.get("/api/movers/stocks/down")
def top_stock_losers(limit: int = 10):
    candidates = _rank_watchlist_movers("down")
    limit = max(1, min(limit, 50))
    if len(candidates) >= limit:
        return candidates[:limit]

    try:
        quotes = _fetch_yahoo_screen("day_losers", 50)
        existing = {c["ticker"] for c in candidates}
        for q in quotes:
            symbol = (q.get("symbol") or "").upper()
            if not symbol or symbol in existing:
                continue
            if "-" in symbol or "=" in symbol or "^" in symbol:
                continue
            price = q.get("regularMarketPrice")
            day_pct = q.get("regularMarketChangePercent")
            volume = q.get("regularMarketVolume") or 0
            if price is None or day_pct is None:
                continue
            if float(price) < 3 or float(volume) < 300_000:
                continue
            if float(day_pct) >= 0:
                continue

            candidates.append({
                "ticker": symbol,
                "price": round(float(price), 4),
                "ret_1h_pct": round(float(day_pct), 2),
                "ret_2h_pct": round(float(day_pct), 2),
                "rsi": None,
                "tech_score": None,
                "momentum_score": round(abs(float(day_pct)), 2),
                "source": "yahoo_day_losers",
            })
            existing.add(symbol)
            if len(candidates) >= limit:
                break
    except Exception as e:
        log.warning(f"Unable to fetch Yahoo day losers fallback: {e}")

    candidates.sort(key=lambda x: x.get("momentum_score", 0), reverse=True)
    return candidates[:limit]


@app.get("/api/stock/{ticker}/analysis")
def stock_analysis(ticker: str):
    symbol = ticker.upper()
    ranges = {
        "1d": ("1d", "5m"),
        "5d": ("5d", "30m"),
        "1mo": ("1mo", "1d"),
        "6mo": ("6mo", "1d"),
        "1y": ("1y", "1d"),
        "5y": ("5y", "1wk"),
        "max": ("max", "1mo"),
    }

    performance = {}
    for key, (period, interval) in ranges.items():
        closes = _fetch_yahoo_chart_closes(symbol, period=period, interval=interval)
        if len(closes) < 2:
            performance[key] = None
            continue
        start = closes[0]
        end = closes[-1]
        performance[key] = round(((end - start) / max(start, 1e-9)) * 100, 2)

    closes_1y = _fetch_yahoo_chart_closes(symbol, period="1y", interval="1d")
    closes_max = _fetch_yahoo_chart_closes(symbol, period="max", interval="1mo")
    if len(closes_1y) < 40:
        return {
            "ticker": symbol,
            "performance_pct": performance,
            "lifetime_points": len(closes_max),
            "pattern_match": None,
            "today_bias": "unknown",
            "today_bias_confidence": 0,
            "explanation": "Not enough history for pattern comparison yet.",
        }

    # Find a similar historical 20-day pattern in the prior year.
    window = 20
    latest = closes_1y[-window:]
    latest_base = latest[0]
    latest_norm = [(x / max(latest_base, 1e-9)) - 1 for x in latest]
    best_idx = None
    best_score = -1.0
    next_move = None
    for i in range(0, len(closes_1y) - (window * 2)):
        seq = closes_1y[i:i + window]
        base = seq[0]
        seq_norm = [(x / max(base, 1e-9)) - 1 for x in seq]
        dot = sum(a * b for a, b in zip(latest_norm, seq_norm))
        mag_a = math.sqrt(sum(a * a for a in latest_norm)) + 1e-9
        mag_b = math.sqrt(sum(b * b for b in seq_norm)) + 1e-9
        corr = dot / (mag_a * mag_b)
        if corr > best_score:
            best_score = corr
            best_idx = i
            p0 = closes_1y[i + window - 1]
            p1 = closes_1y[i + window]
            next_move = ((p1 - p0) / max(p0, 1e-9)) * 100

    bias = "flat"
    confidence = min(95, max(0, int(best_score * 100)))
    if next_move is not None:
        if next_move > 0.4:
            bias = "rise"
        elif next_move < -0.4:
            bias = "fall"

    return {
        "ticker": symbol,
        "performance_pct": performance,
        "lifetime_points": len(closes_max),
        "pattern_match": {
            "lookback_days": window,
            "best_similarity": round(best_score, 3) if best_score >= 0 else None,
            "historical_next_day_move_pct": round(next_move, 2) if next_move is not None else None,
            "historical_match_index": best_idx,
        },
        "today_bias": bias,
        "today_bias_confidence": confidence,
        "explanation": (
            f"Based on the closest 20-day pattern in the last year, "
            f"the next-day historical move was {round(next_move, 2) if next_move is not None else 'n/a'}%."
        ),
    }


@app.get("/api/signals")
def all_signals():
    return get_all_signals()


@app.get("/api/signal/{ticker}")
def ticker_signal(ticker: str):
    sig = get_signal(ticker.upper())
    if not sig:
        sig = generate_signal(ticker.upper())
    return sig


@app.post("/api/signal/{ticker}/refresh")
async def refresh_signal(ticker: str):
    sig = generate_signal(ticker.upper())
    await broadcast({"type": "signal", "data": sig})
    return sig


@app.get("/api/news/{ticker}")
def ticker_news(ticker: str):
    return get_news(ticker.upper())


@app.get("/api/events")
def corporate_events(limit: int = 50):
    return get_events(limit)


@app.get("/api/earnings")
def earnings():
    return get_earnings_calendar()


@app.get("/api/prices/{ticker}")
def price_history(ticker: str, limit: int = 100):
    df = get_price_df(ticker.upper())
    if df.empty:
        return []
    tail = df.tail(limit)
    return tail.to_dict(orient="records")


@app.get("/health")
def health():
    return {"status": "ok"}
