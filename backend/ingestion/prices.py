"""
ingestion/prices.py
Polls yfinance every 60 seconds for the latest price on each ticker.
Swap out for Polygon WebSocket (see commented section) with a paid key.
"""
import asyncio
import os
import logging
import random
from datetime import datetime, timezone

import httpx
import yfinance as yf

from db.store import push_tick, get_price_df

log = logging.getLogger(__name__)

STOCK_TICKERS = [t.strip() for t in os.getenv("STOCK_WATCHLIST", "AAPL,TSLA,NVDA").split(",")]
CRYPTO_TICKERS = [t.strip() for t in os.getenv("CRYPTO_WATCHLIST", "BTC-USD,ETH-USD").split(",")]
ALL_TICKERS = STOCK_TICKERS + CRYPTO_TICKERS


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
INTERVAL_SECONDS = {
    "1m": 60,
    "2m": 120,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "60m": 3600,
    "90m": 5400,
    "1h": 3600,
    "1d": 86400,
}
PERIOD_SECONDS = {
    "1d": 86400,
    "5d": 5 * 86400,
    "1mo": 30 * 86400,
    "3mo": 90 * 86400,
}


async def _fetch_yahoo_chart(ticker: str, *, period: str, interval: str) -> list[tuple[datetime, float, float]]:
    """Fetch chart candles directly from Yahoo to avoid yfinance parser failures."""
    url = YAHOO_CHART_URL.format(ticker=ticker)
    params = {
        "range": period,
        "interval": interval,
        "includePrePost": "false",
        "events": "div,splits",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 TradePulse/1.0",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart API error for {ticker}: {chart['error']}")

    results = chart.get("result") or []
    if not results:
        return []

    item = results[0]
    timestamps = item.get("timestamp") or []
    indicators = item.get("indicators", {}).get("quote", [])
    if not indicators:
        return []

    quote = indicators[0]
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []
    candles: list[tuple[datetime, float, float]] = []
    for idx, ts in enumerate(timestamps):
        if idx >= len(closes):
            break
        close = closes[idx]
        if close is None:
            continue
        volume = volumes[idx] if idx < len(volumes) and volumes[idx] is not None else 0
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        candles.append((dt, float(close), float(volume)))
    return candles


def _finnhub_symbol(ticker: str) -> tuple[str, bool]:
    """Return Finnhub symbol + whether it's a crypto symbol."""
    if ticker.endswith("-USD"):
        base = ticker.replace("-USD", "")
        return f"BINANCE:{base}USDT", True
    return ticker, False


def _finnhub_resolution(interval: str) -> str:
    if interval.endswith("m"):
        return interval[:-1]
    if interval in {"1h", "60m"}:
        return "60"
    if interval == "1d":
        return "D"
    return "5"


async def _fetch_finnhub_candles(
    ticker: str, *, period: str, interval: str
) -> list[tuple[datetime, float, float]]:
    """Fallback candles from Finnhub when Yahoo is unavailable/rate-limited."""
    api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not api_key or api_key.lower().startswith("your_"):
        return []

    symbol, is_crypto = _finnhub_symbol(ticker)
    endpoint = "https://finnhub.io/api/v1/crypto/candle" if is_crypto else "https://finnhub.io/api/v1/stock/candle"

    now = int(datetime.now(tz=timezone.utc).timestamp())
    window = PERIOD_SECONDS.get(period, 5 * 86400)
    from_ts = max(0, now - window)

    params = {
        "symbol": symbol,
        "resolution": _finnhub_resolution(interval),
        "from": from_ts,
        "to": now,
        "token": api_key,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(endpoint, params=params)
        response.raise_for_status()
        payload = response.json()

    if payload.get("s") != "ok":
        return []

    ts_list = payload.get("t") or []
    closes = payload.get("c") or []
    volumes = payload.get("v") or []
    candles: list[tuple[datetime, float, float]] = []
    for idx, ts in enumerate(ts_list):
        if idx >= len(closes):
            break
        close = closes[idx]
        if close is None:
            continue
        volume = volumes[idx] if idx < len(volumes) and volumes[idx] is not None else 0
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        candles.append((dt, float(close), float(volume)))
    return candles


async def fetch_latest_price(ticker: str) -> dict | None:
    """Fetch the latest price using yfinance (free, no key needed)."""
    # Prefer Yahoo chart API because yfinance can fail with JSONDecodeError on some setups.
    try:
        candles = await _fetch_yahoo_chart(ticker, period="1d", interval="1m")
        if candles:
            _, price, volume = candles[-1]
            return {"ticker": ticker, "price": price, "volume": volume}
    except Exception as e:
        log.warning(f"Direct price fetch failed for {ticker}: {e}")

    try:
        candles = await _fetch_finnhub_candles(ticker, period="1d", interval="1m")
        if candles:
            _, price, volume = candles[-1]
            return {"ticker": ticker, "price": price, "volume": volume}
    except Exception as e:
        log.warning(f"Finnhub latest price fetch failed for {ticker}: {e}")

    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, lambda: yf.Ticker(ticker).fast_info)
        price = data.get("last_price") or data.get("regularMarketPrice")
        volume = data.get("three_month_average_volume", 0)
        if price:
            return {"ticker": ticker, "price": float(price), "volume": float(volume or 0)}
    except Exception as e:
        log.warning(f"Price fetch failed for {ticker}: {e}")
    return None


async def load_historical(ticker: str, period: str = "5d", interval: str = "5m"):
    """Load OHLCV history into the store for technical indicator calculation."""
    loop = asyncio.get_event_loop()
    retries = 3
    for attempt in range(1, retries + 1):
        try:
            candles = await _fetch_yahoo_chart(ticker, period=period, interval=interval)
            if not candles:
                # Empty can happen when Yahoo throttles or temporarily blocks requests.
                if attempt < retries:
                    wait = attempt * 2 + random.uniform(0.2, 0.8)
                    log.warning(
                        f"Historical data empty for {ticker}, retrying in {wait:.1f}s "
                        f"(attempt {attempt}/{retries})"
                    )
                    await asyncio.sleep(wait)
                    continue
                log.warning(f"Historical load returned empty for {ticker} after {retries} attempts")
                return

            for ts, close, volume in candles:
                push_tick(ticker, close, volume, ts)
            log.info(f"Loaded {len(candles)} candles for {ticker}")
            return
        except Exception as e:
            # Fallback to yfinance in case direct Yahoo chart API is unavailable.
            try:
                df = await loop.run_in_executor(
                    None,
                    lambda: yf.download(
                        ticker,
                        period=period,
                        interval=interval,
                        progress=False,
                        threads=False,
                    ),
                )
                if not df.empty:
                    for _, row in df.iterrows():
                        push_tick(ticker, float(row["Close"]), float(row["Volume"]), row.name.to_pydatetime())
                    log.info(f"Loaded {len(df)} candles for {ticker} (yfinance fallback)")
                    return
            except Exception:
                pass
            try:
                candles = await _fetch_finnhub_candles(ticker, period=period, interval=interval)
                if candles:
                    for ts, close, volume in candles:
                        push_tick(ticker, close, volume, ts)
                    log.info(f"Loaded {len(candles)} candles for {ticker} (Finnhub fallback)")
                    return
            except Exception:
                pass

            if attempt < retries:
                wait = attempt * 2 + random.uniform(0.2, 0.8)
                log.warning(
                    f"Historical load failed for {ticker}: {e}. "
                    f"Retrying in {wait:.1f}s (attempt {attempt}/{retries})"
                )
                await asyncio.sleep(wait)
                continue
            log.warning(f"Historical load failed for {ticker} after {retries} attempts: {e}")


async def price_poll_loop(broadcast_fn, interval_seconds: int = 60):
    """Main polling loop — runs forever, broadcasts new ticks."""
    log.info("Starting price poll loop...")

    # Load historical data first so technicals have data to work with.
    # Do this sequentially with slight spacing to reduce Yahoo rate-limit errors.
    await asyncio.sleep(1.0)
    for ticker in ALL_TICKERS:
        await load_historical(ticker)
        await asyncio.sleep(0.6)
    log.info("Historical data loaded for all tickers.")

    while True:
        for ticker in ALL_TICKERS:
            result = await fetch_latest_price(ticker)
            if result:
                push_tick(ticker, result["price"], result["volume"])
                await broadcast_fn({"type": "tick", "data": result})
        await asyncio.sleep(interval_seconds)


# ── Polygon WebSocket (uncomment with a paid Polygon key) ─────────────────
# from polygon import WebSocketClient
# from polygon.websocket.models import Market, Feed
#
# async def polygon_stream(broadcast_fn):
#     client = WebSocketClient(
#         api_key=os.getenv("POLYGON_API_KEY"),
#         market=Market.Stocks,
#         feed=Feed.RealTime,
#     )
#     await client.subscribe("T.*")
#     async for msgs in client:
#         for m in msgs:
#             if m.symbol in ALL_TICKERS:
#                 push_tick(m.symbol, m.price, m.size)
#                 await broadcast_fn({"type": "tick",
#                                     "data": {"ticker": m.symbol,
#                                              "price": m.price,
#                                              "volume": m.size}})
