"""
trades/main.py
TradePulse Trader — Robinhood MCP (Claude CLI) for account/orders,
Yahoo Finance for live quotes and price history.
Port: 8009
"""

import asyncio
import json
import logging
from pathlib import Path

import httpx
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

import mcp_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(title="TradePulse Trader", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_HTML = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

_WL_FILE = Path(__file__).parent / "watchlist.json"

def _load_watchlist() -> list[str]:
    try:
        if _WL_FILE.exists():
            return json.loads(_WL_FILE.read_text())
    except Exception:
        pass
    return ["AAOI", "SPCX", "SMCI", "MSTR", "NVDA"]

def _save_watchlist(tickers: list[str]) -> None:
    _WL_FILE.write_text(json.dumps(list(dict.fromkeys(tickers))))

_watchlist: list[str] = _load_watchlist()


# ── Yahoo Finance helpers (no auth) ───────────────────────────────────────────

_YF_HDR = {"User-Agent": "Mozilla/5.0 (compatible; TradePulse/1.0)"}

async def _yf_quote(ticker: str) -> dict:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1d"
    async with httpx.AsyncClient(timeout=8, headers=_YF_HDR) as c:
        r = await c.get(url)
        if r.status_code != 200:
            return {}
        results = r.json().get("chart", {}).get("result", [])
        if not results:
            return {}
        meta = results[0].get("meta", {})
        price = float(meta.get("regularMarketPrice", 0) or 0)
        prev  = float(meta.get("previousClose", meta.get("chartPreviousClose", price)) or price)
        chg_raw = meta.get("regularMarketChangePercent")
        chg_pct = float(chg_raw) if chg_raw is not None else (((price - prev) / prev * 100) if prev else 0)
        return {
            "price":     price,
            "prev_close": prev,
            "chg_pct":   round(chg_pct, 2),
            "volume":    meta.get("regularMarketVolume"),
            "high_52w":  float(meta.get("fiftyTwoWeekHigh", 0) or 0),
            "low_52w":   float(meta.get("fiftyTwoWeekLow", 0) or 0),
        }


async def _yf_quotes_bulk(tickers: list[str]) -> dict[str, dict]:
    async with httpx.AsyncClient(timeout=10, headers=_YF_HDR) as c:
        async def _one(t: str):
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}?interval=1d&range=1d"
                r = await c.get(url)
                if r.status_code == 200:
                    res = r.json().get("chart", {}).get("result", [])
                    if res:
                        meta  = res[0]["meta"]
                        price = float(meta.get("regularMarketPrice", 0) or 0)
                        prev  = float(meta.get("previousClose", price) or price)
                        chg   = ((price - prev) / prev * 100) if prev else 0
                        return t, {"price": price, "chg_pct": round(chg, 2)}
            except Exception:
                pass
            return t, None
        results = await asyncio.gather(*[_one(t) for t in tickers])
        return {t: d for t, d in results if d is not None}


async def _yf_history(ticker: str, span: str = "week") -> list[float]:
    interval_map = {"day": "5m", "week": "1h", "month": "1d", "3month": "1d", "year": "1wk"}
    range_map    = {"day": "1d", "week": "5d", "month": "1mo", "3month": "3mo", "year": "1y"}
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval={interval_map.get(span, '1h')}&range={range_map.get(span, '5d')}"
    )
    try:
        async with httpx.AsyncClient(timeout=10, headers=_YF_HDR) as c:
            r = await c.get(url)
            if r.status_code != 200:
                return []
            results = r.json().get("chart", {}).get("result", [])
            if not results:
                return []
            closes = results[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
            return [float(p) for p in closes if p is not None]
    except Exception:
        return []


# ── UI ────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def ui():
    return _HTML


# ── Status (Claude CLI + Robinhood MCP) ──────────────────────────────────────

@app.get("/api/trader/status")
async def status():
    cli = mcp_engine.claude_available()
    mcp = mcp_engine.check_robinhood_mcp() if cli else False
    return {"claude_available": cli, "robinhood_mcp": mcp}


# ── Account + Portfolio ───────────────────────────────────────────────────────

@app.get("/api/trader/account")
async def account():
    try:
        return await mcp_engine.fetch_account_via_mcp()
    except Exception as exc:
        log.error("account fetch failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/trader/portfolio")
async def portfolio():
    try:
        positions = await mcp_engine.fetch_positions_via_mcp()
        tickers   = [p["ticker"] for p in positions if p.get("ticker")]
        if tickers:
            prices = await _yf_quotes_bulk(tickers)
            for p in positions:
                t         = p.get("ticker", "")
                q         = prices.get(t, {})
                cur       = float(q.get("price", 0) or 0)
                avg       = float(p.get("avg_cost", 0) or 0)
                qty       = float(p.get("quantity", 0) or 0)
                p["current_price"]  = cur
                p["equity"]         = round(cur * qty, 2)
                p["percent_change"] = round((cur - avg) / avg * 100, 2) if avg else 0
                p["chg_pct_today"]  = float(q.get("chg_pct", 0) or 0)
        return {"positions": positions}
    except Exception as exc:
        log.error("portfolio fetch failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


# ── Quotes (Yahoo Finance — no auth) ─────────────────────────────────────────

@app.get("/api/trader/quote/{ticker}")
async def quote(ticker: str):
    try:
        data = await _yf_quote(ticker.upper())
        if not data:
            return JSONResponse({"error": "No data returned"}, status_code=404)
        return data
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/trader/history/{ticker}")
async def history(ticker: str, span: str = "week"):
    prices = await _yf_history(ticker.upper(), span)
    return {"prices": prices}


# ── Watchlist ─────────────────────────────────────────────────────────────────

@app.get("/api/trader/watchlist")
async def get_watchlist():
    return {"tickers": _watchlist}


@app.post("/api/trader/watchlist")
async def add_to_watchlist(payload: dict = Body(...)):
    global _watchlist
    ticker = payload.get("ticker", "").upper().strip()
    if ticker and ticker not in _watchlist:
        _watchlist.append(ticker)
        _save_watchlist(_watchlist)
    return {"tickers": _watchlist}


@app.delete("/api/trader/watchlist/{ticker}")
async def remove_from_watchlist(ticker: str):
    global _watchlist
    _watchlist = [t for t in _watchlist if t != ticker.upper()]
    _save_watchlist(_watchlist)
    return {"tickers": _watchlist}


@app.post("/api/trader/watchlist/sync")
async def sync_watchlist(payload: dict = Body(...)):
    """Batch-add tickers from portfolio without removing manually added ones."""
    global _watchlist
    tickers = [str(t).upper().strip() for t in payload.get("tickers", []) if t]
    added = [t for t in tickers if t and t not in _watchlist]
    if added:
        _watchlist.extend(added)
        _save_watchlist(_watchlist)
    return {"tickers": _watchlist, "added": added}


# ── Live price SSE (Yahoo Finance — no auth) ──────────────────────────────────

@app.get("/api/trader/stream")
async def price_stream():
    """SSE — pushes watchlist prices every 10 s via Yahoo Finance (no auth required)."""
    async def generate():
        while True:
            if not _watchlist:
                await asyncio.sleep(5)
                continue
            try:
                prices = await _yf_quotes_bulk(_watchlist)
                yield f"data: {json.dumps(prices)}\n\n"
            except Exception as exc:
                log.warning("Price stream error: %s", exc)
            await asyncio.sleep(10)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Orders (via Robinhood MCP) ────────────────────────────────────────────────

@app.post("/api/trader/order")
async def place_order(payload: dict = Body(...)):
    side          = str(payload.get("side",          "")).lower().strip()
    ticker        = str(payload.get("ticker",        "")).upper().strip()
    quantity      = float(payload.get("quantity",    0) or 0)
    order_type    = str(payload.get("order_type",    "market")).lower().strip()
    limit_price   = payload.get("limit_price")
    stop_price    = payload.get("stop_price")
    trail_amount  = payload.get("trail_amount")
    trail_type    = str(payload.get("trail_type",    "percentage") or "percentage")
    time_in_force = str(payload.get("time_in_force", "gfd") or "gfd").lower()

    if not ticker or not side or quantity <= 0:
        return JSONResponse(
            {"error": "ticker, side, and quantity > 0 are required"}, status_code=400
        )

    try:
        result = await mcp_engine.place_order_via_mcp(
            side=side,
            ticker=ticker,
            quantity=quantity,
            order_type=order_type,
            limit_price=float(limit_price) if limit_price else None,
            stop_price=float(stop_price) if stop_price else None,
            trail_amount=float(trail_amount) if trail_amount else None,
            trail_type=trail_type,
            time_in_force=time_in_force,
        )
        if not result.get("success"):
            return JSONResponse(result, status_code=422)
        return result
    except Exception as exc:
        log.error("Order placement failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/trader/orders/open")
async def open_orders():
    try:
        return await mcp_engine.fetch_open_orders_via_mcp()
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/trader/orders/history")
async def order_history(limit: int = 30):
    try:
        return await mcp_engine.fetch_order_history_via_mcp(limit)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.delete("/api/trader/orders/{order_id}")
async def cancel_order(order_id: str):
    try:
        return await mcp_engine.cancel_order_via_mcp(order_id)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)
