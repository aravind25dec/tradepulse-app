"""
trades/rh.py
Robinhood API wrapper — authentication + all order types.
Uses robin_stocks; session token is cached locally after first login.
"""

import logging
import robin_stocks.robinhood as r

log = logging.getLogger(__name__)
_logged_in = False


# ── Auth ──────────────────────────────────────────────────────────────────────

def login(username: str, password: str, mfa_code: str = "") -> dict:
    global _logged_in
    try:
        kwargs: dict = {"store_session": True, "login_type": "sms"}
        if mfa_code:
            kwargs["mfa_code"] = mfa_code
        r.login(username, password, **kwargs)
        _logged_in = True
        return {"success": True}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def logout() -> None:
    global _logged_in
    try:
        r.logout()
    except Exception:
        pass
    _logged_in = False


def is_logged_in() -> bool:
    return _logged_in


def try_restore_session() -> bool:
    """Try to restore a cached Robinhood session on startup."""
    global _logged_in
    try:
        r.login("", "", store_session=True, login_type="sms")
        _logged_in = True
        return True
    except Exception:
        return False


# ── Account ───────────────────────────────────────────────────────────────────

def get_account() -> dict:
    try:
        profile = r.profiles.load_portfolio_profile()
        return {
            "equity":         float(profile.get("equity") or 0),
            "extended_equity": float(profile.get("extended_hours_equity") or 0),
            "day_return":     float(profile.get("equity_previous_close") or 0),
        }
    except Exception as exc:
        return {"error": str(exc)}


def get_portfolio() -> list[dict]:
    try:
        holdings = r.build_holdings()
        result = []
        for ticker, d in holdings.items():
            result.append({
                "ticker":          ticker,
                "quantity":        float(d.get("quantity") or 0),
                "avg_cost":        float(d.get("average_buy_price") or 0),
                "current_price":   float(d.get("price") or 0),
                "equity":          float(d.get("equity") or 0),
                "percent_change":  float(d.get("percent_change") or 0),
                "equity_change":   float(d.get("equity_change") or 0),
            })
        return result
    except Exception as exc:
        log.error("Portfolio fetch failed: %s", exc)
        return []


# ── Quotes ────────────────────────────────────────────────────────────────────

def get_quote(ticker: str) -> dict:
    try:
        q    = r.stocks.get_quotes(ticker)[0]
        fund = r.stocks.get_fundamentals(ticker)[0]
        prev = float(q.get("adjusted_previous_close") or q.get("previous_close") or 0)
        cur  = float(q.get("last_trade_price") or q.get("ask_price") or 0)
        chg_pct = ((cur - prev) / prev * 100) if prev else 0
        return {
            "price":       cur,
            "prev_close":  prev,
            "chg_pct":     round(chg_pct, 2),
            "ask":         float(q.get("ask_price") or 0),
            "bid":         float(q.get("bid_price") or 0),
            "ask_size":    q.get("ask_size", ""),
            "bid_size":    q.get("bid_size", ""),
            "volume":      fund.get("volume") or "—",
            "market_cap":  fund.get("market_cap") or "—",
            "pe_ratio":    fund.get("pe_ratio") or "—",
            "high_52w":    float(fund.get("high_52_weeks") or 0),
            "low_52w":     float(fund.get("low_52_weeks") or 0),
            "description": fund.get("description", "")[:300],
        }
    except Exception as exc:
        return {"error": str(exc)}


def get_historicals(ticker: str, span: str = "week") -> list[float]:
    """Return closing prices for sparkline. span: day|week|month|3month|year."""
    try:
        hist = r.stocks.get_stock_historicals(
            ticker, interval="5minute" if span == "day" else "hour",
            span=span, bounds="regular",
        )
        return [float(h["close_price"]) for h in hist if h.get("close_price")]
    except Exception:
        return []


# ── Orders ────────────────────────────────────────────────────────────────────

def place_order(
    side:        str,    # "buy" | "sell"
    ticker:      str,
    quantity:    float,
    order_type:  str,    # see ORDER_TYPES below
    limit_price: float | None = None,
    stop_price:  float | None = None,
    trail_amount: float | None = None,
    trail_type:  str = "percentage",  # "percentage" | "amount"
    time_in_force: str = "gfd",       # gfd | gtc | ioc | opg
) -> dict:
    """
    ORDER_TYPES:
      market           — execute immediately at market price
      limit            — execute only at limit_price or better
      stop_loss        — market order triggered when price hits stop_price
      stop_limit       — limit order triggered when price hits stop_price
      trailing_stop    — tracks best price; triggers when price moves trail_amount against you
    """
    try:
        kw = {"timeInForce": time_in_force}
        if side == "buy":
            if order_type == "market":
                res = r.orders.order_buy_market(ticker, quantity, **kw)
            elif order_type == "limit":
                res = r.orders.order_buy_limit(ticker, quantity, limit_price, **kw)
            elif order_type == "stop_loss":
                res = r.orders.order_buy_stop_loss(ticker, quantity, stop_price, **kw)
            elif order_type == "stop_limit":
                res = r.orders.order_buy_stop_limit(ticker, quantity, limit_price, stop_price, **kw)
            elif order_type == "trailing_stop":
                res = r.orders.order_buy_trailing_stop(ticker, quantity, trail_amount, trail_type, **kw)
            else:
                return {"success": False, "error": f"Unknown order type: {order_type}"}
        else:
            if order_type == "market":
                res = r.orders.order_sell_market(ticker, quantity, **kw)
            elif order_type == "limit":
                res = r.orders.order_sell_limit(ticker, quantity, limit_price, **kw)
            elif order_type == "stop_loss":
                res = r.orders.order_sell_stop_loss(ticker, quantity, stop_price, **kw)
            elif order_type == "stop_limit":
                res = r.orders.order_sell_stop_limit(ticker, quantity, limit_price, stop_price, **kw)
            elif order_type == "trailing_stop":
                res = r.orders.order_sell_trailing_stop(ticker, quantity, trail_amount, trail_type, **kw)
            else:
                return {"success": False, "error": f"Unknown order type: {order_type}"}

        if res and res.get("id"):
            return {"success": True, "order_id": res["id"], "state": res.get("state", "queued")}
        return {"success": False, "error": res or "No response"}
    except Exception as exc:
        log.error("Order failed: %s", exc)
        return {"success": False, "error": str(exc)}


def get_open_orders() -> list[dict]:
    try:
        orders = r.orders.get_all_open_stock_orders() or []
        result = []
        for o in orders:
            instr  = o.get("instrument", "")
            ticker = ""
            if instr:
                try:
                    ticker = r.stocks.get_instrument_by_url(instr).get("symbol", "")
                except Exception:
                    pass
            result.append({
                "id":          o.get("id", ""),
                "ticker":      ticker,
                "side":        o.get("side", ""),
                "order_type":  o.get("type", ""),
                "quantity":    float(o.get("quantity") or 0),
                "filled_qty":  float(o.get("cumulative_quantity") or 0),
                "limit_price": float(o.get("price") or 0) or None,
                "stop_price":  float(o.get("stop_price") or 0) or None,
                "state":       o.get("state", ""),
                "created_at":  o.get("created_at", ""),
                "time_in_force": o.get("time_in_force", ""),
            })
        return result
    except Exception as exc:
        log.error("Open orders failed: %s", exc)
        return []


def get_order_history(limit: int = 20) -> list[dict]:
    try:
        orders = r.orders.get_all_stock_orders() or []
        result = []
        for o in list(orders)[:limit]:
            instr  = o.get("instrument", "")
            ticker = ""
            if instr:
                try:
                    ticker = r.stocks.get_instrument_by_url(instr).get("symbol", "")
                except Exception:
                    pass
            avg_px = o.get("average_price") or o.get("price") or 0
            result.append({
                "id":          o.get("id", ""),
                "ticker":      ticker,
                "side":        o.get("side", ""),
                "order_type":  o.get("type", ""),
                "quantity":    float(o.get("quantity") or 0),
                "filled_qty":  float(o.get("cumulative_quantity") or 0),
                "avg_price":   float(avg_px or 0),
                "state":       o.get("state", ""),
                "created_at":  o.get("created_at", ""),
            })
        return result
    except Exception as exc:
        log.error("Order history failed: %s", exc)
        return []


def cancel_order(order_id: str) -> dict:
    try:
        r.orders.cancel_stock_order(order_id)
        return {"success": True}
    except Exception as exc:
        return {"success": False, "error": str(exc)}
