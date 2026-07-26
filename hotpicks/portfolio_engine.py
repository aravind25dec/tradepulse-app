"""
hotpicks/portfolio_engine.py
Portfolio analysis — connects Robinhood positions to the signal engine and
produces position-aware recommendations for each holding.

Security note:
  - Credentials are held in-process memory only (store_session=False).
  - Nothing is written to disk. For personal/local use only.
  - The unofficial Robinhood API (robin_stocks) may change without notice.
"""

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import robin_stocks.robinhood as _rh
    ROBINHOOD_AVAILABLE = True
except ImportError:
    _rh = None
    ROBINHOOD_AVAILABLE = False
    logger.warning("robin_stocks not installed — Robinhood login unavailable. Run: pip install robin_stocks")

# In-process login state (no disk writes)
_logged_in = False

# Cache instrument URL → ticker to avoid repeated HTTP calls
_instrument_cache: dict[str, str] = {}

_ACCOUNT_LABELS = {
    "individual":       "Brokerage",
    "cash":             "Cash Account",
    "margin":           "Margin Account",
    "traditional_ira":  "Traditional IRA",
    "roth_ira":         "Roth IRA",
    "rollover_ira":     "Rollover IRA",
    "sep_ira":          "SEP IRA",
}


# ── Robinhood auth ─────────────────────────────────────────────────────────────

def robinhood_login(username: str, password: str, mfa_code: Optional[str] = None) -> dict:
    """Attempt Robinhood login. Returns success, needs_mfa, or error."""
    global _logged_in
    if not ROBINHOOD_AVAILABLE:
        return {"success": False, "error": "robin_stocks not installed. Run: pip install robin_stocks"}
    try:
        _rh.login(
            username,
            password,
            mfa_code=mfa_code or None,
            store_session=False,   # never write credentials to disk
        )
        _logged_in = True
        _instrument_cache.clear()
        return {"success": True}
    except Exception as exc:
        msg = str(exc).lower()
        if any(k in msg for k in ("mfa", "two factor", "2fa", "verification", "challenge")):
            return {"success": False, "needs_mfa": True}
        return {"success": False, "error": str(exc)}


def robinhood_logout() -> None:
    global _logged_in
    try:
        if ROBINHOOD_AVAILABLE:
            _rh.logout()
    except Exception:
        pass
    _logged_in = False
    _instrument_cache.clear()


def is_logged_in() -> bool:
    return _logged_in


# ── Multi-account helpers ──────────────────────────────────────────────────────

def get_all_accounts() -> list[dict]:
    """Return all Robinhood accounts with type label and buying power."""
    if not ROBINHOOD_AVAILABLE or not _logged_in:
        raise RuntimeError("Not logged in to Robinhood")

    raw = _get_all_accounts_raw()
    logger.info(
        "Robinhood raw accounts (%d) key fields: %s", len(raw),
        [{k: v for k, v in (a or {}).items()
          if k in ("account_number", "type", "brokerage_account_type",
                   "account_type", "buying_power", "cash", "deactivated", "state")}
         for a in raw],
    )

    result = []
    for acct in raw:
        if not acct:
            continue
        acct_num = acct.get("account_number", "")
        if not acct_num:
            continue
        if acct.get("deactivated") or acct.get("state") in ("deactivated", "closed"):
            continue
        acct_type = (
            acct.get("brokerage_account_type")
            or acct.get("account_type")
            or acct.get("type", "individual")
        )
        label = _ACCOUNT_LABELS.get(
            acct_type,
            acct_type.replace("_", " ").title() if acct_type else "Account",
        )
        bp = float(acct.get("buying_power") or acct.get("cash") or acct.get("portfolio_cash") or 0)
        result.append({
            "account_number": acct_num,
            "type":           acct_type or "individual",
            "label":          label,
            "buying_power":   round(bp, 2),
        })
    return result


def _get_all_accounts_raw() -> list[dict]:
    """
    Fetch the raw /accounts/ list, trying multiple methods.
    robin_stocks' get_all_accounts() sometimes only returns the first page
    or omits accounts it doesn't recognise; we follow pagination manually.
    """
    try:
        from robin_stocks.robinhood.helper import request_get
        # Follow next-page links so we get ALL accounts (not just the first page)
        url  = "https://api.robinhood.com/accounts/"
        all_results: list[dict] = []
        while url:
            resp = request_get(url, "regular") or {}
            page = resp.get("results", []) if isinstance(resp, dict) else (resp or [])
            all_results.extend(page)
            url = resp.get("next") if isinstance(resp, dict) else None
        if all_results:
            return all_results
    except Exception as exc:
        logger.warning("Manual pagination fetch failed: %s — falling back to robin_stocks helper", exc)

    # Fallback: standard robin_stocks call
    return _rh.account.get_all_accounts() or []


def _resolve_ticker(instrument_url: str) -> str:
    """Resolve a Robinhood instrument URL to a ticker symbol (cached)."""
    if not instrument_url:
        return ""
    if instrument_url in _instrument_cache:
        return _instrument_cache[instrument_url]
    try:
        data   = _rh.stocks.get_instrument_by_url(instrument_url) or {}
        ticker = data.get("symbol", "").upper()
    except Exception:
        ticker = ""
    _instrument_cache[instrument_url] = ticker
    return ticker


def _positions_for_account(account_number: str) -> list[dict]:
    """Fetch open non-zero positions for a specific Robinhood account number."""
    raw = None
    # Try account-specific URL first (robin_stocks >= 2.x)
    try:
        raw = _rh.account.get_open_stock_positions(account_number=account_number) or []
        logger.info("Account %s — %d position(s) via account_number kwarg", account_number, len(raw))
    except TypeError:
        pass

    # Fallback: hit the account-positions URL directly via robin_stocks internals
    if raw is None:
        try:
            import robin_stocks.robinhood.urls as _urls
            from robin_stocks.robinhood.helper import request_get
            url = f"https://api.robinhood.com/accounts/{account_number}/positions/"
            raw = request_get(url, "results", payload={"nonzero": "true"}) or []
            logger.info("Account %s — %d position(s) via direct URL", account_number, len(raw))
        except Exception as exc:
            logger.warning("Direct URL fallback failed for %s: %s", account_number, exc)
            raw = []

    positions = []
    for pos in (raw or []):
        qty = float(pos.get("quantity", 0))
        if qty <= 0:
            continue
        ticker = _resolve_ticker(pos.get("instrument", ""))
        if not ticker:
            continue
        avg_cost = float(pos.get("average_buy_price", 0))
        positions.append({
            "ticker":   ticker,
            "quantity": round(qty, 4),
            "avg_cost": round(avg_cost, 2),
        })
    return positions


def get_robinhood_positions() -> list[dict]:
    """
    Return all open positions across ALL accounts.
    Each entry includes account_number and account_label so the UI can group/filter.
    Falls back to the default account if multi-account calls fail.
    """
    if not ROBINHOOD_AVAILABLE or not _logged_in:
        raise RuntimeError("Not logged in to Robinhood")

    try:
        accounts = get_all_accounts()
    except Exception as exc:
        logger.warning("Could not fetch accounts list: %s — falling back to default", exc)
        accounts = []

    all_positions: list[dict] = []

    if accounts:
        seen_accounts: set[str] = set()
        for acct in accounts:
            acct_num = acct["account_number"]
            if acct_num in seen_accounts:
                continue
            seen_accounts.add(acct_num)
            try:
                for pos in _positions_for_account(acct_num):
                    pos["account_number"] = acct_num
                    pos["account_label"]  = acct["label"]
                    pos["account_type"]   = acct["type"]
                    all_positions.append(pos)
            except Exception as exc:
                logger.warning("Positions fetch failed for account %s: %s", acct_num, exc)
    else:
        # No account list — use build_holdings() for the default account
        holdings = _rh.account.build_holdings() or {}
        for ticker, info in holdings.items():
            qty = float(info.get("quantity", 0))
            if qty <= 0:
                continue
            all_positions.append({
                "ticker":         ticker.upper(),
                "quantity":       round(qty, 4),
                "avg_cost":       round(float(info.get("average_buy_price", 0)), 2),
                "account_number": "",
                "account_label":  "Brokerage",
                "account_type":   "individual",
            })

    return sorted(all_positions, key=lambda x: (x.get("account_label", ""), x["ticker"]))


# ── Order placement ────────────────────────────────────────────────────────────

def place_order(
    ticker:         str,
    side:           str,
    order_type:     str,
    quantity:       float,
    limit_price:    Optional[float] = None,
    account_number: Optional[str]   = None,
    time_in_force:  str             = "gtc",
) -> dict:
    """
    Place a buy or sell order on Robinhood.
    side: 'buy' | 'sell'
    order_type: 'market' | 'limit'
    """
    if not ROBINHOOD_AVAILABLE or not _logged_in:
        return {"success": False, "error": "Not logged in to Robinhood"}

    ticker     = ticker.upper().strip()
    side       = side.lower().strip()
    order_type = order_type.lower().strip()

    if side not in ("buy", "sell"):
        return {"success": False, "error": f"Invalid side '{side}' — must be buy or sell"}
    if order_type not in ("market", "limit"):
        return {"success": False, "error": f"Invalid order type '{order_type}' — must be market or limit"}
    if quantity <= 0:
        return {"success": False, "error": "Quantity must be greater than zero"}
    if order_type == "limit" and not limit_price:
        return {"success": False, "error": "Limit price is required for limit orders"}

    # GTC (Good Till Cancelled) is safer than GFD (Good For Day) — GFD orders
    # placed outside market hours expire at close and never execute, which is
    # why "success" orders appear to do nothing.  Default to GTC so orders
    # persist until explicitly cancelled.
    tif = time_in_force if time_in_force in ("gtc", "gfd", "ioc", "opg") else "gtc"

    kwargs: dict = {"timeInForce": tif}
    if account_number:
        kwargs["account_number"] = account_number

    logger.info("Placing order: %s %s %s qty=%s tif=%s acct=%s",
                side.upper(), order_type, ticker, quantity, tif, account_number or "default")
    try:
        if side == "buy" and order_type == "market":
            res = _rh.orders.order_buy_market(ticker, quantity, **kwargs)
        elif side == "buy" and order_type == "limit":
            res = _rh.orders.order_buy_limit(ticker, quantity, limit_price, **kwargs)
        elif side == "sell" and order_type == "market":
            res = _rh.orders.order_sell_market(ticker, quantity, **kwargs)
        else:  # sell limit
            res = _rh.orders.order_sell_limit(ticker, quantity, limit_price, **kwargs)
        logger.info("Order response: %s", {k: res.get(k) for k in ("id","state","side","quantity","type","price") if res and k in res})

        if not res:
            return {"success": False, "error": "No response from Robinhood — order may not have been placed"}

        return {
            "success":        True,
            "order_id":       res.get("id"),
            "state":          res.get("state", "queued"),
            "ticker":         ticker,
            "side":           side,
            "type":           order_type,
            "quantity":       quantity,
            "limit_price":    limit_price,
            "account_number": account_number,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ── Position-aware recommendation ─────────────────────────────────────────────

def _portfolio_recommendation(result: dict) -> dict:
    """Given a signal result + position data, return an action recommendation."""
    grade   = result.get("grade", "NEUTRAL")
    danger  = result.get("danger", 0)
    pnl_pct = result.get("pnl_pct")
    sig     = result.get("signals", {})
    rsi     = sig.get("rsi", 50.0)
    bb_pct  = sig.get("bb_pct", 0.5)
    in_profit = pnl_pct is not None and pnl_pct > 0
    in_loss   = pnl_pct is not None and pnl_pct < 0

    if danger >= 4:
        if in_loss:
            return {"action": "CUT LOSS",    "cls": "rec-sell",    "icon": "⛔",
                    "reason": f"Multiple danger signals + position down {abs(pnl_pct):.1f}% — protect remaining capital"}
        return     {"action": "TAKE PROFIT", "cls": "rec-warn",    "icon": "💰",
                    "reason": f"Multiple danger signals — lock in {pnl_pct:+.1f}% gain before reversal"}

    if rsi > 72 and bb_pct > 0.85 and in_profit:
        return     {"action": "TRIM 25–50%", "cls": "rec-warn",    "icon": "✂",
                    "reason": f"Overbought (RSI {rsi:.0f}, BB {bb_pct:.0%}) — sell partial position to lock gains"}

    if grade == "SKIP":
        if in_loss:
            return {"action": "CUT LOSS",    "cls": "rec-sell",    "icon": "⛔",
                    "reason": f"Bearish signals + down {abs(pnl_pct):.1f}% — stop loss territory, exit"}
        return     {"action": "TAKE PROFIT", "cls": "rec-warn",    "icon": "💰",
                    "reason": f"Bearish setup — book {pnl_pct:+.1f}% gains before price drops"}

    if grade == "STRONG BUY":
        return     {"action": "ADD MORE",    "cls": "rec-buy",     "icon": "✅",
                    "reason": "Strong momentum confirmed — setup supports adding to your position"}

    if grade == "BUY":
        return     {"action": "HOLD & BUY",  "cls": "rec-buy",     "icon": "📈",
                    "reason": "Positive setup — hold and consider adding on a pullback to VWAP"}

    if grade == "WATCH":
        if in_loss:
            return {"action": "WATCH & WAIT","cls": "rec-neutral", "icon": "👀",
                    "reason": f"Mixed signals + down {abs(pnl_pct):.1f}% — set a tight stop, do not average down yet"}
        return     {"action": "HOLD",        "cls": "rec-neutral", "icon": "⏸",
                    "reason": "No strong signal — hold your position and check back in 15 min"}

    if in_loss:
        return     {"action": "REVIEW",      "cls": "rec-warn",    "icon": "⚠",
                    "reason": f"Weak signals + down {abs(pnl_pct):.1f}% — reassess your original thesis"}
    return         {"action": "HOLD",        "cls": "rec-neutral", "icon": "⏸",
                    "reason": "No strong signal either way — maintain position"}


# ── Portfolio analysis ─────────────────────────────────────────────────────────

async def analyze_portfolio(positions: list[dict]) -> dict:
    """
    Run the full signal pipeline on each held position.
    Positions can include account_number / account_label fields from multi-account fetch.
    AI is called only for positions with BUY/STRONG BUY or high-danger signals.
    """
    from engine import (
        fetch_market_context, _scan_ticker, _apply_ai_plan, _HEADERS,
    )
    import httpx

    news_api_key  = os.getenv("NEWS_API_KEY", "")
    sem           = asyncio.Semaphore(4)

    # Deduplicate tickers for scanning but keep all position entries (same ticker, diff accounts)
    unique_tickers = list(dict.fromkeys(p["ticker"] for p in positions))
    # pos_map: ticker → list of positions (multi-account may have same ticker in multiple accounts)
    pos_map: dict[str, list[dict]] = {}
    for p in positions:
        pos_map.setdefault(p["ticker"], []).append(p)

    async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
        market_ctx = await fetch_market_context(client)
        logger.info(
            "Portfolio scan: %d tickers | regime=%s SPY%+.1f%% QQQ%+.1f%%",
            len(unique_tickers), market_ctx["regime"],
            market_ctx["spy_chg"], market_ctx["qqq_chg"],
        )
        raw = await asyncio.gather(
            *[_scan_ticker(t, client, news_api_key, sem, market_ctx) for t in unique_tickers],
            return_exceptions=True,
        )

    results, ai_queue = [], []
    for r in raw:
        if not r or isinstance(r, Exception):
            continue
        ticker   = r["ticker"]
        price    = r["signals"].get("price", 0.0)
        pos_list = pos_map.get(ticker, [{}])

        # Expand one signal result into one row per account holding
        for pos in pos_list:
            entry = dict(r)  # shallow copy — signals/plan/history are shared (read-only)
            avg_cost = float(pos.get("avg_cost", 0))
            quantity = float(pos.get("quantity", 0))

            if avg_cost and price:
                pnl_pct = (price - avg_cost) / avg_cost * 100
                pnl_usd = (price - avg_cost) * quantity
            else:
                pnl_pct = pnl_usd = None

            entry["position"]      = pos
            entry["pnl_pct"]       = round(pnl_pct, 2) if pnl_pct is not None else None
            entry["pnl_usd"]       = round(pnl_usd, 2) if pnl_usd is not None else None
            entry["portfolio_rec"] = _portfolio_recommendation(entry)
            entry["account_number"]= pos.get("account_number", "")
            entry["account_label"] = pos.get("account_label",  "Brokerage")
            entry["account_type"]  = pos.get("account_type",   "individual")

            results.append(entry)

            if entry["grade"] in ("STRONG BUY", "BUY") or entry.get("danger", 0) >= 3:
                ai_queue.append(entry)

    # Phase 2 AI — only on high-signal positions (via Claude CLI)
    if ai_queue:
        logger.info("Portfolio AI: enriching %d positions", len(ai_queue))
        await asyncio.gather(
            *[_apply_ai_plan(r, market_ctx) for r in ai_queue],
            return_exceptions=True,
        )

    # Sort: biggest danger first, then worst P&L
    results.sort(key=lambda x: (-x.get("danger", 0), x.get("pnl_pct") or 0))

    total_value   = sum(r["signals"].get("price", 0) * float(r["position"].get("quantity", 0)) for r in results)
    total_cost    = sum(float(r["position"].get("avg_cost", 0)) * float(r["position"].get("quantity", 0)) for r in results)
    total_pnl_usd = sum(r.get("pnl_usd") or 0 for r in results)

    return {
        "market_ctx":    market_ctx,
        "positions":     results,
        "total":         len(results),
        "total_value":   round(total_value,   2),
        "total_cost":    round(total_cost,    2),
        "total_pnl":     round(total_pnl_usd, 2),
        "total_pnl_pct": round((total_pnl_usd / total_cost * 100) if total_cost else 0, 2),
        "danger_count":  sum(1 for r in results if r.get("danger", 0) >= 3),
        "sell_signals":  sum(1 for r in results if r["portfolio_rec"]["action"] in
                             ("CUT LOSS", "TAKE PROFIT", "TRIM 25–50%")),
    }
