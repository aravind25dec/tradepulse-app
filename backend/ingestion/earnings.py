"""
ingestion/earnings.py
Fetches upcoming earnings calendar from Finnhub.
Also detects earnings surprises for tickers we track.
"""
import asyncio
import os
import logging
from datetime import date, timedelta

import finnhub

from db.store import set_earnings_calendar, push_event

log = logging.getLogger(__name__)

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")


def build_client():
    if not FINNHUB_KEY:
        return None
    return finnhub.Client(api_key=FINNHUB_KEY)


def fetch_earnings_calendar(tickers: list[str], days_ahead: int = 14) -> list[dict]:
    client = build_client()
    if not client:
        log.warning("No Finnhub key — earnings calendar unavailable.")
        return []
    try:
        start = date.today().isoformat()
        end = (date.today() + timedelta(days=days_ahead)).isoformat()
        raw = client.earnings_calendar(_from=start, to=end, symbol="")
        calendar = raw.get("earningsCalendar", [])
        # Filter to our watchlist
        filtered = [e for e in calendar if e.get("symbol") in tickers]
        return [
            {
                "ticker": e["symbol"],
                "report_date": e.get("date", ""),
                "hour": e.get("hour", ""),          # bmo=before open, amc=after close
                "eps_estimate": e.get("epsEstimate"),
                "eps_actual": e.get("epsActual"),
                "revenue_estimate": e.get("revenueEstimate"),
                "revenue_actual": e.get("revenueActual"),
                "surprise_pct": _surprise_pct(e.get("epsActual"), e.get("epsEstimate")),
            }
            for e in filtered
        ]
    except Exception as ex:
        log.warning(f"Finnhub earnings fetch error: {ex}")
        return []


def _surprise_pct(actual, estimate) -> float | None:
    if actual is None or estimate is None or estimate == 0:
        return None
    return round((actual - estimate) / abs(estimate) * 100, 2)


def surprise_label(pct: float | None) -> str:
    if pct is None:
        return "pending"
    if pct > 10:
        return "strong_beat"
    if pct > 0:
        return "beat"
    if pct < -10:
        return "strong_miss"
    return "miss"


async def earnings_poll_loop(tickers: list[str], broadcast_fn, interval_seconds: int = 3600):
    """Refresh earnings calendar every hour."""
    log.info("Starting earnings poll loop...")
    while True:
        loop = asyncio.get_event_loop()
        calendar = await loop.run_in_executor(None, lambda: fetch_earnings_calendar(tickers))
        set_earnings_calendar(calendar)

        # Push surprise events for already-reported earnings
        for entry in calendar:
            if entry["eps_actual"] is not None and entry["surprise_pct"] is not None:
                push_event({
                    "ticker": entry["ticker"],
                    "type": "EARNINGS_RESULT",
                    "description": f"EPS {surprise_label(entry['surprise_pct'])}: "
                                   f"actual={entry['eps_actual']} vs est={entry['eps_estimate']} "
                                   f"({entry['surprise_pct']:+.1f}%)",
                    "filing_date": entry["report_date"],
                })

        await broadcast_fn({"type": "earnings", "data": calendar})
        log.info(f"Earnings calendar updated: {len(calendar)} upcoming reports")
        await asyncio.sleep(interval_seconds)
