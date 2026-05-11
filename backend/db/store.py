"""
db/store.py
In-memory store for price history, signals, news, and events.
Replace with TimescaleDB for production by swapping the functions below.
"""
from collections import defaultdict, deque
from datetime import datetime
from typing import Optional
import pandas as pd

# ── In-memory stores ────────────────────────────────────────────────────────
_price_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
_signals: dict[str, dict] = {}
_news: dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
_events: deque = deque(maxlen=200)          # M&A / contract / earnings
_earnings_calendar: list = []


# ── Price history ────────────────────────────────────────────────────────────
def push_tick(ticker: str, price: float, volume: float, timestamp: Optional[datetime] = None):
    _price_history[ticker].append({
        "price": price,
        "volume": volume,
        "ts": (timestamp or datetime.utcnow()).isoformat(),
    })


def get_price_df(ticker: str) -> pd.DataFrame:
    rows = list(_price_history[ticker])
    if not rows:
        return pd.DataFrame(columns=["price", "volume", "ts"])
    df = pd.DataFrame(rows)
    # Price ticks can contain mixed ISO timestamp precisions (with/without microseconds).
    # Use mixed parsing to avoid failures when formats vary within the same series.
    df["ts"] = pd.to_datetime(df["ts"], format="mixed")
    df = df.rename(columns={"price": "close"})
    return df


# ── Signals ──────────────────────────────────────────────────────────────────
def save_signal(signal: dict):
    _signals[signal["ticker"]] = signal


def get_all_signals() -> list:
    return list(_signals.values())


def get_signal(ticker: str) -> Optional[dict]:
    return _signals.get(ticker)


# ── News ─────────────────────────────────────────────────────────────────────
def push_news(ticker: str, articles: list):
    for a in articles:
        _news[ticker].append(a)


def get_news(ticker: str) -> list:
    return list(_news[ticker])


# ── Corporate events (M&A, contracts, earnings) ───────────────────────────
def push_event(event: dict):
    event.setdefault("ts", datetime.utcnow().isoformat())
    _events.appendleft(event)


def get_events(limit: int = 50) -> list:
    return list(_events)[:limit]


# ── Earnings calendar ─────────────────────────────────────────────────────
def set_earnings_calendar(data: list):
    global _earnings_calendar
    _earnings_calendar = data


def get_earnings_calendar() -> list:
    return _earnings_calendar
