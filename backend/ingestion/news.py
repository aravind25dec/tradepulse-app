"""
ingestion/news.py
Fetches news from NewsAPI and SEC EDGAR 8-K filings.
Polls every 10 minutes.
"""
import asyncio
import os
import logging
import re
from datetime import datetime, timedelta

import httpx

from db.store import push_news, push_event, get_events

log = logging.getLogger(__name__)

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
HEADERS_SEC = {"User-Agent": "TradingApp contact@example.com"}  # SEC requires User-Agent


async def fetch_newsapi(ticker: str, client: httpx.AsyncClient) -> list[dict]:
    """Fetch top headlines for a ticker from NewsAPI."""
    if not NEWS_API_KEY:
        return []
    try:
        r = await client.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": ticker,
                "sortBy": "publishedAt",
                "apiKey": NEWS_API_KEY,
                "pageSize": 5,
                "language": "en",
                "from": (datetime.utcnow() - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%S"),
            },
            timeout=10,
        )
        articles = r.json().get("articles", [])
        return [
            {
                "source": a["source"]["name"],
                "title": a["title"],
                "url": a["url"],
                "published": a["publishedAt"],
                "ticker": ticker,
            }
            for a in articles
            if a.get("title") and "[Removed]" not in a.get("title", "")
        ]
    except Exception as e:
        log.warning(f"NewsAPI error for {ticker}: {e}")
        return []


async def fetch_sec_8k(cik: str, ticker: str, client: httpx.AsyncClient) -> list[dict]:
    """Fetch recent SEC 8-K filings and detect M&A / contract keywords."""
    try:
        r = await client.get(
            f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json",
            headers=HEADERS_SEC,
            timeout=15,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        descriptions = filings.get("primaryDocDescription", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])

        events = []
        for i, form in enumerate(forms[:30]):
            if form == "8-K":
                desc = descriptions[i] if i < len(descriptions) else ""
                if re.search(
                    r"(acqui|merger|contract award|definitive agreement|material agreement|joint venture)",
                    desc,
                    re.IGNORECASE,
                ):
                    event = {
                        "ticker": ticker,
                        "type": "SEC_8K",
                        "description": desc,
                        "filing_date": dates[i] if i < len(dates) else "",
                        "accession": accessions[i] if i < len(accessions) else "",
                        "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K",
                    }
                    events.append(event)
                    push_event(event)
        return events
    except Exception as e:
        log.warning(f"SEC EDGAR error for {ticker}/{cik}: {e}")
        return []


# CIK lookup for common tickers (add more as needed)
TICKER_CIK_MAP = {
    "AAPL": "320193",
    "TSLA": "1318605",
    "NVDA": "1045810",
    "MSFT": "789019",
    "AMZN": "1018724",
    "GOOGL": "1652044",
    "META": "1326801",
}


async def news_poll_loop(tickers: list[str], broadcast_fn, interval_seconds: int = 600):
    """Poll news for all tickers every interval_seconds."""
    log.info("Starting news poll loop...")
    async with httpx.AsyncClient() as client:
        while True:
            for ticker in tickers:
                # Stock-only news from NewsAPI
                base_ticker = ticker.split("-")[0]  # handle BTC-USD → BTC
                articles = await fetch_newsapi(base_ticker, client)
                if articles:
                    push_news(ticker, articles)
                    await broadcast_fn({"type": "news", "data": {"ticker": ticker, "articles": articles}})

                # SEC filings for known tickers
                cik = TICKER_CIK_MAP.get(ticker)
                if cik:
                    await fetch_sec_8k(cik, ticker, client)

            await asyncio.sleep(interval_seconds)
