"""
signalforge/data/fetch_history.py
Bulk historical OHLCV downloader for the training universe.

Reuses autotrader/engine.py's fetch_history() (yfinance, free, no key) rather
than a new hand-rolled Yahoo client — that function already handles the
MultiIndex-column quirk and delisted/bad-symbol failures safely.

Writes one Parquet file per ticker to data/raw/{ticker}.parquet. Re-running
overwrites a ticker's own file with fresher data; it does not touch any
already-trained model in models/registry/ (those are separate, immutable).
"""
import logging
import sys
import time
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_AUTOTRADER_DIR = _THIS_DIR.parent.parent / "autotrader"
sys.path.insert(0, str(_THIS_DIR))          # for `from universe import ...` regardless of invocation style
sys.path.insert(0, str(_AUTOTRADER_DIR))    # for `from engine import fetch_history`

from engine import fetch_history as _fetch_history_yf  # type: ignore  # noqa: E402
from universe import build_training_universe  # type: ignore  # noqa: E402

log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent / "raw"
DEFAULT_PERIOD = "10y"
REQUEST_DELAY_SEC = 0.6   # polite delay between per-ticker yfinance requests


def fetch_and_save(ticker: str, period: str = DEFAULT_PERIOD) -> bool:
    df = _fetch_history_yf(ticker, period=period)
    if df is None or df.empty:
        log.warning("Skipping %s — no data returned", ticker)
        return False
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_DIR / f"{ticker}.parquet")
    return True


def fetch_universe(tickers: list[str] | None = None, period: str = DEFAULT_PERIOD,
                    max_tickers: int | None = None) -> dict:
    tickers = tickers or build_training_universe(max_size=max_tickers)
    ok, failed = [], []
    for i, ticker in enumerate(tickers, 1):
        try:
            if fetch_and_save(ticker, period=period):
                ok.append(ticker)
            else:
                failed.append(ticker)
        except Exception as exc:
            log.error("Fetch failed for %s: %s", ticker, exc)
            failed.append(ticker)
        if i % 25 == 0:
            log.info("Progress: %d/%d tickers fetched (%d ok, %d failed)", i, len(tickers), len(ok), len(failed))
        time.sleep(REQUEST_DELAY_SEC)

    log.info("Done: %d ok, %d failed out of %d", len(ok), len(failed), len(tickers))
    return {"ok": ok, "failed": failed, "period": period}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Bulk-download historical OHLCV for the training universe.")
    parser.add_argument("--period", default=DEFAULT_PERIOD, help="yfinance period, e.g. 10y, 5y, max")
    parser.add_argument("--max-tickers", type=int, default=None, help="Cap universe size (useful for a quick test run)")
    args = parser.parse_args()

    result = fetch_universe(period=args.period, max_tickers=args.max_tickers)
    print(f"Fetched {len(result['ok'])} tickers, {len(result['failed'])} failed.")
