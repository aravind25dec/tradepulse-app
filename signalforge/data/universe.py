"""
signalforge/data/universe.py
Training universe: a broad, liquid set of US equities to fetch history for.

Combines this repo's existing hand-picked watchlist (hotpicks/engine.py's
DEFAULT_UNIVERSE) with the free S&P 500 constituent list from Wikipedia
(no API key, no rate limit) for much broader coverage than any single
sibling app's watchlist — a real ML model needs many more tickers x years
of history than a handful of names to generalize.
"""
import importlib.util
import logging
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_HOTPICKS_DIR = Path(__file__).resolve().parent.parent.parent / "hotpicks"


def _load_module(unique_name: str, file_path: Path):
    """Load a module from a file path under a unique sys.modules key.

    hotpicks/engine.py and autotrader/engine.py are both plain `engine.py` —
    a bare `sys.path.insert` + `import engine` (the pattern the rest of this
    repo uses, e.g. hivepicks importing hotpicks/engine.py) works fine when a
    process only ever imports ONE such module, but this app needs BOTH in the
    same process, and a second `import engine` would just return the first
    one back from sys.modules's cache instead of re-importing. Loading each
    by explicit file path under a distinct name sidesteps that collision.
    """
    if unique_name in sys.modules:
        return sys.modules[unique_name]
    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Fallback core watchlist used only if both the hotpicks import and the
# Wikipedia fetch fail — keeps the pipeline runnable offline / air-gapped.
_FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "JPM", "XOM", "JNJ", "PG", "V", "MA", "HD", "KO", "PEP", "WMT",
]


def _hotpicks_universe() -> list[str]:
    try:
        module = _load_module("_signalforge_hotpicks_engine", _HOTPICKS_DIR / "engine.py")
        return list(module.DEFAULT_UNIVERSE)
    except Exception as exc:
        log.warning("Could not import hotpicks DEFAULT_UNIVERSE: %s", exc)
        return []


def _sp500_tickers() -> list[str]:
    """Scrape the free S&P 500 constituent table from Wikipedia. No key, no rate limit.
    Tickers with a dot (e.g. BRK.B) are normalized to a dash for yfinance (BRK-B).
    """
    try:
        tables = pd.read_html(SP500_WIKI_URL)
        symbols = tables[0]["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
        return sorted(set(symbols.tolist()))
    except Exception as exc:
        log.warning("Could not fetch S&P 500 list from Wikipedia: %s", exc)
        return []


def build_training_universe(max_size: int | None = None) -> list[str]:
    """Order-preserving, deduplicated union of the hotpicks watchlist + S&P 500.
    Falls back to a small hardcoded list if both sources are unreachable.
    """
    seen: dict[str, None] = {}
    for src in (_hotpicks_universe(), _sp500_tickers()):
        for t in src:
            seen.setdefault(t.upper(), None)

    if not seen:
        log.warning("Both universe sources failed — using hardcoded fallback list")
        seen = {t: None for t in _FALLBACK_UNIVERSE}

    tickers = list(seen.keys())
    return tickers[:max_size] if max_size else tickers


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    u = build_training_universe()
    print(f"{len(u)} tickers: {u[:20]}{' ...' if len(u) > 20 else ''}")
