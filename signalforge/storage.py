"""
signalforge/storage.py
Two small pieces of durable local state, both plain JSON files on disk
(consistent with this repo's existing convention — backend/usage_stats.json,
localai/memory.db — rather than pulling in a database for a single-user tool):

1. Favorites — the ticker list shown as tiles on the dashboard.
2. Prediction cache — one entry per ticker, shared by BOTH the Favorites
   section and the Robinhood-holdings section. This is what makes "predict
   once, use everywhere" (see AGENTS.md) actually work: whichever section
   renders a ticker first triggers /api/predict; the other section's tile for
   the same ticker reads the same cached entry instead of triggering a second
   Claude CLI call + model inference.

A module-level lock guards read-modify-write on both files — FastAPI can serve
concurrent requests, and favorites/cache updates are infrequent enough that a
simple lock (not a database) is the right amount of engineering here.
"""
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
FAVORITES_PATH = _THIS_DIR / "favorites.json"
CACHE_PATH = _THIS_DIR / "predictions_cache.json"

_lock = threading.Lock()


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to read %s, using default: %s", path, exc)
        return default


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── Favorites ──────────────────────────────────────────────────────────────

def get_favorites() -> list[str]:
    with _lock:
        return _read_json(FAVORITES_PATH, [])


def add_favorite(ticker: str) -> list[str]:
    ticker = ticker.upper().strip()
    with _lock:
        favorites = _read_json(FAVORITES_PATH, [])
        if ticker not in favorites:
            favorites.append(ticker)
            _write_json(FAVORITES_PATH, favorites)
        return favorites


def remove_favorite(ticker: str) -> list[str]:
    ticker = ticker.upper().strip()
    with _lock:
        favorites = _read_json(FAVORITES_PATH, [])
        if ticker in favorites:
            favorites.remove(ticker)
            _write_json(FAVORITES_PATH, favorites)
        return favorites


# ── Prediction cache ───────────────────────────────────────────────────────

def get_cached_prediction(ticker: str) -> dict | None:
    ticker = ticker.upper().strip()
    with _lock:
        cache = _read_json(CACHE_PATH, {})
        return cache.get(ticker)


def set_cached_prediction(ticker: str, prediction: dict) -> dict:
    """Stamps and stores a prediction result. Returns the stamped dict
    (with cached_at added) so the caller can hand the same object straight
    back in the HTTP response — one source of truth for the timestamp."""
    ticker = ticker.upper().strip()
    stamped = {**prediction, "cached_at": datetime.now(timezone.utc).isoformat()}
    with _lock:
        cache = _read_json(CACHE_PATH, {})
        cache[ticker] = stamped
        _write_json(CACHE_PATH, cache)
    return stamped
