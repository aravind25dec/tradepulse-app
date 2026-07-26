"""
backend/usage_tracker.py
Shared Claude token-usage tracker for all TradePulse apps.

Records token counts extracted from Claude CLI stream-json "result" events.
Persists stats to backend/usage_stats.json — survives server restarts.

Tracks three windows:
  - 5-hour rolling window  (Claude Pro/Max actual rate-limit window)
  - Daily totals
  - Weekly totals (resets Mon 00:00 UTC)
"""

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

_FILE = Path(__file__).parent / "usage_stats.json"
_LOCK = threading.Lock()

H5_WINDOW = 5 * 3600   # seconds in the Claude Pro rate-limit window

# Conservative estimates — Anthropic does not publish exact token limits
PLAN_LIMITS = {
    "Pro":  {"weekly": 5_000_000, "h5": 1_000_000},
    "Max":  {"weekly": 25_000_000, "h5": 5_000_000},
}
DEFAULT_PLAN = "Pro"


# ── Time helpers ─────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _week_bounds() -> tuple[str, str]:
    """(week_start_iso, week_reset_iso) — Mon 00:00 UTC to next Mon 00:00 UTC."""
    now    = _now()
    monday = (now - timedelta(days=now.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    return monday.isoformat(), (monday + timedelta(days=7)).isoformat()


def _today_key() -> str:
    return _now().strftime("%Y-%m-%d")


# ── Storage ───────────────────────────────────────────────────────────────────

def _blank() -> dict:
    ws, wr = _week_bounds()
    now_iso = _now().isoformat()
    return {
        "plan":           DEFAULT_PLAN,
        # 5-hour window
        "h5_start":       now_iso,
        "h5_requests":    0,
        "h5_input":       0,
        "h5_output":      0,
        # daily
        "today_key":      _today_key(),
        "today_requests": 0,
        "today_input":    0,
        "today_output":   0,
        # weekly
        "week_start":     ws,
        "week_reset":     wr,
        "week_requests":  0,
        "week_input":     0,
        "week_output":    0,
        "week_cache":     0,
        "week_cost_usd":  0.0,
        # metadata
        "last_request_at": None,
        "session_start":  now_iso,
        "apps":           {},
    }


def _load() -> dict:
    try:
        if _FILE.exists():
            data = json.loads(_FILE.read_text(encoding="utf-8"))
            ws, _ = _week_bounds()
            if data.get("week_start") == ws:
                return data
    except Exception:
        pass
    return _blank()


def _save(data: dict) -> None:
    _FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _maybe_reset_h5(d: dict) -> dict:
    """Reset the 5-hour window counters if the window has expired."""
    h5_start_str = d.get("h5_start")
    if not h5_start_str:
        d["h5_start"]    = _now().isoformat()
        d["h5_requests"] = d["h5_input"] = d["h5_output"] = 0
        return d
    try:
        h5_start = datetime.fromisoformat(h5_start_str)
        if (_now() - h5_start).total_seconds() >= H5_WINDOW:
            d["h5_start"]    = _now().isoformat()
            d["h5_requests"] = d["h5_input"] = d["h5_output"] = 0
    except Exception:
        d["h5_start"]    = _now().isoformat()
        d["h5_requests"] = d["h5_input"] = d["h5_output"] = 0
    return d


# ── Public API ────────────────────────────────────────────────────────────────

def record(
    *,
    input_tokens:      int   = 0,
    output_tokens:     int   = 0,
    cache_read_tokens: int   = 0,
    cost_usd:          float = 0.0,
    app:               str   = "unknown",
) -> None:
    """Call this after every Claude CLI subprocess with the result event's usage."""
    with _LOCK:
        d     = _load()
        today = _today_key()

        # 5h window — reset if expired
        d = _maybe_reset_h5(d)
        d["h5_requests"] += 1
        d["h5_input"]    += input_tokens
        d["h5_output"]   += output_tokens

        # daily — reset if new day
        if d.get("today_key") != today:
            d["today_key"] = today
            d["today_requests"] = d["today_input"] = d["today_output"] = 0
        d["today_requests"] += 1
        d["today_input"]    += input_tokens
        d["today_output"]   += output_tokens

        # weekly
        d["week_requests"]  += 1
        d["week_input"]     += input_tokens
        d["week_output"]    += output_tokens
        d["week_cache"]     += cache_read_tokens
        d["week_cost_usd"]  += cost_usd

        d["last_request_at"] = _now().isoformat()

        a = d["apps"].setdefault(app, {"requests": 0, "input": 0, "output": 0})
        a["requests"] += 1
        a["input"]    += input_tokens
        a["output"]   += output_tokens

        _save(d)


def get_stats() -> dict:
    with _LOCK:
        d = _load()
        d = _maybe_reset_h5(d)

    limits = PLAN_LIMITS.get(d.get("plan", DEFAULT_PLAN), PLAN_LIMITS[DEFAULT_PLAN])

    total_h5    = d["h5_input"]    + d["h5_output"]
    total_today = d["today_input"] + d["today_output"]
    total_week  = d["week_input"]  + d["week_output"]

    h5_pct   = round(min(100.0, total_h5   / limits["h5"]     * 100), 2) if limits["h5"]     else 0.0
    week_pct = round(min(100.0, total_week / limits["weekly"]  * 100), 2) if limits["weekly"] else 0.0

    # Minutes remaining in current 5h window
    h5_reset_in_min = 0
    try:
        h5_start    = datetime.fromisoformat(d["h5_start"])
        h5_end      = h5_start + timedelta(seconds=H5_WINDOW)
        remaining_s = (h5_end - _now()).total_seconds()
        h5_reset_in_min = max(0, int(remaining_s // 60))
    except Exception:
        h5_reset_in_min = 0

    return {
        **d,
        # 5h computed
        "h5_total_tokens":    total_h5,
        "h5_usage_pct":       h5_pct,
        "h5_limit":           limits["h5"],
        "h5_reset_in_min":    h5_reset_in_min,
        # daily / weekly computed
        "today_total_tokens": total_today,
        "week_total_tokens":  total_week,
        "plan_limit_weekly":  limits["weekly"],
        "week_usage_pct":     week_pct,
    }


def reset_stats() -> None:
    with _LOCK:
        _save(_blank())


def set_plan(plan: str) -> None:
    with _LOCK:
        d = _load()
        d["plan"] = plan
        _save(d)
