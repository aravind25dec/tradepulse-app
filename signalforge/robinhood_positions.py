"""
signalforge/robinhood_positions.py
Fetches current stock positions across ALL Robinhood accounts via the
robinhood-trading MCP, through the Claude Code CLI (agent_engine.oneshot_claude)
— same technique as hotpicks/agent_engine.py's fetch_positions_via_mcp, adapted
here to just the distinct-tickers list this app's tiles need.

Requires the MCP already registered once via:
  claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading
"""
import json
import logging
import re

import agent_engine

log = logging.getLogger(__name__)

_POSITIONS_PROMPT = (
    "Use the robinhood-trading MCP tools to fetch ALL my current stock positions "
    "across every account (Brokerage, Traditional IRA, Roth IRA, and any others).\n\n"
    "IMPORTANT: Respond with ONLY a raw JSON array — no markdown fences, no explanation:\n"
    '[{"ticker":"AAPL","quantity":"10.0000","avg_cost":"180.50",'
    '"account_label":"Brokerage","account_number":"5RY12345"}]\n\n'
    "Rules:\n"
    "- Only positions where quantity > 0\n"
    "- ticker is the stock symbol (AAPL, not Apple Inc)\n"
    "- avg_cost is average cost per share (number only, no $ sign)\n"
    "- account_label: Brokerage / Traditional IRA / Roth IRA\n"
    "- If no positions exist return: []"
)


def _parse_markdown_positions(text: str) -> list[dict]:
    """Fallback parser for Claude's markdown-table response — see hotpicks/agent_engine.py
    for the original, more heavily-commented version this is copied from."""
    positions: list[dict] = []
    current_label = "Brokerage"
    current_acct = ""

    for line in text.splitlines():
        line = line.strip()
        hdr = re.match(r"^#{1,3}\s+(.+?)(?:\s+[—–-]+\s+[•*]+(\d{3,4}))?\s*$", line)
        if hdr:
            raw = hdr.group(1).strip()
            current_acct = hdr.group(2) or ""
            low = raw.lower()
            if "traditional" in low and "ira" in low:
                current_label = "Traditional IRA"
            elif "roth" in low:
                current_label = "Roth IRA"
            elif "individual" in low or "brokerage" in low or "default" in low:
                current_label = "Brokerage"
            else:
                current_label = raw
            continue

        if "|" not in line:
            continue
        if re.search(r"symbol|ticker|qty|quantity|avg|cost", line, re.I):
            continue
        if re.match(r"^\|[-| ]+\|$", line):
            continue

        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 2:
            continue

        ticker = re.sub(r"[^A-Za-z]", "", parts[0]).upper()
        qty_raw = re.sub(r"[^0-9.]", "", parts[1])
        avg_raw = re.sub(r"[^0-9.]", "", parts[2]) if len(parts) > 2 else "0"

        if not ticker or not qty_raw:
            continue
        try:
            if float(qty_raw) <= 0:
                continue
        except ValueError:
            continue

        positions.append({
            "ticker": ticker, "quantity": qty_raw, "avg_cost": avg_raw or "0",
            "account_label": current_label, "account_number": current_acct,
        })

    return positions


async def fetch_positions() -> list[dict]:
    """Ask Claude to use the Robinhood MCP and return current positions.
    Raises on failure — caller (main.py) turns that into a 503/500 JSON error."""
    text = await agent_engine.oneshot_claude(_POSITIONS_PROMPT, timeout=60.0, app="signalforge-robinhood")
    log.debug("MCP positions raw response: %s", text[:600])

    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)

    start, end = cleaned.find("["), cleaned.rfind("]")
    if start >= 0 and end > start:
        try:
            positions = json.loads(cleaned[start : end + 1])
            result = [p for p in positions if float(p.get("quantity", 0) or 0) > 0]
            log.info("Parsed %d positions from JSON", len(result))
            return result
        except (json.JSONDecodeError, ValueError):
            log.warning("JSON parse failed; falling back to markdown parser")

    result = _parse_markdown_positions(text)
    if result:
        log.info("Parsed %d positions from markdown table", len(result))
        return result

    raise ValueError(f"Could not parse positions from Claude response: {text[:400]}")


# Positions below this many total shares are excluded from the dashboard —
# dividend-reinvestment/round-up fractional remnants (e.g. 0.01164 shares of
# MSFT) aren't holdings anyone is actually tracking, and cluttered a real
# 36-ticker account down to mostly noise. Chosen from a real gap in observed
# data: intentional-looking positions started at 15+ shares, dust topped out
# under 2.5.
MIN_HOLDING_QUANTITY = 1.0


async def fetch_distinct_tickers() -> list[dict]:
    """Return one entry per distinct ticker held (merging quantity across accounts),
    which is what the dashboard's Robinhood tiles actually need. Tickers whose
    TOTAL quantity across all accounts is still under MIN_HOLDING_QUANTITY are
    dropped — see the constant's comment for why."""
    positions = await fetch_positions()
    by_ticker: dict[str, dict] = {}
    for p in positions:
        t = p["ticker"].upper()
        qty = float(p.get("quantity", 0) or 0)
        if t not in by_ticker:
            by_ticker[t] = {"ticker": t, "total_quantity": 0.0, "accounts": []}
        by_ticker[t]["total_quantity"] += qty
        by_ticker[t]["accounts"].append({
            "account_label": p.get("account_label", "Brokerage"),
            "quantity": qty,
            "avg_cost": float(p.get("avg_cost", 0) or 0),
        })
    return [h for h in by_ticker.values() if h["total_quantity"] >= MIN_HOLDING_QUANTITY]
