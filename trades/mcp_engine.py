"""
trades/mcp_engine.py
Robinhood via Claude CLI + MCP — all account/order operations.
No username/password; auth is handled by the Claude browser session.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ── Locate claude CLI ─────────────────────────────────────────────────────────

def _find_claude() -> str | None:
    for name in ("claude", "claude.cmd", "claude.exe"):
        p = shutil.which(name)
        if p:
            return p

    if sys.platform != "win32":
        return None

    candidates: list[Path] = []
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        candidates.append(Path(appdata) / "npm" / "claude.cmd")
    candidates.append(Path.home() / "AppData" / "Roaming" / "npm" / "claude.cmd")

    try:
        r = subprocess.run(
            "npm config get prefix",
            shell=True, capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            candidates.append(Path(r.stdout.strip()) / "claude.cmd")
    except Exception:
        pass

    for p in candidates:
        if p.is_file():
            log.info("claude found at: %s", p)
            return str(p)

    return None


def claude_available() -> bool:
    return _find_claude() is not None


def check_robinhood_mcp() -> bool:
    exe = _find_claude()
    if not exe:
        return False
    try:
        r = subprocess.run(
            [exe, "mcp", "list"],
            capture_output=True, text=True,
            shell=(sys.platform == "win32"),
            timeout=8,
        )
        return "robinhood" in r.stdout.lower()
    except Exception:
        return False


# ── Core subprocess helper ────────────────────────────────────────────────────

async def _oneshot(prompt: str, timeout: float = 90.0) -> str:
    """Run a single prompt through Claude CLI and return the full text response."""
    exe = _find_claude()
    if not exe:
        raise RuntimeError("Claude CLI not found")

    cmd = [
        exe, "--print", "--verbose", "--output-format", "stream-json",
        "--dangerously-skip-permissions", prompt,
    ]
    loop  = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    # Strip ANTHROPIC_API_KEY so CLI uses Pro subscription, not API credits
    clean_env = os.environ.copy()
    clean_env.pop("ANTHROPIC_API_KEY", None)
    proc_ref: list = []

    def _worker():
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=clean_env, shell=(sys.platform == "win32"),
            )
            proc_ref.append(proc)
            for raw in proc.stdout:
                loop.call_soon_threadsafe(queue.put_nowait, ("line", raw))
            proc.wait()
            if proc.returncode not in (0, None):
                err = proc.stderr.read().decode("utf-8", errors="replace").strip()
                if err:
                    loop.call_soon_threadsafe(queue.put_nowait, ("err", err))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("err", str(exc)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    threading.Thread(target=_worker, daemon=True).start()

    full_text = ""
    last_len  = 0
    try:
        while True:
            tag, val = await asyncio.wait_for(queue.get(), timeout=timeout)
            if tag == "done":
                break
            if tag == "err":
                raise RuntimeError(f"Claude CLI error: {val}")
            line = val.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                if line:
                    full_text += line + "\n"
                continue
            if evt.get("type") == "assistant":
                for block in evt.get("message", {}).get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if len(text) > last_len:
                            full_text += text[last_len:]
                            last_len = len(text)
    except asyncio.TimeoutError:
        if proc_ref:
            try:
                proc_ref[0].kill()
            except Exception:
                pass
        raise RuntimeError(f"MCP call timed out after {timeout:.0f}s")

    return full_text


def _parse_json(text: str, default: Any) -> Any:
    """Strip markdown fences and parse the first JSON value found."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)
    for start_c, end_c in (("[", "]"), ("{", "}")):
        start = cleaned.find(start_c)
        end   = cleaned.rfind(end_c)
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    return default


# ── Account ───────────────────────────────────────────────────────────────────

async def fetch_account_via_mcp() -> dict:
    prompt = (
        "Use the robinhood-trading MCP tools to get my total portfolio equity "
        "and today's gain/loss percentage.\n\n"
        "Respond with ONLY a raw JSON object — no markdown, no explanation:\n"
        '{"equity": 52340.50, "day_return_pct": 1.25}\n\n'
        "- equity: total portfolio value in dollars\n"
        "- day_return_pct: today's return as a percentage (positive = gain, negative = loss)"
    )
    text = await _oneshot(prompt)
    data = _parse_json(text, {})
    return {
        "equity":         float(data.get("equity", 0) or 0),
        "day_return_pct": float(data.get("day_return_pct", 0) or 0),
    }


# ── Positions ─────────────────────────────────────────────────────────────────

async def fetch_positions_via_mcp() -> list[dict]:
    prompt = (
        "Use the robinhood-trading MCP tools to fetch ALL my current stock positions "
        "across every account (Brokerage, Traditional IRA, Roth IRA, and any others).\n\n"
        "Respond with ONLY a raw JSON array — no markdown fences, no explanation:\n"
        '[{"ticker":"AAPL","quantity":"10.0000","avg_cost":"180.50",'
        '"account_label":"Brokerage","account_number":"5RY12345"}]\n\n'
        "Rules:\n"
        "- Only include positions where quantity > 0\n"
        "- ticker is the stock symbol (AAPL, not Apple Inc)\n"
        "- avg_cost is average cost per share, number only, no $ sign\n"
        "- account_label: Brokerage / Traditional IRA / Roth IRA\n"
        "- If no positions exist return: []"
    )
    text = await _oneshot(prompt)
    data = _parse_json(text, [])
    if not isinstance(data, list):
        return []
    return [p for p in data if float(p.get("quantity", 0) or 0) > 0]


# ── Open Orders ───────────────────────────────────────────────────────────────

async def fetch_open_orders_via_mcp() -> list[dict]:
    prompt = (
        "Use the robinhood-trading MCP tools to get all my currently open stock orders.\n\n"
        "Respond with ONLY a raw JSON array — no markdown, no explanation:\n"
        '[{"id":"abc-123","ticker":"AAPL","side":"buy","order_type":"limit",'
        '"quantity":10,"filled_qty":0,"limit_price":180.00,"stop_price":null,'
        '"state":"confirmed","created_at":"2024-01-01T10:00:00Z","time_in_force":"gtc"}]\n\n'
        "If there are no open orders return: []"
    )
    text = await _oneshot(prompt)
    data = _parse_json(text, [])
    return data if isinstance(data, list) else []


# ── Order History ─────────────────────────────────────────────────────────────

async def fetch_order_history_via_mcp(limit: int = 30) -> list[dict]:
    prompt = (
        f"Use the robinhood-trading MCP tools to get my last {limit} stock orders "
        "(including filled, cancelled, and any other state).\n\n"
        "Respond with ONLY a raw JSON array — no markdown, no explanation:\n"
        '[{"id":"abc-123","ticker":"AAPL","side":"buy","order_type":"market",'
        '"quantity":5,"filled_qty":5,"avg_price":182.50,'
        '"state":"filled","created_at":"2024-01-01T10:00:00Z"}]\n\n'
        "If no order history return: []"
    )
    text = await _oneshot(prompt)
    data = _parse_json(text, [])
    return data if isinstance(data, list) else []


# ── Place Order ───────────────────────────────────────────────────────────────

async def place_order_via_mcp(
    side:          str,
    ticker:        str,
    quantity:      float,
    order_type:    str,
    limit_price:   float | None = None,
    stop_price:    float | None = None,
    trail_amount:  float | None = None,
    trail_type:    str = "percentage",
    time_in_force: str = "gfd",
) -> dict:
    type_desc = {
        "market":        "market order (execute at current market price)",
        "limit":         f"limit order at ${limit_price:.2f}" if limit_price else "limit order",
        "stop_loss":     f"stop loss order triggered at ${stop_price:.2f}" if stop_price else "stop loss order",
        "stop_limit":    f"stop limit order (stop price ${stop_price:.2f}, limit price ${limit_price:.2f})" if stop_price and limit_price else "stop limit order",
        "trailing_stop": f"trailing stop order with trail of {trail_amount}{'%' if trail_type == 'percentage' else '$'}" if trail_amount else "trailing stop order",
    }.get(order_type, order_type)

    tif_desc = {
        "gfd": "good for day (GFD)",
        "gtc": "good till cancelled (GTC)",
        "ioc": "immediate or cancel (IOC)",
        "opg": "market on open (OPG)",
    }.get(time_in_force, time_in_force.upper())

    prompt = (
        f"Use the robinhood-trading MCP tools to place the following stock order:\n"
        f"  Action:        {side.upper()}\n"
        f"  Symbol:        {ticker}\n"
        f"  Quantity:      {quantity} shares\n"
        f"  Order type:    {type_desc}\n"
        f"  Time in force: {tif_desc}\n\n"
        "After placing the order, respond with ONLY a raw JSON object — no markdown:\n"
        '{"success": true, "order_id": "abc-123", "state": "confirmed"}\n\n'
        "If the order fails respond with:\n"
        '{"success": false, "error": "specific reason for failure"}'
    )
    text = await _oneshot(prompt)
    data = _parse_json(text, {"success": False, "error": "No response from MCP"})
    if not isinstance(data, dict):
        return {"success": False, "error": str(data)}
    return data


# ── Cancel Order ──────────────────────────────────────────────────────────────

async def cancel_order_via_mcp(order_id: str) -> dict:
    prompt = (
        f"Use the robinhood-trading MCP tools to cancel the open stock order with ID: {order_id}\n\n"
        "After cancelling, respond with ONLY a raw JSON object — no markdown:\n"
        '{"success": true}\n\n'
        "If it cannot be cancelled:\n"
        '{"success": false, "error": "reason"}'
    )
    text = await _oneshot(prompt)
    data = _parse_json(text, {"success": False, "error": "No response"})
    return data if isinstance(data, dict) else {"success": False, "error": str(data)}
