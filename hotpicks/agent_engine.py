"""
agent_engine.py
Routes chat through the LOCAL Claude Code CLI — uses your Claude Pro/Max subscription,
no Anthropic API credits needed.

Claude Code is running in your IDE right now; this file finds the same executable,
calls it with --print, and streams the response back to the web UI.

For Robinhood MCP access (all accounts + trading), run once in a terminal:
  claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading
"""
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

log = logging.getLogger(__name__)

_session_id: str | None = None   # maintained for multi-turn continuity

# ── "Pull Historicals" skill ─────────────────────────────────────────────────
# Opt-in instruction prepended to a message when the chat UI's historicals toggle
# is on. The chat agent otherwise has no fixed technical-analysis behavior — it's
# a bare pass-through to the Claude CLI — so without this it only pulls candle
# data if the wording of the question happens to prompt it to.
_HISTORICALS_SKILL = """Before answering, use the robinhood-trading MCP's get_equity_historicals tool to pull \
recent OHLCV candle data for the ticker(s) this question is about — interval=5minute/span=day for an intraday \
question, interval=day/span=month or 3month for a trend/swing question. Then work through the actual numbers: \
VWAP position, RSI, EMA9 vs EMA21, Bollinger Band position, and the shape of the last 1-2 candles (doji, \
bearish/bullish engulfing, shooting star, etc.). Cite the specific values you found (e.g. "RSI 62, price is 0.4% \
above VWAP") rather than vague language like "looks bullish." If the question isn't about a specific ticker's \
price action, skip this and just answer normally."""

# Idle timeout between stream-json events from the Claude CLI subprocess. A slow/hung
# MCP tool call (e.g. get_equity_historicals over a flaky remote connection) otherwise
# leaves stream_chat() awaiting forever with no error and no way for the UI to recover.
_STREAM_IDLE_TIMEOUT = 90.0


# ── "Optimize Query" skill (LangChain) ───────────────────────────────────────
# Opt-in preprocessing step: rewrites the latest message into a clear, self-contained
# query before it goes to the main agent, using recent chat history (including the
# agent's own prior response) to resolve short/ambiguous follow-ups like a lone ticker
# symbol. Runs through the same local Claude Pro CLI session as the rest of this app
# (via _oneshot_claude) rather than a separate paid API — LangChain here provides the
# prompt/chain structure, not a different model.

class _ClaudeCliLLM(LLM):
    """LangChain LLM wrapper around the local Claude Code CLI (_oneshot_claude).
    Async-only — this app has no sync call path, so _call is intentionally unimplemented.
    """

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    def _call(self, prompt: str, stop: Optional[list[str]] = None,
               run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        raise NotImplementedError("_ClaudeCliLLM is async-only — use ainvoke/_acall")

    async def _acall(self, prompt: str, stop: Optional[list[str]] = None,
                      run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        return await _oneshot_claude(prompt, timeout=30.0)


_OPTIMIZE_PROMPT = ChatPromptTemplate.from_template(
    """You are a query optimizer for a trading-analysis AI agent. Rewrite the user's latest message \
into a single, clear, self-contained, and specific query for the agent to answer directly. Resolve \
pronouns and short/ambiguous follow-ups (e.g. a lone ticker symbol sent right after the agent asked \
what to analyze) using the chat history below. Strip filler words, but preserve every specific ticker, \
number, or constraint the user mentioned. Output ONLY the rewritten query — no preamble, no quotes, \
no explanation.

Chat history:
{chat_history}

Latest message: {question}

Rewritten query:"""
)

_optimize_chain = _OPTIMIZE_PROMPT | _ClaudeCliLLM() | StrOutputParser()


async def optimize_query(messages: list) -> str:
    """Rewrite messages[-1] into a complete, optimized standalone query using the last
    couple of turns (which already include the agent's own prior response) as context.
    Best-effort — falls back to the raw message on any failure since this is a quality-of-life
    enhancement, never a hard dependency for the chat to function.
    """
    question = messages[-1]["content"]
    history_turns = messages[:-1][-4:]   # last ~2 exchanges is enough context to disambiguate
    chat_history = "\n".join(f"{m['role']}: {m['content']}" for m in history_turns) or "(no prior turns)"
    try:
        result = await _optimize_chain.ainvoke({"chat_history": chat_history, "question": question})
        result = result.strip().strip('"')
        return result or question
    except Exception as exc:
        log.warning("Query optimization failed, using raw message: %s", exc)
        return question

# Shared usage tracker (backend/usage_tracker.py) — graceful no-op if absent
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
    import usage_tracker as _ut
    _HAS_TRACKER = True
except ImportError:
    _HAS_TRACKER = False

def _record_usage(result_event: dict, app: str) -> None:
    if not _HAS_TRACKER:
        return
    try:
        u = result_event.get("usage") or {}
        _ut.record(
            input_tokens      = u.get("input_tokens", 0),
            output_tokens     = u.get("output_tokens", 0),
            cache_read_tokens = u.get("cache_read_input_tokens", 0),
            cost_usd          = float(
                result_event.get("total_cost_usd")
                or result_event.get("cost_usd") or 0
            ),
            app=app,
        )
    except Exception as exc:
        log.debug("usage record skipped: %s", exc)


# ── Locate claude CLI ─────────────────────────────────────────────────────────

def _find_claude() -> str | None:
    """
    Find the claude executable.
    On Windows, npm installs it as claude.cmd which shutil.which may miss
    when PATHEXT isn't fully set in the uvicorn process environment.
    """
    # 1. Standard PATH lookup — covers Linux/Mac and most Windows terminals
    for name in ("claude", "claude.cmd", "claude.exe"):
        p = shutil.which(name)
        if p:
            return p

    if sys.platform != "win32":
        return None

    # 2. Windows fallback: check common npm global bin locations
    candidates: list[Path] = []
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        candidates.append(Path(appdata) / "npm" / "claude.cmd")
    candidates.append(Path.home() / "AppData" / "Roaming" / "npm" / "claude.cmd")

    # 3. Ask npm where its global prefix is
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


def reset_session() -> None:
    global _session_id
    _session_id = None


# ── MCP status ────────────────────────────────────────────────────────────────

def check_robinhood_mcp() -> bool:
    """Return True if the Robinhood MCP is registered in Claude Code."""
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


# ── One-shot helper (no session, returns full text) ───────────────────────────

async def _oneshot_claude(prompt: str, timeout: float = 60.0) -> str:
    """
    Run a single prompt through claude CLI without any session tracking.
    Returns the full accumulated text response.
    Used for structured data extraction (e.g. positions JSON from MCP).
    """
    exe = _find_claude()
    if not exe:
        raise RuntimeError("Claude CLI not found")

    # The prompt is piped via stdin rather than passed as a trailing CLI argument.
    # On Windows, shell=True routes the whole command line through cmd.exe, which
    # treats embedded newlines as command separators and silently truncates/mangles
    # any multi-line prompt passed as an argv element — stdin sidesteps that entirely.
    cmd = [
        exe, "--print", "--verbose", "--output-format", "stream-json",
        "--dangerously-skip-permissions",
    ]

    loop  = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    clean_env = os.environ.copy()
    clean_env.pop("ANTHROPIC_API_KEY", None)

    proc_ref: list = []

    def _worker():
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE,
                env=clean_env, shell=(sys.platform == "win32"),
            )
            proc_ref.append(proc)
            proc.stdin.write(prompt.encode("utf-8"))
            proc.stdin.close()
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
            etype = evt.get("type", "")
            if etype == "result":
                _record_usage(evt, "mcp-fetch")
            if etype == "assistant":
                for block in evt.get("message", {}).get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if len(text) > last_len:
                            full_text += text[last_len:]
                            last_len = len(text)
    except asyncio.TimeoutError:
        log.warning("_oneshot_claude timed out after %.0fs — killing subprocess", timeout)
        if proc_ref:
            try:
                proc_ref[0].kill()
            except Exception:
                pass
        raise RuntimeError(f"Claude CLI did not respond within {timeout:.0f}s")

    return full_text


def _parse_markdown_positions(text: str) -> list[dict]:
    """
    Fallback parser: extract positions from Claude's markdown table response.

    Handles format like:
        ## Individual (Default) — ••••7947
        | Symbol | Qty | Avg Cost |
        |--------|-----|----------|
        | MSTR   | 60  | $131.78  |
    """
    import re

    positions: list[dict] = []
    current_label  = "Brokerage"
    current_acct   = ""

    for line in text.splitlines():
        line = line.strip()

        # Section header: ## Individual (Default) — ••••7947
        hdr = re.match(r"^#{1,3}\s+(.+?)(?:\s+[—–-]+\s+[•*]+(\d{3,4}))?\s*$", line)
        if hdr:
            raw = hdr.group(1).strip()
            current_acct = hdr.group(2) or ""
            # Normalise common Robinhood account name patterns
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

        # Skip non-table lines
        if "|" not in line:
            continue
        # Skip header / divider rows
        if re.search(r"symbol|ticker|qty|quantity|avg|cost", line, re.I):
            continue
        if re.match(r"^\|[-| ]+\|$", line):
            continue

        # Parse data row: | MSTR | 60 | $131.78 |
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
            "ticker":         ticker,
            "quantity":       qty_raw,
            "avg_cost":       avg_raw or "0",
            "account_label":  current_label,
            "account_number": current_acct,
        })

    return positions


async def fetch_positions_via_mcp() -> list[dict]:
    """
    Ask Claude to use the Robinhood MCP and return current positions.
    Tries JSON array first; falls back to parsing Claude's markdown table output.
    Returns list of dicts: [{ticker, quantity, avg_cost, account_label, account_number}]
    """
    import re

    prompt = (
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

    log.info("Fetching positions via Robinhood MCP through Claude CLI…")
    text = await _oneshot_claude(prompt)
    log.debug("MCP positions raw response: %s", text[:600])

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)

    # ── Try JSON array first ──────────────────────────────────────────────────
    start = cleaned.find("[")
    end   = cleaned.rfind("]")
    if start >= 0 and end > start:
        try:
            positions = json.loads(cleaned[start : end + 1])
            result = [p for p in positions if float(p.get("quantity", 0)) > 0]
            log.info("Parsed %d positions from JSON", len(result))
            return result
        except (json.JSONDecodeError, ValueError):
            log.warning("JSON parse failed; falling back to markdown parser")

    # ── Fallback: parse markdown tables ──────────────────────────────────────
    result = _parse_markdown_positions(text)
    if result:
        log.info("Parsed %d positions from markdown table", len(result))
        return result

    raise ValueError(f"Could not parse positions from Claude response: {text[:400]}")


# ── Chat streaming ────────────────────────────────────────────────────────────

async def stream_chat(messages: list, include_historicals: bool = False,
                       optimize: bool = False) -> AsyncGenerator[str, None]:
    """
    Run the latest user message through `claude --print --output-format stream-json`
    and stream the assistant response.

    Uses subprocess.Popen in a background thread (asyncio subprocesses don't work
    on Windows SelectorEventLoop) bridged to this async generator via asyncio.Queue.

    include_historicals=True prepends _HISTORICALS_SKILL to this turn's message,
    nudging Claude to pull OHLCV candles via the Robinhood MCP and walk through
    the technicals before answering — the "Pull Historicals" toggle in agent.html.

    optimize=True runs messages through optimize_query() first (LangChain, backed by this
    same CLI) to rewrite a short/ambiguous message into a complete standalone query using
    recent chat history — the "Optimize Query" toggle in agent.html.
    """
    global _session_id

    exe = _find_claude()
    if not exe:
        yield (
            "⚠ **Claude Code CLI not found.**\n\n"
            "1. Install: `npm install -g @anthropic-ai/claude-code`\n"
            "2. Authenticate: run `claude` in your terminal once\n"
            "3. Restart the TradePulse server\n\n"
            f"Searched in: `{os.environ.get('PATH','(empty PATH)')}`"
        )
        return

    if not messages:
        return

    user_msg = messages[-1]["content"]

    if optimize:
        optimized = await optimize_query(messages)
        if optimized and optimized != user_msg:
            yield f"\n\n_🧠 optimized query: \"{optimized}\"_\n\n"
            user_msg = optimized

    if include_historicals:
        user_msg = f"{_HISTORICALS_SKILL}\n\n---\n\n{user_msg}"

    # ── Build CLI command ──────────────────────────────────────────────────
    # The prompt is piped via stdin rather than passed as a trailing CLI argument —
    # see the matching comment in _oneshot_claude for why (cmd.exe mangles embedded
    # newlines in argv on Windows when shell=True).
    cmd = [
        exe,
        "--print",
        "--verbose",
        "--output-format", "stream-json",
        "--dangerously-skip-permissions",   # allow MCP tool calls without prompts
    ]

    if _session_id and len(messages) > 1:
        # Resume existing session — Claude already has the full history
        cmd += ["--resume", _session_id]
        stdin_prompt = user_msg
    else:
        if len(messages) > 1:
            # Embed prior turns on first message or after session reset
            prior = []
            for m in messages[:-1]:
                role = "User" if m["role"] == "user" else "Assistant"
                prior.append(f"{role}: {m['content']}")
            stdin_prompt = "[Prior context]\n" + "\n\n".join(prior) + f"\n\n[Current question]\n{user_msg}"
        else:
            stdin_prompt = user_msg

    log.info("Claude CLI: %s --print --output-format stream-json …", exe)

    # ── Launch in thread, bridge via Queue ────────────────────────────────
    loop:  asyncio.AbstractEventLoop = asyncio.get_event_loop()
    queue: asyncio.Queue             = asyncio.Queue()

    # Strip ANTHROPIC_API_KEY so the CLI uses the Pro subscription
    # (browser-authenticated), not API credits.
    clean_env = os.environ.copy()
    clean_env.pop("ANTHROPIC_API_KEY", None)

    proc_ref: list = []

    def _worker():
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                env=clean_env,
                # shell=True lets cmd.exe resolve .cmd files on Windows
                shell=(sys.platform == "win32"),
            )
            proc_ref.append(proc)
            proc.stdin.write(stdin_prompt.encode("utf-8"))
            proc.stdin.close()
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

    # ── Consume the queue and yield text ─────────────────────────────────
    last_len = 0   # track cumulative text to yield only new delta per event
    announced_tools: set = set()   # avoid re-announcing the same tool_use block

    while True:
        try:
            tag, val = await asyncio.wait_for(queue.get(), timeout=_STREAM_IDLE_TIMEOUT)
        except asyncio.TimeoutError:
            log.warning("stream_chat idle for %.0fs — killing subprocess (likely a hung tool call)", _STREAM_IDLE_TIMEOUT)
            if proc_ref:
                try:
                    proc_ref[0].kill()
                except Exception:
                    pass
            _session_id = None   # subprocess was killed mid-turn — don't resume into a stuck state
            yield (
                f"\n\n⚠ **No response for {_STREAM_IDLE_TIMEOUT:.0f}s — the agent looked stuck** "
                f"(likely a slow or hung tool call, e.g. Robinhood MCP). The session was reset; please try again."
            )
            break

        if tag == "done":
            break

        if tag == "err":
            log.error("Claude CLI stderr: %s", val)
            yield f"\n\n⚠ Claude error:\n```\n{val}\n```"
            break

        line = val.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            # Plain text output (non-JSON build) — pass through
            if line:
                yield line + "\n"
            continue

        etype = evt.get("type", "")

        if etype == "system" and evt.get("subtype") == "init":
            sid = evt.get("session_id")
            if sid:
                _session_id = sid
                log.debug("Session: %s", _session_id)
            continue

        if etype == "user":
            continue
        if etype == "result":
            _record_usage(evt, "agent")
            continue

        if etype == "assistant":
            for block in evt.get("message", {}).get("content", []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    key = block.get("id") or block.get("name")
                    if key and key not in announced_tools:
                        announced_tools.add(key)
                        yield f"\n\n_🔧 calling `{block.get('name', 'tool')}`…_\n\n"
                elif block.get("type") == "text":
                    full = block.get("text", "")
                    if len(full) > last_len:
                        yield full[last_len:]
                        last_len = len(full)
                    elif last_len == 0 and full:
                        yield full
                        last_len = len(full)
