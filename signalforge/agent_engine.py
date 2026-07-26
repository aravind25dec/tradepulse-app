"""
signalforge/agent_engine.py
Routes a single one-shot prompt through the LOCAL Claude Code CLI — uses your
Claude Pro/Max subscription, no Anthropic API credits needed.

Trimmed copy of hotpicks/agent_engine.py: this app only needs a one-shot
structured-JSON live snapshot per ticker (see live_context.py), not multi-turn
chat sessions or Robinhood MCP brokerage access, so the session-resume,
streaming-chat, and Robinhood position-fetching code from the original isn't
carried over.
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

log = logging.getLogger(__name__)

# Idle timeout between stream-json events from the Claude CLI subprocess. A slow/hung
# tool call otherwise leaves _oneshot_claude() awaiting forever with no error.
_STREAM_IDLE_TIMEOUT = 90.0

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
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            cache_read_tokens=u.get("cache_read_input_tokens", 0),
            cost_usd=float(result_event.get("total_cost_usd") or result_event.get("cost_usd") or 0),
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
        r = subprocess.run("npm config get prefix", shell=True, capture_output=True, text=True, timeout=5)
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


# ── One-shot helper (no session, returns full text) ───────────────────────────

async def oneshot_claude(prompt: str, timeout: float = 60.0) -> str:
    """
    Run a single prompt through the claude CLI without any session tracking.
    Returns the full accumulated text response.
    """
    exe = _find_claude()
    if not exe:
        raise RuntimeError("Claude CLI not found — install with: npm install -g @anthropic-ai/claude-code")

    # The prompt is piped via stdin rather than passed as a trailing CLI argument.
    # On Windows, shell=True routes the whole command line through cmd.exe, which
    # treats embedded newlines as command separators and silently truncates/mangles
    # any multi-line prompt passed as an argv element — stdin sidesteps that entirely.
    cmd = [exe, "--print", "--verbose", "--output-format", "stream-json", "--dangerously-skip-permissions"]

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    # Strip ANTHROPIC_API_KEY so the CLI uses the Pro subscription (browser-authenticated),
    # not API credits.
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
    last_len = 0
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
                full_text += line + "\n"
                continue
            etype = evt.get("type", "")
            if etype == "result":
                _record_usage(evt, "signalforge-live-context")
            if etype == "assistant":
                for block in evt.get("message", {}).get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if len(text) > last_len:
                            full_text += text[last_len:]
                            last_len = len(text)
    except asyncio.TimeoutError:
        log.warning("oneshot_claude timed out after %.0fs — killing subprocess", timeout)
        if proc_ref:
            try:
                proc_ref[0].kill()
            except Exception:
                pass
        raise RuntimeError(f"Claude CLI did not respond within {timeout:.0f}s")

    return full_text
