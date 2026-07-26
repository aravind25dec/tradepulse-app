# TradePulse AutoTrader

An automated stock screening + trading agent. It auto-discovers today's most active movers
(on top of a small core watchlist), computes a technical indicator matrix locally, classifies
each ticker as **BUY / SELL / HOLD** with a confidence score, and — only when you opt in — can
automatically submit a real bracket order through the [Alpaca](https://alpaca.markets) Trade API
for high-confidence BUY signals.

> ⚠️ **Not financial advice.** Ships pointed at Alpaca's **paper trading** endpoint by default.
> Treat `AUTO_EXECUTE` as a live-fire switch — read the [Execution & risk](#execution--risk) section
> before turning it on against a funded account.

---

## What this app is

It's a self-contained implementation of the screening spec in [`../trade-app.md`](../trade-app.md),
structured the same way as the other single-purpose apps in this repo (`hotpicks/`, `screener/`,
`smartpicks/`, ...): a `main.py` FastAPI server, an `engine.py` with all the logic, and a static
`index.html` dashboard — no build step, no frontend framework.

What it does, end to end:

1. Build today's scan universe: a fixed core watchlist (`AAPL`, `MSFT`, `NVDA`, `AMD`, `TSLA`)
   plus whatever Yahoo Finance's `day_gainers` / `day_losers` / `most_actives` screeners surface
   right now (via `yfinance.screen()`), deduped and capped at 40 tickers so a single request
   doesn't have to fetch an unbounded ticker list. Pass an explicit `tickers=` list to skip
   discovery and scan only what you ask for.
2. Pull ~2 years of daily OHLCV per ticker from Yahoo Finance (via `yfinance`, no API key).
3. Compute RSI(14), SMA(50), SMA(200), MACD(12,26,9), Bollinger Bands(20, 2σ), and OBV locally
   with `pandas_ta` — no paid indicator API.
4. Run the day's candle through a 4-category confirmation matrix (momentum, trend, volume,
   volatility). All four must agree for a BUY or SELL to fire; otherwise it's a HOLD.
5. For BUY signals at **HIGH** confidence, check Alpaca account cash, size a position at 2%
   portfolio risk, and (if `auto_execute` is on) submit a bracket order: limit entry + 3%
   stop-loss + 9% take-profit as nested child orders.

---

## How it's designed

```
autotrader/
  engine.py          Data + analytics + decision matrix + Alpaca execution (no FastAPI imports)
  main.py             FastAPI server — caching, background scans, HTTP surface (port 8010)
  index.html          Dark-themed dashboard, polls the API, no build step
  requirements.txt     fastapi, yfinance, pandas_ta, alpaca-trade-api, ...
```

**`engine.py`** is deliberately framework-agnostic — every function is a plain Python function
that takes a ticker or DataFrame and returns a dict. That's what makes `python engine.py AAPL TSLA`
work standalone (see below) and what makes the decision matrix independently unit-testable.
It's organized in the same four sections as the spec:

- **Data Engine** — `fetch_history()`. Returns `None` on a bad/delisted ticker or network error
  instead of raising, so one bad symbol never kills a whole batch scan. `discover_top_movers()`
  and `build_scan_universe()` sit alongside it: they query Yahoo's predefined screeners for
  today's gainers/losers/most-actives, filter out illiquid names (<$3 or <300k volume), merge
  with `DEFAULT_WATCHLIST`, and tag every ticker with a `source` (`core` / `day_gainers` /
  `day_losers` / `most_actives` / `custom`) so downstream results and the UI can show *why* a
  ticker was scanned. Both are best-effort — a Yahoo screener outage just shrinks the discovered
  set instead of failing the scan.
- **Analytics Engine** — `compute_indicators()`. Appends every indicator onto the DataFrame via
  pandas_ta's `df.ta.*` accessor, then normalizes the Bollinger Band column names (pandas_ta's
  std-suffix formatting changes between versions) into stable `BB_LOWER` / `BB_UPPER` / `BB_WIDTH`
  columns so the rest of the code doesn't care which pandas_ta version is installed.
- **Decision Matrix** — `evaluate_ticker()`. Pure function: DataFrame in, signal dict out.
  Confidence is HIGH when the volume surge is unusually strong (≥1.5x the 20-day average for BUY,
  or a >2% down move on high volume for SELL), MEDIUM otherwise, LOW when a HOLD only weakly
  leans one direction.
- **Execution Engine** — `execute_bracket_order()` and friends. Everything that talks to Alpaca
  lives behind `_get_alpaca_client()`, which returns `None` (never raises) if the SDK isn't
  installed or credentials aren't set — the screener always works in signal-only mode even with
  zero Alpaca setup.

**`main.py`** wraps `engine.py` in a small stateful HTTP service: an in-memory `_cache` dict holds
the latest scan result, a `running` flag, and the `auto_execute` toggle, following the same
"kick off a scan 15s after startup, cache the result, poll on a timer" pattern as `hotpicks/main.py`.
The startup scan (and any `/api/autotrader/run` call with no `tickers` query param) passes
`tickers=None` through to `engine.run_screen()`, which triggers `build_scan_universe()` — so by
default the app is always scanning whatever's actually moving today, not a hardcoded list.
Because `yfinance`/`pandas_ta` are blocking, scans run via `loop.run_in_executor(...)` so they don't
stall the event loop.

**`index.html`** is a single polling page (no WebSocket) — it hits `/api/autotrader/results` every
few seconds while a scan is running and backs off once idle. The auto-execute toggle requires a
confirm dialog before it can be switched on, since flipping it changes the behavior of the *next*
scan to place real orders.

---

## How to start it

Uses its own `requirements.txt` (yfinance + pandas_ta + alpaca-trade-api aren't in the shared
backend venv). One-time setup:

```bash
cd autotrader
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

Add Alpaca paper-trading keys to `backend/.env` (get free ones at alpaca.markets):

```env
APCA_API_KEY_ID=your_alpaca_key_here
APCA_API_SECRET_KEY=your_alpaca_secret_here
APCA_API_BASE_URL=https://paper-api.alpaca.markets
```

If `backend/.env` already has `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` / `ALPACA_BASE_URL` set (the
names the main backend uses), `_get_alpaca_client()` falls back to those automatically — you don't
need to duplicate the same keys under both naming conventions.

Then run:

```bash
python -m uvicorn main:app --reload --port 8010
```

- UI: http://localhost:8010
- API docs: http://localhost:8010/docs
- Or via the repo-wide launcher: `start.bat` (Windows) / `start.sh` (Mac/Linux) from the project root

Without Alpaca keys set, the app still runs fully — it screens tickers and reports BUY/SELL/HOLD,
it just reports `"alpaca": {"connected": false}` and any execute attempt returns
`{"executed": false, "reason": "Alpaca API not configured..."}` instead of erroring.

### Command-line mode

`engine.py` also runs standalone and prints the spec's exact JSON array to stdout:

```bash
python engine.py AAPL MSFT NVDA AMD TSLA
```

### Key endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/autotrader/results` | Latest cached scan results (each result includes a `source` field) |
| `POST` | `/api/autotrader/run?tickers=AAPL,MSFT` | Trigger a background scan. Omit `tickers` to auto-discover today's universe (core watchlist + top gainers/losers/most-actives) |
| `GET`  | `/api/autotrader/movers` | Preview today's auto-discovered universe (`{tickers, sources, count}`) without running a full screen |
| `GET`  | `/api/autotrader/status` | Scan state + Alpaca account snapshot |
| `POST` | `/api/autotrader/auto-execute` | Toggle automatic bracket-order submission on future scans (`{"enabled": true}`) |
| `POST` | `/api/autotrader/execute` | Manually fire one bracket order (`{"ticker": "AAPL", "entry_price": 200}`) |
| `GET`  | `/api/autotrader/orders/{order_id}` | Poll a submitted order's fill status |

---

## Execution & risk

- **`auto_execute` defaults to `False`.** A scan only ever screens tickers until you explicitly
  flip the toggle in the UI (behind a confirm dialog) or `POST` to `/api/autotrader/auto-execute`.
- Position sizing risks exactly 2% of `portfolio_value` per trade, based on the distance from
  entry to the 3% stop-loss — not a fixed share count. It's also capped by available cash, so a
  BUY never oversizes past what the account can actually afford.
- Every Alpaca call is wrapped so a rejected order, a dropped connection, or an
  insufficient-buying-power response comes back as `{"executed": false, "reason": "..."}"` instead
  of a stack trace — a bad order never takes down a scan of the rest of the watchlist.
- `APCA_API_BASE_URL` controls paper vs. live. Double-check it before enabling `auto_execute`
  against a funded account.

---

## How to improvise / extend

Ideas roughly in order of effort:

- **Saved custom watchlists.** `POST /api/autotrader/run?tickers=...` already accepts an ad-hoc
  list that bypasses discovery entirely — wire a saved-watchlist file (see `trades/watchlist.json`
  for the pattern already used elsewhere in this repo) as another selectable source alongside
  "auto-discover".
- **Tune the discovery filters.** `MIN_MOVER_PRICE`, `MIN_MOVER_VOLUME`, `MAX_PER_SCREEN`, and
  `MAX_UNIVERSE_SIZE` at the top of `engine.py` control how aggressive `build_scan_universe()` is.
  A larger `MAX_UNIVERSE_SIZE` finds more opportunities but scans take longer since each ticker is
  a separate sequential `yf.download()` call — worth parallelizing with a thread pool if this grows
  much past 40-50.
- **Recurring scans.** The startup task only fires once. Add an `asyncio` loop that reschedules
  `_run_and_cache` every N minutes during market hours (`hotpicks/main.py` doesn't do this either,
  so you'd be establishing the pattern — a simple `while True: await asyncio.sleep(...)` task
  works fine).
- **Order tracking UI.** `check_order_status()` exists but nothing in `index.html` polls it after
  a bracket order is submitted. Add a small "open orders" panel that polls
  `/api/autotrader/orders/{id}` until the entry leg fills, so partial fills are visible instead of
  just fire-and-forget.
- **Backtesting the decision matrix.** `evaluate_ticker()` is a pure function of a DataFrame — you
  can replay it against historical windows (loop over `df.iloc[:i]` for increasing `i`) to measure
  the historical hit rate of BUY/SELL signals before trusting `auto_execute` with real money.
  `hotpicks/engine.py`'s `_historical_patterns()` shows the same idea already implemented for a
  different engine — a good reference for the windowing logic.
- **Loosen or tighten the confirmation matrix.** All four categories currently must agree
  (spec-mandated). If BUY signals feel too rare in practice, an easy first experiment is requiring
  3-of-4 categories instead of 4-of-4 and comparing signal frequency/quality — do this in
  `evaluate_ticker()` by changing `all(buy_categories)` to a `sum(buy_categories) >= 3` check.
- **Risk parameters are constants today** (`RISK_PER_TRADE_PCT`, `STOP_LOSS_PCT`,
  `TAKE_PROFIT_PCT` at the top of `engine.py`). Making them per-request overrides (query params or
  UI fields) would let you A/B different risk profiles without editing code.
- **Sentiment/news enrichment.** `screener_engine.py` and `hotpicks/engine.py` both layer a
  Claude-scored news sentiment pass on top of the technicals — the same pattern (fetch headlines,
  score with the shared `_RISK_SYSTEM` prompt via `agent_engine._oneshot_claude`) would drop in
  here as an optional confidence adjustment.
