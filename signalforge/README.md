# SignalForge

A locally-trained stock direction predictor fused with live context pulled through
the **Claude Code CLI** (your Pro/Max subscription, not the Anthropic API), with a
small locally fine-tuned LLM narrator layered on top for the write-up.

> ⚠️ **Not financial advice.** Educational project. See the root [README.md](../README.md) disclaimer.

---

## 1. What this is (and isn't)

- **The numeric prediction always comes from a LightGBM classifier trained on this
  machine**, on real historical OHLCV + technical-indicator data. It's saved as a
  versioned, immutable package under `models/registry/` — every training run adds a
  new version and never touches a previous one.
- **No accuracy number here is ever hardcoded.** Every prediction ships with the
  model's own confidence *and* the real, out-of-sample backtested accuracy for
  predictions at that confidence level (see `GET /api/models`). A "99% accurate"
  stock predictor isn't a real thing for any method on real markets — this app
  reports honestly, and transparently, how often it's actually been right.
- **The narrative text comes from a small fine-tuned local LLM (`signalforge-narrator`),
  served through Ollama**, with a plain templated fallback if Ollama or the narrator
  is unavailable. It **only restyles** the numbers it's handed — it never predicts
  the signal or confidence itself. That job belongs to the classifier, full stop.
- **The dashboard is tile-based**: a Favorites section (tickers you've searched, stored
  permanently server-side) and a Robinhood Holdings section (fetched live from your
  connected accounts via the Robinhood MCP). A ticker held in both is predicted
  **once** and shared between both tiles — see §6.

## 2. Architecture

```
data/
  universe.py         Training universe: hotpicks' watchlist + free S&P 500 list (Wikipedia)
  fetch_history.py     Bulk yfinance historical downloader -> data/raw/{ticker}.parquet
features/
  build_features.py    Shared feature engineering (RSI/SMA/MACD/Bollinger/OBV via
                        autotrader/engine.py's pandas_ta indicators + derived ratios) —
                        the SAME function is used at training time and predict time
train/
  train_classifier.py  LightGBM, time-ordered walk-forward split, confidence calibration
  registry.py           Versioned model packages on disk (models/registry/v{N}_{timestamp}/)
models/registry/        model.txt, feature_list.json, metrics.json, training_config.json per version
agent_engine.py         Claude Code CLI bridge (subprocess, --print --output-format stream-json)
live_context.py         Asks Claude for a live price/sentiment/candle read on a ticker
robinhood_positions.py  Fetches distinct tickers across ALL Robinhood accounts via the
                        robinhood-trading MCP (through the Claude Code CLI, same technique
                        as hotpicks/agent_engine.py's fetch_positions_via_mcp)
storage.py              Favorites list + shared per-ticker prediction cache — both plain
                        JSON files on disk (favorites.json, predictions_cache.json,
                        both gitignored — personal runtime state, not source)
narrator/
  build_dataset.py      Bootstraps (prompt -> narration) training pairs via the Claude CLI
  train_narrator.py     LoRA fine-tune (Qwen2.5-0.5B-Instruct) -> merge -> GGUF -> register with Ollama
  registry.json          Tracks every narrator version ever registered (never overwritten)
  dataset/v*.jsonl       Generated training data (gitignored)
  merged/, gguf/         Generated model artifacts (gitignored)
pipeline.py             LangGraph StateGraph tying it all together (see below)
main.py                 FastAPI app — GET /api/predict, GET /api/models, POST /api/models/{v}/promote
index.html              Single-page UI
```

### Predict flow (`pipeline.py`)

```
                 +-> build_features -> score_model -+
START -> fetch_ohlcv                                 +-> narrate -> assemble_response -> END
                 +-------------> live_context -------+
```

`score_model` (fast, local) and `live_context` (the Claude Code CLI subprocess call,
by far the slowest step) have no data dependency on each other, so they run as
parallel LangGraph branches and `narrate` joins on both. `narrate` then tries the
registered Ollama narrator (`signalforge-narrator:<latest>`) and falls back to a
plain mechanical narrative built directly from the numbers if Ollama/the narrator
call fails for any reason — a narrator hiccup must never break the actual prediction.

## 3. Setup

```bash
cd signalforge
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

`requirements-narrator.txt` (torch/transformers/peft/datasets/gguf) is **only**
needed on the machine doing the narrator fine-tuning step — not for training the
classifier or serving predictions day-to-day (the fine-tuned model is served by
Ollama, not by this Python process).

```bash
pip install -r requirements-narrator.txt
```

## 4. Training the classifier (the actual predictor)

```bash
cd data
python fetch_history.py --period 10y          # bulk-download historical OHLCV
cd ../train
python train_classifier.py                     # builds features, trains, writes a new registry version
```

Each run prints the holdout accuracy and writes a full `metrics.json` (per-class
precision/recall, log-loss, and a confidence-calibration table) to a brand-new
`models/registry/v{N}_{timestamp}/` directory, then points `current.json` at it.
Older versions are never modified or deleted — roll back anytime with:

```bash
curl -X POST http://localhost:8011/api/models/v2_20260101T000000Z/promote
```

**Current status on this machine**: 3 versions trained (`v1`/`v2`/`v3`, all on the
same ~33-ticker fallback universe — see §7), holdout accuracy ~42.7% on a 3-class
UP/FLAT/DOWN problem (33% = chance). Retraining with a broader universe is the
most impactful next step (see §7).

## 5. Narrator fine-tuning (the write-up layer)

This is a genuinely slow, CPU-only, multi-hour job on hardware without a discrete
GPU. Treat it as a batch job you kick off and check on, not something interactive.

```bash
cd narrator
python build_dataset.py --n-per-class 100      # ~300 examples, ~1 Claude CLI call each
python train_narrator.py --dataset dataset/v2.jsonl --epochs 3
```

`build_dataset.py` samples real historical (ticker, date, features) rows, scores
them with the **current** classifier (so the numbers narrated match what the model
actually outputs), synthesizes a plausible live-context sentence locally (no Claude
call needed for that part — see the docstring for why), and asks Claude once per
sample to polish a plain mechanical draft into fluent analyst prose. That polished
text is the fine-tuning target.

`train_narrator.py`:
1. LoRA fine-tunes a small base model (`Qwen/Qwen2.5-0.5B-Instruct` by default) via
   `peft` — **checkpoints every 20 steps** (`save_strategy="steps"`) and
   **automatically resumes from the latest checkpoint** if you rerun it after an
   interruption. This matters: on this hardware, a full run is ~6-8 hours at
   ~110-150s/step, and *will* get interrupted by sleep/reboots/session restarts —
   see §8 for exactly this happening twice during development.
2. Merges the adapter into the base weights and converts to GGUF via a shallow
   clone of `llama.cpp` (`narrator/.llama_cpp/`, one-time setup, auto-cloned).
3. Registers the result with Ollama via `POST /api/create` over HTTP — **not** the
   `ollama` CLI, which isn't reliably on PATH in every shell/user context. The
   HTTP API is confirmed reachable at `localhost:11434` regardless.
4. Records the new version in `narrator/registry.json` (append-only, mirrors the
   classifier registry's "never overwrite" philosophy) — `pipeline.py` always uses
   the last entry, no restart needed once a new version registers.

**Current status**: two versions trained and registered —
`signalforge-narrator:v1` (9 examples, proof-of-concept, prone to filler) and
`signalforge-narrator:v2` (294 examples, 3 epochs, the real one). `pipeline.py`
uses `v2` automatically.

## 6. Running the API + dashboard

```bash
uvicorn main:app --reload --port 8011
```

The UI (http://localhost:8011) is a tile dashboard with two sections:

- **⭐ Favorites** — every ticker you've ever searched, added automatically
  (`POST /api/favorites/{ticker}`) and stored permanently in `favorites.json`.
  Each tile has an **✕** to remove it and a **↻** to force a fresh prediction.
- **🏦 Robinhood Holdings** — fetched live on page load from every connected
  Robinhood account via the MCP (`GET /api/robinhood/positions`), read-only
  (no ✕ — holdings aren't something you "remove" from the dashboard, they
  reflect your actual account).

**One prediction per ticker, shared across both sections.** If a ticker is both
a favorite and an actual holding, it's predicted once and both tiles read the
same cached result (`storage.py`'s `predictions_cache.json`, keyed by ticker).
Clicking **↻** on either tile force-refreshes that ticker's cache entry —
because both tiles read from the same underlying entry, the other section's
tile for that ticker reflects the refresh too, without needing its own click.
Tiles show a compact signal/confidence/key-feature summary and a "last
updated" timestamp; click a tile to open a modal with the full analyst
narrative, a **plain-English explanation** (rule-based, always available even
if the narrator/Ollama is down — see `pipeline._layman_explanation`), live
context, and the raw feature values.

Every distinct ticker is predicted **fully in parallel** on load (each is its
own Claude CLI subprocess, ~20-45s) — after the first load, every ticker is
cached (`storage.py`) so reloading the page is near-instant until a ↻ is
clicked. Robinhood holdings are also filtered to real positions: anything
under `robinhood_positions.MIN_HOLDING_QUANTITY` (1 full share) total across
all accounts is dropped — dividend-reinvestment/round-up fractional dust
(e.g. 0.0116 shares of MSFT) isn't a holding anyone is tracking, and a real
36-ticker account was mostly this kind of noise before the filter.

**API reference:**

| Endpoint | Behavior |
|---|---|
| `GET /api/predict?ticker=X` | Full prediction (signal, confidence, backtested accuracy, narrative, layman explanation, live Claude context, features). Served from `predictions_cache.json` if present — near-instant, no CLI call, `from_cache: true` |
| `GET /api/predict?ticker=X&force_refresh=true` | Bypasses the cache, re-runs the full pipeline, **overwrites** the shared cache entry — this is what the tile's ↻ button calls |
| `GET /api/favorites` | List of favorited tickers |
| `POST /api/favorites/{ticker}` | Add to favorites (idempotent) |
| `DELETE /api/favorites/{ticker}` | Remove from favorites |
| `GET /api/robinhood/status` | `{claude_available, robinhood_mcp}` — whether the CLI/MCP are ready |
| `GET /api/robinhood/positions` | Distinct tickers held across all accounts, with per-account quantity/avg-cost |
| `GET /api/models` | Every trained classifier version + its real backtested metrics |
| `POST /api/models/{version}/promote` | Roll back/forward without retraining |

Requires the [Claude Code CLI](https://github.com/anthropics/claude-code) installed
and authenticated (`npm install -g @anthropic-ai/claude-code`, then run `claude`
once) for both the live-context step and the Robinhood MCP fetch. For the latter,
also register the MCP once: `claude mcp add robinhood-trading --transport http
https://agent.robinhood.com/mcp/trading`. If the CLI or Ollama/the narrator is
unavailable, prediction still works with graceful fallbacks (`live_context.available:
false`, mechanical narrative) — see `AGENTS.md` for known flakiness on this machine
and how to check/recover without retraining.

## 7. Known limitations

- **Training universe is small.** The Wikipedia S&P 500 fetch in `data/universe.py`
  fails with an SSL certificate error in this environment (looks like a
  system-clock/cert-freshness mismatch, not a code bug — Yahoo's longer-lived certs
  aren't affected the same way). Training so far has only used the ~33-ticker
  fallback watchlist from `hotpicks/engine.py`. More tickers = more rows = a model
  that generalizes better; this is the highest-leverage next step once the SSL
  issue is sorted (or point `build_training_universe()` at another free ticker-list
  source).
- **Ollama has been intermittently unreachable on this machine** during
  development (confirmed working, then refusing connections minutes later, with no
  clear trigger identified). `pipeline.py` degrades gracefully when this happens —
  predictions keep working, just with the mechanical narrative instead of the
  fine-tuned one. See `AGENTS.md` §"Ollama is down / flaky" for the triage steps.
- **CPU LoRA training is slow and its pace is not stable** — per-step time drifted
  from ~110s to ~145-170s over a single run in testing, for reasons not fully
  root-caused (thermal throttling, memory pressure, and background load are all
  plausible). Budget 6-8+ hours for a ~300-example/3-epoch run and rely on the
  checkpointing to survive interruptions rather than assuming a clean single sitting.

## 8. Roadmap

1. ~~Data pipeline + feature engineering + LightGBM classifier + versioned registry~~ — done
2. ~~Claude Code CLI live-context agent + LangGraph orchestration API~~ — done
3. ~~Fine-tuned local LLM narrator, served through Ollama as a versioned tag~~ — done (`v2`)
4. Broaden the training universe past the current ~33-ticker fallback (§7)
5. Larger/higher-quality narrator dataset once universe is broader (more diverse
   historical setups to sample from)

See `AGENTS.md` for a debugging/triage guide and architecture notes aimed at
whoever (human or agent) picks this codebase up next.
