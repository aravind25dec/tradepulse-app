# AGENTS.md — SignalForge

This file is for whoever (human or AI agent) next works on this codebase. It covers
architecture, the specific bugs already hit and fixed during development (so you
don't re-discover them the hard way), how to triage the failure modes that are
likely to recur, and how to extend the system safely. Pair with `README.md` for
setup/usage — this file is the "how it actually works and how to fix it" doc.

## 1. Design invariants — do not violate these

1. **The LightGBM classifier (`train/train_classifier.py`) is the only thing that
   predicts a number.** The narrator (Ollama-served fine-tuned LLM) only restyles
   numbers it's handed. If you're tempted to have the narrator "double check" or
   adjust the signal/confidence, don't — that's exactly the scope creep this
   architecture was built to prevent.
2. **No accuracy number is ever hardcoded.** Every claim about accuracy traces back
   to `metrics.json`'s real walk-forward backtest, computed fresh each training run.
3. **Feature computation must be IDENTICAL at train time and predict time.**
   `features/build_features.py`'s `engineer_features()` is called from both
   `build_training_table()` (training) and `compute_latest_features()` (predict) —
   never duplicate this logic elsewhere. If you add a feature, add it here once.
4. **Model registries are append-only.** `train/registry.py` (classifier) and
   `narrator/registry.json` (narrator) never overwrite or delete a previous
   version — every training run adds a new one and a `current`/latest pointer
   moves forward. This is what makes rollback possible and makes a bad retrain
   non-destructive.
5. **A narrator/live-context failure must never break `/api/predict`'s numeric
   result.** Both `live_context.get_live_snapshot()` and `pipeline._call_narrator()`
   catch everything and return a clearly-labeled fallback (`available: false` /
   the mechanical narrative) rather than raising.

## 2. Architecture, file by file

| File | Responsibility |
|---|---|
| `data/universe.py` | Training universe list. Imports `hotpicks/engine.py`'s `DEFAULT_UNIVERSE` via `importlib` (see §3.1 for why not a plain import), plus a Wikipedia S&P 500 scrape (currently broken, see §3.5) |
| `data/fetch_history.py` | Bulk historical OHLCV via `autotrader/engine.py`'s `fetch_history()` (yfinance) → `data/raw/{ticker}.parquet` |
| `features/build_features.py` | `engineer_features()` (indicators + derived ratios, via `autotrader/engine.py`'s `compute_indicators()`), `add_forward_labels()` (training only), `compute_latest_features()` (predict-time entry point), `FEATURE_COLUMNS` (the canonical, ordered feature list) |
| `train/train_classifier.py` | Time-ordered walk-forward split (`_time_split`), LightGBM training, confidence calibration (`_confidence_calibration` — bucketed real accuracy by confidence, numeric `confidence_min`/`confidence_max` bounds, not parsed pandas Interval strings) |
| `train/registry.py` | `save_version()`, `load_model()`, `list_versions()`, `set_current()` (promote/rollback) — versioned dirs under `models/registry/` |
| `agent_engine.py` | Trimmed copy of `hotpicks/agent_engine.py` — finds the `claude` CLI cross-platform, runs it via subprocess with `--output-format stream-json`, strips `ANTHROPIC_API_KEY` so it rides the Pro/Max subscription. Only `oneshot_claude()` — no sessions, no Robinhood MCP |
| `live_context.py` | `get_live_snapshot(ticker)` — one `agent_engine.oneshot_claude()` call asking for a JSON price/sentiment/candle read. Best-effort, never raises |
| `narrator/build_dataset.py` | Samples historical rows, scores them with the current classifier, synthesizes a plausible (not real) live-context sentence, asks Claude to polish `pipeline._mechanical_narrative()`'s draft into fluent prose. Writes `narrator/dataset/v{N}.jsonl` incrementally (one line per example, flushed immediately) |
| `narrator/train_narrator.py` | LoRA fine-tune (`peft`) → merge → GGUF convert (shallow `llama.cpp` clone) → register with Ollama over HTTP (`POST /api/create`, not the CLI) → append to `narrator/registry.json` |
| `pipeline.py` | LangGraph `StateGraph` — see §2.1 |
| `main.py` | FastAPI app: `/api/predict`, `/api/models`, `/api/models/{v}/promote` |

### 2.1 `pipeline.py`'s graph

```
                 +-> build_features -> score_model -+
START -> fetch_ohlcv                                 +-> narrate -> assemble_response -> END
                 +-------------> live_context -------+
```

State is a single `TypedDict` (`PredictState`) threaded through every node;
each node returns a partial dict that LangGraph merges into the shared state.
`score_model` and `live_context` run in parallel (no data dependency between
them) and `narrate` joins on both.

**The join is `graph.add_edge(["score_model", "live_context"], "narrate")` — a
single call with a list of sources.** This was a real bug during development:
two *separate* `add_edge(...)` calls into the same node each independently
trigger it as soon as *their own* source completes, rather than waiting for
all predecessors. Since `live_context` is one hop from `fetch_ohlcv` and
`score_model` is two hops (via `build_features`), `narrate` fired after
`live_context` alone finished, before `model_output` existed in state, and
crashed with `KeyError: 'model_output'`. If you add more parallel branches,
use the list-form `add_edge` for any real join point.

`narrate_node` tries `_call_narrator()` (POST to Ollama `/api/chat` with the
tag from `narrator/registry.json`'s last entry) and falls back to
`_mechanical_narrative()` on any failure — timeout, connection refused, empty
response, anything.

## 3. Known bugs already hit (and their fixes) — read before you "fix" these again

### 3.1 Module name collision: `hotpicks/engine.py` vs `autotrader/engine.py`

Both are literally named `engine.py`. The rest of this repo's convention
(`sys.path.insert(0, dir); import engine`) works fine when a process only ever
imports *one* of them — but this app needs `hotpicks`'s `DEFAULT_UNIVERSE` *and*
`autotrader`'s `fetch_history`/`compute_indicators` in the same process. A
second `import engine` just returns the first one back from `sys.modules`'s
cache instead of re-importing, silently giving you the wrong module.

**Fix in place**: `data/universe.py`'s `_load_module(unique_name, file_path)`
uses `importlib.util.spec_from_file_location` to load each `engine.py` under a
distinct `sys.modules` key (`_signalforge_hotpicks_engine`). If you need to
import a third same-named sibling module, do the same — don't add another bare
`sys.path.insert` + `import engine`.

### 3.2 `transformers`/`peft` API drift between versions

This environment resolved `transformers==5.14.1`, `peft==0.19.1`,
`datasets==5.0.0` — much newer than the `>=` floors in
`requirements-narrator.txt`, and two real breaking changes surfaced:

- `tokenizer.apply_chat_template(..., tokenize=True)` returns a `BatchEncoding`
  in this version, not a plain `list[int]` — `prompt_ids + completion_ids`
  crashed with `TypeError: unsupported operand type(s) for +`. **Fix**: render
  with `tokenize=False` to get a string, then tokenize that string yourself
  (`train_narrator.py::_finetune._tokenize`). Version-independent, don't revert.
- `TrainingArguments(no_cuda=True)` — `no_cuda` was removed. **Fix**: just don't
  pass it; CPU-only runs fine without it since there's no CUDA device to select
  anyway.

If you upgrade these packages further, re-run the tiny-dataset validation path
(§5.1) before trusting a multi-hour run again.

### 3.3 `| tee` masks the real exit code — always redirect directly

The first full-scale training attempt was launched as
`python train_narrator.py ... 2>&1 | tee log.txt`. In bash, a pipeline's exit
status is the *last* command's (`tee`, which exits 0 as long as it can write to
the file) — not the actual Python process's. The training process was killed
partway through (see §3.4), but the wrapper reported success, and this wasn't
caught until the log was inspected manually and found to stop mid-training with
no merge/GGUF/registration ever happening.

**Fix in place**: launch as `python train_narrator.py ... > log.txt 2>&1;
echo "EXIT_CODE=$?"` — direct redirection, no pipe, with the real exit code
echoed into the same log so it's unambiguous on inspection. Never pipe a
long-running training command through `tee`/`grep`/anything else without
`set -o pipefail` (or just avoid the pipe entirely, as above).

### 3.4 Long CPU training WILL get interrupted — checkpoint or lose everything

Two full-scale runs died before one succeeded:

- Run 1: `save_strategy="no"`. Died silently around step 145/222 (system sleep
  or similar, never fully diagnosed) with **nothing recoverable** — the entire
  ~5 hours of compute was lost, because there was no checkpoint to resume from.
- Run 2: after adding checkpointing, died around step 180/222, this time
  because the session/harness process itself was torn down (visible in the
  task-notification: "may have been running when the previous Claude Code
  process exited"). This time `checkpoint-180` was intact.
- Run 3 (resumed from `checkpoint-180`): completed training + merge + GGUF
  conversion successfully, but the Ollama registration call failed
  (`ConnectError ... actively refused`) because Ollama itself wasn't running
  at that moment. Registration was then redone standalone (§5.2) once Ollama
  came back up, **without repeating any training** — the GGUF file was already
  on disk.

**Fix in place**: `TrainingArguments(save_strategy="steps", save_steps=20,
save_total_limit=2)`, and `_finetune()` auto-detects the latest
`checkpoints/checkpoint-*` dir and passes it to
`trainer.train(resume_from_checkpoint=...)`. If you change dataset size or
epoch count enough to move step boundaries around, this still works — it just
resumes from whatever checkpoint exists under that version's `output_dir`.

**Practical implication**: never assume a multi-hour background job here ran
to completion just because a task-tracking notification says so. Always
verify by grepping the actual log for the terminal markers
(`grep -E "Registered:|Traceback"`) and/or checking that the expected output
file (`narrator/gguf/v{N}.gguf`, a new `narrator/registry.json` entry) actually
exists.

### 3.5 SSL certificate error fetching the Wikipedia S&P 500 list

`data/universe.py`'s `_sp500_tickers()` fails with
`[SSL: CERTIFICATE_VERIFY_FAILED] certificate has expired` when calling
`pd.read_html` on the Wikipedia page. This looks like a system-clock/cert-
freshness mismatch in this environment (Yahoo Finance's HTTPS calls, used
everywhere else in this app, don't hit the same issue — plausibly because
Yahoo's cert has a longer validity window than Wikipedia's shorter-lived one
relative to whatever this machine's clock thinks "now" is). **Do not "fix"
this by disabling certificate verification** — that's a real security
regression for a one-off data-fetch convenience. Either resolve the underlying
clock/cert issue at the OS level, or swap in a different free ticker-list
source. Until then, training silently falls back to the ~33-ticker
`hotpicks` watchlist — check `universe.py`'s log output to confirm which path
was actually used before trusting a training run's ticker coverage.

### 3.6 Windows backslashes in the Ollama `Modelfile`'s `FROM` line

A `Modelfile`'s `FROM <path>` line chokes on Windows-style backslashes (read
as escape sequences by the parser). **Fix in place**:
`train_narrator.py::_register_with_ollama` uses `gguf_path.resolve().as_posix()`
rather than the raw `Path` string. If you construct a Modelfile anywhere else,
do the same.

## 4. Triage guide for live issues

### "Ollama is down / flaky"

Symptom: `pipeline.py` logs `Narrator call to signalforge-narrator:v...
failed: ...` and `/api/predict` silently returns the mechanical narrative
instead of fluent prose (this is correct, expected degradation — check §1
invariant 5 — but you may still want the real narrator working).

1. Check reachability directly: `curl -m 5 http://localhost:11434/api/tags`.
   If this hangs/refuses, Ollama itself isn't running — this has happened
   intermittently on this machine with no root cause identified yet (confirmed
   reachable, then refused minutes later, no clear trigger).
2. The `ollama` CLI is **not reliably on PATH** in every shell/user context on
   this machine (confirmed: `Get-Command ollama` and a filesystem search for
   `ollama.exe` under this user's profile and common install dirs both came up
   empty, even while the Ollama *service* was actively serving models over
   HTTP — it appears to run under a different user/session context). This is
   exactly why registration goes through the HTTP API (`POST /api/create`),
   never the CLI. Don't reintroduce a CLI dependency.
3. If Ollama comes back up and you just need to (re-)register an already-built
   GGUF without retraining, see §5.2 — do **not** rerun the full
   `train_narrator.py` pipeline, which would reload the base model and
   resume/redo training unnecessarily.

### "A background training job says it completed but something's off"

Don't trust the completion notification alone (§3.3, §3.4). Verify:

```bash
grep -aE "Merging|Converting|Model successfully exported|Registering|Registered:|Traceback|Error|EXIT_CODE" <logfile>
```

If you see `Traceback` without a subsequent `Registered:`, something failed —
read the traceback. If the log just stops mid-progress-bar with no traceback
at all, the process was killed externally (sleep, OOM, session teardown) —
check `narrator/merged/v{N}/checkpoints/` for the latest checkpoint and just
rerun `train_narrator.py` with the same `--dataset` arg; it will resume.

### "Port 8011 already in use"

This repo allocates one port per app: backend 8000, screener 8001/8002,
hotpicks 8003, smartpicks 8004, hivepicks 8005, debateroom 8006, analyst 8007,
localai 8008, trades 8009, autotrader 8010, **signalforge 8011**. If 8011 is
taken, something else (often a previous `uvicorn` you forgot about) is
listening — find and stop it:

```bash
netstat -ano | grep ":8011"          # get the PID in the last column
powershell -Command "Stop-Process -Id <pid> -Force"
```

### "Predict is slow / times out"

`/api/predict`'s dominant cost is the `live_context` node's Claude Code CLI
subprocess call — 20-45s is normal. `agent_engine.oneshot_claude()`'s
`timeout` parameter (default 60s in `live_context.py`) kills a hung subprocess
rather than hanging forever; a timeout there degrades gracefully
(`live_context.available: false`), it doesn't fail the whole request.

## 5. How to extend

### 5.1 Validate any code change to the narrator pipeline cheaply first

Before trusting a multi-hour run, always smoke-test end to end with a tiny
dataset:

```bash
cd narrator
python build_dataset.py --n-per-class 3          # ~9 examples, a few minutes
python train_narrator.py --dataset dataset/v1.jsonl --epochs 1   # a few minutes total
```

This exercises every stage (dataset build → LoRA train → merge → GGUF →
Ollama registration) end to end fast enough to catch API-version breakage
(§3.2) or logic bugs before committing CPU-hours to it.

### 5.2 Re-register an already-built GGUF without retraining

If training/merge/GGUF-conversion succeeded but Ollama registration failed
(or you just want to re-register under a different tag), skip straight to it:

```python
import train_narrator as tn
from pathlib import Path

tn._register_with_ollama(Path("gguf/v2.gguf"), "signalforge-narrator:v2")
tn._record_version("v2", "signalforge-narrator:v2", Path("dataset/v2.jsonl"), tn.BASE_MODEL, 294)
```

### 5.3 Add a new engineered feature

Add it inside `features/build_features.py::engineer_features()` and append its
name to `FEATURE_COLUMNS`. That's the only place to touch — both training
(`build_training_table`) and prediction (`compute_latest_features`) pick it up
automatically. Retrain the classifier afterward; the new `feature_list.json`
in the new registry version will reflect the change, and old versions remain
loadable with their original (shorter) feature list.

### 5.4 Retrain / roll back the classifier

```bash
cd train
python train_classifier.py                 # new version, auto-promoted to current
```

```bash
curl -X POST http://localhost:8011/api/models/v2_20260101T000000Z/promote
```

### 5.5 Swap the narrator's base model

Change `BASE_MODEL` in `narrator/train_narrator.py` (must be a causal-LM
instruction-tuned model with a chat template, small enough for CPU LoRA — this
repo used `Qwen/Qwen2.5-0.5B-Instruct`; anything bigger will make the
already-slow (~110-170s/step) CPU training meaningfully slower). Rerun
`train_narrator.py` — it creates a new version, doesn't touch the old one.

## 6. Quick health check sequence

```bash
curl -s http://localhost:8011/health                      # {"status":"ok","current_model_version":"..."}
curl -s http://localhost:8011/api/models                   # every classifier version + real metrics
curl -s http://localhost:11434/api/tags                    # Ollama reachable? narrator tag present?
curl -s "http://localhost:8011/api/predict?ticker=AAPL"     # full pipeline, ~20-45s (dominated by Claude CLI)
```
