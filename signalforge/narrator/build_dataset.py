"""
signalforge/narrator/build_dataset.py
Bootstraps a (prompt -> narration) fine-tuning dataset for the narrator model.

The narrator's job is purely stylistic: turn the SAME structured inputs
pipeline.py's narrate() node already has (ticker, signal, confidence,
backtested accuracy, features, live context text) into fluent analyst prose —
it never predicts anything itself. So the training target doesn't need real
historical sentiment for arbitrary past dates; it needs Claude to demonstrate,
once per sample, the house STYLE we want the small fine-tuned model to imitate
when handed the same structured inputs at inference time.

Each sample:
  1. Take a real historical (ticker, date, features) row from
     data/processed/dataset.parquet and score it with the CURRENT registry
     model, exactly like pipeline.py's score_model node would.
  2. Synthesize a plausible live-context sentence locally (a template, not a
     Claude call — this is just a stand-in for the "live_context" text slot;
     at real inference time that slot holds REAL Claude Code CLI output
     instead, but the narrator only needs to learn how to weave whatever text
     is in that slot into the final narrative, not judge its truth).
  3. Ask Claude (one _oneshot_claude call) to rewrite the plain mechanical
     narrative (pipeline._mechanical_narrative) into fluent, analyst-style
     prose — this is the fine-tuning TARGET. Reusing the mechanical draft as
     Claude's input keeps every number in the target text traceable back to
     the model's real output; only the phrasing improves.

Written incrementally to narrator/dataset/v{N}.jsonl so a crash partway
through (this takes one Claude CLI call per sample, tens of seconds each)
doesn't lose earlier progress.
"""
import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))
sys.path.insert(0, str(_THIS_DIR.parent / "train"))
sys.path.insert(0, str(_THIS_DIR.parent / "features"))

import agent_engine  # type: ignore  # noqa: E402
import registry  # type: ignore  # noqa: E402
from build_features import FEATURE_COLUMNS, LABEL_NAMES, PROCESSED_DIR  # type: ignore  # noqa: E402
from pipeline import _mechanical_narrative, _estimate_bucket_accuracy, _SIGNAL_NAMES  # type: ignore  # noqa: E402

log = logging.getLogger(__name__)

DATASET_DIR = _THIS_DIR / "dataset"

_SENTIMENT_TEMPLATES = {
    "bullish": [
        "{ticker} has been drawing steady buying interest, with recent coverage framing the setup as constructive.",
        "Chatter around {ticker} has turned more upbeat over the past few sessions, consistent with the price action.",
    ],
    "bearish": [
        "{ticker} has faced some selling pressure lately, with recent coverage flagging caution.",
        "Sentiment around {ticker} has cooled off in the past few sessions, in line with the softer price action.",
    ],
    "neutral": [
        "{ticker} hasn't seen much notable news flow lately — trading looks mostly technically driven right now.",
        "Coverage of {ticker} has been fairly quiet recently, with no clear catalyst pushing sentiment either way.",
    ],
}


def _synthetic_live_context(ticker: str, label: int) -> dict:
    """Plausible-but-fabricated live-context stand-in for a past date — see
    module docstring for why this doesn't need to be factually real."""
    sentiment = {2: "bullish", 0: "bearish", 1: "neutral"}[label]
    sentence = random.choice(_SENTIMENT_TEMPLATES[sentiment]).format(ticker=ticker)
    return {
        "ticker": ticker,
        "available": True,
        "current_price": None,
        "sentiment": sentiment,
        "sentiment_summary": sentence,
        "candle_read": "Recent candles show a pattern consistent with the technicals below.",
    }


def _sample_rows(dataset: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    frames = []
    for label in (0, 1, 2):
        subset = dataset[dataset["label"] == label]
        n = min(n_per_class, len(subset))
        frames.append(subset.sample(n=n, random_state=rng.randint(0, 10_000)))
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


_POLISH_PROMPT = """You are polishing a stock-analysis narrative for a personal trading-signal app called SignalForge. \
Rewrite the DRAFT below into 2-4 fluent, confident, analyst-style sentences. Keep every number, ticker, and fact \
exactly as given — do not invent new figures, do not change the signal or confidence, do not add hedging disclaimers \
beyond what's already implied. Just make it read like a sharp human analyst wrote it instead of a template.

DRAFT:
{draft}

Respond with ONLY the rewritten narrative text — no preamble, no quotes, no markdown."""


async def _build_one(row: pd.Series, booster, feature_columns: list[str], metrics: dict) -> dict | None:
    ticker = row["ticker"]
    features = {c: float(row[c]) for c in FEATURE_COLUMNS}

    x = pd.DataFrame([[features[c] for c in feature_columns]], columns=feature_columns)
    proba = booster.predict(x)[0]
    pred_idx = int(proba.argmax())
    confidence = float(proba[pred_idx])

    model_output = {
        "signal": _SIGNAL_NAMES[pred_idx],
        "confidence": round(confidence, 4),
        "probabilities": {"SELL": round(float(proba[0]), 4), "HOLD": round(float(proba[1]), 4), "BUY": round(float(proba[2]), 4)},
        "model_version": registry.get_current_version(),
        "backtested_accuracy": {
            "overall": metrics.get("accuracy"),
            "for_this_confidence_level": _estimate_bucket_accuracy(confidence, metrics.get("confidence_calibration", [])),
            "test_date_range": metrics.get("test_date_range"),
        },
    }
    live = _synthetic_live_context(ticker, int(row["label"]))
    draft = _mechanical_narrative(ticker, model_output, features, live)

    try:
        polished = await agent_engine.oneshot_claude(_POLISH_PROMPT.format(draft=draft), timeout=45.0)
        polished = polished.strip().strip('"')
        if not polished:
            return None
    except Exception as exc:
        log.warning("Polish call failed for %s: %s", ticker, exc)
        return None

    prompt_payload = {
        "ticker": ticker,
        "model_output": model_output,
        "features": features,
        "live_context": live,
    }
    return {"prompt": json.dumps(prompt_payload, indent=2), "completion": polished}


async def build_dataset(n_per_class: int = 50, seed: int = 42) -> Path:
    if not (PROCESSED_DIR / "dataset.parquet").exists():
        raise FileNotFoundError(
            f"{PROCESSED_DIR / 'dataset.parquet'} not found — run train/train_classifier.py "
            "(or features/build_features.py) at least once first."
        )
    dataset = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")
    booster, feature_columns, metrics = registry.load_model()

    sampled = _sample_rows(dataset, n_per_class, seed)
    log.info("Sampled %d rows (%d per class) — generating narration pairs via Claude CLI...", len(sampled), n_per_class)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(DATASET_DIR.glob("v*.jsonl"))
    version = f"v{len(existing) + 1}"
    out_path = DATASET_DIR / f"{version}.jsonl"

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in sampled.iterrows():
            example = await _build_one(row, booster, feature_columns, metrics)
            if example is not None:
                f.write(json.dumps(example) + "\n")
                f.flush()
                n_written += 1
            if (i + 1) % 10 == 0:
                log.info("Progress: %d/%d sampled, %d written", i + 1, len(sampled), n_written)

    log.info("Wrote %d examples to %s", n_written, out_path)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Bootstrap the narrator fine-tuning dataset via the Claude Code CLI.")
    parser.add_argument("--n-per-class", type=int, default=50, help="Samples per label class (DOWN/FLAT/UP)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = asyncio.run(build_dataset(n_per_class=args.n_per_class, seed=args.seed))
    print(f"Dataset written to: {path}")
