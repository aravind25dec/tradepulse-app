"""
signalforge/pipeline.py
LangGraph StateGraph for GET /api/predict.

                 +-> build_features -> score_model -+
START -> fetch_ohlcv                                 +-> narrate -> assemble_response -> END
                 +-------------> live_context -------+

score_model and live_context have no data dependency on each other (one only
needs computed features, the other only needs the ticker string), so they run
as parallel branches and narrate joins on both — this is the actual reason to
use LangGraph here instead of a plain top-to-bottom function pipeline: the
Claude Code CLI subprocess (live_context) is comfortably the slowest step, and
serializing the fast local model-scoring step behind it would be pure waste.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Any, TypedDict

import httpx
import pandas as pd
from langgraph.graph import StateGraph, START, END

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "train"))
sys.path.insert(0, str(_THIS_DIR / "features"))
sys.path.insert(0, str(_THIS_DIR / "narrator"))
sys.path.insert(0, str(_THIS_DIR.parent / "autotrader"))

import registry  # type: ignore  # noqa: E402
from build_features import compute_latest_features, FEATURE_COLUMNS  # type: ignore  # noqa: E402
from engine import fetch_history as _fetch_history_yf  # type: ignore  # noqa: E402

import live_context as _live_context_mod  # noqa: E402
# Lightweight at module level (torch/transformers/peft are only imported
# inside train_narrator._finetune(), not at import time) — safe to pull in
# just the constants/registry helpers here without dragging in the heavy deps.
from train_narrator import (  # type: ignore  # noqa: E402
    NARRATOR_REGISTRY, OLLAMA_BASE as NARRATOR_OLLAMA_BASE, SYSTEM_PROMPT as NARRATOR_SYSTEM_PROMPT,
)

log = logging.getLogger(__name__)

_SIGNAL_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}
_OHLCV_PERIOD = "2y"   # comfortably covers the 200-day SMA warmup window
_NARRATOR_TIMEOUT = 60.0


class PredictState(TypedDict, total=False):
    ticker: str
    ohlcv: Any             # pd.DataFrame — not included in the final API response
    features: dict
    model_output: dict
    live_context: dict
    narrative: str
    response: dict


# ── Nodes ──────────────────────────────────────────────────────────────────

def fetch_ohlcv(state: PredictState) -> dict:
    ticker = state["ticker"]
    df = _fetch_history_yf(ticker, period=_OHLCV_PERIOD)
    if df is None or df.empty:
        # autotrader.engine.fetch_history() returns None both for an invalid/delisted
        # symbol AND for a real, valid ticker that just doesn't have 200+ trading days
        # yet (recent IPO) — it needs that much history for the 200-day SMA feature.
        # It doesn't expose which case applies, so the message covers both rather
        # than misleadingly suggesting the symbol itself is wrong.
        raise ValueError(
            f"Could not fetch enough OHLCV history for '{ticker}'. Either the symbol is "
            f"wrong/delisted, or it's a real but recently-listed ticker that doesn't have "
            f"the 200+ trading days this model's features require yet — that won't change "
            f"no matter how many times you refresh."
        )
    return {"ohlcv": df}


def build_features_node(state: PredictState) -> dict:
    features = compute_latest_features(state["ohlcv"])
    return {"features": features}


def _estimate_bucket_accuracy(confidence: float, calibration: list[dict]) -> float | None:
    """Look up the backtested actual-accuracy bucket this confidence falls into
    (see train/train_classifier.py's _confidence_calibration) — this, not the
    model's own probability, is the number that should be trusted as 'how often
    predictions like this one turned out right historically.'"""
    for bucket in calibration:
        if bucket["confidence_min"] <= confidence <= bucket["confidence_max"]:
            return bucket["actual_accuracy"]
    return None


def score_model_node(state: PredictState) -> dict:
    booster, feature_columns, metrics = registry.load_model()
    row = pd.DataFrame([[state["features"][c] for c in feature_columns]], columns=feature_columns)
    proba = booster.predict(row)[0]
    pred_idx = int(proba.argmax())
    confidence = float(proba[pred_idx])

    return {
        "model_output": {
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
    }


async def live_context_node(state: PredictState) -> dict:
    snapshot = await _live_context_mod.get_live_snapshot(state["ticker"])
    return {"live_context": snapshot}


def _mechanical_narrative(ticker: str, model_output: dict, features: dict, live: dict) -> str:
    """Phase-2 fallback narrative — plain templated text built directly from the
    numbers, used whenever the Phase-3 fine-tuned Ollama narrator isn't
    registered yet. Never invents a number that isn't already in model_output/features."""
    signal = model_output["signal"]
    conf_pct = model_output["confidence"] * 100
    bucket_acc = model_output["backtested_accuracy"]["for_this_confidence_level"]
    bucket_txt = f"{bucket_acc * 100:.0f}%" if bucket_acc is not None else "not yet enough backtest samples"

    parts = [
        f"{ticker}: model signal is {signal} with {conf_pct:.0f}% confidence "
        f"(historically, predictions at this confidence level were right {bucket_txt} of the time in backtesting).",
        f"RSI {features['rsi']:.1f}, price is {features['price_to_sma50'] * 100:+.1f}% vs 50-day average and "
        f"{features['price_to_sma200'] * 100:+.1f}% vs 200-day average, "
        f"MACD {'bullish' if features['macd_bullish'] else 'bearish'}, "
        f"volume {features['vol_ratio']:.2f}x the 20-day average.",
    ]
    if live.get("available"):
        parts.append(f"Live context: {live.get('sentiment', 'neutral')} sentiment — {live.get('sentiment_summary', '')}")
    else:
        parts.append("Live qualitative context was unavailable this run (Claude Code CLI not reachable).")
    return " ".join(parts)


def _latest_narrator_tag() -> str | None:
    if not NARRATOR_REGISTRY.exists():
        return None
    try:
        versions = json.loads(NARRATOR_REGISTRY.read_text())
        return versions[-1]["ollama_tag"] if versions else None
    except Exception:
        return None


async def _call_narrator(tag: str, ticker: str, model_output: dict, features: dict, live: dict) -> str | None:
    """Best-effort call to the fine-tuned Ollama narrator. Returns None on any
    failure so the caller falls back to the mechanical narrative — a narrator
    hiccup must never take down /api/predict's actual numeric prediction."""
    payload = {"ticker": ticker, "model_output": model_output, "features": features, "live_context": live}
    user_msg = f"Structured inputs:\n{json.dumps(payload, indent=2)}\n\nWrite the narrative."
    try:
        async with httpx.AsyncClient(timeout=_NARRATOR_TIMEOUT) as client:
            resp = await client.post(
                f"{NARRATOR_OLLAMA_BASE}/api/chat",
                json={
                    "model": tag,
                    "messages": [
                        {"role": "system", "content": NARRATOR_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.4, "num_predict": 200},
                },
            )
        if resp.status_code != 200:
            log.warning("Narrator call to %s failed: HTTP %d", tag, resp.status_code)
            return None
        text = resp.json().get("message", {}).get("content", "").strip()
        return text or None
    except Exception as exc:
        log.warning("Narrator call to %s failed: %s", tag, exc)
        return None


async def narrate_node(state: PredictState) -> dict:
    mechanical = _mechanical_narrative(state["ticker"], state["model_output"], state["features"], state["live_context"])
    tag = _latest_narrator_tag()
    if tag is None:
        return {"narrative": mechanical}

    narrated = await _call_narrator(tag, state["ticker"], state["model_output"], state["features"], state["live_context"])
    return {"narrative": narrated or mechanical}


def _layman_explanation(ticker: str, model_output: dict, features: dict) -> str:
    """Plain-English companion to the analyst narrative — deliberately rule-based
    rather than a second LLM round-trip, so it's instant, free, and available even
    when Ollama/the narrator is down (same resilience principle as the mechanical
    narrative fallback)."""
    signal = model_output["signal"]
    conf_pct = model_output["confidence"] * 100
    bucket_acc = model_output["backtested_accuracy"].get("for_this_confidence_level")

    if signal == "BUY":
        lean = f"leans toward thinking {ticker} is more likely to go UP"
    elif signal == "SELL":
        lean = f"leans toward thinking {ticker} is more likely to go DOWN"
    else:
        lean = f"doesn't see a strong signal either way for {ticker} right now — it's a coin-flip zone"

    above_200 = features["price_to_sma200"] > 0
    trend_word = "above" if above_200 else "below"
    trend_read = "a positive sign for the medium-term trend" if above_200 else "a warning sign for the medium-term trend"

    parts = [
        f"In plain terms: the model {lean}, at about {conf_pct:.0f}% confidence.",
        "Think of that confidence like a weather forecast — it's not a guarantee, just how strongly the model leans.",
    ]
    if bucket_acc is not None:
        parts.append(
            f"In the past, when the model was this confident, it turned out to be right about "
            f"{bucket_acc * 100:.0f}% of the time — worth keeping in mind before acting on this."
        )
    else:
        parts.append("There isn't enough backtesting history yet at this exact confidence level to say how reliable calls like this one have been.")
    parts.append(
        f"One thing behind the call: the stock is currently trading {trend_word} its long-term "
        f"(200-day) average price, which many traders read as {trend_read}."
    )
    return " ".join(parts)


def assemble_response_node(state: PredictState) -> dict:
    response = {
        "ticker": state["ticker"],
        **state["model_output"],
        "narrative": state["narrative"],
        "layman_explanation": _layman_explanation(state["ticker"], state["model_output"], state["features"]),
        "live_context": state["live_context"],
        "features": state["features"],
    }
    return {"response": response}


# ── Graph ──────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(PredictState)
    graph.add_node("fetch_ohlcv", fetch_ohlcv)
    graph.add_node("build_features", build_features_node)
    graph.add_node("score_model", score_model_node)
    graph.add_node("live_context", live_context_node)
    graph.add_node("narrate", narrate_node)
    graph.add_node("assemble_response", assemble_response_node)

    graph.add_edge(START, "fetch_ohlcv")
    graph.add_edge("fetch_ohlcv", "build_features")
    graph.add_edge("fetch_ohlcv", "live_context")
    graph.add_edge("build_features", "score_model")
    # A single add_edge call with a list of sources is LangGraph's "wait for
    # ALL of these" join — two separate add_edge calls each independently
    # trigger the target as soon as THEIR OWN source completes, which fired
    # narrate prematurely (before score_model was done) since live_context
    # completes one superstep earlier (it's a single hop from fetch_ohlcv,
    # while score_model is two hops via build_features).
    graph.add_edge(["score_model", "live_context"], "narrate")
    graph.add_edge("narrate", "assemble_response")
    graph.add_edge("assemble_response", END)
    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def predict(ticker: str) -> dict:
    result = await get_graph().ainvoke({"ticker": ticker.upper().strip()})
    return result["response"]
