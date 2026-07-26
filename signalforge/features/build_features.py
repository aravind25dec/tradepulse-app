"""
signalforge/features/build_features.py
Shared feature engineering — the SAME function path is used at training time
(over years of historical bars) and at predict time (over a freshly fetched
OHLCV window), so the model never sees features computed a different way at
inference than it was trained on.

Indicator math itself (RSI, SMA50/200, MACD, Bollinger, OBV, volume-vs-average)
is reused from autotrader/engine.py's compute_indicators() rather than
reimplemented — see that module for the pandas_ta calls. This file only adds
the model-friendly derived ratios on top, plus (training-only) forward-return
labels.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_AUTOTRADER_DIR = _THIS_DIR.parent.parent / "autotrader"
sys.path.insert(0, str(_AUTOTRADER_DIR))
from engine import compute_indicators  # type: ignore  # noqa: E402

log = logging.getLogger(__name__)

RAW_DIR = _THIS_DIR.parent / "data" / "raw"
PROCESSED_DIR = _THIS_DIR.parent / "data" / "processed"

# The exact, ordered feature set the model trains and predicts on. Persisted
# verbatim into each registry version's feature_list.json so a predict-time
# mismatch (missing/extra/reordered column) fails loudly instead of silently
# feeding the model garbage.
FEATURE_COLUMNS = [
    "rsi",
    "price_to_sma50",
    "price_to_sma200",
    "sma50_to_sma200",
    "macd_hist",
    "macd_bullish",
    "bb_pct",
    "vol_ratio",
    "obv_chg_10",
    "ret_5d",
    "ret_20d",
]

DEFAULT_HORIZON = 5        # trading days ahead the label looks
DEFAULT_UP_THRESH = 0.02   # forward return >= +2% => UP
DEFAULT_DOWN_THRESH = -0.02  # forward return <= -2% => DOWN; between the two => FLAT

LABEL_NAMES = {0: "DOWN", 1: "FLAT", 2: "UP"}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach FEATURE_COLUMNS to an OHLCV DataFrame (lowercase open/high/low/close/volume,
    as returned by autotrader.engine.fetch_history). Mutates and returns df.
    Rows before indicator warmup (first ~200 for SMA_200) will have NaNs — callers
    drop those before training or take the last row for prediction (by then warm).
    """
    df = compute_indicators(df)

    close = df["close"]
    df["price_to_sma50"] = close / df["SMA_50"] - 1
    df["price_to_sma200"] = close / df["SMA_200"] - 1
    df["sma50_to_sma200"] = df["SMA_50"] / df["SMA_200"] - 1
    df["macd_hist"] = df["MACDh_12_26_9"]
    df["macd_bullish"] = (df["MACD_12_26_9"] > df["MACDs_12_26_9"]).astype(float)

    bb_range = df["BB_UPPER"] - df["BB_LOWER"]
    df["bb_pct"] = np.where(bb_range > 0, (close - df["BB_LOWER"]) / bb_range, 0.5)

    df["vol_ratio"] = df["volume"] / df["VOL_SMA_20"]
    df["obv_chg_10"] = df["OBV"].pct_change(10)
    df["rsi"] = df["RSI_14"]
    df["ret_5d"] = close.pct_change(5)
    df["ret_20d"] = close.pct_change(20)

    return df


def add_forward_labels(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON,
                        up_thresh: float = DEFAULT_UP_THRESH,
                        down_thresh: float = DEFAULT_DOWN_THRESH) -> pd.DataFrame:
    """Training-only: forward return over `horizon` trading days and a 3-way
    UP/FLAT/DOWN label with a deadzone between down_thresh and up_thresh.
    The last `horizon` rows of each ticker have no future data yet and must be
    dropped by the caller (fwd_return is NaN there).
    """
    df["fwd_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df["label"] = np.select(
        [df["fwd_return"] >= up_thresh, df["fwd_return"] <= down_thresh],
        [2, 0],
        default=1,
    )
    return df


def compute_latest_features(df_ohlcv: pd.DataFrame) -> dict:
    """Predict-time entry point: engineer features over a freshly fetched OHLCV
    window and return just the latest row's FEATURE_COLUMNS as a plain dict.
    """
    df = engineer_features(df_ohlcv.copy())
    last = df.iloc[-1]
    missing = [c for c in FEATURE_COLUMNS if pd.isna(last.get(c))]
    if missing:
        raise ValueError(
            f"Insufficient history to compute features {missing} — need a longer OHLCV window "
            f"(SMA_200 alone needs 200+ trading days)."
        )
    return {c: float(last[c]) for c in FEATURE_COLUMNS}


def build_training_table(raw_dir: Path = RAW_DIR, horizon: int = DEFAULT_HORIZON,
                          up_thresh: float = DEFAULT_UP_THRESH,
                          down_thresh: float = DEFAULT_DOWN_THRESH) -> pd.DataFrame:
    """Load every data/raw/{ticker}.parquet, engineer features + forward labels,
    drop warmup/lookahead NaN rows, and concatenate into one labeled table
    ready for train/train_classifier.py.
    """
    frames = []
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw parquet files in {raw_dir} — run data/fetch_history.py first.")

    for f in files:
        ticker = f.stem
        try:
            df = pd.read_parquet(f)
            df = engineer_features(df)
            df = add_forward_labels(df, horizon=horizon, up_thresh=up_thresh, down_thresh=down_thresh)
            df["ticker"] = ticker
            df = df.dropna(subset=FEATURE_COLUMNS + ["fwd_return", "label"])
            if not df.empty:
                frames.append(df[["ticker"] + FEATURE_COLUMNS + ["fwd_return", "label"]].reset_index().rename(
                    columns={df.index.name or "index": "date"}
                ))
        except Exception as exc:
            log.warning("Skipping %s — feature build failed: %s", ticker, exc)

    if not frames:
        raise ValueError("No ticker produced usable labeled rows — check raw data / horizon settings.")

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sort_values("date").reset_index(drop=True)
    log.info("Built training table: %d rows, %d tickers, label counts=%s",
              len(dataset), dataset["ticker"].nunique(), dataset["label"].value_counts().to_dict())
    return dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    table = build_training_table()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "dataset.parquet"
    table.to_parquet(out_path)
    print(f"Saved {len(table)} rows to {out_path}")
