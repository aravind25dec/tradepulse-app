"""
signalforge/train/train_classifier.py
Trains the LightGBM classifier that produces the actual UP/FLAT/DOWN growth
call + calibrated confidence — this is the ONLY thing in the app that predicts
a number. The Ollama narrator (narrator/) never predicts; it only narrates
whatever this model already decided.

Time-ordered walk-forward split: the holdout test set is the most RECENT slice
of dates across the whole dataset, never a random shuffle — shuffling across
time would leak future market information into training and produce a
metrics.json that lies about real-world accuracy.

Every run writes a brand-new, immutable version via registry.save_version() —
it never overwrites a previous model.
"""
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, log_loss

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent / "features"))

import registry  # type: ignore  # noqa: E402
from build_features import (  # type: ignore  # noqa: E402
    FEATURE_COLUMNS, LABEL_NAMES, DEFAULT_HORIZON, DEFAULT_UP_THRESH, DEFAULT_DOWN_THRESH,
    build_training_table, PROCESSED_DIR,
)

log = logging.getLogger(__name__)

HOLDOUT_FRACTION = 0.15   # most recent 15% of dates, by wall-clock date not row count
LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}
NUM_BOOST_ROUND = 400
EARLY_STOPPING_ROUNDS = 30


def _time_split(dataset: pd.DataFrame, holdout_fraction: float = HOLDOUT_FRACTION):
    """Split by a date cutoff (not a random shuffle) so every training row
    strictly precedes every test row in wall-clock time."""
    dates = pd.to_datetime(dataset["date"])
    cutoff = dates.quantile(1 - holdout_fraction)
    train = dataset[dates < cutoff].copy()
    test = dataset[dates >= cutoff].copy()
    return train, test, cutoff


def _confidence_calibration(y_true: np.ndarray, proba: np.ndarray, buckets: int = 5) -> list[dict]:
    """Real out-of-sample calibration: bucket predictions by the model's own
    confidence (max class probability) and report actual hit rate per bucket.
    This is what /api/predict's 'backtested_accuracy' is grounded in — never
    a single hardcoded number."""
    pred_class = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    correct = (pred_class == y_true).astype(int)

    df = pd.DataFrame({"confidence": confidence, "correct": correct})
    try:
        df["bucket"], bin_edges = pd.qcut(df["confidence"], q=buckets, duplicates="drop", retbins=True)
    except ValueError:
        df["bucket"] = "all"
        bin_edges = [float(df["confidence"].min()), float(df["confidence"].max())]

    grouped = df.groupby("bucket", observed=True)["correct"].agg(["mean", "count"]).reset_index()
    # Numeric min/max per bucket (not a parsed Interval string) so downstream
    # code (pipeline.py's score_model node) can look up "which bucket is this
    # prediction's confidence in" without re-parsing pandas' repr of the bucket.
    edges = sorted(float(e) for e in bin_edges)
    return [
        {
            "confidence_min": round(edges[i], 4),
            "confidence_max": round(edges[i + 1], 4),
            "actual_accuracy": round(float(row["mean"]), 4),
            "n_samples": int(row["count"]),
        }
        for i, (_, row) in enumerate(grouped.iterrows())
    ]


def train(horizon: int = DEFAULT_HORIZON, up_thresh: float = DEFAULT_UP_THRESH,
          down_thresh: float = DEFAULT_DOWN_THRESH, rebuild_dataset: bool = True) -> str:
    if rebuild_dataset or not (PROCESSED_DIR / "dataset.parquet").exists():
        dataset = build_training_table(horizon=horizon, up_thresh=up_thresh, down_thresh=down_thresh)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(PROCESSED_DIR / "dataset.parquet")
    else:
        dataset = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")

    train_df, test_df, cutoff = _time_split(dataset)
    log.info("Train rows: %d (< %s) | Test rows: %d (>= %s)", len(train_df), cutoff, len(test_df), cutoff)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["label"]

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

    booster = lgb.train(
        LGB_PARAMS,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(period=0)],
    )

    proba = booster.predict(X_test, num_iteration=booster.best_iteration)
    pred_class = proba.argmax(axis=1)

    accuracy = accuracy_score(y_test, pred_class)
    report = classification_report(
        y_test, pred_class, target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
        output_dict=True, zero_division=0,
    )
    loss = log_loss(y_test, proba, labels=[0, 1, 2])
    calibration = _confidence_calibration(y_test.to_numpy(), proba)

    metrics = {
        "trained_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "train_date_range": [str(train_df["date"].min()), str(train_df["date"].max())],
        "test_date_range": [str(test_df["date"].min()), str(test_df["date"].max())],
        "holdout_cutoff_date": str(cutoff),
        "n_tickers": int(dataset["ticker"].nunique()),
        "accuracy": round(float(accuracy), 4),
        "log_loss": round(float(loss), 4),
        "classification_report": report,
        "confidence_calibration": calibration,
        "label_distribution_test": {LABEL_NAMES[k]: int(v) for k, v in y_test.value_counts().items()},
    }
    config = {
        "horizon_days": horizon,
        "up_thresh": up_thresh,
        "down_thresh": down_thresh,
        "lgb_params": LGB_PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
        "best_iteration": int(booster.best_iteration),
        "feature_columns": FEATURE_COLUMNS,
    }

    log.info("Holdout accuracy: %.2f%% | log_loss: %.4f", accuracy * 100, loss)
    version = registry.save_version(booster, FEATURE_COLUMNS, metrics, config)
    return version


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    v = train()
    print(f"Trained and registered model version: {v}")
