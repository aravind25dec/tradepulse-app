"""
signalforge/train/registry.py
Versioned model packages on local disk. Every training run creates a brand
new, immutable directory under models/registry/ — nothing is ever overwritten
in place, so a bad retrain can't silently clobber a model that was working.

Layout:
  models/registry/
    v1_20260701T120000Z/
      model.txt              LightGBM Booster (native text format)
      feature_list.json      ordered feature column names the model expects
      metrics.json           real walk-forward backtest results (never hardcoded)
      training_config.json   hyperparams, horizon/thresholds, data date range
    current.json             {"version": "v1_20260701T120000Z"} — the pointer served by the API
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb

log = logging.getLogger(__name__)

REGISTRY_DIR = Path(__file__).resolve().parent.parent / "models" / "registry"
CURRENT_POINTER = REGISTRY_DIR / "current.json"


def _next_version_id() -> str:
    existing = [p for p in REGISTRY_DIR.glob("v*_*") if p.is_dir()]
    n = len(existing) + 1
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"v{n}_{ts}"


def save_version(booster: lgb.Booster, feature_columns: list[str],
                  metrics: dict, config: dict, promote: bool = True) -> str:
    """Write a new immutable registry version. Returns the version id.
    Set promote=False to save a version without making it the one /api/predict serves.
    """
    version = _next_version_id()
    version_dir = REGISTRY_DIR / version
    version_dir.mkdir(parents=True, exist_ok=False)

    booster.save_model(str(version_dir / "model.txt"))
    (version_dir / "feature_list.json").write_text(json.dumps(feature_columns, indent=2))
    (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (version_dir / "training_config.json").write_text(json.dumps(config, indent=2, default=str))

    log.info("Saved model registry version %s", version)
    if promote:
        set_current(version)
    return version


def list_versions() -> list[dict]:
    """All versions, oldest first, each with its metrics for the transparency
    endpoint (/api/models) — no accuracy number is ever hardcoded elsewhere."""
    out = []
    for d in sorted(p for p in REGISTRY_DIR.glob("v*_*") if p.is_dir()):
        metrics = {}
        metrics_path = d / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
        out.append({"version": d.name, "metrics": metrics})
    return out


def get_current_version() -> str | None:
    if not CURRENT_POINTER.exists():
        return None
    return json.loads(CURRENT_POINTER.read_text()).get("version")


def set_current(version: str) -> None:
    """Point the API at a specific (already-saved) version — also doubles as
    a rollback: promote an older version without retraining."""
    if not (REGISTRY_DIR / version).is_dir():
        raise ValueError(f"No such registry version: {version}")
    CURRENT_POINTER.write_text(json.dumps({"version": version}, indent=2))
    log.info("current.json now points at %s", version)


def load_model(version: str | None = None) -> tuple[lgb.Booster, list[str], dict]:
    """Load a specific version, or the current one if version is None.
    Returns (booster, feature_columns, metrics)."""
    version = version or get_current_version()
    if version is None:
        raise RuntimeError("No trained model available yet — run train/train_classifier.py first.")

    version_dir = REGISTRY_DIR / version
    booster = lgb.Booster(model_file=str(version_dir / "model.txt"))
    feature_columns = json.loads((version_dir / "feature_list.json").read_text())
    metrics = json.loads((version_dir / "metrics.json").read_text())
    return booster, feature_columns, metrics
