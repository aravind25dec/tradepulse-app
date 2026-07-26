#!/usr/bin/env bash
# TradePulse — start all services
# Run from the project root: bash start.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting TradePulse from $ROOT"

# ── Backend (port 8000) ───────────────────────────────────────────────────────
source "$ROOT/backend/venv/Scripts/activate" 2>/dev/null \
  || source "$ROOT/backend/venv/bin/activate"   # fallback for Linux/Mac

cd "$ROOT/backend"
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
echo "  ✓ Backend started  (PID $BACKEND_PID) → http://localhost:8000"

# ── Screener API (port 8001) ──────────────────────────────────────────────────
cd "$ROOT/screener"
uvicorn main:app --reload --port 8001 &
SCREENER_PID=$!
echo "  ✓ Screener started  (PID $SCREENER_PID) → http://localhost:8001"

# ── Hot Picks API (port 8003) ─────────────────────────────────────────────────
cd "$ROOT/hotpicks"
uvicorn main:app --reload --port 8003 &
HOTPICKS_PID=$!
echo "  ✓ Hot Picks started (PID $HOTPICKS_PID) → http://localhost:8003"

# ── DebateRoom (port 8006) ────────────────────────────────────────────────────
cd "$ROOT/debateroom"
uvicorn main:app --reload --port 8006 &
DEBATE_PID=$!
echo "  ✓ DebateRoom started (PID $DEBATE_PID) → http://localhost:8006"

# ── AutoTrader / Alpaca (port 8010) — uses its own deps, install first if needed:
##   cd autotrader && pip install -r requirements.txt
cd "$ROOT/autotrader"
uvicorn main:app --reload --port 8010 &
AUTOTRADER_PID=$!
echo "  ✓ AutoTrader started (PID $AUTOTRADER_PID) → http://localhost:8010"

# ── SignalForge / locally-trained predictor (port 8011) — uses its OWN venv
##   (heavier/different deps: lightgbm, langgraph, pyarrow), not the shared backend one.
##   First-time setup: cd signalforge && python -m venv venv && pip install -r requirements.txt
(
  cd "$ROOT/signalforge"
  source venv/Scripts/activate 2>/dev/null || source venv/bin/activate
  uvicorn main:app --reload --port 8011
) &
SIGNALFORGE_PID=$!
echo "  ✓ SignalForge started (PID $SIGNALFORGE_PID) → http://localhost:8011"

# ── Frontend (port 3000) ──────────────────────────────────────────────────────
cd "$ROOT/frontend"
npm start &
FRONTEND_PID=$!
echo "  ✓ Frontend started  (PID $FRONTEND_PID) → http://localhost:3000"

echo ""
echo "  Screener UI  → open screener/index.html in your browser"
echo "  Hot Picks UI → http://localhost:8003"
echo "  DebateRoom   → http://localhost:8006"
echo "  AutoTrader   → http://localhost:8010"
echo "  SignalForge  → http://localhost:8011"
echo ""
echo "  Press Ctrl+C to stop all services"

# Stop all on Ctrl+C
trap "kill $BACKEND_PID $SCREENER_PID $HOTPICKS_PID $DEBATE_PID $AUTOTRADER_PID $SIGNALFORGE_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
