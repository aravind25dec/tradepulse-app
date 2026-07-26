@echo off
echo Starting TradePulse...
echo.

:: ── TradePulse Backend (port 8000) ───────────────────────────────────────────
start "TradePulse Backend" cmd /k "cd /d %~dp0backend && call venv\Scripts\activate && python -m uvicorn main:app --reload --port 8000"

:: ── Screener Backend (port 8001) ─────────────────────────────────────────────
:: Uses the same venv — install screener deps first if needed:
::   cd screener && pip install -r requirements.txt
start "TradePulse Screener API" cmd /k "cd /d %~dp0screener && call %~dp0backend\venv\Scripts\activate && python -m uvicorn main:app --reload --port 8002"

:: ── Hot Picks (port 8003) ─────────────────────────────────────────────────────
start "TradePulse Hot Picks" cmd /k "cd /d %~dp0hotpicks && call %~dp0backend\venv\Scripts\activate && python -m uvicorn main:app --reload --port 8003"

:: ── Smart Picks (port 8004) ────────────────────────────────────────────────────
start "TradePulse Smart Picks" cmd /k "cd /d %~dp0smartpicks && call %~dp0backend\venv\Scripts\activate && python -m uvicorn main:app --reload --port 8004"

:: ── Hive Picks (port 8005) ────────────────────────────────────────────────────
start "TradePulse Hive Picks" cmd /k "cd /d %~dp0hivepicks && call %~dp0backend\venv\Scripts\activate && python -m uvicorn main:app --reload --port 8005"

:: ── DebateRoom (port 8006) ──────────────────────────────────────────────────
start "TradePulse DebateRoom" cmd /k "cd /d %~dp0debateroom && call %~dp0backend\venv\Scripts\activate && python -m uvicorn main:app --reload --port 8006"

:: ── Single-Call Analyst (port 8007) ──────────────────────────────────────────
start "TradePulse Analyst" cmd /k "cd /d %~dp0analyst && call %~dp0backend\venv\Scripts\activate && python -m uvicorn main:app --reload --port 8007"

:: ── Local AI / Ollama (port 8008) ────────────────────────────────────────────
start "TradePulse LocalAI" cmd /k "cd /d %~dp0localai && call %~dp0backend\venv\Scripts\activate && python -m uvicorn main:app --reload --port 8008"

:: ── Trader / Robinhood (port 8009) ────────────────────────────────────────────
start "TradePulse Trader" cmd /k "cd /d %~dp0trades && call %~dp0backend\venv\Scripts\activate && python -m uvicorn main:app --reload --port 8009"

:: ── AutoTrader / Alpaca (port 8010) ──────────────────────────────────────────
:: Uses its own deps (yfinance, pandas_ta, alpaca-trade-api) — install first if needed:
::   cd autotrader && pip install -r requirements.txt
start "TradePulse AutoTrader" cmd /k "cd /d %~dp0autotrader && call %~dp0backend\venv\Scripts\activate && python -m uvicorn main:app --reload --port 8010"

:: ── SignalForge / locally-trained predictor (port 8011) ──────────────────────
:: Uses its OWN venv (heavier/different deps: lightgbm, langgraph, pyarrow), not
:: the shared backend one. First-time setup:
::   cd signalforge && python -m venv venv && venv\Scripts\pip install -r requirements.txt
start "TradePulse SignalForge" cmd /k "cd /d %~dp0signalforge && call venv\Scripts\activate && python -m uvicorn main:app --reload --port 8011"

:: Give backends a moment to boot
timeout /t 2 /nobreak >nul

:: ── Frontend (port 3000) ──────────────────────────────────────────────────────
start "TradePulse Frontend" cmd /k "cd /d %~dp0frontend && npm start"

echo  TradePulse Backend:  http://localhost:8000/docs
echo  Screener API:        http://localhost:8002/docs
echo  Hot Picks API:       http://localhost:8003/docs
echo  Smart Picks API:     http://localhost:8004/docs
echo  Hive Picks API:      http://localhost:8005/docs
echo  DebateRoom API:      http://localhost:8006/docs
echo  Dashboard:           http://localhost:3000
echo.
echo  Screener UI:         open screener\index.html in your browser
echo  Hot Picks UI:        http://localhost:8003
echo  Smart Picks UI:      http://localhost:8004
echo  Hive Picks UI:       http://localhost:8005
echo  DebateRoom UI:       http://localhost:8006
echo  Analyst UI:          http://localhost:8007
echo  Local AI UI:         http://localhost:8008
echo  AutoTrader UI:       http://localhost:8010
echo  SignalForge UI:      http://localhost:8011
echo.
echo Close the terminal windows to stop each service.
