# TradePulse — AI Day Trading Prediction App

Real-time dashboard combining live prices, AI sentiment analysis, technical indicators,
earnings calendars, and SEC 8-K corporate event detection.

> ⚠️ **NOT FINANCIAL ADVICE.** This is an educational project. Always paper-trade first.
> Never risk money you cannot afford to lose.

---

## Features

- **Live price streaming** — yfinance polling (free), upgradeable to Polygon WebSocket
- **AI signal engine** — Claude scores news sentiment and combines it with RSI, MACD, VWAP, Bollinger Bands
- **Trade signals** — LONG / SHORT / HOLD with confidence % and reason breakdown
- **News feed** — NewsAPI headlines per ticker
- **SEC 8-K watcher** — detects M&A, contract awards, definitive agreements from EDGAR
- **Earnings calendar** — upcoming reports with EPS estimates and surprise detection (Finnhub)
- **Stock screener** — standalone screener UI served on port 8001
- **Both US stocks and crypto** — configure your watchlist in `.env`
- **WebSocket** — all data streams live to the React dashboard

---

## Quick Start

### 1. Get API Keys (all have free tiers)

| Key | Where | Cost |
|-----|-------|------|
| `ANTHROPIC_API_KEY` | https://console.anthropic.com | Pay-per-use |
| `NEWS_API_KEY` | https://newsapi.org | Free tier (100 req/day) |
| `FINNHUB_API_KEY` | https://finnhub.io | Free tier |
| `POLYGON_API_KEY` | https://polygon.io | Free (WebSocket requires paid) |

`ANTHROPIC_API_KEY` unlocks AI sentiment. Without it, signals run on technicals only.
All others are optional — the app degrades gracefully.

---

### Option A — One-command startup (after first-time setup below)

**Windows:**
```bat
start.bat
```

**Mac / Linux:**
```bash
bash start.sh
```

This opens three terminal windows:
| Service | URL |
|---------|-----|
| TradePulse Backend | http://localhost:8000 · [API docs](http://localhost:8000/docs) |
| Screener API | http://localhost:8002 · [API docs](http://localhost:8002/docs) |
| React Dashboard | http://localhost:3000 |

---

### Option B — Manual setup (first time)

#### 2. Backend

```bash
cd backend
cp .env.example .env        # Windows: copy .env.example .env
# Fill in your API keys in .env

python -m venv venv
# Activate the venv:
source venv/bin/activate    # Mac / Linux
venv\Scripts\activate       # Windows cmd
.\venv\Scripts\Activate.ps1 # Windows PowerShell

pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Backend starts at http://localhost:8000 — API docs at http://localhost:8000/docs

#### 3. Screener API

The screener shares the same Python venv as the backend.

```bash
# In a new terminal (venv already activated from step 2, or re-activate it)
cd screener
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8002
```

Screener API at http://localhost:8001 — open `screener/index.html` in your browser for the UI.

#### 4. Frontend

```bash
cd frontend
npm install
npm start
```

Dashboard opens at http://localhost:3000

---

## Architecture

```
backend/
  main.py              FastAPI app + WebSocket hub
  .env                 API keys (copy from .env.example)
  requirements.txt     Python dependencies
  ingestion/
    prices.py          yfinance poller (60s interval) → store + broadcast
    news.py            NewsAPI + SEC EDGAR 8-K watcher (10min interval)
    earnings.py        Finnhub earnings calendar (1hr interval)
  signals/
    technical.py       RSI, MACD, VWAP, Bollinger Bands, EMA cross
    sentiment.py       Claude API → structured JSON sentiment
    engine.py          Combines technical + sentiment → LONG/SHORT/HOLD
  db/
    store.py           In-memory store (swap for TimescaleDB in production)

screener/
  main.py              FastAPI screener API (port 8001)
  screener_engine.py   Scanning logic
  index.html           Standalone screener UI
  requirements.txt     Python dependencies (shares backend venv)

signalforge/           Locally-trained stock predictor (port 8011) — see signalforge/README.md
  data/, features/, train/, models/registry/   Historical data → features → versioned LightGBM classifier
  agent_engine.py, live_context.py             Claude Code CLI agent for live price/sentiment context
  pipeline.py                                  LangGraph pipeline fusing the trained model + live context
  main.py, index.html                          FastAPI app + UI

frontend/
  src/
    App.jsx            Main dashboard layout (3-column grid)
    useTradeSocket.js  WebSocket hook with auto-reconnect
    components/
      SignalPanel.jsx       Watchlist with signals + confidence bars
      PriceChart.jsx        Recharts area chart + indicator strip
      NewsStream.jsx        Live news feed per ticker
      EarningsCalendar.jsx  Upcoming earnings table
      EventFeed.jsx         M&A / contract / SEC event feed

start.bat              Windows one-click launcher
start.sh               Mac/Linux one-click launcher
```

---

## Customise your watchlist

Edit `backend/.env`:

```env
STOCK_WATCHLIST=AAPL,TSLA,NVDA,MSFT,AMZN,GOOGL
CRYPTO_WATCHLIST=BTC-USD,ETH-USD,SOL-USD,DOGE-USD
```

---

## Upgrade to real-time Polygon WebSocket

1. Get a Polygon paid plan
2. In `backend/ingestion/prices.py`, uncomment the `polygon_stream` function
3. In `backend/main.py`, replace the `price_poll_loop` task with `polygon_stream`

---

## Production deployment

| Service | What to deploy |
|---------|---------------|
| Railway | Backend (add Postgres addon for persistent TimescaleDB) |
| Vercel  | Frontend (set `REACT_APP_WS_URL=wss://your-backend.railway.app/ws`) |

---

## Backtesting

Run from the backend directory (venv activated):

```python
import yfinance as yf
import sys; sys.path.insert(0, '.')
from signals.technical import compute_signals

df = yf.download('AAPL', period='60d', interval='5m', progress=False)
df = df.rename(columns={'Close':'close','Volume':'volume'})
result = compute_signals(df)
print(result)
```

---

## Disclaimer

Day trading involves substantial risk. This software is for educational purposes only.
Past signal accuracy does not guarantee future performance. Always consult a licensed
financial advisor before trading with real capital. Paper trade for a minimum of 30 days
before considering live deployment.
