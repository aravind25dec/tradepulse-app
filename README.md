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

### 2. Backend setup

```bash
cd backend
cp .env.example .env
# Fill in your API keys in .env

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```

The backend starts at http://localhost:8000
API docs: http://localhost:8000/docs

### 3. Frontend setup

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

frontend/
  src/
    App.jsx            Main dashboard layout (3-column grid)
    useTradeSocket.js  WebSocket hook with auto-reconnect
    components/
      SignalPanel.jsx  Watchlist with signals + confidence bars
      PriceChart.jsx   Recharts area chart + indicator strip
      NewsStream.jsx   Live news feed per ticker
      EarningsCalendar.jsx  Upcoming earnings table
      EventFeed.jsx    M&A / contract / SEC event feed
```

---

## Customise your watchlist

Edit `.env`:

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

Run from the backend directory:

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
