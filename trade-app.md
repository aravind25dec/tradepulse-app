You are an expert Quantitative Trading Developer and Senior Python Engineer. Your task is to build a modular, high-performance automated stock screening script that acts as an autonomous trading agent. 

The application must fetch historical market data using free data libraries, calculate a specific matrix of technical indicators locally, evaluate the data based on precise confirmation rules, and generate a structured JSON output of tickers to "BUY", "SELL", or "HOLD".

### 1. Technical Architecture & Stack
- Language: Python 3.10+
- Core Libraries: Use `yfinance` to extract raw OHLCV market data (since it is completely free and requires no API keys). 
- Calculations: Use `pandas` and `pandas_ta` (Pandas Technical Analysis library) to calculate all indicators locally. Do not use external API endpoints to fetch pre-calculated indicators.

### 2. Required Indicators & Parameters
The application must compute the following metrics using daily historical candles (fetching at least 300 days of history per ticker to ensure the 200-day SMA is perfectly accurate):
1. Momentum: Relative Strength Index (RSI) - 14-period standard.
2. Trend: 
   - 50-day and 200-day Simple Moving Averages (SMA).
   - MACD (12 fast, 26 slow, 9 signal).
3. Volume:
   - On-Balance Volume (OBV) trend direction over the last 10 trading days.
   - Intra-day Volume comparison (Current volume vs. 20-day average volume).
4. Volatility: Bollinger Bands (20-period SMA, 2 standard deviations).

*Note on VWAP:* Since VWAP is inherently an intraday metric requiring tick or minute-by-minute data, substitute it at the daily level with the Daily Volume-Weighted Average Close, or ensure you fall back gracefully to checking if the price is above/below the 20-day volume-weighted baseline.

### 3. Execution & Confirmation Logic
Implement a strict multi-category logic matrix to filter the tickers. A ticker should only be flagged if it meets the collective criteria:

#### CRITERIA FOR A "BUY" SIGNAL:
- Momentum: RSI is bouncing upward out of oversold territory (crossed above 30) OR holds steady above the 50 centerline.
- Trend: Current price is structurally trading ABOVE its rising 200-day SMA, AND the MACD line has crossed above the MACD Signal line within the last 3 candles.
- Volume: Current day volume is at least 1.2x higher than the 20-day moving average volume, and the 10-day OBV trend is positive (accumulation).
- Volatility: Price is bouncing off or holding the lower Bollinger Band, or the bands are expanding after a tight squeeze.

#### CRITERIA FOR A "SELL" SIGNAL:
- Momentum: RSI is dropping out of overbought territory (crossed below 70) OR falls below the 50 centerline.
- Trend: Current price breaks BELOW the 50-day SMA, OR the MACD line crosses below the MACD Signal line.
- Volume: Price drops on above-average volume.
- Volatility: Price hits or exceeds the upper Bollinger Band and shows signs of rejection.

If a ticker does not definitively meet the thresholds for a clear Buy or Sell, classify it as "HOLD".

### 4. Code Base Structure
Provide the complete, clean code in a single executable script or modular design:
1. Data Engine: Uses `yf.download()` to pull the required historical time frame for an array of tickers. Handles missing data or delisted tickers safely using try-except blocks.
2. Analytics Engine: Applies `df.ta.rsi()`, `df.ta.macd()`, `df.ta.bbands()`, and `df.ta.obv()` to the DataFrame.
3. Decision Matrix: Evaluates the last row (most current day) of the DataFrame against the logic thresholds.
4. Main Orchestrator: Accepts a list of watchlisted tickers (e.g., AAPL, MSFT, NVDA, AMD, TSLA), executes the screen, and prints the final report.

### 5. Expected Output Format
The script must print a clean, valid JSON array to the console structured exactly like this:

[
  {
    "ticker": "AAPL",
    "signal": "BUY",
    "confidence_score": "HIGH",
    "metrics": {
      "rsi": 34.2,
      "above_200sma": true,
      "macd_crossover": true,
      "high_volume": true
    },
    "reasoning": "RSI bouncing from oversold while holding long-term structural support at the 200 SMA, confirmed by a fresh bullish MACD cross and positive volume spikes."
  }
]

Please write the complete, robust Python implementation now. Include clean comments explaining how the pandas_ta integration handles the calculations.