import json, math

filepath = r'C:/Users/aravi/.claude/projects/C--Users-aravi-Downloads-tradepulse-app-hotpicks/39a4f289-7514-4eb6-9f04-cd86169d0d8a/tool-results/mcp-robinhood-trading-get_equity_historicals-1781740313548.txt'
with open(filepath, 'r', encoding='utf-8') as f:
    raw = json.load(f)

results = raw['data']['results']

# ---- Helper functions ----

def ema_series(values, period):
    k = 2.0 / (period + 1)
    out = [None] * len(values)
    if len(values) < period:
        return out
    sma = sum(values[:period]) / period
    out[period - 1] = sma
    for i in range(period, len(values)):
        out[i] = values[i] * k + out[i-1] * (1 - k)
    return out

def rsi14(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsi_vals = []
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi_vals.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_vals.append(100 - 100 / (1 + rs))
    return rsi_vals[-1] if rsi_vals else None

def bollinger(closes, period=10, num_std=2):
    if len(closes) < period:
        return None, None, None
    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    std = math.sqrt(variance)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower

def characterize(bars):
    closes = [float(b['close_price']) for b in bars]
    n = len(closes)
    open_p = float(bars[0]['open_price'])
    close_p = closes[-1]
    high_p = max(float(b['high_price']) for b in bars)

    q1_end = max(n // 4, 1)
    q3_start = min(3 * n // 4, n - 1)

    morning_high = max(closes[:q1_end])
    mid_vals = closes[q1_end:q3_start]
    mid_avg = sum(mid_vals) / len(mid_vals) if mid_vals else closes[q1_end]
    late_vals = closes[q3_start:]
    late_avg = sum(late_vals) / len(late_vals) if late_vals else close_p
    late_high = max(late_vals) if late_vals else close_p

    pct_chg = (close_p - open_p) / open_p * 100
    close_vs_high_pct = (high_p - close_p) / high_p * 100
    late_vs_mid_pct = (late_avg - mid_avg) / abs(mid_avg) * 100 if mid_avg != 0 else 0
    morning_vs_close_pct = (morning_high - close_p) / abs(close_p) * 100 if close_p != 0 else 0

    if pct_chg > 1.5 and close_vs_high_pct < 3.0 and late_vs_mid_pct >= -1.0:
        return "all-day runner"
    elif morning_high > 1.03 * close_p and morning_vs_close_pct > 2.0 and pct_chg < 0.5:
        return "morning spike + fade"
    elif late_vs_mid_pct > 1.5 and late_high > 1.015 * mid_avg:
        return "late-day surge"
    elif pct_chg > 0.5:
        return "mild uptrend / mixed"
    elif pct_chg < -0.5:
        return "mild downtrend / mixed"
    else:
        return "flat / choppy"


# ---- Main analysis loop ----

for ticker_data in results:
    sym = ticker_data['symbol']
    bars = ticker_data['bars']

    reg_bars = [b for b in bars if b.get('session', 'reg') in ('reg', 'regular', '')]
    if not reg_bars:
        reg_bars = bars

    opens  = [float(b['open_price'])  for b in reg_bars]
    closes = [float(b['close_price']) for b in reg_bars]
    highs  = [float(b['high_price'])  for b in reg_bars]
    lows   = [float(b['low_price'])   for b in reg_bars]
    vols   = [float(b['volume'])      for b in reg_bars]
    times  = [b['begins_at'] for b in reg_bars]

    open_price  = opens[0]
    close_price = closes[-1]
    hod = max(highs)
    lod = min(lows)
    pct_chg = (close_price - open_price) / open_price * 100

    typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    total_tpv = sum(tp * v for tp, v in zip(typical_prices, vols))
    total_vol  = sum(vols)
    vwap = total_tpv / total_vol if total_vol > 0 else 0
    above_vwap = close_price > vwap

    avg_vol = total_vol / len(vols) if vols else 1
    last_vol = vols[-1]
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 0

    rsi_val = rsi14(closes, 14)

    ema9  = ema_series(closes, 9)
    ema21 = ema_series(closes, 21)
    ema9_final  = next((v for v in reversed(ema9)  if v is not None), None)
    ema21_final = next((v for v in reversed(ema21) if v is not None), None)
    ema9_gt_21  = (ema9_final > ema21_final) if (ema9_final and ema21_final) else None

    bb_mid, bb_upper, bb_lower = bollinger(closes, 10, 2)
    if bb_upper is not None and bb_lower is not None and bb_upper != bb_lower:
        bb_pct = (close_price - bb_lower) / (bb_upper - bb_lower)
    else:
        bb_pct = None

    pattern = characterize(reg_bars)

    print("=" * 60)
    print(f"  TICKER: {sym}")
    print("=" * 60)
    print(f"  Bars analyzed        : {len(reg_bars)}  (5-min, regular session)")
    print(f"  First bar (UTC)      : {times[0]}")
    print(f"  Last  bar (UTC)      : {times[-1]}")
    print(f"  Open                 : ${open_price:.4f}")
    print(f"  Close                : ${close_price:.4f}")
    print(f"  High of Day          : ${hod:.4f}")
    print(f"  Low  of Day          : ${lod:.4f}")
    print(f"  % Change (O->C)      : {pct_chg:+.2f}%")
    print(f"  VWAP                 : ${vwap:.4f}")
    abv = 'ABOVE' if above_vwap else 'BELOW'
    print(f"  Price vs VWAP        : {abv}  (diff {close_price - vwap:+.4f})")
    print(f"  Total Volume         : {int(total_vol):,}")
    print(f"  Avg Bar Volume       : {int(avg_vol):,}")
    print(f"  Last Bar Volume      : {int(last_vol):,}")
    print(f"  Vol Ratio (last/avg) : {vol_ratio:.2f}x")
    if rsi_val is not None:
        print(f"  RSI(14)              : {rsi_val:.2f}")
    else:
        print(f"  RSI(14)              : N/A (insufficient bars)")
    if ema9_final:
        print(f"  EMA9                 : ${ema9_final:.4f}")
    if ema21_final:
        print(f"  EMA21                : ${ema21_final:.4f}")
    print(f"  EMA9 > EMA21         : {ema9_gt_21}")
    if bb_pct is not None:
        zone = "upper half" if bb_pct > 0.5 else "lower half"
        extra = ""
        if bb_pct > 1.0: extra = " >> above upper band"
        elif bb_pct < 0: extra = " << below lower band"
        print(f"  BB(10,2) Lower       : ${bb_lower:.4f}")
        print(f"  BB(10,2) Mid         : ${bb_mid:.4f}")
        print(f"  BB(10,2) Upper       : ${bb_upper:.4f}")
        print(f"  BB %B                : {bb_pct:.3f}  ({zone}{extra})")
    print(f"  Pattern              : {pattern}")
    print()
