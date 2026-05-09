import { useEffect, useState } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine, CartesianGrid,
} from 'recharts';
import axios from 'axios';
import { format } from 'date-fns';

const API = 'http://localhost:8000';

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{
      background: 'var(--bg3)', border: '1px solid var(--border2)',
      borderRadius: 6, padding: '8px 12px', fontSize: 12, fontFamily: 'var(--font-mono)',
    }}>
      <div style={{ color: 'var(--text2)', marginBottom: 2 }}>{d.ts ? format(new Date(d.ts), 'HH:mm:ss') : ''}</div>
      <div style={{ color: 'var(--green)', fontWeight: 500 }}>
        ${Number(d.close).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })}
      </div>
      {d.volume > 0 && <div style={{ color: 'var(--text3)' }}>vol {Number(d.volume).toLocaleString()}</div>}
    </div>
  );
};

export function PriceChart({ ticker, signal, ticks }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!ticker) return;
    setLoading(true);
    axios.get(`${API}/api/prices/${ticker}?limit=200`)
      .then(r => setHistory(r.data || []))
      .catch(() => setHistory([]))
      .finally(() => setLoading(false));
  }, [ticker]);

  // Append latest tick to chart data
  useEffect(() => {
    const tick = ticks?.[ticker];
    if (tick?.price) {
      setHistory(prev => {
        const last = prev[prev.length - 1];
        if (last && Math.abs(last.close - tick.price) < 0.0001) return prev;
        return [...prev.slice(-199), { close: tick.price, volume: tick.volume, ts: new Date().toISOString() }];
      });
    }
  }, [ticks, ticker]);

  const prices = history.map(d => d.close).filter(Boolean);
  const minP = prices.length ? Math.min(...prices) * 0.9995 : 0;
  const maxP = prices.length ? Math.max(...prices) * 1.0005 : 1;
  const vwap = signal?.tech?.vwap;
  const isUp = prices.length >= 2 && prices[prices.length - 1] >= prices[0];
  const lineColor = isUp ? 'var(--green)' : 'var(--red)';

  return (
    <div style={{ position: 'relative' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16, padding: '0 4px' }}>
        <div>
          <div style={{ fontSize: 22, fontWeight: 700, letterSpacing: '.01em' }}>{ticker}</div>
          {prices.length > 0 && (
            <div className="mono" style={{ fontSize: 28, fontWeight: 500, color: lineColor, marginTop: 2 }}>
              ${Number(prices[prices.length - 1]).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })}
            </div>
          )}
        </div>
        {signal && (
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: 11, color: 'var(--text3)', marginBottom: 4, letterSpacing: '.06em', textTransform: 'uppercase' }}>AI signal</div>
            <span className={`badge badge-${signal.signal?.toLowerCase()}`} style={{ fontSize: 14, padding: '4px 12px' }}>
              {signal.signal}
            </span>
            <div className="mono small" style={{ color: 'var(--text3)', marginTop: 4 }}>
              {signal.confidence}% confidence
            </div>
          </div>
        )}
      </div>

      {/* Chart */}
      {loading ? (
        <div style={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text3)' }}>
          loading price data…
        </div>
      ) : history.length < 2 ? (
        <div style={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text3)' }}>
          waiting for price data…
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={history} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id={`grad-${ticker}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={lineColor} stopOpacity={0.15} />
                <stop offset="95%" stopColor={lineColor} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="ts" hide />
            <YAxis domain={[minP, maxP]} hide />
            <Tooltip content={<CustomTooltip />} />
            {vwap && (
              <ReferenceLine
                y={vwap}
                stroke="var(--amber)"
                strokeDasharray="4 4"
                strokeWidth={1}
                label={{ value: 'VWAP', position: 'right', fill: 'var(--amber)', fontSize: 10 }}
              />
            )}
            <Area
              type="monotone"
              dataKey="close"
              stroke={lineColor}
              strokeWidth={1.5}
              fill={`url(#grad-${ticker})`}
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      )}

      {/* Technical indicators strip */}
      {signal?.tech && !signal.tech.error && (
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 12, paddingTop: 12, borderTop: '1px solid var(--border)' }}>
          {[
            { label: 'RSI', value: signal.tech.rsi, color: signal.tech.rsi < 30 ? 'var(--green)' : signal.tech.rsi > 70 ? 'var(--red)' : 'var(--text2)' },
            { label: 'MACD', value: signal.tech.macd_bullish ? 'Bull' : 'Bear', color: signal.tech.macd_bullish ? 'var(--green)' : 'var(--red)' },
            { label: 'VWAP', value: signal.tech.price_above_vwap ? 'Above' : 'Below', color: signal.tech.price_above_vwap ? 'var(--green)' : 'var(--red)' },
            { label: 'BB%', value: `${signal.tech.bb_pct}%`, color: signal.tech.bb_pct < 20 ? 'var(--green)' : signal.tech.bb_pct > 80 ? 'var(--red)' : 'var(--text2)' },
            { label: 'EMA', value: signal.tech.ema_bullish ? 'Bull' : 'Bear', color: signal.tech.ema_bullish ? 'var(--green)' : 'var(--red)' },
          ].map(ind => (
            <div key={ind.label} style={{ minWidth: 70 }}>
              <div style={{ fontSize: 10, color: 'var(--text3)', marginBottom: 2, letterSpacing: '.06em' }}>{ind.label}</div>
              <div className="mono" style={{ fontSize: 13, fontWeight: 500, color: ind.color }}>
                {typeof ind.value === 'number' ? ind.value.toFixed(1) : ind.value}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
