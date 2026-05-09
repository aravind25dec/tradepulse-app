import { useState } from 'react';
import axios from 'axios';

const API = 'http://localhost:8000';

function signalColor(s) {
  if (s === 'LONG')  return 'var(--green)';
  if (s === 'SHORT') return 'var(--red)';
  return 'var(--amber)';
}

function SignalCard({ sig, active, onClick }) {
  const [refreshing, setRefreshing] = useState(false);

  const refresh = async (e) => {
    e.stopPropagation();
    setRefreshing(true);
    try { await axios.post(`${API}/api/signal/${sig.ticker}/refresh`); }
    catch (_) {}
    finally { setRefreshing(false); }
  };

  const price = sig.current_price;
  const conf  = sig.confidence || 0;

  return (
    <div
      onClick={onClick}
      style={{
        padding: '12px 14px',
        borderRadius: 8,
        border: `1px solid ${active ? signalColor(sig.signal) + '55' : 'var(--border)'}`,
        background: active ? 'var(--bg3)' : 'transparent',
        cursor: 'pointer',
        transition: 'all .15s',
        marginBottom: 6,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 6 }}>
        <div>
          <span style={{ fontSize: 15, fontWeight: 700, letterSpacing: '.02em' }}>{sig.ticker}</span>
          {sig.earnings_caution && (
            <span style={{ marginLeft: 6, fontSize: 10, color: 'var(--amber)' }}>⚠ earnings</span>
          )}
        </div>
        <span className={`badge badge-${sig.signal?.toLowerCase()}`}>
          {sig.signal}
        </span>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <span className="mono small" style={{ color: 'var(--text2)' }}>
          {price != null ? `$${Number(price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })}` : '—'}
        </span>
        <span className="mono small" style={{ color: 'var(--text3)' }}>
          conf {conf}%
        </span>
      </div>

      {/* Confidence bar */}
      <div style={{ height: 3, background: 'var(--bg3)', borderRadius: 2, overflow: 'hidden', marginBottom: 8 }}>
        <div style={{
          height: '100%',
          width: `${conf}%`,
          background: signalColor(sig.signal),
          borderRadius: 2,
          transition: 'width .4s ease',
        }} />
      </div>

      {sig.reasons && sig.reasons.length > 0 && (
        <div style={{ fontSize: 11, color: 'var(--text3)', lineHeight: 1.4 }}>
          {sig.reasons.slice(0, 2).map((r, i) => <div key={i}>· {r}</div>)}
        </div>
      )}

      <button
        onClick={refresh}
        disabled={refreshing}
        style={{
          marginTop: 8,
          fontSize: 11,
          color: 'var(--text3)',
          padding: '2px 0',
          opacity: refreshing ? 0.5 : 1,
        }}
      >
        {refreshing ? '↻ refreshing…' : '↻ refresh signal'}
      </button>
    </div>
  );
}

export function SignalPanel({ signals, tickers, ticks, activeTicker, onSelect }) {
  const sorted = tickers
    .map(t => signals[t])
    .filter(Boolean)
    .sort((a, b) => Math.abs(b.confidence || 0) - Math.abs(a.confidence || 0));

  // Tickers with no signal yet
  const missing = tickers.filter(t => !signals[t]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 12, color: 'var(--text2)', letterSpacing: '.08em', textTransform: 'uppercase' }}>Signals</span>
        <span className="mono small" style={{ color: 'var(--text3)' }}>{sorted.length} active</span>
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: '10px 12px' }}>
        {sorted.map(sig => (
          <SignalCard
            key={sig.ticker}
            sig={sig}
            active={activeTicker === sig.ticker}
            onClick={() => onSelect(sig.ticker)}
          />
        ))}
        {missing.map(t => (
          <div key={t} style={{ padding: '10px 14px', marginBottom: 6, borderRadius: 8, border: '1px solid var(--border)', color: 'var(--text3)', fontSize: 13 }}>
            <span style={{ fontWeight: 600 }}>{t}</span>
            <span style={{ marginLeft: 8, fontSize: 11 }}>loading…</span>
          </div>
        ))}
      </div>
    </div>
  );
}
