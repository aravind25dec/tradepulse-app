import { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { useTradeSocket } from './useTradeSocket';
import { SignalPanel } from './components/SignalPanel';
import { PriceChart } from './components/PriceChart';
import { NewsStream } from './components/NewsStream';
import { EarningsCalendar } from './components/EarningsCalendar';
import { EventFeed } from './components/EventFeed';

const API = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const SAVED_STOCKS_KEY = 'tradepulse_custom_stocks_v1';
const SAVED_CRYPTO_KEY = 'tradepulse_custom_crypto_v1';

function StatusDot({ connected }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 11, color: connected ? 'var(--green)' : 'var(--red)', fontFamily: 'var(--font-mono)' }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: connected ? 'var(--green)' : 'var(--red)', display: 'inline-block' }} />
      {connected ? 'live' : 'reconnecting…'}
    </span>
  );
}

function Section({ title, children }) {
  return (
    <div className="card" style={{ overflow: 'hidden' }}>
      <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 11, letterSpacing: '.08em', textTransform: 'uppercase', color: 'var(--text3)', fontWeight: 500 }}>{title}</span>
      </div>
      {children}
    </div>
  );
}

export default function App() {
  const { signals, events, earnings, tickers, stockTickers, cryptoTickers, ticks, connected } = useTradeSocket();
  const [activeTicker, setActiveTicker] = useState(null);
  const [watchMode, setWatchMode] = useState('all');
  const [viewMode, setViewMode] = useState('dashboard');
  const [topStocks, setTopStocks] = useState([]);
  const [topLosers, setTopLosers] = useState([]);
  const [topTab, setTopTab] = useState('up');
  const [analysisTicker, setAnalysisTicker] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [newTicker, setNewTicker] = useState('');
  const [addingTicker, setAddingTicker] = useState(false);
  const rehydratedRef = useRef(false);

  useEffect(() => {
    if (rehydratedRef.current) return;
    rehydratedRef.current = true;
    // Rehydrate customer watchlist from local storage into backend runtime list.
    let savedStocks = [];
    let savedCrypto = [];
    try { savedStocks = JSON.parse(localStorage.getItem(SAVED_STOCKS_KEY) || '[]'); } catch (_) {}
    try { savedCrypto = JSON.parse(localStorage.getItem(SAVED_CRYPTO_KEY) || '[]'); } catch (_) {}
    const allSaved = [...savedStocks, ...savedCrypto];
    allSaved.forEach((ticker) => {
      axios.post(`${API}/api/watchlist/add`, { ticker }).catch(() => {});
    });
  }, []);

  useEffect(() => {
    let mounted = true;
    const loadTopStocks = async () => {
      try {
        const [upRes, downRes] = await Promise.all([
          axios.get(`${API}/api/movers/stocks?limit=10`),
          axios.get(`${API}/api/movers/stocks/down?limit=10`),
        ]);
        if (!mounted) return;
        setTopStocks(Array.isArray(upRes.data) ? upRes.data : []);
        setTopLosers(Array.isArray(downRes.data) ? downRes.data : []);
      } catch (_) {
        if (!mounted) return;
        setTopStocks([]);
        setTopLosers([]);
      }
    };
    loadTopStocks();
    const id = setInterval(loadTopStocks, 120000);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, []);

  const filteredTickers = useMemo(() => {
    if (watchMode === 'stocks') return stockTickers;
    if (watchMode === 'crypto') return cryptoTickers;
    return tickers;
  }, [watchMode, tickers, stockTickers, cryptoTickers]);

  const displayTicker = (activeTicker && filteredTickers.includes(activeTicker))
    ? activeTicker
    : (filteredTickers[0] || tickers[0] || 'AAPL');
  const activeSignal = signals[displayTicker];

  const addTicker = async () => {
    const ticker = newTicker.trim().toUpperCase();
    if (!ticker) return;
    setAddingTicker(true);
    try {
      await axios.post(`${API}/api/watchlist/add`, { ticker });
      const key = ticker.endsWith('-USD') ? SAVED_CRYPTO_KEY : SAVED_STOCKS_KEY;
      const existing = JSON.parse(localStorage.getItem(key) || '[]');
      if (!existing.includes(ticker)) {
        localStorage.setItem(key, JSON.stringify([...existing, ticker]));
      }
      setNewTicker('');
      setActiveTicker(ticker);
      setWatchMode(ticker.endsWith('-USD') ? 'crypto' : 'stocks');
    } catch (_) {
      // Keep silent in UI for now; could replace with toast/snackbar later.
    } finally {
      setAddingTicker(false);
    }
  };

  const openAnalysis = async (ticker) => {
    setAnalysisTicker(ticker);
    setAnalysisLoading(true);
    setAnalysisData(null);
    try {
      const { data } = await axios.get(`${API}/api/stock/${ticker}/analysis`);
      setAnalysisData(data || null);
    } catch (_) {
      setAnalysisData(null);
    } finally {
      setAnalysisLoading(false);
    }
  };

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

      {/* Top bar */}
      <header style={{
        height: 48,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 20px',
        borderBottom: '1px solid var(--border)',
        background: 'var(--bg2)',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 16, fontWeight: 700, letterSpacing: '.04em' }}>TradePulse</span>
          <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>
            AI Day Trading Signals
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{ display: 'inline-flex', border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden' }}>
            {[
              { id: 'dashboard', label: 'Dashboard' },
              { id: 'top10', label: 'Top 10 Buy' },
            ].map(v => (
              <button
                key={v.id}
                onClick={() => setViewMode(v.id)}
                style={{
                  fontSize: 11,
                  padding: '4px 8px',
                  background: viewMode === v.id ? 'var(--bg3)' : 'transparent',
                  color: viewMode === v.id ? 'var(--text1)' : 'var(--text3)',
                  borderRight: v.id !== 'top10' ? '1px solid var(--border)' : 'none',
                }}
              >
                {v.label}
              </button>
            ))}
          </div>
          <div style={{ display: 'inline-flex', border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden' }}>
            {[
              { id: 'all', label: 'All' },
              { id: 'stocks', label: 'Stocks' },
              { id: 'crypto', label: 'Crypto' },
            ].map(mode => (
              <button
                key={mode.id}
                onClick={() => setWatchMode(mode.id)}
                style={{
                  fontSize: 11,
                  padding: '4px 8px',
                  background: watchMode === mode.id ? 'var(--bg3)' : 'transparent',
                  color: watchMode === mode.id ? 'var(--text1)' : 'var(--text3)',
                  borderRight: mode.id !== 'crypto' ? '1px solid var(--border)' : 'none',
                }}
              >
                {mode.label}
              </button>
            ))}
          </div>
          <span style={{ fontSize: 11, color: 'var(--red)', fontFamily: 'var(--font-mono)' }}>
            ⚠ NOT FINANCIAL ADVICE — EDUCATIONAL ONLY
          </span>
          <StatusDot connected={connected} />
        </div>
      </header>

      {/* Main layout */}
      {viewMode === 'top10' ? (
        <div style={{ flex: 1, padding: 12, overflow: 'hidden' }}>
          <div className="card" style={{ height: '100%', overflow: 'auto', padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
              <div style={{ fontSize: 12, color: 'var(--text2)', letterSpacing: '.08em', textTransform: 'uppercase' }}>
                {topTab === 'up' ? 'Top 10 Stocks - Moving Up' : 'Top 10 Stocks - Moving Down'}
              </div>
              <div style={{ display: 'inline-flex', border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden' }}>
                <button onClick={() => setTopTab('up')} style={{ fontSize: 11, padding: '4px 8px', background: topTab === 'up' ? 'var(--bg3)' : 'transparent', borderRight: '1px solid var(--border)' }}>Moving Up</button>
                <button onClick={() => setTopTab('down')} style={{ fontSize: 11, padding: '4px 8px', background: topTab === 'down' ? 'var(--bg3)' : 'transparent' }}>Moving Down</button>
              </div>
            </div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ textAlign: 'left', color: 'var(--text3)', borderBottom: '1px solid var(--border)' }}>
                  <th style={{ padding: '8px 6px' }}>Ticker</th>
                  <th style={{ padding: '8px 6px' }}>Price</th>
                  <th style={{ padding: '8px 6px' }}>1h %</th>
                  <th style={{ padding: '8px 6px' }}>2h %</th>
                  <th style={{ padding: '8px 6px' }}>RSI</th>
                  <th style={{ padding: '8px 6px' }}>Score</th>
                </tr>
              </thead>
              <tbody>
                {(topTab === 'up' ? topStocks : topLosers).map((s) => (
                  <tr key={s.ticker} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '8px 6px' }}>
                      <button onClick={() => openAnalysis(s.ticker)} style={{ fontWeight: 700 }}>
                        {s.ticker}
                      </button>
                    </td>
                    <td style={{ padding: '8px 6px' }}>${Number(s.price).toLocaleString(undefined, { maximumFractionDigits: 4 })}</td>
                    <td style={{ padding: '8px 6px', color: s.ret_1h_pct >= 0 ? 'var(--green)' : 'var(--red)' }}>{s.ret_1h_pct}%</td>
                    <td style={{ padding: '8px 6px', color: s.ret_2h_pct >= 0 ? 'var(--green)' : 'var(--red)' }}>{s.ret_2h_pct}%</td>
                    <td style={{ padding: '8px 6px' }}>{s.rsi}</td>
                    <td style={{ padding: '8px 6px' }}>{s.momentum_score}</td>
                  </tr>
                ))}
                {(topTab === 'up' ? topStocks : topLosers).length === 0 && (
                  <tr>
                    <td colSpan={6} style={{ padding: '12px 6px', color: 'var(--text3)' }}>No candidates currently meet filters.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '240px 1fr 280px', gap: 12, padding: 12, overflow: 'hidden', minHeight: 0 }}>

        {/* Left: Signal watchlist */}
        <div style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <div className="card" style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '10px 12px', borderBottom: '1px solid var(--border)', display: 'flex', gap: 8 }}>
              <input
                value={newTicker}
                onChange={(e) => setNewTicker(e.target.value)}
                placeholder="Add ticker (e.g. META or DOGE-USD)"
                style={{ flex: 1, fontSize: 11, padding: '6px 8px', border: '1px solid var(--border)', borderRadius: 6, background: 'var(--bg2)', color: 'var(--text1)' }}
              />
              <button onClick={addTicker} disabled={addingTicker} style={{ fontSize: 11, padding: '6px 8px' }}>
                {addingTicker ? 'Adding...' : 'Add'}
              </button>
            </div>
            <SignalPanel
              signals={signals}
              tickers={filteredTickers}
              ticks={ticks}
              activeTicker={displayTicker}
              onSelect={setActiveTicker}
            />
          </div>
        </div>

        {/* Center: Chart + News */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, overflow: 'hidden', minHeight: 0 }}>

          {/* Price chart + indicators */}
          <div className="card" style={{ padding: 16, flexShrink: 0 }}>
            <PriceChart ticker={displayTicker} signal={activeSignal} ticks={ticks} />
          </div>

          {/* Signal detail */}
          {activeSignal && (
            <div className="card" style={{ padding: 16, flexShrink: 0 }}>
              <div style={{ fontSize: 11, color: 'var(--text3)', letterSpacing: '.08em', textTransform: 'uppercase', marginBottom: 10 }}>AI Analysis — {displayTicker}</div>
              {activeSignal.summary && (
                <p style={{ fontSize: 13, color: 'var(--text2)', marginBottom: 10, lineHeight: 1.5 }}>{activeSignal.summary}</p>
              )}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                {activeSignal.catalysts?.length > 0 && (
                  <div>
                    <div style={{ fontSize: 11, color: 'var(--green)', marginBottom: 5, fontWeight: 500 }}>Catalysts</div>
                    {activeSignal.catalysts.map((c, i) => (
                      <div key={i} style={{ fontSize: 12, color: 'var(--text2)', marginBottom: 3 }}>· {c}</div>
                    ))}
                  </div>
                )}
                {activeSignal.risk_factors?.length > 0 && (
                  <div>
                    <div style={{ fontSize: 11, color: 'var(--red)', marginBottom: 5, fontWeight: 500 }}>Risks</div>
                    {activeSignal.risk_factors.map((r, i) => (
                      <div key={i} style={{ fontSize: 12, color: 'var(--text2)', marginBottom: 3 }}>· {r}</div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* News */}
          <Section title={`News — ${displayTicker}`}>
            <NewsStream ticker={displayTicker} />
          </Section>
        </div>

        {/* Right: Events + Earnings */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, overflow: 'hidden', minHeight: 0 }}>
          <Section title="Corporate events">
            <EventFeed events={events} />
          </Section>
          <Section title="Earnings calendar">
            <EarningsCalendar earnings={earnings} />
          </Section>
          <Section title="Top 10 stocks moving up">
            <div style={{ padding: 12, overflowY: 'auto', maxHeight: 240 }}>
              {topStocks.length === 0 ? (
                <div style={{ fontSize: 12, color: 'var(--text3)' }}>No qualifying stocks yet.</div>
              ) : (
                topStocks.map((s) => (
                  <div key={s.ticker} style={{ padding: '8px 0', borderBottom: '1px solid var(--border)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <button
                        onClick={() => openAnalysis(s.ticker)}
                        style={{ fontSize: 12, fontWeight: 700, color: 'var(--text1)' }}
                      >
                        {s.ticker}
                      </button>
                      <span className="mono small" style={{ color: 'var(--green)' }}>{s.momentum_score}</span>
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 4 }}>
                      1h {s.ret_1h_pct}% · 2h {s.ret_2h_pct}% · RSI {s.rsi}
                    </div>
                  </div>
                ))
              )}
            </div>
          </Section>
        </div>

      </div>
      )}
      {analysisTicker && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,.55)', zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20 }}>
          <div className="card" style={{ width: 'min(840px, 95vw)', maxHeight: '85vh', overflow: 'auto', padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <div style={{ fontSize: 14, fontWeight: 700 }}>{analysisTicker} - Multi-timeframe Analysis</div>
              <button onClick={() => setAnalysisTicker(null)} style={{ fontSize: 12 }}>Close</button>
            </div>
            {analysisLoading ? (
              <div style={{ color: 'var(--text3)', fontSize: 12 }}>Loading analysis...</div>
            ) : !analysisData ? (
              <div style={{ color: 'var(--text3)', fontSize: 12 }}>Unable to load analysis.</div>
            ) : (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 8, marginBottom: 12 }}>
                  {Object.entries(analysisData.performance_pct || {}).map(([k, v]) => (
                    <div key={k} style={{ border: '1px solid var(--border)', borderRadius: 8, padding: 8 }}>
                      <div style={{ fontSize: 10, color: 'var(--text3)', textTransform: 'uppercase' }}>{k}</div>
                      <div style={{ fontSize: 13, color: v == null ? 'var(--text3)' : v >= 0 ? 'var(--green)' : 'var(--red)' }}>
                        {v == null ? 'n/a' : `${v}%`}
                      </div>
                    </div>
                  ))}
                </div>
                <div style={{ marginBottom: 8, fontSize: 12 }}>
                  <strong>Lifetime coverage:</strong> {analysisData.lifetime_points} monthly points (max history)
                </div>
                <div style={{ marginBottom: 8, fontSize: 12 }}>
                  <strong>Pattern match:</strong> similarity {analysisData.pattern_match?.best_similarity ?? 'n/a'} | historical next-day move {analysisData.pattern_match?.historical_next_day_move_pct ?? 'n/a'}%
                </div>
                <div style={{ marginBottom: 8, fontSize: 12 }}>
                  <strong>Today bias:</strong>{' '}
                  <span style={{ color: analysisData.today_bias === 'rise' ? 'var(--green)' : analysisData.today_bias === 'fall' ? 'var(--red)' : 'var(--amber)' }}>
                    {analysisData.today_bias}
                  </span>{' '}
                  ({analysisData.today_bias_confidence}% confidence)
                </div>
                <div style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.5 }}>
                  {analysisData.explanation}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
