import { format, parseISO, differenceInDays } from 'date-fns';

function SurpriseBadge({ pct }) {
  if (pct == null) return <span style={{ color: 'var(--text3)', fontSize: 11 }}>pending</span>;
  const cls = pct > 5 ? 'badge-long' : pct < -5 ? 'badge-short' : 'badge-hold';
  return <span className={`badge ${cls}`}>{pct > 0 ? '+' : ''}{pct.toFixed(1)}%</span>;
}

export function EarningsCalendar({ earnings }) {
  if (!earnings || !earnings.length) {
    return (
      <div style={{ padding: '12px 16px', color: 'var(--text3)', fontSize: 12 }}>
        No upcoming earnings. Add a Finnhub API key to .env to enable.
      </div>
    );
  }

  const sorted = [...earnings].sort((a, b) =>
    new Date(a.report_date) - new Date(b.report_date)
  );

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)' }}>
            {['Ticker', 'Date', 'Time', 'EPS est.', 'EPS act.', 'Surprise'].map(h => (
              <th key={h} style={{ padding: '8px 14px', textAlign: 'left', color: 'var(--text3)', fontWeight: 500, letterSpacing: '.05em', fontSize: 11 }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((e, i) => {
            let daysOut = null;
            try { daysOut = differenceInDays(parseISO(e.report_date), new Date()); } catch (_) {}
            const isNear = daysOut != null && daysOut <= 2 && daysOut >= 0;
            return (
              <tr
                key={i}
                style={{
                  borderBottom: '1px solid var(--border)',
                  background: isNear ? 'var(--amber-dim)' : 'transparent',
                  transition: 'background .1s',
                }}
                onMouseEnter={e => e.currentTarget.style.background = 'var(--bg3)'}
                onMouseLeave={ev => ev.currentTarget.style.background = isNear ? 'var(--amber-dim)' : 'transparent'}
              >
                <td style={{ padding: '9px 14px', fontWeight: 700 }}>{e.ticker}</td>
                <td className="mono" style={{ padding: '9px 14px', color: 'var(--text2)' }}>
                  {e.report_date ? format(parseISO(e.report_date), 'MMM dd') : '—'}
                  {daysOut === 0 && <span style={{ marginLeft: 6, color: 'var(--amber)', fontSize: 10 }}>TODAY</span>}
                  {daysOut === 1 && <span style={{ marginLeft: 6, color: 'var(--amber)', fontSize: 10 }}>TMR</span>}
                </td>
                <td style={{ padding: '9px 14px', color: 'var(--text3)' }}>
                  {e.hour === 'bmo' ? 'Pre-mkt' : e.hour === 'amc' ? 'After-mkt' : '—'}
                </td>
                <td className="mono" style={{ padding: '9px 14px', color: 'var(--text2)' }}>
                  {e.eps_estimate != null ? e.eps_estimate.toFixed(2) : '—'}
                </td>
                <td className="mono" style={{ padding: '9px 14px', color: 'var(--text2)' }}>
                  {e.eps_actual != null ? e.eps_actual.toFixed(2) : '—'}
                </td>
                <td style={{ padding: '9px 14px' }}>
                  <SurpriseBadge pct={e.surprise_pct} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
