import { formatDistanceToNow } from 'date-fns';

const TYPE_STYLES = {
  SEC_8K:         { label: '8-K',      color: 'var(--blue)' },
  EARNINGS_RESULT:{ label: 'EARNINGS', color: 'var(--amber)' },
  MA:             { label: 'M&A',      color: 'var(--red)' },
  CONTRACT:       { label: 'CONTRACT', color: 'var(--green)' },
};

function getStyle(type) {
  return TYPE_STYLES[type] || { label: type, color: 'var(--text2)' };
}

export function EventFeed({ events }) {
  if (!events || !events.length) {
    return (
      <div style={{ padding: '12px 16px', color: 'var(--text3)', fontSize: 12 }}>
        No corporate events yet. Events appear as SEC 8-K filings are detected.
      </div>
    );
  }

  return (
    <div style={{ overflowY: 'auto', maxHeight: 240 }}>
      {events.map((ev, i) => {
        const { label, color } = getStyle(ev.type);
        let ago = '';
        try { ago = ev.ts ? formatDistanceToNow(new Date(ev.ts), { addSuffix: true }) : ''; } catch (_) {}
        return (
          <div key={i} style={{
            display: 'flex', gap: 10, alignItems: 'flex-start',
            padding: '10px 16px', borderBottom: '1px solid var(--border)',
          }}>
            <span style={{ fontSize: 10, fontWeight: 600, color, fontFamily: 'var(--font-mono)', marginTop: 2, flexShrink: 0, minWidth: 58 }}>
              {label}
            </span>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 2 }}>
                <span style={{ fontWeight: 700, fontSize: 12 }}>{ev.ticker}</span>
                {ago && <span style={{ fontSize: 11, color: 'var(--text3)' }}>{ago}</span>}
              </div>
              <div style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.4 }}>{ev.description}</div>
              {ev.url && (
                <a href={ev.url} target="_blank" rel="noopener noreferrer" style={{ fontSize: 11, color: 'var(--blue)', textDecoration: 'none', marginTop: 2, display: 'inline-block' }}>
                  View filing →
                </a>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
