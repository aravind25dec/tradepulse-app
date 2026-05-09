import { useEffect, useState } from 'react';
import axios from 'axios';
import { formatDistanceToNow } from 'date-fns';

const API = 'http://localhost:8000';

export function NewsStream({ ticker }) {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!ticker) return;
    setLoading(true);
    axios.get(`${API}/api/news/${ticker}`)
      .then(r => setArticles(r.data || []))
      .catch(() => setArticles([]))
      .finally(() => setLoading(false));
  }, [ticker]);

  if (loading) {
    return (
      <div style={{ padding: '12px 16px', color: 'var(--text3)', fontSize: 12 }}>
        loading news…
      </div>
    );
  }

  if (!articles.length) {
    return (
      <div style={{ padding: '12px 16px', color: 'var(--text3)', fontSize: 12 }}>
        No recent news for {ticker}. Add a NewsAPI key to .env to enable.
      </div>
    );
  }

  return (
    <div style={{ overflowY: 'auto', maxHeight: 260 }}>
      {articles.map((a, i) => {
        let ago = '';
        try { ago = formatDistanceToNow(new Date(a.published), { addSuffix: true }); } catch (_) {}
        return (
          <a
            key={i}
            href={a.url}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'block',
              padding: '10px 16px',
              borderBottom: '1px solid var(--border)',
              textDecoration: 'none',
              color: 'inherit',
              transition: 'background .1s',
            }}
            onMouseEnter={e => e.currentTarget.style.background = 'var(--bg3)'}
            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
          >
            <div style={{ fontSize: 13, lineHeight: 1.45, marginBottom: 4 }}>{a.title}</div>
            <div style={{ display: 'flex', gap: 10, fontSize: 11, color: 'var(--text3)' }}>
              <span>{a.source}</span>
              {ago && <span>{ago}</span>}
            </div>
          </a>
        );
      })}
    </div>
  );
}
