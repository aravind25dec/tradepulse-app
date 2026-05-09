import { useEffect, useRef, useState, useCallback } from 'react';

const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws';

export function useTradeSocket() {
  const [signals, setSignals] = useState({});
  const [events, setEvents] = useState([]);
  const [earnings, setEarnings] = useState([]);
  const [tickers, setTickers] = useState([]);
  const [stockTickers, setStockTickers] = useState([]);
  const [cryptoTickers, setCryptoTickers] = useState([]);
  const [ticks, setTicks] = useState({});
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const intentionalCloseRef = useRef(false);

  const connect = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }
    try {
      const ws = new WebSocket(WS_URL);
      intentionalCloseRef.current = false;
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        if (reconnectTimer.current) {
          clearTimeout(reconnectTimer.current);
          reconnectTimer.current = null;
        }
      };

      ws.onclose = () => {
        setConnected(false);
        // Avoid reconnect storms from intentional closes during reload/unmount.
        if (intentionalCloseRef.current) return;
        if (reconnectTimer.current) return;
        reconnectTimer.current = setTimeout(() => {
          reconnectTimer.current = null;
          connect();
        }, 3000);
      };

      ws.onerror = () => ws.close();

      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          switch (msg.type) {
            case 'init':
              if (msg.data.signals) {
                const sigMap = {};
                msg.data.signals.forEach(s => { sigMap[s.ticker] = s; });
                setSignals(sigMap);
              }
              if (msg.data.events)   setEvents(msg.data.events);
              if (msg.data.earnings) setEarnings(msg.data.earnings);
              if (msg.data.tickers)  setTickers(msg.data.tickers);
              if (msg.data.stock_tickers) setStockTickers(msg.data.stock_tickers);
              if (msg.data.crypto_tickers) setCryptoTickers(msg.data.crypto_tickers);
              break;
            case 'signal':
              setSignals(prev => ({ ...prev, [msg.data.ticker]: msg.data }));
              break;
            case 'tick':
              setTicks(prev => ({ ...prev, [msg.data.ticker]: msg.data }));
              break;
            case 'news':
              // news handled per-ticker in NewsStream component via REST
              break;
            case 'earnings':
              setEarnings(msg.data);
              break;
            default:
              break;
          }
        } catch (_) {}
      };
    } catch (_) {
      reconnectTimer.current = setTimeout(connect, 3000);
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      intentionalCloseRef.current = true;
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  return {
    signals,
    events,
    earnings,
    tickers,
    stockTickers,
    cryptoTickers,
    ticks,
    connected,
  };
}
