"""
hivepicks/hive_engine.py

Multi-agent hive for day-trading stock selection.
5 specialized agents deliberate on each candidate and reach a consensus.

Agent roster:
  1. MarketRegimeAgent  — reads SPY/QQQ, sets market "weather" for all others
  2. TechnicalAgent     — reasons about chart indicators (RSI, MACD, BB, ADX, EMA)
  3. MomentumAgent      — focuses on intraday velocity, volume surge, VWAP flow
  4. CatalystAgent      — judges news headlines and event-driven catalysts
  5. RiskArbiterAgent   — sees ALL verdicts, makes final BUY/SKIP + trade plan

Flow per stock: Technical + Momentum + Catalyst run in parallel, then Arbiter.
"""

import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import httpx

# ── Import shared data layer from hotpicks ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "hotpicks"))
from engine import (
    DEFAULT_UNIVERSE,
    _HEADERS,
    fetch_market_context,
    fetch_intraday,
    fetch_daily,
    fetch_news,
    fetch_earnings_data,
    compute_signals,
    _mechanical_plan,
)
import agent_engine

log = logging.getLogger(__name__)

# ── Risk-first system prompt (injected into every agent) ─────────────────────

_RISK_SYSTEM = """You are a professional risk-first trading analyst. Your primary job is to PROTECT CAPITAL, not find winners. A pick you don't make cannot lose money.

## CORE RULES — NEVER VIOLATE

1. **Assume you will be wrong.** At least 40% of picks will fail. Every recommendation must have a defined stop-loss that limits damage to ≤1.5% of portfolio.

2. **Do not chase moves.** If a stock is already up >3% from open with RSI >68, the move is likely exhausted. Buying here is buying someone else's exit. Default to SKIP unless there is a *confirmed* new catalyst (earnings beat, FDA approval, contract announcement — not a rumor).

3. **Volume must confirm direction.** Rising price + falling or average volume = distribution (insiders/funds selling into retail buyers). Require vol_ratio ≥1.5x for any BUY. Below that, issue WATCH at most.

4. **Overextension is a SKIP, not a buy signal.** Price above upper Bollinger Band + RSI >70 = overextended. These conditions are reversal signals, not continuation signals. Do not issue BUY in this state regardless of other factors.

5. **Market regime overrides everything.** If SPY is down >0.8% intraday and QQQ is down >1.0%, issue no new BUY signals. Individual stocks fall with the market 80% of the time. Wait for regime to stabilize.

6. **Identify ONE specific reason to buy.** Vague reasons like "strong momentum" or "bullish technicals" are not acceptable. The reason must be: a named catalyst (earnings beat, upgrade, product launch), a specific technical setup (confirmed breakout above resistance X with volume), or a quantified data point (revenue growth 40% YoY, 3 consecutive earnings beats).

## BEFORE ISSUING ANY BUY

Answer these four questions explicitly in your reasoning:
- **What is the catalyst?** (Name it specifically — earnings/upgrade/breakout/macro. "Momentum" is not a catalyst.)
- **What would prove this wrong?** (Name the exact price level or condition that invalidates the thesis.)
- **Is the move already priced in?** (If stock is up >4% today, explain who is still left to buy.)
- **What is the exit plan?** (Specific target price AND specific stop-loss price, not percentages alone.)

If you cannot answer all four clearly, issue WATCH or SKIP.

## SIGNAL QUALITY TIERS

- **STRONG BUY:** Fresh named catalyst (today) + price near VWAP (not extended) + vol_ratio ≥2x + RSI 45–65 + market regime neutral or better. All conditions must be true.
- **BUY:** 3 of the above 5 conditions true, with catalyst confirmed.
- **WATCH:** Interesting setup but missing 2+ conditions. Monitor only — do not act until confirmation.
- **SKIP:** Any of these present: RSI >72, price >upper BB, vol-price divergence, market bear regime, no identifiable catalyst, bearish engulfing candle in last 3 bars.

## VETO RULES (SKIP overrides any BUY vote)

If ANY of the following are true, the verdict is SKIP — no exceptions, no overrides:
- Price above upper Bollinger Band (BB% > 100)
- RSI > 75 on intraday OR daily timeframe
- Bearish engulfing candle on the last completed 5-min or daily bar
- Price-volume divergence detected (price rising, volume declining for 3+ consecutive bars)
- Stock is down >2% from open (momentum is against you)
- Broad market (SPY) is in bear regime today

## WHAT YOU MUST NEVER DO

- Never recommend averaging down on a losing position
- Never issue BUY on a stock already at 52-week high without a confirmed new catalyst
- Never cite analyst price targets older than 60 days as a reason to buy
- Never recommend a stock you cannot explain in one clear sentence
- Never issue more than 3 BUY signals in a single session — scarcity of picks improves quality

IMPORTANT: When your specific task below provides its own JSON response schema, output ONLY that JSON — the risk rules above govern your *reasoning and decisions*, not the output format."""

# Tickers the hive focuses on — curated for day-trading liquidity
HIVE_UNIVERSE = [
    "NVDA", "TSLA", "META", "AAPL", "AMZN", "GOOGL", "MSFT", "AMD",
    "PLTR", "SOFI", "COIN", "HOOD", "MSTR", "RIVN",
    "SMCI", "ARM", "UPST", "RKLB",
    "SPY", "QQQ",
]

# ── JSON extraction helper ────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned).strip()
    s = cleaned.find("{")
    e = cleaned.rfind("}")
    if s < 0 or e <= s:
        raise ValueError(f"No JSON object found in: {text[:200]}")
    return json.loads(cleaned[s:e + 1])


def _safe_agent_result(agent_name: str, ticker: str, error: str) -> dict:
    return {
        "agent":      agent_name,
        "ticker":     ticker,
        "verdict":    "SKIP",
        "confidence": 0,
        "reasoning":  f"Agent error: {error}",
        "key_factors": [],
        "error":      True,
    }


# ── Agent 1: Market Regime ────────────────────────────────────────────────────

async def market_regime_agent(market_ctx: dict) -> dict:
    """
    Reads market context (SPY/QQQ change, regime label) and produces a
    structured verdict on how favorable conditions are for day trading today.
    """
    prompt = f"""You are MarketRegimeBot, a macro market analyst for a day trading desk.

Your job: assess today's market conditions and rate how favorable they are for momentum day trading.

LIVE MARKET DATA:
- SPY change: {market_ctx['spy_chg']:+.2f}%
- QQQ change: {market_ctx['qqq_chg']:+.2f}%
- SPY above VWAP: {market_ctx['spy_above_vwap']}
- QQQ above VWAP: {market_ctx['qqq_above_vwap']}
- Regime label: {market_ctx['regime'].upper()}

Rate the market conditions and give direction to the other agents:
- FAVORABLE: momentum plays are high-probability; bias toward BUY setups
- NEUTRAL: selective — only A+ setups with volume confirmation
- CAUTION: markets are choppy; reduce position sizes, be very selective
- AVOID: broad selling pressure; most longs will fail with the market

Respond with ONLY valid JSON (no markdown, no explanation outside JSON):
{{"verdict": "FAVORABLE|NEUTRAL|CAUTION|AVOID", "confidence": <1-10>, "reasoning": "<2 sentences on market conditions>", "bias": "<specific bias for other agents e.g. prefer gap-up plays, avoid extended names>", "risk_multiplier": <0.5-1.5>}}"""

    try:
        raw = await agent_engine._oneshot_claude(f"{_RISK_SYSTEM}\n\n---\n\n{prompt}")
        result = _extract_json(raw)
        result["agent"] = "market_regime"
        result["market_ctx"] = market_ctx
        log.info("MarketRegimeAgent: %s (confidence=%s)", result.get("verdict"), result.get("confidence"))
        return result
    except Exception as exc:
        log.warning("MarketRegimeAgent failed: %s", exc)
        # Fallback using rule-based regime
        verdict_map = {"bull": "FAVORABLE", "neutral": "NEUTRAL", "caution": "CAUTION", "bear": "AVOID"}
        return {
            "agent":          "market_regime",
            "verdict":        verdict_map.get(market_ctx["regime"], "NEUTRAL"),
            "confidence":     5,
            "reasoning":      f"Fallback: SPY {market_ctx['spy_chg']:+.1f}%, QQQ {market_ctx['qqq_chg']:+.1f}%",
            "bias":           "Standard momentum criteria apply.",
            "risk_multiplier": 1.0,
            "market_ctx":     market_ctx,
            "error":          True,
        }


# ── Agent 2: Technical Analyst ────────────────────────────────────────────────

async def technical_agent(ticker: str, signals: dict, market_regime: dict) -> dict:
    """
    Reasons about chart indicators: RSI, MACD, Bollinger Bands, ADX, EMA cross,
    candlestick patterns. Ignores volume and news.
    """
    prompt = f"""You are TechBot, a quantitative technical analyst for a day trading desk.

Analyze {ticker} from a PURE TECHNICAL chart perspective only. Ignore news and volume.
The market is currently {market_regime.get('verdict', 'NEUTRAL')}. {market_regime.get('bias', '')}

CHART SIGNALS:
- Price: ${signals['price']} | Open: ${signals['day_open']}
- Change from open: {signals['chg_open']:+.2f}%
- Gap from prev close: {signals.get('gap_pct', 0):+.2f}%
- RSI (intraday): {signals['rsi']:.1f}
- EMA 9/21: {'Bullish — 9 > 21' if signals['ema_bullish'] else 'Bearish — 9 < 21'}
- MACD (daily): {'Bullish' if signals.get('macd_bullish') else 'Bearish'}, histogram {signals.get('macd_trend', 'flat')}
- ADX: {signals.get('adx', 20):.1f} (trending: {'YES' if signals.get('trending') else 'NO'})
- Bollinger %B: {signals.get('bb_pct', 0.5):.0%} (0%=lower band, 100%=upper band)
- BB squeeze: {'YES — breakout imminent' if signals.get('bb_squeeze') else 'NO'}
- Last candle: {signals.get('candle_pattern', 'none').upper()}
- ATR: {signals.get('atr', 0):.3f} ({signals.get('atr_pct', 0):.2f}% of price)

Respond with ONLY valid JSON (no markdown):
{{"verdict": "BUY|WATCH|SKIP", "confidence": <1-10>, "reasoning": "<2-3 sentences on what the chart says>", "key_factors": ["<factor1>", "<factor2>", "<factor3>"], "suggested_stop": {signals['price'] * 0.99:.2f}, "chart_target": {signals['price'] * 1.015:.2f}}}"""

    try:
        raw = await agent_engine._oneshot_claude(f"{_RISK_SYSTEM}\n\n---\n\n{prompt}")
        result = _extract_json(raw)
        result["agent"] = "technical"
        result["ticker"] = ticker
        log.info("TechnicalAgent %s: %s (conf=%s)", ticker, result.get("verdict"), result.get("confidence"))
        return result
    except Exception as exc:
        log.warning("TechnicalAgent failed for %s: %s", ticker, exc)
        return _safe_agent_result("technical", ticker, str(exc))


# ── Agent 3: Momentum & Volume ────────────────────────────────────────────────

async def momentum_agent(ticker: str, signals: dict, market_regime: dict) -> dict:
    """
    Focuses exclusively on intraday momentum: velocity, volume surge, VWAP
    positioning, relative strength vs market. Ignores daily chart and news.
    """
    prompt = f"""You are MomentumBot, an intraday momentum and flow trader for a day trading desk.

Analyze {ticker} from a MOMENTUM AND VOLUME perspective only. Ignore fundamental/news.
Market regime: {market_regime.get('verdict', 'NEUTRAL')}. {market_regime.get('bias', '')}

MOMENTUM & FLOW DATA:
- Price: ${signals['price']} vs Open: ${signals['day_open']}
- Intraday change: {signals['chg_open']:+.2f}%
- 15-min velocity: {signals['chg_15m']:+.2f}%
- 30-min velocity: {signals['chg_30m']:+.2f}%
- Volume ratio: {signals['vol_ratio']:.2f}x vs daily average (>2.0 = strong surge, <0.8 = weak)
- VWAP: ${signals['vwap']} | Price is {'ABOVE' if signals['above_vwap'] else 'BELOW'} VWAP
- VWAP distance: {signals['vwap_dist']:+.2f}%
- Volume-price divergence: {'YES — price rising but volume fading (fake move)' if signals.get('vol_divergence') else 'NO'}
- Intraday bars: {signals.get('bars', 0)} (out of ~78 in a full day)

Key question: Is there genuine intraday buying pressure with real volume, or is this move unconvincing?

Respond with ONLY valid JSON (no markdown):
{{"verdict": "BUY|WATCH|SKIP", "confidence": <1-10>, "reasoning": "<2-3 sentences on momentum/flow quality>", "key_factors": ["<factor1>", "<factor2>", "<factor3>"], "vwap_entry_zone": {signals['vwap'] * 1.001:.2f}}}"""

    try:
        raw = await agent_engine._oneshot_claude(f"{_RISK_SYSTEM}\n\n---\n\n{prompt}")
        result = _extract_json(raw)
        result["agent"] = "momentum"
        result["ticker"] = ticker
        log.info("MomentumAgent %s: %s (conf=%s)", ticker, result.get("verdict"), result.get("confidence"))
        return result
    except Exception as exc:
        log.warning("MomentumAgent failed for %s: %s", ticker, exc)
        return _safe_agent_result("momentum", ticker, str(exc))


# ── Agent 4: Catalyst & Sentiment ────────────────────────────────────────────

async def catalyst_agent(ticker: str, headlines: list[str], signals: dict, market_regime: dict) -> dict:
    """
    Evaluates news headlines and recent catalysts. A real catalyst (earnings
    beat, FDA approval, contract win) elevates conviction. Absence of news
    means the move is purely technical.
    """
    news_block = "\n".join(f"- {h}" for h in headlines[:6]) if headlines else "No recent news found."

    prompt = f"""You are CatalystBot, a news-driven event trader for a day trading desk.

Analyze {ticker} from a NEWS AND CATALYST perspective only.
Market: {market_regime.get('verdict', 'NEUTRAL')}.

RECENT HEADLINES (last 12 hours):
{news_block}

CONTEXT:
- Stock is {'up' if signals['chg_open'] > 0 else 'down'} {abs(signals['chg_open']):.1f}% from open
- Gap from yesterday: {signals.get('gap_pct', 0):+.1f}%

Assess:
1. Is there a real fundamental catalyst (earnings, guidance raise, product launch, FDA, contract, analyst upgrade)?
2. Is the news BULLISH, BEARISH, or NEUTRAL for intraday price action?
3. Does the headline explain the move, or is it moving on pure technicals (no clear catalyst)?

A stock with a real bullish catalyst gets higher conviction. No news = purely technical move (acceptable but lower conviction).

Respond with ONLY valid JSON (no markdown):
{{"verdict": "BUY|WATCH|SKIP", "confidence": <1-10>, "reasoning": "<2 sentences on catalyst quality>", "key_factors": ["<factor1>", "<factor2>"], "catalyst_type": "earnings|upgrade|contract|fda|macro|technical_only|bearish_news|none", "catalyst_strength": "strong|moderate|weak|none"}}"""

    try:
        raw = await agent_engine._oneshot_claude(f"{_RISK_SYSTEM}\n\n---\n\n{prompt}")
        result = _extract_json(raw)
        result["agent"] = "catalyst"
        result["ticker"] = ticker
        result["headlines"] = headlines[:3]
        log.info("CatalystAgent %s: %s (conf=%s, catalyst=%s)", ticker, result.get("verdict"), result.get("confidence"), result.get("catalyst_type"))
        return result
    except Exception as exc:
        log.warning("CatalystAgent failed for %s: %s", ticker, exc)
        return _safe_agent_result("catalyst", ticker, str(exc))


# ── Agent 5: Risk Arbiter ─────────────────────────────────────────────────────

async def risk_arbiter_agent(
    ticker: str,
    signals: dict,
    market_regime: dict,
    technical_verdict: dict,
    momentum_verdict: dict,
    catalyst_verdict: dict,
) -> dict:
    """
    The final decision-maker. Reads all specialist verdicts, weighs dissent,
    and produces the definitive trade plan with entry/target/stop.
    Veto power: can override unanimous BUY if risk is unacceptable.
    """
    # Build a concise deliberation summary for the arbiter
    def _fmt(v: dict) -> str:
        return (
            f"{v.get('agent','?').upper()}: {v.get('verdict','?')} "
            f"(confidence {v.get('confidence','?')}/10) — {v.get('reasoning','no reasoning')}"
        )

    votes = [technical_verdict.get("verdict", "SKIP"), momentum_verdict.get("verdict", "SKIP"), catalyst_verdict.get("verdict", "SKIP")]
    buy_count  = votes.count("BUY")
    watch_count = votes.count("WATCH")
    skip_count = votes.count("SKIP")

    atr = signals.get("atr", signals["price"] * 0.005)
    mech = _mechanical_plan(signals, 0, "BUY")  # just for reference levels

    prompt = f"""You are RiskArbiterBot, the senior risk officer and final decision-maker for a day trading desk.

You have received verdicts from 3 specialist agents on {ticker}. Your job is to synthesize their views, resolve any disagreements, and issue the FINAL trading decision.

You have VETO power — you can override an apparent BUY consensus if you see unacceptable risk.

═══ MARKET CONDITIONS ═══
Regime: {market_regime.get('verdict', 'NEUTRAL')} (confidence {market_regime.get('confidence', 5)}/10)
{market_regime.get('reasoning', '')}

═══ SPECIALIST VERDICTS ═══
{_fmt(technical_verdict)}
  Key factors: {'; '.join(technical_verdict.get('key_factors', [])[:3])}

{_fmt(momentum_verdict)}
  Key factors: {'; '.join(momentum_verdict.get('key_factors', [])[:3])}

{_fmt(catalyst_verdict)}
  Key factors: {'; '.join(catalyst_verdict.get('key_factors', [])[:3])}
  Catalyst type: {catalyst_verdict.get('catalyst_type', 'none')} | Strength: {catalyst_verdict.get('catalyst_strength', 'none')}

VOTE TALLY: {buy_count} BUY / {watch_count} WATCH / {skip_count} SKIP

═══ RAW MARKET DATA ═══
Price: ${signals['price']} | VWAP: ${signals['vwap']} ({'ABOVE' if signals['above_vwap'] else 'BELOW'})
Volume ratio: {signals['vol_ratio']:.2f}x | RSI: {signals['rsi']:.1f}
ATR: {atr:.3f} ({signals.get('atr_pct', 0):.2f}% of price)
BB %B: {signals.get('bb_pct', 0.5):.0%} | Candle: {signals.get('candle_pattern', 'none').upper()}
Vol-price divergence: {'YES' if signals.get('vol_divergence') else 'NO'}
MACD: {'Bullish' if signals.get('macd_bullish') else 'Bearish'} {signals.get('macd_trend', '')}
Bearish candle: {'YES' if signals.get('bearish_candle') else 'NO'}

ARBITER RULES:
- If 3/3 BUY and market FAVORABLE → "STRONG BUY"
- If 2/3 BUY and market NEUTRAL → "BUY"
- If 1/3 BUY or 2+/3 WATCH → "WATCH"
- If 2+/3 SKIP or market AVOID → "SKIP"
- Override to SKIP if: RSI > 75, price > upper BB, bearish engulfing candle, vol-price divergence + below VWAP
- Set entry near VWAP or current price (use better of the two)
- Stop: exactly 1.2x ATR below entry
- Target: 2.2x ATR above entry (minimum 1:1.8 reward:risk)
- Position size: "full" (3/3 BUY + strong market), "half" (2/3 or weak market), "quarter" (1/3 or disputed)

Respond with ONLY valid JSON (no markdown):
{{"final_verdict": "STRONG BUY|BUY|WATCH|SKIP", "consensus": "unanimous|majority|split|disputed", "entry": <number>, "target": <number>, "stop": <number>, "rr": "<ratio>", "position_size": "full|half|quarter|none", "horizon": "<timeframe>", "reasoning": "<3 sentences synthesizing the debate>", "dissent": "<which agent disagreed and why, or 'none'>", "conviction": "high|medium|low|none", "veto_applied": false}}"""

    try:
        raw = await agent_engine._oneshot_claude(f"{_RISK_SYSTEM}\n\n---\n\n{prompt}")
        result = _extract_json(raw)
        result["agent"] = "arbiter"
        result["ticker"] = ticker
        log.info(
            "RiskArbiter %s: %s | entry=%s target=%s stop=%s | size=%s",
            ticker,
            result.get("final_verdict"),
            result.get("entry"),
            result.get("target"),
            result.get("stop"),
            result.get("position_size"),
        )
        return result
    except Exception as exc:
        log.warning("RiskArbiter failed for %s: %s", ticker, exc)
        mech_plan = _mechanical_plan(signals, 0, "SKIP")
        return {
            "agent":        "arbiter",
            "ticker":       ticker,
            "final_verdict": "SKIP",
            "consensus":    "unknown",
            "entry":        mech_plan["entry"],
            "target":       mech_plan["target"],
            "stop":         mech_plan["stop"],
            "rr":           mech_plan["rr"],
            "position_size": "none",
            "horizon":      "Skip",
            "reasoning":    f"Arbiter error — defaulting to SKIP for safety: {exc}",
            "dissent":      "none",
            "conviction":   "none",
            "veto_applied": True,
            "error":        True,
        }


# ── Per-stock pipeline ────────────────────────────────────────────────────────

async def _run_hive_on_ticker(
    ticker: str,
    client: httpx.AsyncClient,
    news_api_key: str,
    sem: asyncio.Semaphore,
    market_regime: dict,
) -> dict | None:
    """
    Full 4-agent pipeline for a single ticker:
      1. Fetch data
      2. Run Technical, Momentum, Catalyst agents in parallel
      3. Run Risk Arbiter with all 3 verdicts
    """
    async with sem:
        try:
            intraday, daily, headlines, earnings = await asyncio.gather(
                fetch_intraday(ticker, client),
                fetch_daily(ticker, client),
                fetch_news(ticker, news_api_key, client),
                fetch_earnings_data(ticker, client),
            )

            if intraday is None or len(intraday) < 5:
                log.debug("Skipping %s — insufficient intraday data", ticker)
                return None

            signals = compute_signals(intraday, daily)
            if earnings:
                signals.update(earnings)

            # Agents 2–4 run in parallel (independent of each other)
            technical_v, momentum_v, catalyst_v = await asyncio.gather(
                technical_agent(ticker, signals, market_regime),
                momentum_agent(ticker, signals, market_regime),
                catalyst_agent(ticker, headlines, signals, market_regime),
            )

            # Agent 5: arbiter sees all three
            arbiter_v = await risk_arbiter_agent(
                ticker, signals, market_regime,
                technical_v, momentum_v, catalyst_v,
            )

            return {
                "ticker":        ticker,
                "signals":       signals,
                "headlines":     headlines[:3],
                "deliberation":  {
                    "market_regime": market_regime,
                    "technical":     technical_v,
                    "momentum":      momentum_v,
                    "catalyst":      catalyst_v,
                    "arbiter":       arbiter_v,
                },
                "final_verdict": arbiter_v.get("final_verdict", "SKIP"),
                "conviction":    arbiter_v.get("conviction",   "none"),
                "entry":         arbiter_v.get("entry"),
                "target":        arbiter_v.get("target"),
                "stop":          arbiter_v.get("stop"),
                "rr":            arbiter_v.get("rr"),
                "position_size": arbiter_v.get("position_size", "none"),
                "horizon":       arbiter_v.get("horizon", ""),
                "reasoning":     arbiter_v.get("reasoning", ""),
                "consensus":     arbiter_v.get("consensus", "unknown"),
                "dissent":       arbiter_v.get("dissent", "none"),
                "scanned_at":    datetime.utcnow().isoformat(),
            }

        except Exception as exc:
            log.error("Hive pipeline failed for %s: %s", ticker, exc)
            return None
        finally:
            await asyncio.sleep(0.5)  # throttle to avoid hammering Claude CLI


# ── Pre-filter: cheap technical scan (no AI) ──────────────────────────────────

async def _quick_score(ticker: str, client: httpx.AsyncClient, sem: asyncio.Semaphore) -> tuple[str, float]:
    """Return (ticker, score) using the rule-based scorer from hotpicks engine."""
    from engine import score_setup
    async with sem:
        try:
            intraday, daily = await asyncio.gather(
                fetch_intraday(ticker, client),
                fetch_daily(ticker, client),
            )
            if intraday is None or len(intraday) < 5:
                return ticker, -99.0
            signals = compute_signals(intraday, daily)
            # minimal market ctx for quick score — actual regime handled by Agent 1
            ctx = {"regime": "neutral", "spy_chg": 0.0, "qqq_chg": 0.0, "spy_above_vwap": True, "qqq_above_vwap": True}
            score, grade, _, danger = score_setup(signals, ctx)
            # Hard skip: many danger signals or clear skip
            if danger >= 4 or grade == "SKIP":
                return ticker, score
            return ticker, score
        except Exception:
            return ticker, -99.0
        finally:
            await asyncio.sleep(0.2)


# ── Main entry ────────────────────────────────────────────────────────────────

async def run_hive(custom_tickers: list[str] | None = None) -> dict:
    """
    Orchestrates the full hive run:
      Phase 0 — fetch market context
      Phase 1 — MarketRegimeAgent verdict
      Phase 1.5 — quick technical pre-filter (no AI) to pick top candidates
      Phase 2 — run 4-agent hive on each candidate (3 parallel + 1 arbiter)
    """
    tickers      = custom_tickers or HIVE_UNIVERSE
    news_api_key = os.getenv("NEWS_API_KEY", "")

    async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:

        # Phase 0: market data
        market_ctx = await fetch_market_context(client)
        log.info("Market: %s | SPY %+.2f%% | QQQ %+.2f%%",
                 market_ctx["regime"].upper(), market_ctx["spy_chg"], market_ctx["qqq_chg"])

        # Phase 1: MarketRegime agent
        log.info("Phase 1: MarketRegimeAgent assessing conditions…")
        market_regime = await market_regime_agent(market_ctx)
        log.info("Market verdict: %s — %s", market_regime["verdict"], market_regime.get("reasoning", ""))

        # Phase 1.5: quick technical pre-filter — exclude obvious SKIPs before calling AI
        log.info("Phase 1.5: quick technical pre-filter on %d tickers…", len(tickers))
        quick_sem    = asyncio.Semaphore(6)
        quick_tasks  = [_quick_score(t, client, quick_sem) for t in tickers]
        quick_scores = await asyncio.gather(*quick_tasks, return_exceptions=True)

        scored = [(t, s) for t, s in quick_scores if isinstance(s, float) and s > -5.0]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top candidates for full hive treatment
        max_candidates = 6 if market_regime["verdict"] in ("FAVORABLE", "NEUTRAL") else 4
        candidates = [t for t, _ in scored[:max_candidates]]
        log.info("Candidates for hive: %s", candidates)

        # Phase 2: full 4-agent hive on each candidate
        hive_sem = asyncio.Semaphore(2)  # max 2 stocks' Claude calls at once
        log.info("Phase 2: running 4-agent hive on %d candidates…", len(candidates))
        hive_tasks = [
            _run_hive_on_ticker(t, client, news_api_key, hive_sem, market_regime)
            for t in candidates
        ]
        hive_results = await asyncio.gather(*hive_tasks, return_exceptions=True)

    results = [r for r in hive_results if r and not isinstance(r, Exception)]

    # Sort: STRONG BUY → BUY → WATCH → SKIP, then by conviction
    _order = {"STRONG BUY": 0, "BUY": 1, "WATCH": 2, "SKIP": 3}
    results.sort(key=lambda x: (_order.get(x["final_verdict"], 4), -["high", "medium", "low", "none"].index(x.get("conviction", "none"))))

    strong_buys = [r for r in results if r["final_verdict"] == "STRONG BUY"]
    buys        = [r for r in results if r["final_verdict"] == "BUY"]
    watches     = [r for r in results if r["final_verdict"] == "WATCH"]
    skips       = [r for r in results if r["final_verdict"] == "SKIP"]

    return {
        "run_at":        datetime.utcnow().isoformat(),
        "total_scanned": len(tickers),
        "candidates":    len(candidates),
        "market_regime": market_regime,
        "strong_buys":   strong_buys,
        "buys":          buys,
        "watches":       watches,
        "skips":         skips,
        "all_results":   results,
    }


async def analyze_single_hive(ticker: str) -> dict:
    """On-demand full hive analysis for any single ticker."""
    ticker       = ticker.upper().strip()
    news_api_key = os.getenv("NEWS_API_KEY", "")

    async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
        market_ctx    = await fetch_market_context(client)
        market_regime = await market_regime_agent(market_ctx)
        sem           = asyncio.Semaphore(1)
        result        = await _run_hive_on_ticker(ticker, client, news_api_key, sem, market_regime)

    if result is None:
        return {"error": f"Could not fetch data for '{ticker}'. Check the symbol and try again."}

    result["market_regime"] = market_regime
    return result
