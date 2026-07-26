#!/usr/bin/env python3
"""
TradePulse Market Scanner — Unusual Volume for Equities
Equivalent of the options V/OI scanner but for regular stock trades.
Stocks trading well above their 20-day average daily volume bubble to the top.

Requirements:  pip install httpx rich
Source:        Yahoo Finance v8 chart API (no auth required)

Usage:
    python scanner.py                          # default universe, top 25
    python scanner.py --top 30                 # show 30 rows
    python scanner.py --refresh 20             # refresh every 20 s (default: 30)
    python scanner.py --min-ratio 2.0          # only V/AvgVol >= 2x
    python scanner.py --sort gainers           # biggest gainers today
    python scanner.py --sort losers            # biggest losers today
    python scanner.py --min-price 5            # exclude sub-$5 tickers
    python scanner.py --tickers NVDA MSTR AAPL # scan specific tickers only
    python scanner.py --universe tickers.txt   # one ticker per line in file
    python scanner.py --names                  # add company name column
    python scanner.py --fullscreen             # clear terminal on each refresh
    python scanner.py --legend                 # show colour legend at startup
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import httpx
    from rich import box
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("Missing packages.  Run:  pip install httpx rich")
    sys.exit(1)

console = Console()

# ── Default scan universe ──────────────────────────────────────────────────────
DEFAULT_UNIVERSE = [
    # Mega-cap
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","UNH","V",
    "XOM","LLY","JNJ","WMT","MA","AVGO","HD","CVX","MRK","ABBV",
    "COST","PEP","KO","ADBE","CRM","ACN","AMD","TMO","LIN","MCD",
    "DHR","ABT","DIS","CMCSA","TXN","NEE","VZ","INTC","WFC","BAC",
    "GS","MS","C","SCHW","AXP","BX","USB","PGR","TRV","LOW",
    # High-beta / momentum
    "MSTR","PLTR","SOFI","RIVN","LCID","HOOD","RBLX","SNAP","COIN",
    "MARA","RIOT","CLSK","CORZ","WULF","IREN","BTBT","HUT",
    "AAOI","SMCI","ALAB","CCJ","APP","ON","MU","QCOM","AMAT","ARM",
    "LRCX","KLAC","ENTG","MRVL","TSM","NFLX","UBER","ABNB","DASH",
    # Leveraged ETFs (often show spikes)
    "SOXS","SOXL","TQQQ","SQQQ","UVXY","SPXL","SPXS","LABU","LABD","TNA",
    "NVDL","TSLL","MSTU","MSTX",
    # Broad ETFs
    "SPY","QQQ","IWM","DIA","GLD","SLV","TLT","HYG","VXX",
    "XLF","XLK","XLE","XLV","XLI","ARKK","ARKG",
    # Growth / tech
    "PINS","SPOT","ZM","DDOG","NET","SNOW","MDB","FTNT","PANW","CRWD",
    "ZS","OKTA","DOCN","ENPH","FSLR","PLUG","CHPT",
    # Commodities / materials
    "NEM","FCX","GOLD","SLB","HAL","OXY","DVN","FANG","X","CLF",
    "AA","NUE","STLD","RS","ATI",
]

_YF_HDR = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, */*",
}


# ── Data fetching via v8 chart API (no auth) ───────────────────────────────────

async def _fetch_one(client: httpx.AsyncClient, ticker: str) -> dict | None:
    """
    Fetch 1-month daily data for ticker.
    Uses the v8 chart endpoint which needs no authentication.
    Computes 20-day avg volume from historical bars.
    """
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&range=1mo"
    )
    try:
        r = await client.get(url)
        if r.status_code != 200:
            return None

        result = r.json().get("chart", {}).get("result", [])
        if not result:
            return None

        res        = result[0]
        meta       = res.get("meta", {})
        indicators = res.get("indicators", {}).get("quote", [{}])[0]

        closes  = [c for c in (indicators.get("close")  or []) if c is not None]
        volumes = [v for v in (indicators.get("volume") or []) if v is not None]

        cur_price = float(meta.get("regularMarketPrice") or 0)
        today_vol = int(meta.get("regularMarketVolume")  or 0)

        if not cur_price or not today_vol:
            return None

        # Yesterday's close is the second-to-last bar (last bar = today, partial)
        prev_close = closes[-2] if len(closes) >= 2 else (closes[-1] if closes else cur_price)
        chg_abs    = cur_price - prev_close
        chg_pct    = chg_abs / prev_close * 100 if prev_close else 0

        # 20-day average volume excludes today's partial bar
        hist_vols = volumes[:-1] if len(volumes) > 1 else volumes
        avg_vol   = int(sum(hist_vols[-20:]) / len(hist_vols[-20:])) if hist_vols else None
        v_ratio   = round(today_vol / avg_vol, 2) if avg_vol and avg_vol > 0 else None

        hi52    = float(meta.get("fiftyTwoWeekHigh") or 0) or None
        lo52    = float(meta.get("fiftyTwoWeekLow")  or 0) or None
        day_hi  = float(meta.get("regularMarketDayHigh") or 0) or None
        day_lo  = float(meta.get("regularMarketDayLow")  or 0) or None

        return {
            "ticker":   ticker,
            "name":     (meta.get("shortName") or meta.get("longName") or ticker)[:24],
            "exchange": meta.get("exchangeName", ""),
            "price":    cur_price,
            "chg_pct":  chg_pct,
            "chg_abs":  chg_abs,
            "volume":   today_vol,
            "avg_vol":  avg_vol,
            "v_ratio":  v_ratio,
            "hi52":     hi52,
            "lo52":     lo52,
            "day_hi":   day_hi,
            "day_lo":   day_lo,
            "days":     len(hist_vols),     # how many history bars we have
        }
    except Exception:
        return None


async def fetch_all(tickers: list[str], min_price: float) -> list[dict]:
    """Fetch all tickers in parallel, batched to stay friendly to Yahoo."""
    results: list[dict] = []
    batch_size = 20
    async with httpx.AsyncClient(headers=_YF_HDR, timeout=12) as client:
        for i in range(0, len(tickers), batch_size):
            batch   = tickers[i : i + batch_size]
            fetched = await asyncio.gather(*[_fetch_one(client, t) for t in batch])
            results.extend(r for r in fetched if r and r["price"] >= min_price)
            if i + batch_size < len(tickers):
                await asyncio.sleep(0.15)
    return results


# ── Formatters ────────────────────────────────────────────────────────────────

def _fvol(n: int | None) -> str:
    if n is None: return "—"
    if n >= 1_000_000_000: return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:     return f"{n/1_000_000:.2f}M"
    if n >= 1_000:         return f"{n/1_000:.0f}K"
    return str(n)

def _fprice(p: float) -> str:
    if p >= 10_000: return f"${p:,.0f}"
    if p >= 100:    return f"${p:.2f}"
    if p >= 10:     return f"${p:.2f}"
    return          f"${p:.3f}"

def _f52w(price: float, lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None or hi <= lo: return "—"
    pct = (price - lo) / (hi - lo) * 100
    bar = "▓" * int(pct / 10) + "░" * (10 - int(pct / 10))
    return f"{pct:.0f}%"

def _ratio_style(r: float | None) -> str:
    if r is None:  return "bright_black"
    if r >= 10:    return "bold red"
    if r >= 5:     return "bold dark_orange3"
    if r >= 3:     return "bold yellow"
    if r >= 2:     return "yellow"
    if r >= 1.5:   return "white"
    return "bright_black"

def _chg_style(pct: float) -> str:
    if pct >  5:   return "bold bright_green"
    if pct >  0:   return "green"
    if pct < -5:   return "bold red"
    if pct <  0:   return "red"
    return "bright_black"


# ── Table builder ─────────────────────────────────────────────────────────────

def build_table(
    rows:      list[dict],
    top:       int,
    min_ratio: float,
    sort_by:   str,
    show_name: bool,
) -> Table:

    # Filter by minimum ratio
    filtered = [r for r in rows if (r["v_ratio"] or 0) >= min_ratio]

    # Sort
    key_fn = {
        "ratio":   lambda r: r["v_ratio"] or 0,
        "volume":  lambda r: r["volume"],
        "gainers": lambda r: r["chg_pct"],
        "losers":  lambda r: -r["chg_pct"],
        "change":  lambda r: abs(r["chg_pct"]),
    }.get(sort_by, lambda r: r["v_ratio"] or 0)

    filtered.sort(key=key_fn, reverse=True)
    filtered = filtered[:top]

    now   = datetime.now().strftime("%H:%M:%S")
    title = (
        f"[bold green]UNUSUAL VOLUME — EQUITIES[/]  "
        f"[bright_black](sorted by {sort_by.upper()} · {now})[/]"
    )
    table = Table(
        title=title,
        box=box.SIMPLE_HEAD,
        border_style="bright_black",
        header_style="bold bright_black",
        title_style="bold",
        show_lines=False,
        expand=False,
    )

    table.add_column("Rank",    justify="right",  width=4,  style="bright_black", no_wrap=True)
    table.add_column("Symbol",  justify="left",   width=7,  style="bold white",   no_wrap=True)
    if show_name:
        table.add_column("Name", justify="left",  width=24, style="bright_black", no_wrap=True)
    table.add_column("Price",   justify="right",  width=10, style="white",        no_wrap=True)
    table.add_column("Chg %",   justify="right",  width=8,                        no_wrap=True)
    table.add_column("Chg $",   justify="right",  width=9,                        no_wrap=True)
    table.add_column("Volume",  justify="right",  width=9,  style="white",        no_wrap=True)
    table.add_column("AvgVol",  justify="right",  width=9,  style="bright_black", no_wrap=True)
    table.add_column("V/Avg",   justify="right",  width=7,                        no_wrap=True)
    table.add_column("Day Hi",  justify="right",  width=9,  style="bright_black", no_wrap=True)
    table.add_column("Day Lo",  justify="right",  width=9,  style="bright_black", no_wrap=True)
    table.add_column("52W %",   justify="right",  width=6,  style="bright_black", no_wrap=True)

    for rank, row in enumerate(filtered, 1):
        pct     = row["chg_pct"]
        vr      = row["v_ratio"]
        sign    = "+" if pct >= 0 else ""
        abs_s   = "+" if row["chg_abs"] >= 0 else ""

        cells = [
            str(rank),
            row["ticker"],
        ]
        if show_name:
            cells.append(row["name"])
        cells += [
            _fprice(row["price"]),
            Text(f"{sign}{pct:.2f}%",             style=_chg_style(pct)),
            Text(f"{abs_s}{_fprice(abs(row['chg_abs']))}", style=_chg_style(pct)),
            Text(_fvol(row["volume"]),             style="white"),
            _fvol(row["avg_vol"]),
            Text(f"{vr:.1f}x" if vr else "—",     style=_ratio_style(vr)),
            _fprice(row["day_hi"]) if row["day_hi"] else "—",
            _fprice(row["day_lo"]) if row["day_lo"] else "—",
            _f52w(row["price"], row["lo52"], row["hi52"]),
        ]
        table.add_row(*cells)

    # Caption / footer
    total   = len(rows)
    shown   = len(filtered)
    gainers = sum(1 for r in filtered if r["chg_pct"] > 0)
    losers  = sum(1 for r in filtered if r["chg_pct"] < 0)
    below   = sum(1 for r in rows if (r["v_ratio"] or 0) < min_ratio)

    table.caption = (
        f"[bright_black]Showing {shown}/{total} scanned  ·  "
        f"[green]{gainers} ▲[/]  [red]{losers} ▼[/]  ·  "
        f"{below} below {min_ratio}x threshold  ·  20-day avg vol  ·  Ctrl+C to exit[/bright_black]"
    )
    return table


# ── Legend ────────────────────────────────────────────────────────────────────

def print_legend() -> None:
    console.rule("[bold]V/Avg colour guide[/]")
    console.print("  [bold red]≥ 10x[/]  extreme spike — potential halt, earnings, major news")
    console.print("  [bold dark_orange3]≥  5x[/]  very high  — strong directional bet or news catalyst")
    console.print("  [bold yellow]≥  3x[/]  elevated  — worth watching closely")
    console.print("  [yellow]≥  2x[/]  above average — moderate unusual activity")
    console.print("  [white]≥  1.5x[/]  mild uptick in interest")
    console.rule("[bold]Column guide[/]")
    console.print("  V/Avg  = Today's volume ÷ 20-day average daily volume")
    console.print("  52W %  = Price position in 52-week range (0% = 52W low, 100% = 52W high)")
    console.print("  Chg $  = Absolute dollar change vs. yesterday's close")
    console.print("  AvgVol = 20-trading-day average (1-month window)")
    console.rule()
    console.print()


# ── Main loop ─────────────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    # Build universe
    if args.universe:
        path = Path(args.universe)
        if not path.exists():
            console.print(f"[red]File not found: {path}[/]")
            sys.exit(1)
        tickers = [l.strip().upper() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = DEFAULT_UNIVERSE

    tickers = list(dict.fromkeys(tickers))   # deduplicate, preserve order

    console.print(
        f"\n[bold green]TradePulse Market Scanner[/]  "
        f"[bright_black]universe: {len(tickers)} tickers  ·  "
        f"refresh: {args.refresh}s  ·  "
        f"min V/Avg: {args.min_ratio}x  ·  "
        f"sort: {args.sort}[/]\n"
    )
    if args.legend:
        print_legend()

    console.print("[bright_black]Fetching initial data…[/]\n")

    with Live(
        console=console,
        refresh_per_second=0.5,
        screen=args.fullscreen,
        vertical_overflow="visible",
    ) as live:
        while True:
            t0   = time.monotonic()
            rows = await fetch_all(tickers, args.min_price)

            if not rows:
                live.update(
                    Text(
                        "⚠  No data returned — check connection or Yahoo Finance rate limits.",
                        style="red",
                    )
                )
            else:
                table = build_table(
                    rows,
                    top       = args.top,
                    min_ratio = args.min_ratio,
                    sort_by   = args.sort,
                    show_name = args.names,
                )
                live.update(table)

            wait = max(0.0, args.refresh - (time.monotonic() - t0))
            await asyncio.sleep(wait)


def main() -> None:
    p = argparse.ArgumentParser(
        description="TradePulse Unusual Volume Scanner for equities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--tickers",   nargs="+", metavar="T", help="Specific tickers to scan (replaces default universe)")
    p.add_argument("--universe",  metavar="FILE",          help="Text file of tickers, one per line")
    p.add_argument("--top",       type=int,   default=25,  help="Rows to show (default: 25)")
    p.add_argument("--refresh",   type=int,   default=30,  help="Refresh interval seconds (default: 30)")
    p.add_argument("--min-ratio", type=float, default=1.0, dest="min_ratio",
                   help="Minimum V/AvgVol to display (default: 1.0)")
    p.add_argument("--min-price", type=float, default=0.5, dest="min_price",
                   help="Minimum stock price (default: $0.50)")
    p.add_argument("--sort",      choices=["ratio","volume","gainers","losers","change"],
                   default="ratio",  help="Sort column (default: ratio)")
    p.add_argument("--names",     action="store_true", help="Show company name column")
    p.add_argument("--legend",    action="store_true", help="Print colour legend at startup")
    p.add_argument("--fullscreen",action="store_true", help="Fullscreen/clear mode")

    args = p.parse_args()

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        console.print("\n[bright_black]Scanner stopped.[/]")


if __name__ == "__main__":
    main()
