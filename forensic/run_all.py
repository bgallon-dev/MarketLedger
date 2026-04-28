"""
Standalone forensic scanner — runs the Capital Allocation Archaeologist, Sector
Contagion Tracer, and/or Longitudinal Credibility Scorer on every ticker in the
database (or a specified subset).

The three modules write to research_outputs/forensic_all.csv by default.

Usage:
    py -m forensic.run_all                         # all three modules, all DB tickers
    py -m forensic.run_all --archaeology-only      # fast path, no EDGAR calls
    py -m forensic.run_all --no-contagion          # skip EDGAR-heavy contagion scan
    py -m forensic.run_all --tickers AAPL MSFT     # specific tickers only
    py -m forensic.run_all --output my_scan.csv    # custom output path
    py -m forensic.run_all --max-workers 8         # tune parallelism

Notes:
  - Archaeology is fast (uses existing DB data, no network).
  - Contagion and credibility fetch SEC EDGAR filings on first run; results are
    cached in the SQLite DB, so subsequent runs are near-instant.
  - On first run across 6 000+ tickers, contagion can take 30–60 min due to
    EDGAR rate limits (10 req/sec). Run --archaeology-only first to get
    immediate results, then add --no-archaeology later to fill in EDGAR data.
  - Contagion needs all sector peers present to compute leadership scores, so
    it always runs on the full ticker list even when --tickers is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Ensure project root is on sys.path when run as a module or directly
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from database.database import (
    get_connection,
    get_financial_data_bulk,
    get_ticker_history_bulk,
    get_ticker_sectors,
)


# ── Helpers mirroring main.py private functions ────────────────────────────────

def _history_map_from_bulk(df: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    if df is None or df.empty:
        return {}
    out: Dict[str, pd.DataFrame] = {}
    work = df.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    for symbol, grp in work.groupby("symbol", sort=False):
        cols = [c for c in grp.columns if c != "symbol"]
        out[str(symbol)] = grp[cols].sort_values("date").reset_index(drop=True)
    return out


def _financial_map_from_bulk(df: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    if df is None or df.empty:
        return {}
    out: Dict[str, pd.DataFrame] = {}
    work = df.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    for symbol, grp in work.groupby("symbol", sort=False):
        out[str(symbol)] = grp.pivot(index="metric", columns="period", values="value")
    return out


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _get_tickers_with_data() -> List[str]:
    """All tickers that have at least balance sheet data in the DB."""
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT DISTINCT t.symbol
            FROM tickers t
            JOIN balance_sheet b ON b.ticker_id = t.id
            ORDER BY t.symbol
        """).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def _build_prefetched(symbols: List[str]) -> Dict[str, Any]:
    """Load financial data for a single chunk of symbols."""
    history = get_ticker_history_bulk(symbols)
    balance_sheet = get_financial_data_bulk(symbols, "balance_sheet")
    income_statement = get_financial_data_bulk(symbols, "income_statement")
    cash_flow = get_financial_data_bulk(symbols, "cash_flow")
    sectors = get_ticker_sectors(symbols)
    return {
        "history_by_symbol": _history_map_from_bulk(history),
        "balance_sheet_by_symbol": _financial_map_from_bulk(balance_sheet),
        "income_statement_by_symbol": _financial_map_from_bulk(income_statement),
        "cash_flow_by_symbol": _financial_map_from_bulk(cash_flow),
        "sectors": sectors,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def run_all_forensics(
    tickers: Optional[List[str]] = None,
    run_archaeology: bool = True,
    run_contagion: bool = True,
    run_credibility: bool = True,
    output_path: Optional[str] = None,
    max_workers: Optional[int] = None,
    chunk_size: int = 300,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run selected forensic modules and write results to a CSV.

    Archaeology and credibility load financial data in chunks to stay within
    memory limits. Contagion always runs on the full universe (sectors only,
    no financial data) so that sector peer groups are complete.

    Parameters
    ----------
    tickers:
        Explicit list of tickers. If None, uses all tickers with DB data.
    run_archaeology / run_contagion / run_credibility:
        Toggle individual modules.
    output_path:
        Destination CSV. Defaults to research_outputs/forensic_all.csv.
    max_workers:
        Parallel workers passed to each module. None = auto.
    chunk_size:
        Number of tickers to load financial data for at a time (default 300).
    verbose:
        Print progress.

    Returns
    -------
    DataFrame with all forensic columns merged on Ticker.
    """
    # ── Resolve ticker list ────────────────────────────────────────────────────
    if tickers:
        scan_tickers = [t.strip().upper() for t in tickers]
        if verbose:
            print(f"[forensic-all] Scanning {len(scan_tickers)} specified tickers.")
    else:
        if verbose:
            print("[forensic-all] Loading all tickers with DB data...")
        scan_tickers = _get_tickers_with_data()
        if verbose:
            print(f"[forensic-all] Found {len(scan_tickers)} tickers.")

    candidates_df = pd.DataFrame({"Ticker": scan_tickers})
    result_df = candidates_df.copy()

    # ── 1 & 3. Archaeology + Credibility — chunked financial data load ─────────
    need_financials = run_archaeology or run_credibility
    if need_financials:
        if run_archaeology:
            from forensic.capital_archaeologist import run_capital_archaeology_scan
        if run_credibility:
            from forensic.credibility_scorer import run_credibility_scan

        arch_parts: List[pd.DataFrame] = []
        cred_parts: List[pd.DataFrame] = []
        n_chunks = (len(scan_tickers) + chunk_size - 1) // chunk_size

        for i in range(0, len(scan_tickers), chunk_size):
            chunk = scan_tickers[i : i + chunk_size]
            chunk_num = i // chunk_size + 1
            if verbose:
                print(
                    f"\n[forensic-all] Chunk {chunk_num}/{n_chunks} "
                    f"({len(chunk)} tickers) — loading financial data..."
                )
            prefetched = _build_prefetched(chunk)
            chunk_df = pd.DataFrame({"Ticker": chunk})

            if run_archaeology:
                arch_parts.append(
                    run_capital_archaeology_scan(
                        chunk_df,
                        prefetched=prefetched,
                        verbose=verbose,
                        max_workers=max_workers,
                    )
                )

            if run_credibility:
                cred_parts.append(
                    run_credibility_scan(
                        chunk_df,
                        prefetched=prefetched,
                        verbose=verbose,
                        max_workers=max_workers,
                    )
                )

        if run_archaeology and arch_parts:
            arch_df = pd.concat(arch_parts, ignore_index=True)
            result_df = result_df.merge(arch_df, on="Ticker", how="left")

        if run_credibility and cred_parts:
            cred_df = pd.concat(cred_parts, ignore_index=True)
            result_df = result_df.merge(cred_df, on="Ticker", how="left")

    # ── 2. Sector Contagion Tracer — full universe, sectors only ──────────────
    # Needs all sector peers present simultaneously for leadership scores.
    # Uses only the sectors dict (no financial data) so memory is trivial.
    if run_contagion:
        if verbose:
            print("\n[forensic-all] Running Sector Contagion Tracer (full universe)...")
            if tickers:
                print(
                    "  Note: using all DB tickers for sector peer groups; "
                    "results filtered back to requested tickers."
                )
        all_tickers = scan_tickers if not tickers else _get_tickers_with_data()
        full_candidates = pd.DataFrame({"Ticker": all_tickers})
        contagion_prefetched = {"sectors": get_ticker_sectors(all_tickers)}

        from forensic.contagion import run_contagion_scan
        contagion_df = run_contagion_scan(
            full_candidates,
            prefetched=contagion_prefetched,
            verbose=verbose,
            max_workers=max_workers,
        )
        contagion_subset = contagion_df[contagion_df["Ticker"].isin(candidates_df["Ticker"])]
        result_df = result_df.merge(contagion_subset, on="Ticker", how="left")

    # ── Write output ──────────────────────────────────────────────────────────
    if output_path is None:
        out_dir = _ROOT / "research_outputs"
        out_dir.mkdir(exist_ok=True)
        output_path = str(out_dir / "forensic_all.csv")

    result_df.to_csv(output_path, index=False)
    if verbose:
        print(f"\n[forensic-all] Done — {len(result_df)} rows written to {output_path}")

    return result_df


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run forensic modules on all DB tickers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Specific tickers to scan (default: all tickers with DB data)",
    )
    parser.add_argument(
        "--no-archaeology", action="store_true",
        help="Skip Capital Allocation Archaeologist",
    )
    parser.add_argument(
        "--no-contagion", action="store_true",
        help="Skip Sector Contagion Tracer (suppresses all EDGAR calls for contagion)",
    )
    parser.add_argument(
        "--no-credibility", action="store_true",
        help="Skip Longitudinal Credibility Scorer",
    )
    parser.add_argument(
        "--archaeology-only", action="store_true",
        help="Run only the Capital Allocation Archaeologist (fast, no EDGAR)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=None, metavar="N",
        help="Max parallel workers per module (default: auto)",
    )
    parser.add_argument(
        "--output", metavar="PATH",
        help="Output CSV path (default: research_outputs/forensic_all.csv)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    run_arch = not args.no_archaeology
    run_cont = not args.no_contagion
    run_cred = not args.no_credibility

    if args.archaeology_only:
        run_cont = False
        run_cred = False

    run_all_forensics(
        tickers=args.tickers,
        run_archaeology=run_arch,
        run_contagion=run_cont,
        run_credibility=run_cred,
        output_path=args.output,
        max_workers=args.max_workers,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
