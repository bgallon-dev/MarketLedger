#!/usr/bin/env python3
"""
Discovery Chronicle Pipeline

Runs 8 disconfirmation pattern detectors against every selected pick (or a
specified subset) and stores flagged signal-disclosure tensions in the
discovery_entries table.

Usage:
    py dashboard/brain/discovery_pipeline.py                     # selected picks only
    py dashboard/brain/discovery_pipeline.py --tickers MNR AAPL  # specific tickers
    py dashboard/brain/discovery_pipeline.py --all-tickers        # full DB universe
    py dashboard/brain/discovery_pipeline.py --verbose
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

_THIS_DIR = Path(__file__).parent
_MARKET_LEDGER = _THIS_DIR.parent.parent
_DB_PATH = _MARKET_LEDGER / "database" / "financial_data.db"
_YAHOO_PROJECT = _MARKET_LEDGER.parent
_DECISION_LOG_PATH = _YAHOO_PROJECT / "buy_list_with_projections_decision_log.csv"

sys.path.insert(0, str(_MARKET_LEDGER))

from dashboard.brain.discovery import run_all_detectors
from database.database import get_financial_data_bulk


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table() -> None:
    conn = _db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS discovery_entries (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker              TEXT NOT NULL,
            run_date            TEXT NOT NULL,
            pattern_type        TEXT NOT NULL,
            severity            TEXT NOT NULL,
            investment_signal   TEXT,
            mos_pct             REAL,
            source_type         TEXT NOT NULL,
            source_form_type    TEXT,
            source_filing_date  TEXT,
            excerpt             TEXT,
            quant_metric        TEXT,
            quant_value         REAL,
            quant_threshold     REAL,
            discovered_at       TEXT NOT NULL,
            UNIQUE (ticker, run_date, pattern_type, source_type, source_filing_date)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_discovery_ticker ON discovery_entries(ticker)"
    )
    conn.commit()
    conn.close()


def _load_tickers_from_decision_log(
    path: Path, all_tickers: bool = False
) -> list:
    """Return tickers from the decision log. If all_tickers=False, selected picks only."""
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if all_tickers:
        return df["Ticker"].dropna().str.upper().unique().tolist()
    # Selected = passed all gates
    selected_mask = (
        df.get("Decision", pd.Series()).str.lower().isin(["selected", "buy"])
        | df.get("Investment Signal", pd.Series()).str.contains("Buy", na=False, case=False)
    )
    return df.loc[selected_mask, "Ticker"].dropna().str.upper().unique().tolist()


def _load_filing_data(ticker: str) -> list:
    """Return last 5 cached filings (10-K + 10-Q) for ticker from sec_filings."""
    conn = _db()
    try:
        rows = conn.execute(
            """
            SELECT filed_date, form_type, item_1a_text, item_7_text
            FROM sec_filings
            WHERE ticker = ?
            ORDER BY filed_date DESC
            LIMIT 5
            """,
            (ticker.upper(),),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _load_signal_meta(ticker: str, decision_log_path: Path) -> tuple:
    """Return (investment_signal, mos_pct) for ticker from decision log."""
    if not decision_log_path.exists():
        return None, None
    try:
        df = pd.read_csv(decision_log_path)
        row = df[df["Ticker"].str.upper() == ticker.upper()]
        if row.empty:
            return None, None
        sig = row.iloc[0].get("Investment Signal")
        mos = row.iloc[0].get("Undervalued %")
        return (str(sig) if pd.notna(sig) else None,
                float(mos) if pd.notna(mos) else None)
    except Exception:
        return None, None


def _write_entries(entries: list) -> int:
    if not entries:
        return 0
    conn = _db()
    written = 0
    try:
        for e in entries:
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO discovery_entries
                        (ticker, run_date, pattern_type, severity,
                         investment_signal, mos_pct,
                         source_type, source_form_type, source_filing_date,
                         excerpt, quant_metric, quant_value, quant_threshold,
                         discovered_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        e["ticker"], e["run_date"], e["pattern_type"], e["severity"],
                        e.get("investment_signal"), e.get("mos_pct"),
                        e["source_type"], e.get("source_form_type"),
                        e.get("source_filing_date"),
                        e.get("excerpt"), e.get("quant_metric"),
                        e.get("quant_value"), e.get("quant_threshold"),
                        e["discovered_at"],
                    ),
                )
                written += 1
            except sqlite3.IntegrityError:
                pass  # Already exists (UNIQUE constraint)
        conn.commit()
    finally:
        conn.close()
    return written


def run_discovery_pipeline(
    tickers: Optional[list] = None,
    all_tickers: bool = False,
    verbose: bool = False,
) -> int:
    """Run all detectors. Returns total entries written."""
    _ensure_table()

    run_date = datetime.now().strftime("%Y-%m-%d")

    if tickers:
        symbols = [t.upper() for t in tickers]
    elif all_tickers:
        conn = _db()
        rows = conn.execute("SELECT symbol FROM tickers").fetchall()
        conn.close()
        symbols = [r["symbol"] for r in rows]
    else:
        symbols = _load_tickers_from_decision_log(_DECISION_LOG_PATH, all_tickers=False)
        if not symbols:
            print("No selected tickers found in decision log. Use --all-tickers to run on full universe.")
            return 0

    if verbose:
        print(f"Running discovery pass on {len(symbols)} tickers (run_date={run_date})...")

    # Bulk load financial data
    cf_metrics = [
        "FreeCashFlow", "CashDividendsPaid", "CommonStockDividendsPaid",
        "PaymentOfDividends", "RepurchaseOfCapitalStock", "RepurchaseOfCommonStock",
        "CommonStockRepurchase", "StockRepurchase",
    ]
    bs_metrics = ["TotalDebt", "LongTermDebt", "TotalAssets"]
    cf_long = get_financial_data_bulk(symbols, "cash_flow", metrics=cf_metrics)
    bs_long = get_financial_data_bulk(symbols, "balance_sheet", metrics=bs_metrics)

    total_written = 0
    for ticker in symbols:
        filing_data = _load_filing_data(ticker)
        inv_signal, mos_pct = _load_signal_meta(ticker, _DECISION_LOG_PATH)

        entries = run_all_detectors(
            ticker=ticker,
            run_date=run_date,
            cf_long=cf_long,
            bs_long=bs_long,
            filing_data=filing_data,
            investment_signal=inv_signal,
            mos_pct=mos_pct,
        )

        written = _write_entries(entries)
        total_written += written

        if verbose and entries:
            sev_counts = {}
            for e in entries:
                sev_counts[e["severity"]] = sev_counts.get(e["severity"], 0) + 1
            print(f"  {ticker:8s}  {written} tensions  {sev_counts}")
        elif verbose:
            print(f"  {ticker:8s}  clean")

    print(f"Discovery pass complete: {total_written} tensions written for {len(symbols)} tickers.")
    return total_written


def main():
    parser = argparse.ArgumentParser(
        description="Run disconfirmation pattern detectors on selected picks."
    )
    parser.add_argument("--tickers", nargs="+", metavar="TICKER",
                        help="Specific tickers to scan (default: selected picks from decision log)")
    parser.add_argument("--all-tickers", action="store_true",
                        help="Scan full DB universe regardless of pipeline selection")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_discovery_pipeline(
        tickers=args.tickers,
        all_tickers=args.all_tickers,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
