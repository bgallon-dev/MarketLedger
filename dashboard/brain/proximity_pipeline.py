#!/usr/bin/env python3
"""
Proximity Engine Pipeline

Builds or updates the proximity_vectors index: a state-vector per ticker-quarter
encoding structural financial characteristics. Also computes 12M forward price
returns and dividend-cut indicators for each historical period.

Usage:
    py dashboard/brain/proximity_pipeline.py                          # full universe
    py dashboard/brain/proximity_pipeline.py --tickers MNR KMI WMB   # subset
    py dashboard/brain/proximity_pipeline.py --rebuild                # force recompute all
    py dashboard/brain/proximity_pipeline.py --verbose
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).parent
_MARKET_LEDGER = _THIS_DIR.parent.parent
_DB_PATH = _MARKET_LEDGER / "database" / "financial_data.db"

sys.path.insert(0, str(_MARKET_LEDGER))

from dashboard.brain.proximity import build_state_vector, _vec_to_blob, _FALLBACKS


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table() -> None:
    conn = _db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS proximity_vectors (
            ticker          TEXT NOT NULL,
            period          TEXT NOT NULL,
            state_vector    BLOB NOT NULL,
            fwd_return_12m  REAL,
            div_cut         INTEGER,
            computed_at     TEXT NOT NULL,
            PRIMARY KEY (ticker, period)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_proximity_ticker ON proximity_vectors(ticker)"
    )
    conn.commit()
    conn.close()


# ── Period enumeration ─────────────────────────────────────────────────────────

def _periods_for_ticker(conn: sqlite3.Connection, ticker: str) -> list:
    """Return unique YYYY-QN periods for a ticker from balance_sheet data."""
    try:
        df = pd.read_sql(
            """
            SELECT DISTINCT f.period
            FROM balance_sheet f
            JOIN tickers t ON f.ticker_id = t.id
            WHERE t.symbol = ?
            ORDER BY f.period DESC
            """,
            conn, params=(ticker.upper(),),
        )
    except Exception:
        return []
    if df.empty:
        return []

    periods = []
    seen = set()
    for raw in df["period"]:
        try:
            dt = datetime.strptime(raw[:10], "%Y-%m-%d")
            q = (dt.month - 1) // 3 + 1
            key = f"{dt.year}-Q{q}"
            if key not in seen:
                seen.add(key)
                periods.append(key)
        except Exception:
            continue
    return periods


# ── Forward outcome computation ────────────────────────────────────────────────

def _period_end_date(period: str) -> Optional[str]:
    try:
        year, q = period.split("-Q")
        month = int(q) * 3
        return f"{year}-{month:02d}-28"
    except Exception:
        return None


def _compute_fwd_return_12m(
    ticker: str, period: str, conn: sqlite3.Connection
) -> Optional[float]:
    """12M price return starting from approximate period end."""
    end_date = _period_end_date(period)
    if not end_date:
        return None
    try:
        fwd_date = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=365)).strftime("%Y-%m-%d")
        rows = conn.execute(
            """
            SELECT h.date, h.close
            FROM history h
            JOIN tickers t ON h.ticker_id = t.id
            WHERE t.symbol = ? AND (h.date >= ? OR h.date >= ?)
            ORDER BY h.date ASC
            LIMIT 2
            """,
            (ticker.upper(), end_date, fwd_date),
        ).fetchall()
        if len(rows) < 1:
            return None
        # Find start and end prices
        start_rows = [r for r in rows if r["date"] >= end_date]
        end_rows = conn.execute(
            """
            SELECT h.close FROM history h
            JOIN tickers t ON h.ticker_id = t.id
            WHERE t.symbol = ? AND h.date >= ?
            ORDER BY h.date ASC LIMIT 1
            """,
            (ticker.upper(), fwd_date),
        ).fetchone()
        start_rows2 = conn.execute(
            """
            SELECT h.close FROM history h
            JOIN tickers t ON h.ticker_id = t.id
            WHERE t.symbol = ? AND h.date >= ?
            ORDER BY h.date ASC LIMIT 1
            """,
            (ticker.upper(), end_date),
        ).fetchone()
        if start_rows2 is None or end_rows is None:
            return None
        p0 = float(start_rows2["close"])
        p1 = float(end_rows["close"])
        if p0 <= 0:
            return None
        return (p1 - p0) / p0
    except Exception:
        return None


def _compute_div_cut(
    ticker: str, period: str, conn: sqlite3.Connection
) -> Optional[int]:
    """1 if dividends dropped >20% in 4 quarters after period, 0 otherwise."""
    end_date = _period_end_date(period)
    if not end_date:
        return None
    try:
        fwd_date = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=365)).strftime("%Y-%m-%d")
        prior_start = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")

        prior = conn.execute(
            """
            SELECT SUM(h.dividends) as total FROM history h
            JOIN tickers t ON h.ticker_id = t.id
            WHERE t.symbol = ? AND h.date >= ? AND h.date <= ?
            """,
            (ticker.upper(), prior_start, end_date),
        ).fetchone()
        after = conn.execute(
            """
            SELECT SUM(h.dividends) as total FROM history h
            JOIN tickers t ON h.ticker_id = t.id
            WHERE t.symbol = ? AND h.date > ? AND h.date <= ?
            """,
            (ticker.upper(), end_date, fwd_date),
        ).fetchone()

        p = float(prior["total"] or 0)
        a = float(after["total"] or 0)
        if p <= 0:
            return None  # No prior dividends — cut indicator not meaningful
        return 1 if a < p * 0.8 else 0
    except Exception:
        return None


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_proximity_pipeline(
    tickers: Optional[list] = None,
    rebuild: bool = False,
    verbose: bool = False,
) -> int:
    """Build/update the proximity_vectors index. Returns count of vectors written."""
    _ensure_table()

    conn = _db()
    try:
        if tickers:
            symbols = [t.upper() for t in tickers]
        else:
            rows = conn.execute("SELECT symbol FROM tickers").fetchall()
            symbols = [r["symbol"] for r in rows]
    finally:
        conn.close()

    if not symbols:
        print("No tickers found.")
        return 0

    if verbose:
        print(f"Building proximity index for {len(symbols)} tickers...")

    total_written = 0
    for ticker in symbols:
        conn = _db()
        try:
            periods = _periods_for_ticker(conn, ticker)
            if not periods:
                continue

            for period in periods:
                # Skip if already computed and not rebuilding
                if not rebuild:
                    existing = conn.execute(
                        "SELECT 1 FROM proximity_vectors WHERE ticker=? AND period=?",
                        (ticker, period),
                    ).fetchone()
                    if existing:
                        continue

                vec = build_state_vector(ticker, period, conn)
                if vec is None:
                    continue

                fwd_return = _compute_fwd_return_12m(ticker, period, conn)
                div_cut = _compute_div_cut(ticker, period, conn)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO proximity_vectors
                        (ticker, period, state_vector, fwd_return_12m, div_cut, computed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticker, period, _vec_to_blob(vec),
                        fwd_return, div_cut,
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()
                total_written += 1

                if verbose:
                    fwd_str = f"{fwd_return:.1%}" if fwd_return is not None else "pending"
                    print(f"  {ticker:8s} {period}  fwd={fwd_str}  div_cut={div_cut}")

        except Exception as exc:
            print(f"  ERROR {ticker}: {exc}")
        finally:
            conn.close()

    print(f"Proximity index: {total_written} vectors written for {len(symbols)} tickers.")
    return total_written


def main():
    parser = argparse.ArgumentParser(
        description="Build proximity state-vector index for historical analog retrieval."
    )
    parser.add_argument("--tickers", nargs="+", metavar="TICKER")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force recompute even if already indexed")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_proximity_pipeline(
        tickers=args.tickers,
        rebuild=args.rebuild,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
