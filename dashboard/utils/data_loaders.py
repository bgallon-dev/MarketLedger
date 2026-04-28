"""Cached data loaders for the MarketLedger dashboard."""
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
_DASHBOARD_DIR  = Path(__file__).parent.parent          # dashboard/
_MARKET_LEDGER  = _DASHBOARD_DIR.parent                 # MarketLedger/
_YAHOO_PROJECT  = _MARKET_LEDGER.parent                 # Yahoo_project/
_RESEARCH_DIR   = _MARKET_LEDGER / "research_outputs"

if str(_MARKET_LEDGER) not in sys.path:
    sys.path.insert(0, str(_MARKET_LEDGER))

from dashboard.utils.db import get_financial_db, get_paper_db


# ── BLOB / dtype helpers ───────────────────────────────────────────────────────

def _coerce_blob(val, cast_fn):
    if isinstance(val, (bytes, bytearray)):
        return cast_fn(int.from_bytes(val, "little"))
    try:
        return cast_fn(val) if pd.notna(val) else None
    except (TypeError, ValueError):
        return None


def _fix_positions_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce any BLOB-serialised integer/float columns in the positions table."""
    for col in ("rv_gate_count", "hold_days"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: _coerce_blob(x, int))
    for col in ("shares",):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: _coerce_blob(x, float))
    # Warn about any remaining object columns that look like bytes
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(5)
        if any(isinstance(v, (bytes, bytearray)) for v in sample):
            warnings.warn(f"positions.{col} still contains BLOB bytes after coercion")
    return df


# ── Pipeline CSVs ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_decision_log() -> pd.DataFrame:
    path = _YAHOO_PROJECT / "buy_list_with_projections_decision_log.csv"
    if not path.exists():
        path = _YAHOO_PROJECT / "buy_list_with_projections.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_pipeline_results() -> pd.DataFrame:
    path = _YAHOO_PROJECT / "buy_list_with_projections.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_research_csvs() -> dict[str, pd.DataFrame]:
    result = {}
    if _RESEARCH_DIR.exists():
        for csv_path in _RESEARCH_DIR.glob("*.csv"):
            try:
                result[csv_path.stem] = pd.read_csv(csv_path)
            except Exception:
                pass
    return result


# ── Paper trading DB ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_positions() -> pd.DataFrame:
    try:
        conn = get_paper_db()
        df = pd.read_sql_query("SELECT * FROM positions", conn)
        conn.close()
    except Exception:
        return pd.DataFrame()
    return _fix_positions_dtypes(df)


@st.cache_data(ttl=3600)
def load_transactions() -> pd.DataFrame:
    try:
        conn = get_paper_db()
        df = pd.read_sql_query("SELECT * FROM transactions ORDER BY id", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_rebalance_log() -> pd.DataFrame:
    try:
        conn = get_paper_db()
        df = pd.read_sql_query("SELECT * FROM rebalance_log ORDER BY id", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_portfolio_meta() -> dict:
    try:
        conn = get_paper_db()
        rows = conn.execute("SELECT key, value FROM portfolio_meta").fetchall()
        conn.close()
        return {r["key"]: r["value"] for r in rows}
    except Exception:
        return {}


# ── Financial history ─────────────────────────────────────────────────────────

@st.cache_data(ttl=1800)
def load_ticker_history(ticker: str, days: int = 730) -> pd.DataFrame:
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        conn = get_financial_db()
        df = pd.read_sql_query(
            """
            SELECT h.date, h.open, h.high, h.low, h.close, h.volume
            FROM history h
            JOIN tickers t ON h.ticker_id = t.id
            WHERE t.symbol = ? AND h.date >= ?
            ORDER BY h.date
            """,
            conn,
            params=(ticker.upper(), start),
        )
        conn.close()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    return df


@st.cache_data(ttl=300)
def load_latest_closes(symbols: tuple[str, ...]) -> pd.Series:
    """Return a Series {symbol: latest_close} for the given symbols.

    Uses a 7-day window so we only touch recent rows — still hits the index.
    """
    if not symbols:
        return pd.Series(dtype=float)
    start = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    try:
        from database.database import get_ticker_history_bulk
        df = get_ticker_history_bulk(list(symbols), start_date=start)
    except Exception:
        return pd.Series(dtype=float)
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    return df.sort_values("date").groupby("symbol")["close"].last()


@st.cache_data(ttl=3600)
def load_all_tickers() -> pd.DataFrame:
    try:
        conn = get_financial_db()
        df = pd.read_sql_query(
            "SELECT symbol, name, sector, industry FROM tickers ORDER BY symbol",
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_financial_table(ticker: str, table: str) -> pd.DataFrame:
    """Return a pivoted (metric × period) DataFrame for one ticker + table."""
    valid_tables = {"balance_sheet", "income_statement", "cash_flow", "financials", "earnings"}
    if table not in valid_tables:
        return pd.DataFrame()
    try:
        conn = get_financial_db()
        df = pd.read_sql_query(
            f"""
            SELECT f.metric, f.period, f.value
            FROM {table} f
            JOIN tickers t ON f.ticker_id = t.id
            WHERE t.symbol = ?
            ORDER BY f.metric, f.period
            """,
            conn,
            params=(ticker.upper(),),
        )
        conn.close()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    return df.pivot_table(index="metric", columns="period", values="value", aggfunc="first")
