"""
Proximity Engine — state vector construction and nearest-neighbor retrieval.

Each ticker-quarter pair is encoded as a 9-dimensional state vector capturing
structural financial characteristics. Cosine similarity over the full index
surfaces the closest historical analogs with their 12M forward price outcomes.

Dimensions:
  0  fcf_margin       FreeCashFlow / TotalRevenue
  1  leverage         TotalDebt / TotalAssets
  2  wc_ratio         WorkingCapital / TotalAssets
  3  rev_growth_yoy   YoY revenue growth rate
  4  altman_z         Altman Z-Score proxy
  5  fcf_coverage     FreeCashFlow / (dividends + buybacks)
  6  momentum_gap     (close - 200MA) / 200MA
  7  tail_encoded     Light=1, Normal=2, Moderate=3, Heavy=4
  8  cfc_score        Cash Flow Coherence from mood_vectors (or 0.0)
"""

import io
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_MARKET_LEDGER = Path(__file__).parent.parent.parent
_DB_PATH = _MARKET_LEDGER / "database" / "financial_data.db"

_DIM = 9
_FALLBACKS = np.array([0.0, 0.5, 0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0])

_DIV_METRICS = ["CashDividendsPaid", "CommonStockDividendsPaid", "PaymentOfDividends"]
_BUY_METRICS = [
    "RepurchaseOfCapitalStock", "RepurchaseOfCommonStock",
    "CommonStockRepurchase", "StockRepurchase",
]
_TAIL_ENCODE = {"Light": 1.0, "Normal": 2.0, "Moderate": 3.0, "Heavy": 4.0}


# ── BLOB serialisation ─────────────────────────────────────────────────────────

def _vec_to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return buf.getvalue()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob)).astype(np.float64)


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _first_val(wide: pd.DataFrame, metrics: list, period: str) -> Optional[float]:
    """Get the most recent non-null value for any of the given metrics."""
    for m in metrics:
        if m in wide.index:
            row = wide.loc[m]
            # Try the exact period first, then the most recent available
            if period in row.index and pd.notna(row[period]):
                return float(row[period])
            val = row.dropna()
            if not val.empty:
                return float(val.iloc[0])
    return None


def _period_to_date(period: str) -> Optional[str]:
    """Convert 'YYYY-QN' to approximate period-end date."""
    try:
        year, q = period.split("-Q")
        month = int(q) * 3
        return f"{year}-{month:02d}-30"
    except Exception:
        return None


# ── State vector construction ──────────────────────────────────────────────────

def build_state_vector(
    ticker: str,
    period: str,
    conn: sqlite3.Connection,
) -> Optional[np.ndarray]:
    """
    Build a 9-dimensional state vector for ticker at period ('YYYY-QN').
    Returns None if fewer than 5 of 9 dimensions are available.
    """
    ticker = ticker.upper()
    period_date = _period_to_date(period)

    # Load financials from DB
    try:
        bs_df = pd.read_sql(
            """
            SELECT f.metric, f.period, f.value
            FROM balance_sheet f
            JOIN tickers t ON f.ticker_id = t.id
            WHERE t.symbol = ?
            """,
            conn, params=(ticker,),
        )
        cf_df = pd.read_sql(
            """
            SELECT f.metric, f.period, f.value
            FROM cash_flow f
            JOIN tickers t ON f.ticker_id = t.id
            WHERE t.symbol = ?
            """,
            conn, params=(ticker,),
        )
        is_df = pd.read_sql(
            """
            SELECT f.metric, f.period, f.value
            FROM income_statement f
            JOIN tickers t ON f.ticker_id = t.id
            WHERE t.symbol = ?
            """,
            conn, params=(ticker,),
        )
    except Exception:
        return None

    def _wide(df):
        if df.empty:
            return pd.DataFrame()
        return df.pivot(index="metric", columns="period", values="value")

    bs = _wide(bs_df)
    cf = _wide(cf_df)
    is_ = _wide(is_df)

    vec = np.full(_DIM, np.nan)

    # dim 0: fcf_margin
    fcf = _first_val(cf, ["FreeCashFlow"], period)
    rev = _first_val(is_, ["TotalRevenue", "OperatingRevenue", "Revenue"], period)
    if fcf is not None and rev is not None and rev != 0:
        vec[0] = fcf / rev

    # dim 1: leverage
    debt = _first_val(bs, ["TotalDebt", "LongTermDebt"], period)
    assets = _first_val(bs, ["TotalAssets"], period)
    if debt is not None and assets is not None and assets > 0:
        vec[1] = debt / assets

    # dim 2: wc_ratio
    wc = _first_val(bs, ["WorkingCapital"], period)
    if wc is not None and assets is not None and assets > 0:
        vec[2] = wc / assets

    # dim 3: rev_growth_yoy
    if rev is not None and not is_.empty and "TotalRevenue" in is_.index:
        sorted_periods = sorted(is_.columns)
        if period in sorted_periods:
            idx = sorted_periods.index(period)
            if idx >= 4:
                prior_period = sorted_periods[idx - 4]
                prior_rev = _first_val(is_, ["TotalRevenue", "OperatingRevenue", "Revenue"], prior_period)
                if prior_rev is not None and prior_rev != 0:
                    vec[3] = (rev - prior_rev) / abs(prior_rev)

    # dim 4: altman_z proxy (simplified: EBIT/Assets + Revenue/Assets + WC/Assets)
    ebit = _first_val(is_, ["EBIT", "OperatingIncome"], period)
    re_ = _first_val(bs, ["RetainedEarnings"], period)
    if assets is not None and assets > 0:
        z = 0.0
        count = 0
        if wc is not None:
            z += 1.2 * wc / assets; count += 1
        if re_ is not None:
            z += 1.4 * re_ / assets; count += 1
        if ebit is not None:
            z += 3.3 * ebit / assets; count += 1
        if rev is not None:
            z += 1.0 * rev / assets; count += 1
        if count >= 2:
            vec[4] = z

    # dim 5: fcf_coverage
    div = None
    for m in _DIV_METRICS:
        if not cf.empty and m in cf.index:
            v = _first_val(cf, [m], period)
            if v is not None:
                div = abs(v)
                break
    buy = None
    for m in _BUY_METRICS:
        if not cf.empty and m in cf.index:
            v = _first_val(cf, [m], period)
            if v is not None:
                buy = abs(v)
                break
    distributions = (div or 0.0) + (buy or 0.0)
    if fcf is not None and distributions > 0:
        vec[5] = fcf / distributions

    # dim 6: momentum_gap (price vs 200-day MA at period end)
    if period_date is not None:
        try:
            hist = pd.read_sql(
                """
                SELECT h.date, h.close
                FROM history h
                JOIN tickers t ON h.ticker_id = t.id
                WHERE t.symbol = ? AND h.date <= ?
                ORDER BY h.date DESC
                LIMIT 250
                """,
                conn, params=(ticker, period_date),
            )
            if len(hist) >= 20:
                ma200 = hist["close"].mean()
                price = hist["close"].iloc[0]
                if ma200 > 0:
                    vec[6] = (price - ma200) / ma200
        except Exception:
            pass

    # dim 7: tail_encoded (from distro fits if available; fallback to neutral)
    try:
        row = conn.execute(
            "SELECT Tail_Risk FROM history WHERE 1=0"  # placeholder
        ).fetchone()
    except Exception:
        pass
    # Tail_Risk isn't stored in the DB — use neutral fallback (2.0 = Normal)
    vec[7] = 2.0

    # dim 8: cfc_score from mood_vectors
    try:
        mv_row = conn.execute(
            "SELECT cash_flow_coherence FROM mood_vectors WHERE ticker=? "
            "ORDER BY filing_date DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        if mv_row and mv_row[0] is not None:
            vec[8] = float(mv_row[0])
        else:
            vec[8] = 0.0
    except Exception:
        vec[8] = 0.0

    # Require at least 5 of 9 dimensions
    valid_count = int(np.sum(~np.isnan(vec)))
    if valid_count < 5:
        return None

    # Fill remaining NaNs with fallbacks
    for i, v in enumerate(vec):
        if np.isnan(v):
            vec[i] = _FALLBACKS[i]

    return vec


def get_current_state(ticker: str, conn: sqlite3.Connection) -> Optional[np.ndarray]:
    """Build a state vector for ticker using its most recent available period."""
    try:
        periods_df = pd.read_sql(
            """
            SELECT DISTINCT f.period
            FROM balance_sheet f
            JOIN tickers t ON f.ticker_id = t.id
            WHERE t.symbol = ?
            ORDER BY f.period DESC
            LIMIT 1
            """,
            conn, params=(ticker.upper(),),
        )
    except Exception:
        return None
    if periods_df.empty:
        return None
    period_raw = periods_df["period"].iloc[0]
    # Convert period string (e.g. '2024-12-31') to YYYY-QN
    try:
        dt = datetime.strptime(period_raw[:10], "%Y-%m-%d")
        q = (dt.month - 1) // 3 + 1
        period = f"{dt.year}-Q{q}"
    except Exception:
        period = period_raw

    return build_state_vector(ticker, period, conn)


# ── Cosine similarity query ────────────────────────────────────────────────────

def find_analogs(
    query_vector: np.ndarray,
    conn: sqlite3.Connection,
    exclude_ticker: str,
    k: int = 4,
) -> list:
    """
    Load all proximity_vectors, z-score normalise, compute cosine similarity
    against query_vector, return top-k excluding exclude_ticker.

    Returns list of dicts: {ticker, period, similarity, fwd_return_12m, div_cut}
    """
    try:
        rows = conn.execute(
            "SELECT ticker, period, state_vector, fwd_return_12m, div_cut FROM proximity_vectors"
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    if not rows:
        return []

    tickers_list = []
    periods_list = []
    fwd_returns = []
    div_cuts = []
    vecs = []

    for row in rows:
        if row["ticker"].upper() == exclude_ticker.upper():
            continue
        try:
            v = _blob_to_vec(row["state_vector"])
            if v.shape[0] != _DIM:
                continue
        except Exception:
            continue
        tickers_list.append(row["ticker"])
        periods_list.append(row["period"])
        fwd_returns.append(row["fwd_return_12m"])
        div_cuts.append(row["div_cut"])
        vecs.append(v)

    if not vecs:
        return []

    matrix = np.stack(vecs)  # shape (N, 9)

    # Z-score normalise each column using matrix statistics
    col_mean = np.nanmean(matrix, axis=0)
    col_std = np.nanstd(matrix, axis=0)
    col_std[col_std == 0] = 1.0

    matrix_norm = (matrix - col_mean) / col_std
    query_norm = (query_vector - col_mean) / col_std

    # Cosine similarity
    matrix_norms = np.linalg.norm(matrix_norm, axis=1, keepdims=True)
    matrix_norms[matrix_norms == 0] = 1.0
    matrix_unit = matrix_norm / matrix_norms

    query_unit = query_norm / max(np.linalg.norm(query_norm), 1e-9)
    similarities = matrix_unit @ query_unit

    # Top-k
    top_indices = np.argsort(similarities)[::-1][:k]

    return [
        {
            "ticker": tickers_list[i],
            "period": periods_list[i],
            "similarity": float(similarities[i]),
            "fwd_return_12m": fwd_returns[i],
            "div_cut": div_cuts[i],
        }
        for i in top_indices
    ]
