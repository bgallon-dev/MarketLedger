#!/usr/bin/env python3
"""
Financial Mood Vector Pipeline

Computes a 4-dimensional filing-derived mood signal per company:
  - Cash Flow Coherence     (FCF vs declared distributions, trend slope)
  - Disclosure Pressure     (YoY risk factor accumulation from 10-K Item 1A)
  - Operational Confidence  (management FLS accuracy via management_claims)
  - Narrative Stability     (semantic distance between consecutive MD&A sections)

Usage:
    py dashboard/experimental/mood_pipeline.py
    py dashboard/experimental/mood_pipeline.py --tickers AAPL MSFT MNR --verbose
    py dashboard/experimental/mood_pipeline.py --max-workers 8
"""

import argparse
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

# ── path setup ─────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).parent
_MARKET_LEDGER = _THIS_DIR.parent.parent
_DB_PATH = _MARKET_LEDGER / "database" / "financial_data.db"

sys.path.insert(0, str(_MARKET_LEDGER))

# ── constants ──────────────────────────────────────────────────────────────────
_FCF_METRICS = ["FreeCashFlow"]
_DIV_METRICS = ["CashDividendsPaid", "CommonStockDividendsPaid", "PaymentOfDividends"]
_BUY_METRICS = [
    "RepurchaseOfCapitalStock", "RepurchaseOfCommonStock",
    "CommonStockRepurchase", "StockRepurchase",
]
_MDA_TRUNCATE = 10_000
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Paragraph openers that indicate body text rather than risk factor headers
_BODY_STARTER_RE = re.compile(
    r"^(we |our |the |this |in |as |if |it |such |these |those |a |an )",
    re.IGNORECASE,
)


# ── DB connection ──────────────────────────────────────────────────────────────

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table() -> None:
    conn = _db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mood_vectors (
            ticker                      TEXT NOT NULL,
            filing_date                 TEXT NOT NULL,
            accession_no                TEXT,
            disclosure_pressure         REAL,
            operational_confidence      REAL,
            cash_flow_coherence         REAL,
            narrative_stability         REAL,
            pct_disclosure_pressure     REAL,
            pct_operational_confidence  REAL,
            pct_cash_flow_coherence     REAL,
            pct_narrative_stability     REAL,
            composite_mood              REAL,
            sector                      TEXT,
            size_bucket                 TEXT,
            computed_at                 TEXT NOT NULL,
            PRIMARY KEY (ticker, filing_date)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mood_vectors_ticker ON mood_vectors(ticker)"
    )
    conn.commit()
    conn.close()


# ── Extractor 1: Cash Flow Coherence ──────────────────────────────────────────

def _first_series(wide: pd.DataFrame, variants: list) -> pd.Series:
    for m in variants:
        if m in wide.index:
            return wide.loc[m]
    return pd.Series(dtype=float)


def compute_cash_flow_coherence(ticker: str, cf_long: pd.DataFrame) -> Optional[float]:
    """
    Slope of FCF-coverage-ratio over trailing 8 reporting periods.
    Positive slope = FCF is increasingly covering distributions (coherent).
    Negative slope = FCF is decreasingly covering distributions (incoherent).
    Returns None if insufficient data. Clipped to [-3, +3].
    """
    if cf_long.empty:
        return None

    t_data = cf_long[cf_long["symbol"] == ticker]
    if t_data.empty:
        return None

    wide = t_data.pivot(index="metric", columns="period", values="value")

    fcf_row = _first_series(wide, _FCF_METRICS)
    div_row = _first_series(wide, _DIV_METRICS)
    buy_row = _first_series(wide, _BUY_METRICS)

    if fcf_row.empty:
        return None

    # Intersect periods across FCF and at least one distribution metric
    dist_periods = set(div_row.index) | set(buy_row.index)
    if not dist_periods:
        return None

    periods = sorted(set(fcf_row.index) & dist_periods)
    if len(periods) < 3:
        return None
    periods = periods[-8:]

    fcf = fcf_row.reindex(periods).astype(float)
    div = (
        div_row.reindex(periods).astype(float).abs()
        if not div_row.empty else pd.Series(0.0, index=periods)
    )
    buy = (
        buy_row.reindex(periods).astype(float).abs()
        if not buy_row.empty else pd.Series(0.0, index=periods)
    )

    distributions = div.fillna(0.0) + buy.fillna(0.0)
    nonzero_mask = distributions != 0.0
    if nonzero_mask.sum() < 3:
        return None

    coverage = (fcf[nonzero_mask] / distributions[nonzero_mask]).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(coverage) < 3:
        return None

    x = np.arange(len(coverage), dtype=float)
    slope = float(np.polyfit(x, coverage.values, 1)[0])
    return float(np.clip(slope, -3.0, 3.0))


# ── Extractor 2: Disclosure Pressure ──────────────────────────────────────────

def _parse_risk_factor_titles(item_1a_text: str) -> set:
    """
    Extract a set of normalised risk-factor title tokens from plain-text Item 1A.
    Heuristic: first line of each double-newline-separated paragraph, filtered
    to lines that look like headers rather than body prose.
    """
    if not item_1a_text:
        return set()

    titles = set()
    for para in re.split(r"\n{2,}", item_1a_text.strip()):
        first_line = para.strip().split("\n")[0].strip()
        if len(first_line) < 15 or len(first_line) > 250:
            continue
        if _BODY_STARTER_RE.match(first_line):
            continue
        # Normalise: lowercase, strip punctuation, collapse whitespace
        token = re.sub(r"[^\w\s]", " ", first_line.lower())
        token = re.sub(r"\s+", " ", token).strip()
        if token:
            titles.add(token)
    return titles


def compute_disclosure_pressure(
    current_1a: Optional[str], prior_1a: Optional[str]
) -> Optional[float]:
    """
    (additions - removals) / len(prior_risks).
    Positive = company added more risk factors than it retired (pressure building).
    Negative = company removed risk factors (pressure easing).
    Returns None if prior text is unavailable or has < 3 identifiable risk factors.
    """
    if not current_1a or not prior_1a:
        return None

    current_set = _parse_risk_factor_titles(current_1a)
    prior_set = _parse_risk_factor_titles(prior_1a)

    if len(prior_set) < 3:
        return None

    additions = len(current_set - prior_set)
    removals = len(prior_set - current_set)
    return (additions - removals) / len(prior_set)


# ── Extractor 3: Operational Confidence ───────────────────────────────────────

def compute_operational_confidence(
    ticker: str, claims_df: pd.DataFrame
) -> Optional[float]:
    """
    Average outcome_met rate from management_claims, rescaled to [-1, +1].
    Requires at least 2 resolved claims. Returns None otherwise.
    """
    t_claims = claims_df[
        (claims_df["ticker"] == ticker) & (claims_df["outcome_met"].notna())
    ].sort_values("filed_date", ascending=False).head(8)

    if len(t_claims) < 2:
        return None

    accuracy = t_claims["outcome_met"].astype(float).mean() * 100.0  # 0–100
    return (accuracy - 50.0) / 50.0  # -1..+1


# ── Extractor 4: Narrative Stability ──────────────────────────────────────────

def compute_narrative_stability(
    current_mda: Optional[str],
    prior_mda: Optional[str],
    model,
) -> Optional[float]:
    """
    Cosine similarity between sentence embeddings of consecutive MD&A sections.
    1 = identical narrative, 0 = completely rewritten.
    Returns None if either text is missing or the model is not loaded.
    """
    if not current_mda or not prior_mda or model is None:
        return None

    from sklearn.metrics.pairwise import cosine_similarity as sk_cos

    emb_curr = model.encode(current_mda[:_MDA_TRUNCATE], show_progress_bar=False)
    emb_prev = model.encode(prior_mda[:_MDA_TRUNCATE], show_progress_bar=False)

    sim = float(sk_cos([emb_curr], [emb_prev])[0][0])
    return float(np.clip(sim, 0.0, 1.0))


# ── Normalisation layer ────────────────────────────────────────────────────────

def _percentile_rank(group_series: pd.Series, value: float) -> float:
    valid = group_series.dropna().values
    if len(valid) < 2:
        return 50.0
    return float(percentileofscore(valid, value, kind="rank"))


def normalize_scores(results: list, tickers_meta: pd.DataFrame) -> list:
    """
    Attach sector + size_bucket, then compute percentile ranks within each
    (sector, size_bucket) peer group. Adds composite_mood as mean of pct dimensions.
    """
    df = pd.DataFrame(results)
    if df.empty:
        return results

    meta = tickers_meta.set_index("ticker")
    sector_map = meta["sector"] if "sector" in meta.columns else pd.Series(dtype=str)
    bucket_map = meta["size_bucket"] if "size_bucket" in meta.columns else pd.Series(dtype=str)

    df["sector"] = df["ticker"].map(sector_map).fillna("Unknown")
    df["size_bucket"] = df["ticker"].map(bucket_map).fillna("unknown")

    dims = [
        "disclosure_pressure", "operational_confidence",
        "cash_flow_coherence", "narrative_stability",
    ]

    # Percentile rank within peer group
    for (sector, bucket), group_df in df.groupby(["sector", "size_bucket"]):
        idxs = group_df.index
        for dim in dims:
            pct_col = f"pct_{dim}"
            for i in idxs:
                val = df.at[i, dim]
                if pd.isna(val) if not isinstance(val, float) else (val != val):
                    df.at[i, pct_col] = np.nan
                else:
                    df.at[i, pct_col] = _percentile_rank(group_df[dim], val)

    pct_cols = [f"pct_{d}" for d in dims]
    df["composite_mood"] = df[pct_cols].mean(axis=1, skipna=True)

    return df.to_dict("records")


# ── Bulk data loaders ──────────────────────────────────────────────────────────

def _load_tickers_meta(conn: sqlite3.Connection, symbols: list) -> pd.DataFrame:
    """Load sector and size_bucket (derived from market cap) per ticker."""
    if symbols:
        placeholders = ",".join("?" * len(symbols))
        tickers_df = pd.read_sql(
            f"SELECT symbol, sector FROM tickers WHERE symbol IN ({placeholders})",
            conn, params=symbols,
        )
    else:
        tickers_df = pd.read_sql("SELECT symbol, sector FROM tickers", conn)

    tickers_df = tickers_df.rename(columns={"symbol": "ticker"})
    tickers_df["size_bucket"] = "unknown"

    # Approximate market cap: latest share count × latest close
    try:
        shares_df = pd.read_sql(
            """
            SELECT t.symbol AS ticker, f.value AS shares, f.period
            FROM balance_sheet f
            JOIN tickers t ON f.ticker_id = t.id
            WHERE f.metric IN ('OrdinarySharesNumber', 'ShareIssued', 'CommonStock')
            """,
            conn,
        )
        close_df = pd.read_sql(
            """
            SELECT t.symbol AS ticker, h.close
            FROM history h
            JOIN tickers t ON h.ticker_id = t.id
            WHERE h.date = (
                SELECT MAX(h2.date) FROM history h2 WHERE h2.ticker_id = h.ticker_id
            )
            """,
            conn,
        )
        if not shares_df.empty and not close_df.empty:
            latest_shares = (
                shares_df.sort_values("period", ascending=False)
                .groupby("ticker")
                .first()[["shares"]]
                .reset_index()
            )
            mc_df = latest_shares.merge(close_df, on="ticker", how="inner")
            mc_df["market_cap"] = mc_df["shares"] * mc_df["close"]

            def _bucket(mc):
                if pd.isna(mc) or mc <= 0:
                    return "unknown"
                if mc < 2e9:
                    return "small"
                if mc < 10e9:
                    return "mid"
                return "large"

            mc_df["size_bucket"] = mc_df["market_cap"].apply(_bucket)
            tickers_df = tickers_df.merge(
                mc_df[["ticker", "size_bucket"]], on="ticker", how="left",
                suffixes=("", "_mc"),
            )
            tickers_df["size_bucket"] = (
                tickers_df["size_bucket_mc"].fillna(tickers_df["size_bucket"])
            )
            tickers_df = tickers_df.drop(columns=["size_bucket_mc"], errors="ignore")
    except Exception:
        pass  # Fall back to 'unknown' bucket if market cap data unavailable

    return tickers_df


def _load_sec_filings(conn: sqlite3.Connection, symbols: list) -> dict:
    """Return {ticker: [current_filing_dict, prior_filing_dict]} (up to 2 per ticker)."""
    try:
        if symbols:
            placeholders = ",".join("?" * len(symbols))
            rows = conn.execute(
                f"""
                SELECT ticker, filed_date, accession_no, item_1a_text, item_7_text
                FROM sec_filings
                WHERE ticker IN ({placeholders})
                ORDER BY ticker, filed_date DESC
                """,
                symbols,
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT ticker, filed_date, accession_no, item_1a_text, item_7_text
                FROM sec_filings
                ORDER BY ticker, filed_date DESC
                """
            ).fetchall()
    except sqlite3.OperationalError:
        return {}  # sec_filings table not yet populated

    filings: dict = {}
    for row in rows:
        t = row["ticker"]
        if t not in filings:
            filings[t] = []
        if len(filings[t]) < 2:
            filings[t].append(dict(row))
    return filings


def _load_management_claims(conn: sqlite3.Connection, symbols: list) -> pd.DataFrame:
    """Load resolved management_claims rows (outcome_met IS NOT NULL)."""
    try:
        if symbols:
            placeholders = ",".join("?" * len(symbols))
            return pd.read_sql(
                f"""
                SELECT ticker, filed_date, outcome_met
                FROM management_claims
                WHERE ticker IN ({placeholders}) AND outcome_met IS NOT NULL
                """,
                conn, params=symbols,
            )
        return pd.read_sql(
            "SELECT ticker, filed_date, outcome_met FROM management_claims "
            "WHERE outcome_met IS NOT NULL",
            conn,
        )
    except sqlite3.OperationalError:
        return pd.DataFrame(columns=["ticker", "filed_date", "outcome_met"])


# ── Per-ticker worker ──────────────────────────────────────────────────────────

def _compute_ticker(
    ticker: str,
    cf_long: pd.DataFrame,
    filings: dict,
    claims_df: pd.DataFrame,
    model,
    verbose: bool,
) -> Optional[dict]:
    filing_list = filings.get(ticker, [])
    current_filing = filing_list[0] if len(filing_list) >= 1 else None
    prior_filing = filing_list[1] if len(filing_list) >= 2 else None

    cfc = compute_cash_flow_coherence(ticker, cf_long)
    dp = compute_disclosure_pressure(
        current_filing.get("item_1a_text") if current_filing else None,
        prior_filing.get("item_1a_text") if prior_filing else None,
    )
    oc = compute_operational_confidence(ticker, claims_df)
    ns = compute_narrative_stability(
        current_filing.get("item_7_text") if current_filing else None,
        prior_filing.get("item_7_text") if prior_filing else None,
        model,
    )

    filing_date = (
        current_filing["filed_date"] if current_filing
        else datetime.now().strftime("%Y-%m-%d")
    )
    accession_no = current_filing.get("accession_no") if current_filing else None

    if verbose:
        print(
            f"  {ticker:8s}  CFC={_fmt(cfc):>8}  DP={_fmt(dp):>8}  "
            f"OC={_fmt(oc):>8}  NS={_fmt(ns):>8}"
        )

    return {
        "ticker": ticker,
        "filing_date": filing_date,
        "accession_no": accession_no,
        "disclosure_pressure": dp,
        "operational_confidence": oc,
        "cash_flow_coherence": cfc,
        "narrative_stability": ns,
        "pct_disclosure_pressure": None,
        "pct_operational_confidence": None,
        "pct_cash_flow_coherence": None,
        "pct_narrative_stability": None,
        "composite_mood": None,
        "computed_at": datetime.now().isoformat(),
    }


def _fmt(val) -> str:
    return f"{val:.3f}" if val is not None else "None"


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_mood_pipeline(
    tickers: Optional[list] = None,
    max_workers: int = 4,
    verbose: bool = False,
) -> int:
    """Compute and persist mood vectors. Returns count of rows written."""
    _ensure_table()

    conn = _db()
    try:
        if tickers:
            symbols = [t.upper() for t in tickers]
        else:
            rows = conn.execute("SELECT symbol FROM tickers").fetchall()
            symbols = [r["symbol"] for r in rows]

        if not symbols:
            print("No tickers found.")
            return 0

        if verbose:
            print(f"Processing {len(symbols)} tickers...")

        from database.database import get_financial_data_bulk

        cf_metrics = _FCF_METRICS + _DIV_METRICS + _BUY_METRICS
        cf_long = get_financial_data_bulk(symbols, "cash_flow", metrics=cf_metrics)
        filings = _load_sec_filings(conn, symbols)
        claims_df = _load_management_claims(conn, symbols)
        tickers_meta = _load_tickers_meta(conn, symbols)
    finally:
        conn.close()

    if verbose:
        print(f"Loading embedding model '{_EMBEDDING_MODEL}'...")
    model = None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(_EMBEDDING_MODEL)
    except ImportError:
        print(
            "WARNING: sentence-transformers not installed — Narrative Stability will be NULL.\n"
            "Install with: py -m pip install sentence-transformers"
        )

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _compute_ticker, t, cf_long, filings, claims_df, model, verbose
            ): t
            for t in symbols
        }
        done = 0
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                result = fut.result()
                if result is not None:
                    results.append(result)
            except Exception as exc:
                print(f"  ERROR {ticker}: {exc}")
            done += 1
            if not verbose and done % 50 == 0:
                print(f"  {done}/{len(symbols)} tickers processed...")

    if not results:
        print("No results to write.")
        return 0

    results = normalize_scores(results, tickers_meta)

    conn = _db()
    written = 0
    try:
        for row in results:
            conn.execute(
                """
                INSERT OR REPLACE INTO mood_vectors
                    (ticker, filing_date, accession_no,
                     disclosure_pressure, operational_confidence,
                     cash_flow_coherence, narrative_stability,
                     pct_disclosure_pressure, pct_operational_confidence,
                     pct_cash_flow_coherence, pct_narrative_stability,
                     composite_mood, sector, size_bucket, computed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row["ticker"], row["filing_date"], row.get("accession_no"),
                    row.get("disclosure_pressure"), row.get("operational_confidence"),
                    row.get("cash_flow_coherence"), row.get("narrative_stability"),
                    row.get("pct_disclosure_pressure"), row.get("pct_operational_confidence"),
                    row.get("pct_cash_flow_coherence"), row.get("pct_narrative_stability"),
                    row.get("composite_mood"), row.get("sector"), row.get("size_bucket"),
                    row["computed_at"],
                ),
            )
            written += 1
        conn.commit()
    finally:
        conn.close()

    print(f"Written {written} mood vectors to database.")
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Compute filing-derived mood vectors and store in mood_vectors table."
    )
    parser.add_argument("--tickers", nargs="+", metavar="TICKER",
                        help="Subset of tickers to process (default: full universe)")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Parallel worker threads (default: 4)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-ticker scores")
    args = parser.parse_args()

    run_mood_pipeline(
        tickers=args.tickers,
        max_workers=args.max_workers,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
