"""Cached loader for filing-derived mood vectors."""
import streamlit as st

from dashboard.utils.db import get_financial_db


@st.cache_data(ttl=3600)
def load_mood_vector(ticker: str) -> dict | None:
    """Return the latest mood vector for *ticker*, or None if not computed."""
    conn = get_financial_db()
    try:
        row = conn.execute(
            """
            SELECT ticker, filing_date, accession_no,
                   disclosure_pressure, operational_confidence,
                   cash_flow_coherence, narrative_stability,
                   pct_disclosure_pressure, pct_operational_confidence,
                   pct_cash_flow_coherence, pct_narrative_stability,
                   composite_mood, sector, size_bucket, computed_at
            FROM mood_vectors
            WHERE ticker = ?
            ORDER BY filing_date DESC
            LIMIT 1
            """,
            (ticker.upper(),),
        ).fetchone()
    except Exception:
        return None
    finally:
        conn.close()
    return dict(row) if row else None
