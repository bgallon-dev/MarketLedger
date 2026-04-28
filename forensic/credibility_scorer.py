"""
Longitudinal Credibility Scorer — forensic module.

Extracts forward-looking statements (FLS) from 10-K Item 7 (MD&A) text,
tracks them against subsequent financial outcomes, and computes a per-ticker
management credibility score over time.

The system is longitudinal: claims accumulate in SQLite across runs, and
outcomes are resolved as new financial data becomes available. A management
team with a three-year track record of accurate guidance scores higher than
one whose promises do not materialize.

Public API:
    run_credibility_scan(candidates_df, prefetched=None, verbose=True,
                         max_workers=None) -> pd.DataFrame
"""

import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

DB_PATH = Path(__file__).parent.parent / "database" / "financial_data.db"

# ── Forward-looking statement triggers ────────────────────────────────────────
# Each entry: (claim_type, trigger_phrases, value_regex_or_None, direction_regex_or_None)
# All matching is case-insensitive.  A sentence must contain ≥1 trigger phrase.

_FLS_TRIGGERS: List[Tuple[str, List[str], Optional[str], Optional[str]]] = [
    (
        "revenue_growth",
        [
            # Require explicit growth/change language near "revenue" — avoids
            # catching gross-margin or utilization figures that mention "revenue"
            r"revenue.*to grow",
            r"revenue.*to increase",
            r"revenues.*grow",
            r"revenue growth of",
            r"revenue.*grow.*by",
            r"revenue.*increase.*by",
            r"revenue.*decline.*by",
            r"revenue.*decrease.*by",
        ],
        r"(\d+\.?\d*)\s*(?:%|percent)",
        r"(increase|grow|decline|decrease)",
    ),
    (
        "margin_recovery",
        [
            # Require directional change language, not just mention of a margin level
            r"margin.*improve by",
            r"margin.*expand by",
            r"margin.*increase by",
            r"margin.*decrease by",
            r"operating margin.*improve",
            r"margin.*recover",
            r"expand.*margin",
            r"margin.*improvement",
        ],
        r"(\d+\.?\d*)\s*(?:%|percent|basis point)",
        r"(improve|increase|expand|recover|decline|decrease|compress)",
    ),
    (
        "capex_plan",
        [
            r"capital expenditure.*approximately", r"capex.*approximately",
            r"expect.*capital expenditure", r"plan to invest",
            r"approximately.*capital", r"capital.*plan.*invest",
        ],
        r"\$\s*(\d+(?:\.\d+)?)\s*(million|billion)",
        None,
    ),
    (
        "dividend",
        [
            r"committed to.*dividend", r"maintain.*dividend", r"increase.*dividend",
            r"dividend.*grow", r"dividend.*commit", r"expect.*dividend",
            r"plan.*dividend", r"dividend.*intend",
        ],
        r"\$\s*(\d+(?:\.\d+)?)\s*(?:per share)?",
        r"(increase|maintain|grow|commit|reduce|cut|suspend|eliminate)",
    ),
    (
        "buyback",
        [
            r"share repurchase", r"repurchase.*share", r"buyback program",
            r"repurchase up to", r"buy back.*share", r"expect.*repurchase",
            r"plan.*repurchase",
        ],
        r"\$\s*(\d+(?:\.\d+)?)\s*(million|billion)",
        r"(repurchase|buy back|retire)",
    ),
    (
        "organic_growth",
        [
            r"organic growth", r"grow.*organically", r"internal.*investment",
            r"without.*acqui", r"organic.*expand", r"focus on organic",
        ],
        None,
        r"(grow|increase|expand|improve|achieve|focus|prioritize)",
    ),
]

# Quantified claims weight 2× vs directional-only claims when computing score
_QUANTIFIED_WEIGHT = 2.0
_DIRECTIONAL_WEIGHT = 1.0

# A genuine forward-looking statement must contain at least one future/intent indicator.
# Past-tense sentences ("the program was completed", "we completed the repurchase")
# are historical disclosures, not forward commitments.
_FUTURE_REQUIRED = re.compile(
    r"\b(will|expect|plan|intend|anticipate|target|aim|project|forecast|"
    r"would|should|could|may|might|going to|look to|seek to|"
    r"expect to|plan to|intend to|are expected to|is expected to|"
    r"we expect|we plan|we intend|we anticipate|we target|we aim|"
    r"we will|company will|company expects|company plans|company intends)\b",
    re.IGNORECASE,
)

# SEC safe harbor boilerplate and HTML-heavy lines — not actual FLS.
_BOILERPLATE_PATTERNS = [
    r"all statements other than.*historical fact",
    r"these forward.looking statements.*include",
    r"such forward.looking statements",
    r"forward.looking statements.*involve.*risk",
    r"cautionary.*forward.looking",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE)

# ── SQLite schema ──────────────────────────────────────────────────────────────

_CREATE_CLAIMS_TABLE = """
CREATE TABLE IF NOT EXISTS management_claims (
    ticker               TEXT NOT NULL,
    filed_date           TEXT NOT NULL,
    claim_type           TEXT NOT NULL,
    claim_text           TEXT NOT NULL,
    claim_value          REAL,
    claim_dir            TEXT,
    outcome_met          INTEGER,
    outcome_checked_date TEXT,
    PRIMARY KEY (ticker, filed_date, claim_type, claim_text)
)
"""

_CREATE_CLAIMS_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_mgmt_claims_ticker "
    "ON management_claims(ticker)"
)


def _ensure_management_tables() -> None:
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    try:
        conn.execute(_CREATE_CLAIMS_TABLE)
        conn.execute(_CREATE_CLAIMS_INDEX)
        conn.commit()
    finally:
        conn.close()


# ── FLS extraction ─────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 20]


def extract_forward_claims(item_7_text: str, filing_date: str) -> List[Dict[str, Any]]:
    """
    Extract forward-looking statements from Item 7 MD&A text.

    Returns a list of dicts with keys:
        filed_date, claim_type, claim_text, claim_value, claim_dir
    """
    if not item_7_text:
        return []

    sentences = _split_sentences(item_7_text)
    claims: List[Dict[str, Any]] = []
    seen: set = set()

    for sentence in sentences:
        # Reject safe harbor boilerplate and HTML-heavy lines early
        if _BOILERPLATE_RE.search(sentence):
            continue
        if sentence.count("&#") > 3:
            continue

        sentence_lower = sentence.lower()
        for claim_type, triggers, value_re, dir_re in _FLS_TRIGGERS:
            if not any(re.search(t, sentence_lower) for t in triggers):
                continue

            # Require at least one future/intent indicator — filters out past-tense
            # historical disclosures ("the program was completed in August 2024")
            if not _FUTURE_REQUIRED.search(sentence):
                continue

            # Deduplicate on (type, first-80-chars-of-sentence)
            dedup_key = (claim_type, sentence[:80])
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            claim_value: Optional[float] = None
            if value_re:
                m = re.search(value_re, sentence, re.IGNORECASE)
                if m:
                    try:
                        raw = float(m.group(1))
                        matched_text = m.group(0).lower()
                        if "basis" in matched_text:
                            # Basis points: 200 bps = 2 percentage points = 0.02
                            claim_value = raw / 10_000.0
                        elif "%" in value_re or "percent" in value_re:
                            claim_value = raw / 100.0
                        else:
                            # Dollar amount — normalize million/billion
                            unit = (m.group(2) or "").lower() if m.lastindex and m.lastindex >= 2 else ""
                            multiplier = 1_000_000_000 if "billion" in unit else 1_000_000
                            claim_value = raw * multiplier
                    except (ValueError, IndexError):
                        pass

            # Plausibility bounds: discard values that are almost certainly
            # misclassified absolute levels (margins, utilization) rather than growth rates.
            # The claim is still recorded as directional if claim_dir is present.
            if claim_type == "revenue_growth" and claim_value is not None:
                if abs(claim_value) > 0.50:   # >50% annual revenue growth is implausible
                    claim_value = None
            elif claim_type == "margin_recovery" and claim_value is not None:
                if abs(claim_value) > 0.20:   # >20pp margin swing is an absolute level, not a change
                    claim_value = None

            claim_dir: Optional[str] = None
            if dir_re:
                m2 = re.search(dir_re, sentence, re.IGNORECASE)
                if m2:
                    claim_dir = m2.group(1).lower()

            claims.append({
                "filed_date": filing_date,
                "claim_type": claim_type,
                "claim_text": sentence[:500],
                "claim_value": claim_value,
                "claim_dir": claim_dir,
            })

    return claims


# ── Outcome resolution ─────────────────────────────────────────────────────────

def _sorted_periods(financials_df: pd.DataFrame) -> List[Tuple[datetime, str]]:
    """Return (datetime, column_str) pairs sorted chronologically."""
    pairs = []
    for col in financials_df.columns:
        try:
            dt = datetime.strptime(str(col)[:10], "%Y-%m-%d")
            pairs.append((dt, str(col)))
        except ValueError:
            pass
    return sorted(pairs)


def _find_subsequent_period(
    financials_df: pd.DataFrame, filing_date_str: str
) -> Optional[str]:
    """Find the first financial period that is 6–24 months after the filing date."""
    try:
        filing_dt = datetime.strptime(filing_date_str[:10], "%Y-%m-%d")
    except ValueError:
        return None
    min_dt = filing_dt + timedelta(days=180)
    max_dt = filing_dt + timedelta(days=730)
    candidates = [
        (dt, col)
        for dt, col in _sorted_periods(financials_df)
        if min_dt <= dt <= max_dt
    ]
    if not candidates:
        return None
    # Prefer the period closest to 12 months out
    target = filing_dt + timedelta(days=365)
    return min(candidates, key=lambda x: abs((x[0] - target).days))[1]


def _prior_period(financials_df: pd.DataFrame, period_col: str) -> Optional[str]:
    """Return the column immediately preceding period_col."""
    pairs = _sorted_periods(financials_df)
    cols = [c for _, c in pairs]
    try:
        idx = cols.index(period_col)
        return cols[idx - 1] if idx > 0 else None
    except ValueError:
        return None


def _get_metric(df: pd.DataFrame, metric: str, period: str) -> Optional[float]:
    if df is None or df.empty or metric not in df.index or period not in df.columns:
        return None
    val = df.loc[metric, period]
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _resolve_revenue_growth(
    claim: Dict[str, Any],
    inc_df: Optional[pd.DataFrame],
) -> Optional[int]:
    if inc_df is None:
        return None
    period = _find_subsequent_period(inc_df, claim["filed_date"])
    if period is None:
        return None
    prev = _prior_period(inc_df, period)
    if prev is None:
        return None
    rev_new = _get_metric(inc_df, "TotalRevenue", period)
    rev_old = _get_metric(inc_df, "TotalRevenue", prev)
    if rev_new is None or rev_old is None or rev_old == 0:
        return None
    actual_growth = (rev_new - rev_old) / abs(rev_old)
    claimed = claim.get("claim_value")
    if claimed is not None:
        # Quantified: met if actual ≥ claimed × 0.8
        return 1 if actual_growth >= claimed * 0.8 else 0
    # Directional
    direction = claim.get("claim_dir", "")
    if direction in ("increase", "grow", "improve"):
        return 1 if actual_growth > 0 else 0
    if direction in ("decline", "decrease"):
        return 1 if actual_growth < 0 else 0
    return None


def _resolve_margin_recovery(
    claim: Dict[str, Any],
    inc_df: Optional[pd.DataFrame],
) -> Optional[int]:
    if inc_df is None:
        return None
    period = _find_subsequent_period(inc_df, claim["filed_date"])
    if period is None:
        return None
    prev = _prior_period(inc_df, period)
    if prev is None:
        return None
    for op_metric in ("OperatingIncome", "EBIT"):
        op_new = _get_metric(inc_df, op_metric, period)
        op_old = _get_metric(inc_df, op_metric, prev)
        rev_new = _get_metric(inc_df, "TotalRevenue", period)
        rev_old = _get_metric(inc_df, "TotalRevenue", prev)
        if all(v is not None and v != 0 for v in [op_new, op_old, rev_new, rev_old]):
            margin_new = op_new / rev_new
            margin_old = op_old / rev_old
            direction = claim.get("claim_dir", "")
            if direction in ("improve", "increase", "expand", "recover"):
                return 1 if margin_new > margin_old else 0
            if direction in ("decline", "decrease", "compress"):
                return 1 if margin_new < margin_old else 0
            return None
    return None


def _resolve_capex_plan(
    claim: Dict[str, Any],
    cf_df: Optional[pd.DataFrame],
) -> Optional[int]:
    if cf_df is None:
        return None
    period = _find_subsequent_period(cf_df, claim["filed_date"])
    if period is None:
        return None
    for cap_metric in ("CapitalExpenditure", "PurchaseOfPPE", "CapitalExpenditures"):
        actual_capex = _get_metric(cf_df, cap_metric, period)
        if actual_capex is not None:
            actual_capex = abs(actual_capex)
            claimed = claim.get("claim_value")
            if claimed is not None and claimed > 0:
                # Met if within ±30% of stated amount
                return 1 if abs(actual_capex - claimed) / claimed <= 0.30 else 0
            return None
    return None


def _resolve_dividend(
    claim: Dict[str, Any],
    history_df: Optional[pd.DataFrame],
    filing_date_str: str,
) -> Optional[int]:
    if history_df is None or history_df.empty:
        return None
    div_col = "dividends" if "dividends" in history_df.columns else "Dividends"
    if div_col not in history_df.columns:
        return None
    date_col = "date" if "date" in history_df.columns else "Date"
    if date_col not in history_df.columns:
        return None
    try:
        filing_dt = datetime.strptime(filing_date_str[:10], "%Y-%m-%d")
    except ValueError:
        return None
    cutoff = pd.Timestamp(filing_dt + timedelta(days=180), tz="UTC")
    future = history_df[pd.to_datetime(history_df[date_col], errors="coerce", utc=True) >= cutoff]
    if future.empty:
        return None
    dividends_paid = future[div_col].sum()
    direction = claim.get("claim_dir", "")
    if direction in ("increase", "grow"):
        # Compare against 12-month trailing before filing
        past_start = pd.Timestamp(filing_dt - timedelta(days=365), tz="UTC")
        filing_ts = pd.Timestamp(filing_dt, tz="UTC")
        dates_utc = pd.to_datetime(history_df[date_col], errors="coerce", utc=True)
        past = history_df[(dates_utc >= past_start) & (dates_utc < filing_ts)]
        past_divs = past[div_col].sum() if not past.empty else 0
        return 1 if dividends_paid > past_divs else 0
    if direction in ("maintain", "commit"):
        return 1 if dividends_paid > 0 else 0
    if direction in ("reduce", "cut", "suspend", "eliminate"):
        return 1 if dividends_paid == 0 else 0
    return None


def _resolve_buyback(
    claim: Dict[str, Any],
    bs_df: Optional[pd.DataFrame],
) -> Optional[int]:
    if bs_df is None:
        return None
    period = _find_subsequent_period(bs_df, claim["filed_date"])
    if period is None:
        return None
    prev = _prior_period(bs_df, period)
    if prev is None:
        return None
    shares_new = _get_metric(bs_df, "OrdinarySharesNumber", period)
    shares_old = _get_metric(bs_df, "OrdinarySharesNumber", prev)
    if shares_new is None or shares_old is None or shares_old == 0:
        return None
    # Buyback confirmed if shares declined
    return 1 if shares_new < shares_old else 0


def _resolve_organic_growth(
    claim: Dict[str, Any],
    bs_df: Optional[pd.DataFrame],
) -> Optional[int]:
    """Organic growth claim is inconsistent if goodwill grew >10% of equity."""
    if bs_df is None:
        return None
    period = _find_subsequent_period(bs_df, claim["filed_date"])
    if period is None:
        return None
    prev = _prior_period(bs_df, period)
    if prev is None:
        return None
    for gw_metric in ("Goodwill", "GoodwillAndIntangibleAssets"):
        gw_new = _get_metric(bs_df, gw_metric, period)
        gw_old = _get_metric(bs_df, gw_metric, prev)
        equity = _get_metric(bs_df, "StockholdersEquity", period)
        if gw_new is not None and gw_old is not None and equity and equity != 0:
            gw_change = gw_new - gw_old
            # Large goodwill increase contradicts organic growth claim
            return 0 if gw_change > abs(equity) * 0.10 else 1
    return None


_RESOLVERS = {
    "revenue_growth": lambda claim, bs, inc, cf, hist: _resolve_revenue_growth(claim, inc),
    "margin_recovery": lambda claim, bs, inc, cf, hist: _resolve_margin_recovery(claim, inc),
    "capex_plan": lambda claim, bs, inc, cf, hist: _resolve_capex_plan(claim, cf),
    "dividend": lambda claim, bs, inc, cf, hist: _resolve_dividend(claim, hist, claim["filed_date"]),
    "buyback": lambda claim, bs, inc, cf, hist: _resolve_buyback(claim, bs),
    "organic_growth": lambda claim, bs, inc, cf, hist: _resolve_organic_growth(claim, bs),
}


def resolve_claim_outcomes(
    claims: List[Dict[str, Any]],
    bs_df: Optional[pd.DataFrame],
    inc_df: Optional[pd.DataFrame],
    cf_df: Optional[pd.DataFrame],
    history_df: Optional[pd.DataFrame],
) -> List[Dict[str, Any]]:
    """
    Attempt to resolve each claim against actual financial data.

    Claims filed within the last 12 months remain unresolved (outcome not yet
    visible in the financial statements).
    """
    cutoff = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    resolved = []
    for claim in claims:
        updated = dict(claim)
        # Claims too recent to have a full year of subsequent data
        if claim.get("filed_date", "9999") > cutoff:
            updated["outcome_met"] = None
            updated["outcome_checked_date"] = None
            resolved.append(updated)
            continue

        resolver = _RESOLVERS.get(claim["claim_type"])
        if resolver is None:
            updated["outcome_met"] = None
        else:
            try:
                outcome = resolver(claim, bs_df, inc_df, cf_df, history_df)
                updated["outcome_met"] = outcome
            except Exception:
                updated["outcome_met"] = None

        updated["outcome_checked_date"] = (
            datetime.now().strftime("%Y-%m-%d") if updated["outcome_met"] is not None else None
        )
        resolved.append(updated)
    return resolved


# ── Persistence ────────────────────────────────────────────────────────────────

def _save_claims(ticker: str, claims: List[Dict[str, Any]]) -> None:
    if not claims:
        return
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    try:
        for c in claims:
            conn.execute(
                """
                INSERT OR IGNORE INTO management_claims
                    (ticker, filed_date, claim_type, claim_text,
                     claim_value, claim_dir, outcome_met, outcome_checked_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker.upper(),
                    c["filed_date"],
                    c["claim_type"],
                    c["claim_text"],
                    c.get("claim_value"),
                    c.get("claim_dir"),
                    c.get("outcome_met"),
                    c.get("outcome_checked_date"),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _update_outcomes(ticker: str, claims: List[Dict[str, Any]]) -> None:
    """Update outcome_met for claims that have been newly resolved."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    try:
        for c in claims:
            if c.get("outcome_met") is not None:
                conn.execute(
                    """
                    UPDATE management_claims
                    SET outcome_met=?, outcome_checked_date=?
                    WHERE ticker=? AND filed_date=? AND claim_type=? AND claim_text=?
                      AND outcome_met IS NULL
                    """,
                    (
                        c["outcome_met"],
                        c.get("outcome_checked_date"),
                        ticker.upper(),
                        c["filed_date"],
                        c["claim_type"],
                        c["claim_text"],
                    ),
                )
        conn.commit()
    finally:
        conn.close()


def _load_resolved_claims(ticker: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM management_claims WHERE ticker=?",
            (ticker.upper(),),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


# ── Score computation ──────────────────────────────────────────────────────────

def compute_credibility_score(ticker: str) -> Dict[str, Any]:
    """
    Compute a credibility score from all resolved claims in the database.

    Returns dict with keys:
        Management_Credibility_Score, Management_Credibility_Label,
        Management_Claims_Resolved, Management_Claims_Met
    """
    all_claims = _load_resolved_claims(ticker)
    resolved = [c for c in all_claims if c.get("outcome_met") is not None]

    n_resolved = len(resolved)
    n_met = sum(1 for c in resolved if c["outcome_met"] == 1)

    if n_resolved < 2:
        return {
            "Ticker": ticker,
            "Management_Credibility_Score": float("nan"),
            "Management_Credibility_Label": "Insufficient Data",
            "Management_Claims_Resolved": n_resolved,
            "Management_Claims_Met": n_met,
        }

    weighted_met = sum(
        (_QUANTIFIED_WEIGHT if c.get("claim_value") is not None else _DIRECTIONAL_WEIGHT)
        for c in resolved
        if c["outcome_met"] == 1
    )
    weighted_total = sum(
        _QUANTIFIED_WEIGHT if c.get("claim_value") is not None else _DIRECTIONAL_WEIGHT
        for c in resolved
    )

    score = (weighted_met / weighted_total) * 100.0 if weighted_total > 0 else 0.0

    if score >= 75:
        label = "Trusted"
    elif score >= 50:
        label = "Moderate"
    else:
        label = "Skeptical"

    return {
        "Ticker": ticker,
        "Management_Credibility_Score": round(score, 1),
        "Management_Credibility_Label": label,
        "Management_Claims_Resolved": n_resolved,
        "Management_Claims_Met": n_met,
    }


# ── Per-ticker pipeline ────────────────────────────────────────────────────────

def _score_single_ticker(
    ticker: str,
    prefetched: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    from forensic.contagion import fetch_filing_history

    empty_result = {
        "Ticker": ticker,
        "Management_Credibility_Score": float("nan"),
        "Management_Credibility_Label": "Insufficient Data",
        "Management_Claims_Resolved": 0,
        "Management_Claims_Met": 0,
    }

    # Retrieve per-ticker financial DataFrames from prefetched maps
    bs_map = prefetched.get("balance_sheet_by_symbol") or {}
    inc_map = prefetched.get("income_statement_by_symbol") or {}
    cf_map = prefetched.get("cash_flow_by_symbol") or {}
    hist_map = prefetched.get("history_by_symbol") or {}

    bs_df = bs_map.get(ticker)
    inc_df = inc_map.get(ticker)
    cf_df = cf_map.get(ticker)
    history_df = hist_map.get(ticker)

    if all(df is None for df in (bs_df, inc_df, cf_df)):
        return empty_result

    # Fetch filing history (Item 7 texts), using cache first
    try:
        filings = fetch_filing_history(ticker, n_filings=5, verbose=verbose)
    except Exception as exc:
        if verbose:
            print(f"  [credibility] Filing fetch failed {ticker}: {exc}")
        filings = []

    # Process each filing: extract claims, resolve, persist
    for filing in filings:
        item_7 = filing.get("item_7_text")
        filed_date = filing.get("filed_date", "")
        if not item_7 or not filed_date:
            continue
        claims = extract_forward_claims(item_7, filed_date)
        if not claims:
            continue
        _save_claims(ticker, claims)
        resolved = resolve_claim_outcomes(claims, bs_df, inc_df, cf_df, history_df)
        _update_outcomes(ticker, resolved)

    return compute_credibility_score(ticker)


# ── Public API ─────────────────────────────────────────────────────────────────

def run_credibility_scan(
    candidates_df: pd.DataFrame,
    prefetched: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run the Longitudinal Credibility Scorer on the candidates DataFrame.

    For each ticker, fetches the last 5 years of 10-K Item 7 (MD&A) filings,
    extracts forward-looking statements, resolves them against actual financial
    outcomes, and computes a weighted credibility score.

    Returns a DataFrame with Ticker + Management_Credibility_* columns,
    left-merged to preserve original row order.
    """
    if "Ticker" not in candidates_df.columns:
        raise ValueError("candidates_df must have a 'Ticker' column")
    if candidates_df.empty:
        return candidates_df[["Ticker"]].copy()

    _ensure_management_tables()

    prefetched = prefetched or {}
    tickers = [str(t).strip().upper() for t in candidates_df["Ticker"].dropna().unique()]

    workers = max(1, min(int(max_workers or 0) or min(4, os.cpu_count() or 2), 8))

    if verbose:
        print(f"  [credibility] Scoring {len(tickers)} tickers ({workers} workers)...")

    results: List[Dict[str, Any]] = []

    if workers == 1 or len(tickers) == 1:
        for ticker in tickers:
            try:
                results.append(_score_single_ticker(ticker, prefetched, verbose))
            except Exception as exc:
                if verbose:
                    print(f"  [credibility] Failed {ticker}: {exc}")
                results.append({
                    "Ticker": ticker,
                    "Management_Credibility_Score": float("nan"),
                    "Management_Credibility_Label": "Insufficient Data",
                    "Management_Claims_Resolved": 0,
                    "Management_Claims_Met": 0,
                })
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for ticker in tickers:
                futures[executor.submit(_score_single_ticker, ticker, prefetched, verbose)] = ticker
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                if verbose:
                    print(f"  [credibility] Failed {ticker}: {exc}")
                results.append({
                    "Ticker": ticker,
                    "Management_Credibility_Score": float("nan"),
                    "Management_Credibility_Label": "Insufficient Data",
                    "Management_Claims_Resolved": 0,
                    "Management_Claims_Met": 0,
                })

    if not results:
        return candidates_df[["Ticker"]].copy()

    result_df = pd.DataFrame(results)
    merged = candidates_df[["Ticker"]].merge(result_df, on="Ticker", how="left")

    if verbose:
        scored = result_df["Management_Credibility_Label"].value_counts()
        print(
            "  [credibility] Done: "
            + ", ".join(f"{v} {k}" for k, v in scored.items())
        )

    return merged
