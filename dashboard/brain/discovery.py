"""
Discovery Chronicle — disconfirmation pattern detectors.

Eight pattern detectors, each independent. Each returns a list of discovery
entry dicts (empty list = no tension found). Callers aggregate and persist
to the discovery_entries table via discovery_pipeline.py.
"""

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_MARKET_LEDGER = Path(__file__).parent.parent.parent
_DB_PATH = _MARKET_LEDGER / "database" / "financial_data.db"

# Dividend/buyback metric variant names (same order as mood_pipeline)
_DIV_METRICS = ["CashDividendsPaid", "CommonStockDividendsPaid", "PaymentOfDividends"]
_BUY_METRICS = [
    "RepurchaseOfCapitalStock", "RepurchaseOfCommonStock",
    "CommonStockRepurchase", "StockRepurchase",
]
_DEBT_METRICS = ["TotalDebt", "LongTermDebt", "ShortLongTermDebt"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _entry(
    ticker: str,
    run_date: str,
    pattern_type: str,
    severity: str,
    excerpt: str,
    source_type: str,
    source_form_type: Optional[str] = None,
    source_filing_date: Optional[str] = None,
    investment_signal: Optional[str] = None,
    mos_pct: Optional[float] = None,
    quant_metric: Optional[str] = None,
    quant_value: Optional[float] = None,
    quant_threshold: Optional[float] = None,
) -> dict:
    return {
        "ticker": ticker,
        "run_date": run_date,
        "pattern_type": pattern_type,
        "severity": severity,
        "investment_signal": investment_signal,
        "mos_pct": mos_pct,
        "source_type": source_type,
        "source_form_type": source_form_type,
        "source_filing_date": source_filing_date,
        "excerpt": excerpt,
        "quant_metric": quant_metric,
        "quant_value": quant_value,
        "quant_threshold": quant_threshold,
        "discovered_at": datetime.now().isoformat(),
    }


def _extract_sentence(text: str, match: re.Match) -> str:
    """Return the sentence (up to 400 chars) containing a regex match."""
    start = max(0, match.start() - 200)
    end = min(len(text), match.end() + 200)
    fragment = text[start:end]
    # Try to trim to nearest sentence boundaries
    sent_start = max(
        fragment.rfind(". ", 0, match.start() - start) + 2,
        fragment.rfind("\n", 0, match.start() - start) + 1,
        0,
    )
    sent_end = fragment.find(". ", match.end() - start)
    if sent_end == -1:
        sent_end = len(fragment)
    else:
        sent_end += 2
    return fragment[sent_start:sent_end].strip()[:400]


def _first_series(wide: pd.DataFrame, variants: list) -> pd.Series:
    for m in variants:
        if m in wide.index:
            return wide.loc[m]
    return pd.Series(dtype=float)


def _to_wide(long_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    t = long_df[long_df["symbol"] == ticker]
    if t.empty:
        return pd.DataFrame()
    return t.pivot(index="metric", columns="period", values="value")


# ── Pattern 1: Distribution Coverage ──────────────────────────────────────────

def check_distribution_coverage(
    ticker: str, run_date: str, cf_long: pd.DataFrame,
    investment_signal: Optional[str] = None, mos_pct: Optional[float] = None,
) -> list:
    wide = _to_wide(cf_long, ticker)
    if wide.empty:
        return []

    fcf_row = _first_series(wide, ["FreeCashFlow"])
    div_row = _first_series(wide, _DIV_METRICS)
    buy_row = _first_series(wide, _BUY_METRICS)

    if fcf_row.empty or (div_row.empty and buy_row.empty):
        return []

    periods = sorted(set(fcf_row.index) & (set(div_row.index) | set(buy_row.index)))
    if len(periods) < 2:
        return []
    periods = periods[-4:]

    div = div_row.reindex(periods).astype(float).abs() if not div_row.empty else pd.Series(0.0, index=periods)
    buy = buy_row.reindex(periods).astype(float).abs() if not buy_row.empty else pd.Series(0.0, index=periods)
    distributions = div.fillna(0.0) + buy.fillna(0.0)
    nonzero = distributions[distributions > 0]
    if nonzero.empty:
        return []

    fcf = fcf_row.reindex(nonzero.index).astype(float)
    coverage = (fcf / distributions[nonzero.index]).replace([np.inf, -np.inf], np.nan).dropna()
    if coverage.empty:
        return []

    mean_cov = float(coverage.mean())
    if mean_cov < 1.0:
        sev = "warning"
    elif mean_cov < 1.2:
        sev = "caution"
    else:
        return []

    return [_entry(
        ticker=ticker, run_date=run_date, pattern_type="distribution_coverage",
        severity=sev,
        excerpt=f"FCF/distributions = {mean_cov:.2f}x (trailing {len(coverage)} periods)",
        source_type="quantitative",
        investment_signal=investment_signal, mos_pct=mos_pct,
        quant_metric="FCF/distributions", quant_value=round(mean_cov, 3),
        quant_threshold=1.0 if mean_cov < 1.0 else 1.2,
    )]


# ── Pattern 2: Leverage Trend ─────────────────────────────────────────────────

def check_leverage_trend(
    ticker: str, run_date: str, bs_long: pd.DataFrame,
    investment_signal: Optional[str] = None, mos_pct: Optional[float] = None,
) -> list:
    wide = _to_wide(bs_long, ticker)
    if wide.empty:
        return []

    debt_row = _first_series(wide, _DEBT_METRICS)
    assets_row = _first_series(wide, ["TotalAssets"])
    if debt_row.empty or assets_row.empty:
        return []

    periods = sorted(set(debt_row.index) & set(assets_row.index))
    if len(periods) < 4:
        return []
    periods = periods[-8:]

    debt = debt_row.reindex(periods).astype(float)
    assets = assets_row.reindex(periods).astype(float)
    ratio = (debt / assets).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratio) < 3:
        return []

    x = np.arange(len(ratio), dtype=float)
    slope = float(np.polyfit(x, ratio.values, 1)[0])

    if slope < 0.05:
        return []

    return [_entry(
        ticker=ticker, run_date=run_date, pattern_type="leverage_trend",
        severity="warning" if slope > 0.1 else "caution",
        excerpt=f"Debt/assets ratio trending +{slope:.1%}/period over trailing {len(ratio)} periods",
        source_type="quantitative",
        investment_signal=investment_signal, mos_pct=mos_pct,
        quant_metric="debt/assets slope", quant_value=round(slope, 4),
        quant_threshold=0.05,
    )]


# ── Text pattern helpers ───────────────────────────────────────────────────────

_GOING_CONCERN_RE = re.compile(
    r"going.concern|substantial\s+doubt.{0,60}(?:ability|continue)", re.IGNORECASE
)
_MATERIAL_WEAKNESS_RE = re.compile(
    r"material\s+weakness|significant\s+deficiency", re.IGNORECASE
)
_COVENANT_RE = re.compile(
    r"covenant\s+(?:waiver|amendment|default)|waived.{0,20}covenant"
    r"|amend.{0,20}credit\s+(?:facility|agreement)|credit\s+agreement.{0,30}amend",
    re.IGNORECASE,
)
_CUSTOMER_CONC_RE = re.compile(
    r"(\d{1,3})%?\s+of\s+(?:our\s+|total\s+|annual\s+)?(?:net\s+)?(?:revenue|sales|net\s+revenues)"
    r".{0,80}(?:customer|client)"
    r"|(?:customer|client).{0,80}(\d{1,3})%?\s+of\s+(?:our\s+|total\s+)?(?:revenue|sales)",
    re.IGNORECASE,
)
_MGMT_CHANGE_RE = re.compile(
    r"(?:departure|resignation|retired|stepped\s+down|no\s+longer\s+serves)"
    r".{0,80}(?:Chief|President|CEO|CFO|COO|CTO|Chairman)"
    r"|(?:Chief|President|CEO|CFO|COO|CTO|Chairman)"
    r".{0,80}(?:departure|resignation|retired|stepped\s+down|no\s+longer)",
    re.IGNORECASE,
)


def _text_pattern_check(
    ticker: str, run_date: str, pattern_type: str, severity: str,
    regex: re.Pattern, text: Optional[str], source_type: str,
    form_type: Optional[str], filing_date: Optional[str],
    investment_signal: Optional[str], mos_pct: Optional[float],
    max_hits: int = 2,
) -> list:
    if not text:
        return []
    results = []
    for m in list(regex.finditer(text))[:max_hits]:
        excerpt = _extract_sentence(text, m)
        results.append(_entry(
            ticker=ticker, run_date=run_date, pattern_type=pattern_type,
            severity=severity, excerpt=excerpt,
            source_type=source_type, source_form_type=form_type,
            source_filing_date=filing_date,
            investment_signal=investment_signal, mos_pct=mos_pct,
        ))
    return results


# ── Patterns 3-7: Text-based ───────────────────────────────────────────────────

def check_going_concern(ticker, run_date, filing_data, investment_signal=None, mos_pct=None):
    results = []
    for f in filing_data:
        for src_type, text in [("item_1a", f.get("item_1a_text")), ("item_7", f.get("item_7_text"))]:
            results += _text_pattern_check(
                ticker, run_date, "going_concern", "warning",
                _GOING_CONCERN_RE, text, src_type,
                f.get("form_type"), f.get("filed_date"),
                investment_signal, mos_pct, max_hits=1,
            )
    return results[:2]


def check_material_weakness(ticker, run_date, filing_data, investment_signal=None, mos_pct=None):
    results = []
    for f in filing_data:
        results += _text_pattern_check(
            ticker, run_date, "material_weakness", "warning",
            _MATERIAL_WEAKNESS_RE, f.get("item_1a_text"), "item_1a",
            f.get("form_type"), f.get("filed_date"),
            investment_signal, mos_pct, max_hits=1,
        )
    return results[:2]


def check_covenant_pressure(ticker, run_date, filing_data, investment_signal=None, mos_pct=None):
    results = []
    for f in filing_data:
        for src_type, text in [("item_1a", f.get("item_1a_text")), ("item_7", f.get("item_7_text"))]:
            results += _text_pattern_check(
                ticker, run_date, "covenant_pressure", "caution",
                _COVENANT_RE, text, src_type,
                f.get("form_type"), f.get("filed_date"),
                investment_signal, mos_pct, max_hits=1,
            )
    return results[:2]


def check_customer_concentration(ticker, run_date, filing_data, investment_signal=None, mos_pct=None):
    results = []
    for f in filing_data:
        for m in list(_CUSTOMER_CONC_RE.finditer(f.get("item_1a_text") or ""))[:2]:
            pct_str = m.group(1) or m.group(2) or "0"
            try:
                pct = int(pct_str)
            except ValueError:
                pct = 0
            severity = "caution" if pct >= 20 else "note"
            excerpt = _extract_sentence(f["item_1a_text"], m)
            results.append(_entry(
                ticker=ticker, run_date=run_date, pattern_type="customer_concentration",
                severity=severity, excerpt=excerpt, source_type="item_1a",
                source_form_type=f.get("form_type"), source_filing_date=f.get("filed_date"),
                investment_signal=investment_signal, mos_pct=mos_pct,
                quant_metric="customer_revenue_pct", quant_value=float(pct),
                quant_threshold=20.0,
            ))
    return results[:2]


def check_management_change(ticker, run_date, filing_data, investment_signal=None, mos_pct=None):
    results = []
    for f in filing_data:
        for src_type, text in [("item_7", f.get("item_7_text"))]:
            results += _text_pattern_check(
                ticker, run_date, "management_change", "note",
                _MGMT_CHANGE_RE, text, src_type,
                f.get("form_type"), f.get("filed_date"),
                investment_signal, mos_pct, max_hits=1,
            )
    return results[:2]


# ── Pattern 8: Guidance Miss Streak ───────────────────────────────────────────

def check_guidance_miss_streak(
    ticker: str, run_date: str,
    investment_signal: Optional[str] = None, mos_pct: Optional[float] = None,
) -> list:
    """Flag if mood_vectors.operational_confidence < -0.3 (persistent under-delivery)."""
    conn = sqlite3.connect(str(_DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT operational_confidence FROM mood_vectors WHERE ticker=? "
            "ORDER BY filing_date DESC LIMIT 1",
            (ticker.upper(),),
        ).fetchone()
    finally:
        conn.close()

    if row is None or row["operational_confidence"] is None:
        return []
    oc = float(row["operational_confidence"])
    if oc >= -0.3:
        return []

    accuracy_pct = (oc * 50.0) + 50.0
    return [_entry(
        ticker=ticker, run_date=run_date, pattern_type="guidance_miss_streak",
        severity="caution",
        excerpt=(
            f"Management credibility at {accuracy_pct:.0f}% claim accuracy "
            f"(score {oc:.2f}). Consistent pattern of under-delivering vs. prior guidance."
        ),
        source_type="quantitative",
        investment_signal=investment_signal, mos_pct=mos_pct,
        quant_metric="operational_confidence", quant_value=round(oc, 3),
        quant_threshold=-0.3,
    )]


# ── Public API ─────────────────────────────────────────────────────────────────

def run_all_detectors(
    ticker: str,
    run_date: str,
    cf_long: pd.DataFrame,
    bs_long: pd.DataFrame,
    filing_data: list,
    investment_signal: Optional[str] = None,
    mos_pct: Optional[float] = None,
) -> list:
    """
    Run all 8 pattern detectors for a ticker. Returns a flat list of discovery
    entry dicts. Empty list = no tensions found.
    """
    results = []
    results += check_distribution_coverage(ticker, run_date, cf_long, investment_signal, mos_pct)
    results += check_leverage_trend(ticker, run_date, bs_long, investment_signal, mos_pct)
    results += check_going_concern(ticker, run_date, filing_data, investment_signal, mos_pct)
    results += check_material_weakness(ticker, run_date, filing_data, investment_signal, mos_pct)
    results += check_covenant_pressure(ticker, run_date, filing_data, investment_signal, mos_pct)
    results += check_customer_concentration(ticker, run_date, filing_data, investment_signal, mos_pct)
    results += check_management_change(ticker, run_date, filing_data, investment_signal, mos_pct)
    results += check_guidance_miss_streak(ticker, run_date, investment_signal, mos_pct)
    return results
