"""
valuation_sanity_gate.py
------------------------
Multi-check valuation sanity gate with typed failure codes and rich diagnostics.

Public API
----------
    # Rich API
    run_sanity_gate(ctx: SanityContext) -> SanityResult

    # Backward-compatible drop-in for old check_valuation_sanity()
    check_valuation_sanity(fair_value, current_price, **kwargs) -> (bool, str)

    # Helper for valuation_projector.py
    _get_metric_series(df, metric_name, n_periods) -> List[float]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1.  TYPED FAILURE CODES
#     Each code maps to a suppression behaviour downstream.
#     "SUPPRESS_SIGNAL"  → keep row in output, blank Investment Signal
#     "EXCLUDE"          → drop row from selected portfolio
#     "WARN"             → keep row and signal, add warning annotation
# ---------------------------------------------------------------------------

SANITY_FAILURE_BEHAVIORS = {
    # Data quality failures — model inputs are known bad
    "NEGATIVE_FCF_DCF":         "SUPPRESS_SIGNAL",   # DCF on negative FCF is nonsense
    "ZERO_PRICE":               "EXCLUDE",            # Can't compute MOS at all
    "ZERO_FAIR_VALUE":          "EXCLUDE",            # Waterfall produced nothing usable
    "SHARES_IMPLAUSIBLE":       "SUPPRESS_SIGNAL",    # Share count looks like a data error
    "REVENUE_IMPLAUSIBLE":      "SUPPRESS_SIGNAL",    # Revenue used in P/S looks wrong

    # Model applicability failures — DCF/P/S doesn't fit the company type
    "SECTOR_EXCLUDED":          "SUPPRESS_SIGNAL",    # Financials, REITs, SPACs
    "NEGATIVE_EQUITY_PS":       "SUPPRESS_SIGNAL",    # Negative equity + P/S fallback
    "HOLDING_COMPANY":          "WARN",               # Revenue mostly from subs, not ops

    # Ratio bound failures — the ratio check from the old gate, now typed
    "FV_TOO_LOW":               "SUPPRESS_SIGNAL",    # FV < 20% of price (sanity floor)
    "FV_TOO_HIGH":              "SUPPRESS_SIGNAL",    # FV > 500% of price (sanity ceiling)
    "FV_EXTREME_HIGH":          "EXCLUDE",            # FV > 1000% of price (data error)

    # Cyclicality / earnings quality flags — soft warnings, not hard failures
    "CYCLICAL_PEAK_RISK":       "WARN",               # FCF looks unusually high vs history
    "CYCLICAL_TROUGH_RISK":     "WARN",               # FCF looks unusually low vs history
    "PS_FALLBACK_LOW_MARGIN":   "WARN",               # P/S used but op margin < 0
    "SINGLE_PERIOD_FCF":        "WARN",               # Only one FCF period available
}

# Sectors where DCF simply doesn't apply — always route to SUPPRESS_SIGNAL
EXCLUDED_SECTORS = {
    "Financial Services",
    "Financials",
    "Banks",
    "Insurance",
    "Real Estate",
    "REIT",
    "Diversified Financials",
}

# Keywords in company name that suggest holding/conglomerate structure
HOLDING_COMPANY_KEYWORDS = {
    "holdings", "holding", "group", "partners", "capital", "investment",
    "acquisition", "acquisitions", "ventures",
}


# ---------------------------------------------------------------------------
# 2.  INPUT CONTEXT — everything the gate needs
# ---------------------------------------------------------------------------

@dataclass
class SanityContext:
    """
    All inputs the sanity gate needs to make its determination.
    Pull these from the same sources already available in
    _analyze_ticker_valuation() and pass as a single object.
    """
    ticker: str
    fair_value: float
    current_price: float
    valuation_method: str           # "DCF", "P/S", "P/B", "N/A"

    # FCF inputs
    fcf_latest: float = 0.0         # Most recent period FCF
    fcf_series: List[float] = field(default_factory=list)   # Last N periods (newest first)

    # Revenue / equity for P/S sanity
    total_revenue: float = 0.0
    stockholders_equity: float = 0.0
    operating_income: float = 0.0

    # Share count plausibility
    shares_outstanding: float = 0.0

    # Company metadata
    sector: Optional[str] = None
    company_name: Optional[str] = None
    market_cap_mm: Optional[float] = None   # millions

    # Strategy context
    strategy: Optional[str] = None


# ---------------------------------------------------------------------------
# 3.  RESULT TYPE
# ---------------------------------------------------------------------------

@dataclass
class SanityResult:
    """
    Structured result from run_sanity_gate().

    passed:        True iff no EXCLUDE or SUPPRESS_SIGNAL failures fired.
    behavior:      The most severe suppression behaviour across all failures.
                   Precedence: EXCLUDE > SUPPRESS_SIGNAL > WARN > None
    failure_codes: All failure codes that fired, in detection order.
    reason:        Human-readable primary reason (first non-WARN failure,
                   or first WARN if all soft).
    diagnostics:   Pipe-delimited string of all fired codes — goes into the
                   Valuation_Diagnostics column for operator inspection.
    """
    passed: bool
    behavior: Optional[str]          # None means fully clean
    failure_codes: List[str]
    reason: str
    diagnostics: str

    # Convenience — mirrors the old (bool, str) return for backward compat
    def as_legacy_tuple(self) -> Tuple[bool, str]:
        return (self.passed, self.reason)


# ---------------------------------------------------------------------------
# 4.  INDIVIDUAL CHECK FUNCTIONS
#     Each returns (failure_code | None).
#     None means the check passed.
# ---------------------------------------------------------------------------

def _check_zero_price(ctx: SanityContext) -> Optional[str]:
    if ctx.current_price <= 0:
        return "ZERO_PRICE"
    return None


def _check_zero_fair_value(ctx: SanityContext) -> Optional[str]:
    if ctx.fair_value <= 0:
        return "ZERO_FAIR_VALUE"
    return None


def _check_negative_fcf_dcf(ctx: SanityContext) -> Optional[str]:
    """DCF was used but the underlying FCF was negative."""
    if ctx.valuation_method == "DCF" and ctx.fcf_latest <= 0:
        return "NEGATIVE_FCF_DCF"
    return None


def _check_sector_exclusion(ctx: SanityContext) -> Optional[str]:
    """Financials, REITs, and SPACs — DCF and P/S produce structural nonsense."""
    if ctx.sector and ctx.sector in EXCLUDED_SECTORS:
        return "SECTOR_EXCLUDED"
    return None


def _check_holding_company(ctx: SanityContext) -> Optional[str]:
    """Soft flag — holding companies have intercompany revenue noise."""
    if ctx.company_name:
        name_lower = ctx.company_name.lower()
        if any(kw in name_lower for kw in HOLDING_COMPANY_KEYWORDS):
            return "HOLDING_COMPANY"
    return None


def _check_shares_plausibility(ctx: SanityContext) -> Optional[str]:
    """Implausible share counts are a common yfinance data error."""
    if ctx.shares_outstanding <= 0:
        return "SHARES_IMPLAUSIBLE"

    if ctx.market_cap_mm and ctx.market_cap_mm > 0 and ctx.current_price > 0:
        implied_mktcap_mm = (ctx.shares_outstanding * ctx.current_price) / 1_000_000
        ratio = implied_mktcap_mm / ctx.market_cap_mm
        if ratio < 0.1 or ratio > 10.0:
            return "SHARES_IMPLAUSIBLE"

    return None


def _check_revenue_plausibility(ctx: SanityContext) -> Optional[str]:
    """P/S fallback with suspicious revenue."""
    if ctx.valuation_method != "P/S":
        return None

    if ctx.total_revenue <= 0:
        return "REVENUE_IMPLAUSIBLE"

    if ctx.market_cap_mm and ctx.market_cap_mm > 0:
        ps_ratio = (ctx.market_cap_mm * 1_000_000) / ctx.total_revenue
        if ps_ratio > 50:
            return "REVENUE_IMPLAUSIBLE"

    return None


def _check_negative_equity_ps(ctx: SanityContext) -> Optional[str]:
    """P/S fallback on a company with deeply negative equity."""
    if ctx.valuation_method == "P/S" and ctx.stockholders_equity < 0:
        return "NEGATIVE_EQUITY_PS"
    return None


def _check_ratio_bounds(ctx: SanityContext) -> Optional[str]:
    """Three-tier ratio bound check replacing the old min/max check."""
    if ctx.current_price <= 0 or ctx.fair_value <= 0:
        return None   # Already caught by earlier checks

    ratio = ctx.fair_value / ctx.current_price

    if ratio < 0.20:
        return "FV_TOO_LOW"
    if ratio > 10.0:
        return "FV_EXTREME_HIGH"
    if ratio > 5.0:
        return "FV_TOO_HIGH"

    return None


def _check_ps_low_margin(ctx: SanityContext) -> Optional[str]:
    """Soft warning: P/S fallback used but operating margin is negative."""
    if ctx.valuation_method != "P/S":
        return None
    if ctx.total_revenue > 0 and ctx.operating_income < 0:
        return "PS_FALLBACK_LOW_MARGIN"
    return None


def _check_cyclicality(ctx: SanityContext) -> Optional[str]:
    """Soft warning: compare latest FCF to multi-period median."""
    if ctx.valuation_method != "DCF":
        return None
    if len(ctx.fcf_series) < 3:
        return None
    if ctx.fcf_latest <= 0:
        return None

    median_fcf = float(np.median([f for f in ctx.fcf_series if f > 0] or [ctx.fcf_latest]))
    if median_fcf <= 0:
        return None

    ratio = ctx.fcf_latest / median_fcf
    if ratio > 2.0:
        return "CYCLICAL_PEAK_RISK"
    if ratio < 0.3:
        return "CYCLICAL_TROUGH_RISK"

    return None


def _check_single_period_fcf(ctx: SanityContext) -> Optional[str]:
    """Soft warning: DCF is based on only one FCF period."""
    if ctx.valuation_method == "DCF" and len(ctx.fcf_series) <= 1:
        return "SINGLE_PERIOD_FCF"
    return None


# ---------------------------------------------------------------------------
# 5.  THE GATE RUNNER
# ---------------------------------------------------------------------------

_HARD_CHECKS = [
    _check_zero_price,
    _check_zero_fair_value,
    _check_sector_exclusion,
    _check_negative_fcf_dcf,
    _check_shares_plausibility,
    _check_revenue_plausibility,
    _check_negative_equity_ps,
    _check_ratio_bounds,
]

_SOFT_CHECKS = [
    _check_holding_company,
    _check_ps_low_margin,
    _check_cyclicality,
    _check_single_period_fcf,
]

_BEHAVIOR_PRECEDENCE = {"EXCLUDE": 3, "SUPPRESS_SIGNAL": 2, "WARN": 1, None: 0}


def run_sanity_gate(ctx: SanityContext) -> SanityResult:
    """
    Run all checks and return a typed SanityResult.

    Hard checks are run first. If any hard check fires an EXCLUDE, soft
    checks are skipped (not actionable when valuation is already excluded).
    """
    fired: List[str] = []
    worst_behavior: Optional[str] = None

    def _record(code: Optional[str]) -> None:
        nonlocal worst_behavior
        if code is None:
            return
        fired.append(code)
        behavior = SANITY_FAILURE_BEHAVIORS.get(code, "WARN")
        if _BEHAVIOR_PRECEDENCE.get(behavior, 0) > _BEHAVIOR_PRECEDENCE.get(worst_behavior, 0):
            worst_behavior = behavior

    has_exclude = False
    for check_fn in _HARD_CHECKS:
        _record(check_fn(ctx))
        if worst_behavior == "EXCLUDE":
            has_exclude = True

    if not has_exclude:
        for check_fn in _SOFT_CHECKS:
            _record(check_fn(ctx))

    passed = worst_behavior not in ("EXCLUDE", "SUPPRESS_SIGNAL")

    reason = "Passed"
    for code in fired:
        behavior = SANITY_FAILURE_BEHAVIORS.get(code, "WARN")
        if behavior in ("EXCLUDE", "SUPPRESS_SIGNAL"):
            reason = _code_to_reason(code)
            break
    else:
        if fired:
            reason = _code_to_reason(fired[0])

    diagnostics = "|".join(fired) if fired else "OK"

    return SanityResult(
        passed=passed,
        behavior=worst_behavior,
        failure_codes=fired,
        reason=reason,
        diagnostics=diagnostics,
    )


def _code_to_reason(code: str) -> str:
    """Human-readable reason for each failure code."""
    return {
        "NEGATIVE_FCF_DCF":       "DCF applied to negative FCF — unreliable",
        "ZERO_PRICE":             "Invalid current price (zero or negative)",
        "ZERO_FAIR_VALUE":        "No valid fair value produced",
        "SHARES_IMPLAUSIBLE":     "Share count looks like a data error",
        "REVENUE_IMPLAUSIBLE":    "Revenue data implausible for P/S valuation",
        "SECTOR_EXCLUDED":        "Sector not suitable for DCF/P/S valuation",
        "NEGATIVE_EQUITY_PS":     "P/S fallback with negative equity",
        "HOLDING_COMPANY":        "Holding company structure — revenue may not reflect operations",
        "FV_TOO_LOW":             "Fair value < 20% of market price",
        "FV_TOO_HIGH":            "Fair value > 500% of market price",
        "FV_EXTREME_HIGH":        "Fair value > 1000% of market price — likely data error",
        "CYCLICAL_PEAK_RISK":     "FCF appears to be at cyclical peak — may overstate value",
        "CYCLICAL_TROUGH_RISK":   "FCF appears to be at cyclical trough — may understate value",
        "PS_FALLBACK_LOW_MARGIN": "P/S fallback used but operating margin is negative",
        "SINGLE_PERIOD_FCF":      "DCF based on single FCF period only",
    }.get(code, code)


# ---------------------------------------------------------------------------
# 6.  BACKWARD-COMPATIBLE WRAPPER
# ---------------------------------------------------------------------------

def check_valuation_sanity(
    fair_value: float,
    current_price: float,
    ticker: str = "",
    valuation_method: str = "DCF",
    fcf_latest: float = 0.0,
    fcf_series: Optional[List[float]] = None,
    total_revenue: float = 0.0,
    stockholders_equity: float = 0.0,
    operating_income: float = 0.0,
    shares_outstanding: float = 0.0,
    sector: Optional[str] = None,
    company_name: Optional[str] = None,
    market_cap_mm: Optional[float] = None,
    strategy: Optional[str] = None,
    config=None,   # kept for signature compat, ignored
) -> Tuple[bool, str]:
    """
    Drop-in replacement for the original check_valuation_sanity().
    Still returns (bool, str) for backward compatibility.
    Pass additional kwargs to enable the richer checks.
    Call run_sanity_gate(ctx) directly for the full SanityResult.
    """
    ctx = SanityContext(
        ticker=ticker,
        fair_value=fair_value,
        current_price=current_price,
        valuation_method=valuation_method,
        fcf_latest=fcf_latest,
        fcf_series=fcf_series or [],
        total_revenue=total_revenue,
        stockholders_equity=stockholders_equity,
        operating_income=operating_income,
        shares_outstanding=shares_outstanding,
        sector=sector,
        company_name=company_name,
        market_cap_mm=market_cap_mm,
        strategy=strategy,
    )
    return run_sanity_gate(ctx).as_legacy_tuple()


# ---------------------------------------------------------------------------
# 7.  HELPER — add to valuation_projector.py alongside _get_latest_metric()
# ---------------------------------------------------------------------------

def _get_metric_series(
    df: Optional[pd.DataFrame],
    metric_name: str,
    n_periods: int = 5,
) -> List[float]:
    """
    Return up to n_periods non-null values for a metric from a pivoted
    financial DataFrame, most-recent first.
    """
    if df is None or df.empty or metric_name not in df.index:
        return []

    row = df.loc[metric_name]
    sorted_cols = sorted(row.index, reverse=True)
    values: List[float] = []
    for col in sorted_cols:
        if len(values) >= n_periods:
            break
        val = row[col]
        if pd.notna(val):
            try:
                values.append(float(val))
            except (TypeError, ValueError):
                continue
    return values


# ---------------------------------------------------------------------------
# 8.  QUICK SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Sanity Gate Self-Test ===\n")

    cases = [
        ("Clean DCF",
         dict(fair_value=120.0, current_price=100.0, valuation_method="DCF",
              fcf_latest=10.0, fcf_series=[10.0, 9.0, 11.0], shares_outstanding=1e6),
         True, None),

        ("Negative FCF in DCF",
         dict(fair_value=50.0, current_price=100.0, valuation_method="DCF",
              fcf_latest=-5.0, fcf_series=[-5.0, 8.0, 7.0], shares_outstanding=1e6),
         False, "SUPPRESS_SIGNAL"),

        ("Financial sector",
         dict(fair_value=120.0, current_price=100.0, valuation_method="DCF",
              fcf_latest=10.0, sector="Financial Services", shares_outstanding=1e6),
         False, "SUPPRESS_SIGNAL"),

        ("FV extreme high",
         dict(fair_value=1200.0, current_price=100.0, valuation_method="P/S",
              total_revenue=1e9, shares_outstanding=1e6),
         False, "EXCLUDE"),

        ("FV too low",
         dict(fair_value=15.0, current_price=100.0, valuation_method="DCF",
              fcf_latest=5.0, shares_outstanding=1e6),
         False, "SUPPRESS_SIGNAL"),

        ("Cyclical peak (soft warning only)",
         dict(fair_value=120.0, current_price=100.0, valuation_method="DCF",
              fcf_latest=30.0, fcf_series=[30.0, 10.0, 11.0, 9.0], shares_outstanding=1e6),
         True, "WARN"),

        ("P/S negative equity",
         dict(fair_value=120.0, current_price=100.0, valuation_method="P/S",
              total_revenue=5e8, stockholders_equity=-1e7, shares_outstanding=1e6),
         False, "SUPPRESS_SIGNAL"),

        ("Zero price",
         dict(fair_value=100.0, current_price=0.0, valuation_method="DCF",
              fcf_latest=5.0, shares_outstanding=1e6),
         False, "EXCLUDE"),
    ]

    all_pass = True
    for label, kwargs, exp_passed, exp_behavior in cases:
        ctx = SanityContext(ticker="TEST", **kwargs)
        result = run_sanity_gate(ctx)
        ok = result.passed == exp_passed and result.behavior == exp_behavior
        status = "OK" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {label}")
        print(f"      passed={result.passed}  behavior={result.behavior}  codes={result.failure_codes}")
        print(f"      reason: {result.reason}")
        if not ok:
            print(f"      EXPECTED: passed={exp_passed}  behavior={exp_behavior}")
        print()

    print("All tests passed." if all_pass else "SOME TESTS FAILED.")
