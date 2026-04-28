"""
Capital Allocation Archaeologist — forensic module.

Analyzes multi-year financial records to detect gaps between management's
disclosed capital allocation priorities and their actual behavior.  A company
that proclaims organic growth while compounding goodwill through acquisitions,
or that pledges dividend commitment while FCF coverage erodes, is showing you
something the quantitative screens miss.

All computation is purely quantitative — uses existing prefetched financials,
no new data sources required.

Public API:
    run_capital_archaeology_scan(candidates_df, prefetched=None, verbose=True,
                                  max_workers=None) -> pd.DataFrame
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ── Financial metric name variants ─────────────────────────────────────────────
# yfinance metric names vary; we try each in order and use the first with data.

_CAPEX_METRICS = ["CapitalExpenditure", "PurchaseOfPPE", "CapitalExpenditures"]
_GOODWILL_METRICS = ["Goodwill", "GoodwillAndIntangibleAssets"]
_DIVS_PAID_METRICS = ["CashDividendsPaid", "CommonStockDividendsPaid", "PaymentOfDividends"]

# ── Series extraction ──────────────────────────────────────────────────────────

def _get_metric_series(
    df: Optional[pd.DataFrame],
    metric: str,
    min_periods: int = 2,
) -> pd.Series:
    """
    Extract a time-sorted float Series for *metric* from a pivoted financial
    DataFrame (index=metric, columns=period strings).  Returns empty Series if
    unavailable or insufficient data.
    """
    if df is None or df.empty or metric not in df.index:
        return pd.Series(dtype=float)
    raw = df.loc[metric].dropna()
    if raw.empty:
        return pd.Series(dtype=float)
    try:
        raw.index = pd.to_datetime(raw.index.astype(str).str[:10], errors="coerce")
    except Exception:
        return pd.Series(dtype=float)
    series = raw.dropna().sort_index().astype(float)
    return series if len(series) >= min_periods else pd.Series(dtype=float)


def _first_nonempty(
    df: Optional[pd.DataFrame], candidates: List[str], min_periods: int = 2
) -> pd.Series:
    """Try each candidate metric name; return the first non-empty Series."""
    for metric in candidates:
        s = _get_metric_series(df, metric, min_periods)
        if not s.empty:
            return s
    return pd.Series(dtype=float)


def _cagr(series: pd.Series) -> Optional[float]:
    """Compound annual growth rate over the full span of a time-indexed Series."""
    if len(series) < 2:
        return None
    v0, vn = series.iloc[0], series.iloc[-1]
    if v0 == 0 or v0 is None:
        return None
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0:
        return None
    try:
        return (abs(vn) / abs(v0)) ** (1 / years) - 1
    except (ZeroDivisionError, ValueError):
        return None


def _trend_label(series: pd.Series, threshold: float = 0.05) -> str:
    """Classify a numeric series as Improving / Stable / Declining."""
    if len(series) < 2:
        return "N/A"
    first_half = series.iloc[: len(series) // 2].mean()
    second_half = series.iloc[len(series) // 2 :].mean()
    if first_half == 0:
        return "N/A"
    change = (second_half - first_half) / abs(first_half)
    if change > threshold:
        return "Improving"
    if change < -threshold:
        return "Declining"
    return "Stable"


# ── Signal detectors ───────────────────────────────────────────────────────────

def detect_acquisition_pattern(
    ticker: str,
    bs_by_symbol: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Detect serial acquisition behavior from goodwill growth vs. equity."""
    bs = bs_by_symbol.get(ticker)
    gw = _first_nonempty(bs, _GOODWILL_METRICS)
    equity = _get_metric_series(bs, "StockholdersEquity")

    if gw.empty or equity.empty:
        return {"is_acquirer": False, "goodwill_cagr": None}

    # Align to common dates
    common = gw.index.intersection(equity.index)
    if len(common) < 2:
        return {"is_acquirer": False, "goodwill_cagr": None}

    gw = gw.loc[common]
    equity = equity.loc[common]

    gw_cagr = _cagr(gw)
    latest_equity = float(equity.iloc[-1]) if not equity.empty else 0
    latest_gw = float(gw.iloc[-1]) if not gw.empty else 0

    # Acquirer if: goodwill CAGR > 10% OR goodwill > 30% of equity
    is_acquirer = bool(
        (gw_cagr is not None and gw_cagr > 0.10)
        or (latest_equity > 0 and latest_gw / latest_equity > 0.30)
    )
    return {"is_acquirer": is_acquirer, "goodwill_cagr": gw_cagr}


def detect_buyback_consistency(
    ticker: str,
    bs_by_symbol: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Detect share buyback discipline from share count trends."""
    bs = bs_by_symbol.get(ticker)
    shares = _get_metric_series(bs, "OrdinarySharesNumber")

    if shares.empty:
        return {"is_cannibal": False, "share_reduction_rate": None}

    share_cagr = _cagr(shares)
    # Cannibal: shares declining at ≥ 2% per year consistently
    is_cannibal = bool(share_cagr is not None and share_cagr < -0.02)
    return {
        "is_cannibal": is_cannibal,
        "share_reduction_rate": share_cagr,
    }


def detect_fcf_dividend_coverage(
    ticker: str,
    cf_by_symbol: Dict[str, pd.DataFrame],
    bs_by_symbol: Dict[str, pd.DataFrame],
    hist_by_symbol: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Detect dividend sustainability from FCF coverage ratio trends."""
    cf = cf_by_symbol.get(ticker)
    fcf = _get_metric_series(cf, "FreeCashFlow")

    # Dividends paid: prefer cash_flow table, fall back to history dividends
    divs_paid = _first_nonempty(cf, _DIVS_PAID_METRICS)

    if fcf.empty:
        return {"coverage_trend": "N/A", "coverage_ratio_latest": None}

    if not divs_paid.empty:
        # Align dates
        common = fcf.index.intersection(divs_paid.index)
        if len(common) >= 2:
            fcf_aligned = fcf.loc[common]
            divs_aligned = divs_paid.loc[common].abs()
            with_divs = divs_aligned[divs_aligned > 0]
            if not with_divs.empty:
                coverage = fcf_aligned.loc[with_divs.index] / with_divs
                coverage = coverage.replace([float("inf"), float("-inf")], None).dropna()
                if not coverage.empty:
                    return {
                        "coverage_trend": _trend_label(coverage),
                        "coverage_ratio_latest": float(coverage.iloc[-1]),
                    }

    # Fall back: derive dividend outflow from history
    hist = hist_by_symbol.get(ticker)
    if hist is not None and not hist.empty:
        div_col = "dividends" if "dividends" in hist.columns else "Dividends"
        date_col = "date" if "date" in hist.columns else "Date"
        if div_col in hist.columns and date_col in hist.columns:
            hist_copy = hist.copy()
            hist_copy[date_col] = pd.to_datetime(hist_copy[date_col], errors="coerce", utc=True)
            hist_copy["year"] = hist_copy[date_col].dt.year
            annual_divs = hist_copy.groupby("year")[div_col].sum()
            if not annual_divs.empty and not fcf.empty:
                # Rough proxy: FCF per share not available, so just check sign
                coverage_latest = (
                    float(fcf.iloc[-1]) / float(annual_divs.iloc[-1])
                    if annual_divs.iloc[-1] != 0
                    else None
                )
                return {
                    "coverage_trend": "N/A",
                    "coverage_ratio_latest": coverage_latest,
                }

    return {"coverage_trend": "N/A", "coverage_ratio_latest": None}


def detect_capex_trajectory(
    ticker: str,
    cf_by_symbol: Dict[str, pd.DataFrame],
    inc_by_symbol: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Detect capital intensity trend (capex as % of revenue)."""
    cf = cf_by_symbol.get(ticker)
    inc = inc_by_symbol.get(ticker)

    capex = _first_nonempty(cf, _CAPEX_METRICS).abs()
    revenue = _get_metric_series(inc, "TotalRevenue")

    if capex.empty or revenue.empty:
        return {"capex_intensity_trend": "N/A", "capex_intensity_latest": None}

    common = capex.index.intersection(revenue.index)
    if len(common) < 2:
        return {"capex_intensity_trend": "N/A", "capex_intensity_latest": None}

    capex_aligned = capex.loc[common]
    rev_aligned = revenue.loc[common]
    intensity = (capex_aligned / rev_aligned.replace(0, None)).dropna()

    if intensity.empty:
        return {"capex_intensity_trend": "N/A", "capex_intensity_latest": None}

    return {
        "capex_intensity_trend": _trend_label(intensity),
        "capex_intensity_latest": float(intensity.iloc[-1]),
    }


def detect_leverage_trajectory(
    ticker: str,
    bs_by_symbol: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Detect debt-to-equity trajectory."""
    bs = bs_by_symbol.get(ticker)
    debt = _get_metric_series(bs, "TotalDebt")
    equity = _get_metric_series(bs, "StockholdersEquity")

    if debt.empty or equity.empty:
        return {"leverage_trend": "N/A", "de_ratio_latest": None}

    common = debt.index.intersection(equity.index)
    if len(common) < 2:
        return {"leverage_trend": "N/A", "de_ratio_latest": None}

    de = (debt.loc[common] / equity.loc[common].replace(0, None)).dropna()
    if de.empty:
        return {"leverage_trend": "N/A", "de_ratio_latest": None}

    return {
        "leverage_trend": _trend_label(de, threshold=0.10),
        "de_ratio_latest": float(de.iloc[-1]),
    }


def detect_roic_trend(
    ticker: str,
    inc_by_symbol: Dict[str, pd.DataFrame],
    bs_by_symbol: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Detect ROIC trend: OperatingIncome / (TotalDebt + StockholdersEquity)."""
    inc = inc_by_symbol.get(ticker)
    bs = bs_by_symbol.get(ticker)

    for op_metric in ("OperatingIncome", "EBIT"):
        op_income = _get_metric_series(inc, op_metric)
        if not op_income.empty:
            break
    else:
        return {"roic_trend": "N/A", "roic_latest": None}

    debt = _get_metric_series(bs, "TotalDebt")
    equity = _get_metric_series(bs, "StockholdersEquity")

    if equity.empty:
        return {"roic_trend": "N/A", "roic_latest": None}

    common = op_income.index.intersection(equity.index)
    if not debt.empty:
        common = common.intersection(debt.index)
    if len(common) < 2:
        return {"roic_trend": "N/A", "roic_latest": None}

    invested = equity.loc[common]
    if not debt.empty:
        invested = invested + debt.loc[common]

    roic = (op_income.loc[common] / invested.replace(0, None)).dropna()
    if roic.empty:
        return {"roic_trend": "N/A", "roic_latest": None}

    return {
        "roic_trend": _trend_label(roic),
        "roic_latest": float(roic.iloc[-1]),
    }


# ── Gap detection ──────────────────────────────────────────────────────────────

def detect_stated_vs_actual_gaps(
    signals: Dict[str, Any],
    item_7_texts: List[str],
) -> List[str]:
    """
    Cross-reference behavioral signals against Item 7 narrative.

    Returns a list of human-readable gap descriptions (empty = no gaps found).
    """
    if not item_7_texts:
        return []

    combined = " ".join(item_7_texts).lower()
    gaps: List[str] = []

    mentions_organic = any(
        phrase in combined for phrase in [
            "organic growth", "grow organically", "without acquisitions", "internal growth"
        ]
    )
    if mentions_organic and signals.get("is_acquirer"):
        gaps.append("Organic growth claimed but acquisitions accelerating (goodwill rising)")

    mentions_div_commitment = any(
        phrase in combined for phrase in [
            "committed to the dividend", "maintain our dividend", "dividend is a priority"
        ]
    )
    if mentions_div_commitment and signals.get("coverage_ratio_latest") is not None:
        if signals["coverage_ratio_latest"] < 0.8:
            gaps.append(
                "Dividend commitment stated but FCF coverage deteriorating "
                f"(coverage {signals['coverage_ratio_latest']:.2f}×)"
            )

    mentions_deleveraging = any(
        phrase in combined for phrase in [
            "reduce leverage", "reduce debt", "deleverage", "pay down debt"
        ]
    )
    if mentions_deleveraging and signals.get("leverage_trend") == "Improving":
        # "Improving" leverage_trend means D/E is going up (higher = worse)
        gaps.append("Deleveraging pledged but debt-to-equity ratio is rising")

    mentions_capex = any(
        phrase in combined for phrase in [
            "invest in our business", "capital investment", "increase capex", "grow capex"
        ]
    )
    if mentions_capex and signals.get("capex_intensity_trend") == "Declining":
        gaps.append("Increased capital investment promised but capex intensity is declining")

    return gaps


# ── Score computation ──────────────────────────────────────────────────────────

def compute_capital_allocation_score(
    signals: Dict[str, Any],
    gaps: List[str],
) -> float:
    """
    Compute a 0–100 capital allocation coherence score.

    Four components, each worth 0–25 points:
      1. Consistency  — stated strategy matches actual allocation
      2. FCF discipline — FCF coverage of capital returned to shareholders
      3. Capital efficiency — ROIC trend
      4. Shareholder alignment — buybacks or dividends, no meaningful dilution
    """
    # 1. Consistency: 25 pts minus 8 per gap flag
    consistency = max(0.0, 25.0 - 8.0 * len(gaps))

    # 2. FCF discipline
    coverage = signals.get("coverage_ratio_latest")
    if coverage is None:
        fcf_score = 12.5  # Neutral when data unavailable
    elif coverage >= 2.0:
        fcf_score = 25.0
    elif coverage >= 1.0:
        fcf_score = 18.0
    elif coverage >= 0.5:
        fcf_score = 8.0
    else:
        fcf_score = 0.0

    # 3. Capital efficiency (ROIC trend)
    roic_trend = signals.get("roic_trend", "N/A")
    roic_map = {"Improving": 25.0, "Stable": 15.0, "Declining": 5.0, "N/A": 12.5}
    efficiency_score = roic_map.get(roic_trend, 12.5)

    # 4. Shareholder alignment
    share_rate = signals.get("share_reduction_rate")  # negative = buybacks
    coverage_ratio = signals.get("coverage_ratio_latest")
    if share_rate is not None and share_rate < -0.02:
        # Consistent buybacks — high alignment
        alignment_score = 25.0
    elif coverage_ratio is not None and coverage_ratio >= 1.0:
        # Dividend well-covered — good alignment
        alignment_score = 20.0
    elif share_rate is not None and share_rate > 0.02:
        # Dilution occurring — poor alignment
        alignment_score = 5.0
    else:
        alignment_score = 12.5

    return round(consistency + fcf_score + efficiency_score + alignment_score, 1)


def _classify_allocation_pattern(signals: Dict[str, Any]) -> str:
    is_cannibal = signals.get("is_cannibal", False)
    is_acquirer = signals.get("is_acquirer", False)
    leverage_trend = signals.get("leverage_trend", "N/A")
    coverage = signals.get("coverage_ratio_latest")
    de_ratio = signals.get("de_ratio_latest")
    capex_trend = signals.get("capex_intensity_trend", "N/A")

    # Strong cannibal: consistent buybacks, low leverage
    if is_cannibal and (de_ratio is None or de_ratio < 1.0):
        return "Cannibal"

    # Active acquirer
    if is_acquirer:
        return "Acquirer"

    # Leverager: rising debt, strained FCF coverage
    if leverage_trend == "Improving" and coverage is not None and coverage < 1.0:
        return "Leverager"

    # Organic-growth oriented: stable/increasing capex, no acquisitions
    if capex_trend in ("Stable", "Improving") and not is_acquirer:
        return "Organic"

    return "Mixed"


# ── Per-ticker pipeline ────────────────────────────────────────────────────────

def _analyze_single_ticker(
    ticker: str,
    prefetched: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    empty = {
        "Ticker": ticker,
        "Capital_Allocation_Score": float("nan"),
        "Capital_Allocation_Pattern": "Insufficient Data",
        "Capital_Allocation_Flags": "",
        "Capital_Allocation_ROIC_Trend": "N/A",
        "Capital_Allocation_FCF_Coverage": None,
    }

    bs_map = prefetched.get("balance_sheet_by_symbol") or {}
    inc_map = prefetched.get("income_statement_by_symbol") or {}
    cf_map = prefetched.get("cash_flow_by_symbol") or {}
    hist_map = prefetched.get("history_by_symbol") or {}

    if not any(m.get(ticker) is not None for m in (bs_map, inc_map, cf_map)):
        return empty

    # Gather all behavioral signals
    acq = detect_acquisition_pattern(ticker, bs_map)
    buyback = detect_buyback_consistency(ticker, bs_map)
    fcf_cov = detect_fcf_dividend_coverage(ticker, cf_map, bs_map, hist_map)
    capex = detect_capex_trajectory(ticker, cf_map, inc_map)
    leverage = detect_leverage_trajectory(ticker, bs_map)
    roic = detect_roic_trend(ticker, inc_map, bs_map)

    signals = {
        **acq,
        **buyback,
        **fcf_cov,
        **capex,
        **leverage,
        **roic,
    }

    # Fetch Item 7 texts for gap detection (from cache — no new EDGAR calls here)
    item_7_texts: List[str] = []
    try:
        from forensic.contagion import _get_cached_filings_n
        cached = _get_cached_filings_n(ticker, 5)
        item_7_texts = [f["item_7_text"] for f in cached if f.get("item_7_text")]
    except Exception:
        pass

    gaps = detect_stated_vs_actual_gaps(signals, item_7_texts)
    score = compute_capital_allocation_score(signals, gaps)
    pattern = _classify_allocation_pattern(signals)

    if verbose and gaps:
        print(f"  [archaeology] {ticker}: {len(gaps)} gap(s): {'; '.join(gaps)}")

    return {
        "Ticker": ticker,
        "Capital_Allocation_Score": score,
        "Capital_Allocation_Pattern": pattern,
        "Capital_Allocation_Flags": "; ".join(gaps) if gaps else "",
        "Capital_Allocation_ROIC_Trend": signals.get("roic_trend", "N/A"),
        "Capital_Allocation_FCF_Coverage": signals.get("coverage_ratio_latest"),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def run_capital_archaeology_scan(
    candidates_df: pd.DataFrame,
    prefetched: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run the Capital Allocation Archaeologist on the candidates DataFrame.

    Detects gaps between stated capital priorities and actual multi-year
    financial behavior using existing prefetched financials.  No new data
    sources or network calls required.

    Returns a DataFrame with Ticker + Capital_Allocation_* columns,
    left-merged to preserve original row order.
    """
    if "Ticker" not in candidates_df.columns:
        raise ValueError("candidates_df must have a 'Ticker' column")
    if candidates_df.empty:
        return candidates_df[["Ticker"]].copy()

    prefetched = prefetched or {}
    tickers = [str(t).strip().upper() for t in candidates_df["Ticker"].dropna().unique()]

    workers = max(1, min(int(max_workers or 0) or min(4, os.cpu_count() or 2), 8))

    if verbose:
        print(f"  [archaeology] Analyzing {len(tickers)} tickers ({workers} workers)...")

    results: List[Dict[str, Any]] = []

    if workers == 1 or len(tickers) == 1:
        for ticker in tickers:
            try:
                results.append(_analyze_single_ticker(ticker, prefetched, verbose))
            except Exception as exc:
                if verbose:
                    print(f"  [archaeology] Failed {ticker}: {exc}")
                results.append({
                    "Ticker": ticker,
                    "Capital_Allocation_Score": float("nan"),
                    "Capital_Allocation_Pattern": "Insufficient Data",
                    "Capital_Allocation_Flags": "",
                    "Capital_Allocation_ROIC_Trend": "N/A",
                    "Capital_Allocation_FCF_Coverage": None,
                })
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for ticker in tickers:
                futures[executor.submit(_analyze_single_ticker, ticker, prefetched, verbose)] = ticker
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                if verbose:
                    print(f"  [archaeology] Failed {ticker}: {exc}")
                results.append({
                    "Ticker": ticker,
                    "Capital_Allocation_Score": float("nan"),
                    "Capital_Allocation_Pattern": "Insufficient Data",
                    "Capital_Allocation_Flags": "",
                    "Capital_Allocation_ROIC_Trend": "N/A",
                    "Capital_Allocation_FCF_Coverage": None,
                })

    if not results:
        return candidates_df[["Ticker"]].copy()

    result_df = pd.DataFrame(results)
    merged = candidates_df[["Ticker"]].merge(result_df, on="Ticker", how="left")

    if verbose:
        patterns = result_df["Capital_Allocation_Pattern"].value_counts()
        print(
            "  [archaeology] Done: "
            + ", ".join(f"{v} {k}" for k, v in patterns.items())
        )

    return merged
