"""
Valuation Projector Module - DCF valuation and price projections.

This module provides intrinsic value calculations using Discounted Cash Flow (DCF)
analysis and projects future prices based on historical growth rates.

Features:
---------
- Strategy-conditioned valuation parameters (moat, cannibal, magic_formula, etc.)
- Bear/Base/Bull scenario analysis with probability-weighted fair value
- Valuation waterfall: DCF → P/S → P/B fallbacks
- Epistemic confidence scoring

Usage:
------
    from valuation.valuation_projector import run_valuation_scan

    # Pass a DataFrame with tickers to analyze
    buy_list_df = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT"],
        "Strategy": ["magic_formula", "moat"]
    })
    result_df = run_valuation_scan(buy_list_df)

    # Result includes: Fair Value (Bear/Base/Bull), Valuation Method,
    # Undervalued %, Model Confidence, Price Projections
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, Iterable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add pyfinancial to path to import database module
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "pyfinancial"))
from pyfinancial.database.database import (
    get_ticker_history,
    get_financial_data,
    get_ticker_history_bulk,
    get_financial_data_bulk,
)


# ==============================================================================
# STRATEGY-CONDITIONED VALUATION PARAMETERS
# ==============================================================================


@dataclass
class ValuationParams:
    """Valuation parameters tied to strategy identity."""

    growth_rate: float
    discount_rate: float
    terminal_multiple: int
    description: str


# Strategy-specific valuation assumptions
STRATEGY_VALUATION_PARAMS: Dict[str, ValuationParams] = {
    "moat": ValuationParams(
        growth_rate=0.05,  # Lower growth - stable, mature businesses
        discount_rate=0.08,  # Lower discount rate - lower risk due to moat
        terminal_multiple=15,  # Higher multiple - durable competitive advantage
        description="Wide moat: stable growth, lower risk premium",
    ),
    "cannibal": ValuationParams(
        growth_rate=0.05,  # Lower revenue growth - returning capital vs reinvesting
        discount_rate=0.10,  # Standard discount rate
        terminal_multiple=18,  # Higher terminal multiple - buybacks boost per-share value
        description="Buyback cannibals: lower growth, enhanced per-share value",
    ),
    "magic_formula": ValuationParams(
        growth_rate=0.08,  # Moderate growth - quality + value blend
        discount_rate=0.12,  # Stricter discount rate - margin of safety
        terminal_multiple=12,  # Conservative multiple
        description="Magic formula: moderate growth, strict margin of safety",
    ),
    "fcf_yield": ValuationParams(
        growth_rate=0.06,  # Moderate-low growth - mature cash generators
        discount_rate=0.10,  # Standard discount rate
        terminal_multiple=12,  # Conservative multiple
        description="FCF yield: cash cows with stable cash generation",
    ),
    "quality": ValuationParams(
        growth_rate=0.10,  # Higher growth - quality businesses can reinvest
        discount_rate=0.10,  # Standard discount rate
        terminal_multiple=14,  # Above-average multiple for quality
        description="Quality: higher sustainable growth, premium valuation",
    ),
    "default": ValuationParams(
        growth_rate=0.08,  # Default moderate growth
        discount_rate=0.10,  # Default 10% discount rate
        terminal_multiple=12,  # Default conservative multiple
        description="Default: balanced assumptions",
    ),
}


def get_strategy_params(strategy: Optional[str]) -> ValuationParams:
    """
    Get valuation parameters for a given strategy.

    Args:
        strategy: Strategy name (moat, cannibal, magic_formula, etc.)

    Returns:
        ValuationParams for the strategy, or default if unknown
    """
    if strategy is None:
        return STRATEGY_VALUATION_PARAMS["default"]

    strategy_lower = strategy.lower().strip()
    return STRATEGY_VALUATION_PARAMS.get(
        strategy_lower, STRATEGY_VALUATION_PARAMS["default"]
    )


# ==============================================================================
# SCENARIO BAND CONFIGURATION
# ==============================================================================


@dataclass
class ScenarioConfig:
    """Configuration for Bear/Base/Bull scenario analysis."""

    # Growth rate adjustments (additive)
    bear_growth_adj: float = -0.03  # -3% from base
    bull_growth_adj: float = +0.03  # +3% from base

    # Discount rate adjustments (additive)
    bear_discount_adj: float = +0.02  # Higher discount in bear case
    bull_discount_adj: float = -0.01  # Lower discount in bull case

    # Terminal multiple adjustments (multiplicative)
    bear_multiple_factor: float = 0.75  # 25% lower multiple
    bull_multiple_factor: float = 1.25  # 25% higher multiple

    # Scenario probabilities (must sum to 1.0)
    bear_probability: float = 0.25
    base_probability: float = 0.50
    bull_probability: float = 0.25


DEFAULT_SCENARIO_CONFIG = ScenarioConfig()


# ==============================================================================
# VALUATION SANITY GATE - Catch unrealistic valuations
# ==============================================================================


@dataclass
class SanityGateConfig:
    """Configuration for valuation sanity checks."""

    min_fair_value: float = 0.0  # Fair value must be positive
    min_price_ratio: float = 0.2  # FV >= 0.2 * Current Price
    max_price_ratio: float = 5.0  # FV <= 5.0 * Current Price


DEFAULT_SANITY_CONFIG = SanityGateConfig()


def check_valuation_sanity(
    fair_value: float,
    current_price: float,
    config: Optional[SanityGateConfig] = None,
) -> Tuple[bool, str]:
    """
    Check if a valuation result passes sanity checks.

    Args:
        fair_value: Calculated fair value
        current_price: Current market price
        config: Sanity gate configuration

    Returns:
        Tuple of (passes_sanity, failure_reason)
    """
    config = config or DEFAULT_SANITY_CONFIG

    if fair_value <= config.min_fair_value:
        return (False, "Non-positive fair value")

    if current_price <= 0:
        return (False, "Invalid current price")

    ratio = fair_value / current_price

    if ratio < config.min_price_ratio:
        return (False, f"FV too low ({ratio:.1f}x price)")

    if ratio > config.max_price_ratio:
        return (False, f"FV too high ({ratio:.1f}x price)")

    return (True, "Passed")


def classify_investment_signal(
    bear_fv: float,
    base_fv: float,
    current_price: float,
    distress_risk: Optional[str] = None,
    tail_risk: Optional[str] = None,
    sanity_passed: bool = True,
) -> Tuple[str, str]:
    """
    Classify investment signal based on Bear/Base scenario analysis.

    Classification Logic:
    - Strong Buy: SAFE, Bear_FV >= Current, Base MOS >= 20%
    - Speculative Buy: SAFE, Base MOS >= 20%, but Bear_FV < Current
    - Hold: SAFE, Base MOS > 0% but < 20%
    - Avoid: DISTRESS risk, Heavy tail risk, or failed sanity
    - Needs Review: Failed sanity checks

    Args:
        bear_fv: Bear case fair value
        base_fv: Base case fair value
        current_price: Current market price
        distress_risk: Altman Z-Score risk classification
        tail_risk: Tail risk from distribution fitting
        sanity_passed: Whether valuation passed sanity checks

    Returns:
        Tuple of (signal, reason)
    """
    # Failed sanity gate
    if not sanity_passed:
        return ("Needs Review", "Failed valuation sanity checks")

    # Avoid distressed companies
    if distress_risk and distress_risk in ["DISTRESS (Risk)", "GREY ZONE"]:
        return ("Avoid", f"Bankruptcy risk: {distress_risk}")

    # Avoid heavy tail risk (volatile)
    if tail_risk and tail_risk == "Heavy":
        return ("Caution", "Heavy tail risk - high volatility")

    # Calculate margins of safety
    if current_price <= 0 or base_fv <= 0:
        return ("Needs Review", "Invalid price data")

    base_mos = (base_fv - current_price) / current_price
    bear_protected = bear_fv >= current_price

    # Strong Buy: Protected even in bear case with good base MOS
    if bear_protected and base_mos >= 0.20:
        return ("Strong Buy", f"Bear-protected, {base_mos*100:.0f}% MOS")

    # Speculative Buy: Good base MOS but bear case shows downside
    if base_mos >= 0.20 and not bear_protected:
        bear_downside = (current_price - bear_fv) / current_price * 100
        return (
            "Speculative Buy",
            f"{base_mos*100:.0f}% MOS, {bear_downside:.0f}% bear downside",
        )

    # Hold: Positive but modest MOS
    if base_mos > 0:
        return ("Hold", f"Modest {base_mos*100:.0f}% MOS")

    # Overvalued
    return ("Overvalued", f"Negative MOS ({base_mos*100:.0f}%)")


def calculate_intrinsic_value(
    fcf_per_share: float,
    sales_per_share: float = 0.0,
    book_value_per_share: float = 0.0,
    growth_rate: float = 0.08,
    discount_rate: float = 0.10,
    years: int = 5,
    terminal_multiple: int = 12,
) -> tuple[float, str]:
    """
    Calculate intrinsic value using a waterfall of valuation methods.

    Valuation Waterfall:
    1. DCF (if FCF > 0) - Traditional discounted cash flow
    2. Price-to-Sales (if Revenue > 0) - For growth stocks with negative FCF
    3. Price-to-Book (if Book Value > 0) - For pre-revenue companies

    Args:
        fcf_per_share: Free cash flow per share
        sales_per_share: Revenue per share (used for P/S fallback)
        book_value_per_share: Book value per share (used for P/B fallback)
        growth_rate: Expected growth rate for DCF projection
        discount_rate: Discount rate for present value calculation
        years: Number of years to project
        terminal_multiple: Multiple for terminal value calculation

    Returns:
        Tuple of (intrinsic_value, valuation_method_used)
    """
    # 1. Standard DCF for positive FCF companies
    if fcf_per_share > 0:
        future_cash_flows = []
        current_fcf = fcf_per_share

        # Project Future Cash Flows
        for _ in range(years):
            current_fcf *= 1 + growth_rate
            future_cash_flows.append(current_fcf)

        # Calculate Terminal Value (Exit Multiple)
        terminal_value = future_cash_flows[-1] * terminal_multiple

        # Discount back to present
        dcf_value = 0
        for i, cf in enumerate(future_cash_flows):
            dcf_value += cf / ((1 + discount_rate) ** (i + 1))

        # Discount Terminal Value
        pv_terminal = terminal_value / ((1 + discount_rate) ** years)

        return (dcf_value + pv_terminal, "DCF")

    # 2. Fallback: Price-to-Sales Valuation for Growth Stocks
    # If FCF is negative but has revenue, use P/S ratio
    if sales_per_share > 0:
        fair_ps_ratio = 4.0  # Conservative multiple for growth stocks
        return (sales_per_share * fair_ps_ratio, "P/S")

    # 3. Fallback: Price-to-Book for Pre-Revenue Companies
    # For speculative/development-stage companies
    if book_value_per_share > 0:
        fair_pb_ratio = 2.0  # Conservative P/B for speculative stocks
        return (book_value_per_share * fair_pb_ratio, "P/B")

    return (0.0, "N/A")


def calculate_scenario_valuations(
    fcf_per_share: float,
    sales_per_share: float = 0.0,
    book_value_per_share: float = 0.0,
    strategy: Optional[str] = None,
    scenario_config: Optional[ScenarioConfig] = None,
    years: int = 5,
) -> Dict[str, Any]:
    """
    Calculate Bear/Base/Bull scenario valuations with probability-weighted fair value.

    This replaces single-path DCF with scenario bands for more robust valuation.

    Args:
        fcf_per_share: Free cash flow per share
        sales_per_share: Revenue per share (for P/S fallback)
        book_value_per_share: Book value per share (for P/B fallback)
        strategy: Strategy name for parameter selection
        scenario_config: Custom scenario configuration (uses default if None)
        years: Projection years for DCF

    Returns:
        Dictionary containing:
        - bear_value, base_value, bull_value: Scenario intrinsic values
        - prob_weighted_value: Probability-weighted fair value
        - valuation_method: Method used (DCF, P/S, P/B)
        - scenario_spread: Bull-Bear spread as % of base
    """
    config = scenario_config or DEFAULT_SCENARIO_CONFIG
    params = get_strategy_params(strategy)

    # Calculate Base case
    base_value, method = calculate_intrinsic_value(
        fcf_per_share=fcf_per_share,
        sales_per_share=sales_per_share,
        book_value_per_share=book_value_per_share,
        growth_rate=params.growth_rate,
        discount_rate=params.discount_rate,
        terminal_multiple=params.terminal_multiple,
        years=years,
    )

    # For non-DCF methods, apply simpler adjustments
    if method != "DCF":
        # P/S and P/B use multiple-based adjustments
        bear_value = base_value * config.bear_multiple_factor
        bull_value = base_value * config.bull_multiple_factor
    else:
        # Bear case: lower growth, higher discount, lower multiple
        bear_growth = max(0.0, params.growth_rate + config.bear_growth_adj)
        bear_discount = params.discount_rate + config.bear_discount_adj
        bear_multiple = int(params.terminal_multiple * config.bear_multiple_factor)

        bear_value, _ = calculate_intrinsic_value(
            fcf_per_share=fcf_per_share,
            growth_rate=bear_growth,
            discount_rate=bear_discount,
            terminal_multiple=bear_multiple,
            years=years,
        )

        # Bull case: higher growth, lower discount, higher multiple
        bull_growth = params.growth_rate + config.bull_growth_adj
        bull_discount = max(0.05, params.discount_rate + config.bull_discount_adj)
        bull_multiple = int(params.terminal_multiple * config.bull_multiple_factor)

        bull_value, _ = calculate_intrinsic_value(
            fcf_per_share=fcf_per_share,
            growth_rate=bull_growth,
            discount_rate=bull_discount,
            terminal_multiple=bull_multiple,
            years=years,
        )

    # Probability-weighted fair value
    prob_weighted = (
        config.bear_probability * bear_value
        + config.base_probability * base_value
        + config.bull_probability * bull_value
    )

    # Calculate scenario spread (uncertainty measure)
    scenario_spread = (
        (bull_value - bear_value) / base_value * 100 if base_value > 0 else 0
    )

    return {
        "bear_value": bear_value,
        "base_value": base_value,
        "bull_value": bull_value,
        "prob_weighted_value": prob_weighted,
        "valuation_method": method,
        "scenario_spread": scenario_spread,
        "strategy_params": params.description,
    }


# ==============================================================================
# EPISTEMIC CONFIDENCE SCORING
# ==============================================================================


def calculate_model_confidence(
    valuation_method: str,
    fcf_available: bool,
    revenue_available: bool,
    book_value_available: bool,
    tail_risk: Optional[str] = None,
    scenario_spread: float = 0.0,
    data_years: int = 0,
) -> Tuple[str, float, str]:
    """
    Calculate epistemic confidence score based on data quality and model reliability.

    Prevents over-trust in clean-looking but fragile outputs by explicitly
    measuring uncertainty in the valuation.

    Args:
        valuation_method: Method used (DCF, P/S, P/B, N/A)
        fcf_available: Whether FCF data was available
        revenue_available: Whether revenue data was available
        book_value_available: Whether book value data was available
        tail_risk: Tail risk from distribution fitting (Heavy/Normal/Thin)
        scenario_spread: Bull-Bear spread as % of base value
        data_years: Years of historical data available

    Returns:
        Tuple of (confidence_level, confidence_score, explanation)
        - confidence_level: "High", "Medium", or "Low"
        - confidence_score: 0-100 numeric score
        - explanation: Reason for the confidence level
    """
    score = 0
    factors = []

    # 1. VALUATION METHOD QUALITY (0-35 points)
    # DCF is gold standard, fallbacks are less reliable
    method_scores = {
        "DCF": 35,
        "P/S": 20,
        "P/B": 10,
        "N/A": 0,
    }
    method_score = method_scores.get(valuation_method, 0)
    score += method_score

    if valuation_method == "DCF":
        factors.append("DCF valuation (gold standard)")
    elif valuation_method == "P/S":
        factors.append("P/S fallback (no positive FCF)")
    elif valuation_method == "P/B":
        factors.append("P/B fallback (no revenue data)")
    else:
        factors.append("No valuation possible")

    # 2. DATA AVAILABILITY (0-25 points)
    data_score = 0
    if fcf_available:
        data_score += 12
    if revenue_available:
        data_score += 8
    if book_value_available:
        data_score += 5
    score += data_score

    if data_score >= 20:
        factors.append("Complete financial data")
    elif data_score >= 10:
        factors.append("Partial financial data")
    else:
        factors.append("Limited financial data")

    # 3. TAIL RISK STABILITY (0-20 points)
    # Heavy tails = less predictable = lower confidence
    tail_scores = {
        "Thin": 20,
        "Normal": 15,
        "Heavy": 5,
        None: 10,  # Unknown
    }
    tail_score = tail_scores.get(tail_risk, 10)
    score += tail_score

    if tail_risk == "Heavy":
        factors.append("Heavy tail risk (volatile)")
    elif tail_risk == "Thin":
        factors.append("Stable return distribution")
    elif tail_risk == "Normal":
        factors.append("Normal tail behavior")

    # 4. SCENARIO SPREAD / UNCERTAINTY (0-20 points)
    # Tighter spread = higher confidence
    if scenario_spread < 50:
        spread_score = 20
        factors.append("Tight scenario range")
    elif scenario_spread < 100:
        spread_score = 15
        factors.append("Moderate scenario uncertainty")
    elif scenario_spread < 200:
        spread_score = 8
        factors.append("Wide scenario range")
    else:
        spread_score = 0
        factors.append("Extreme valuation uncertainty")
    score += spread_score

    # Determine confidence level
    if score >= 70:
        confidence_level = "High"
    elif score >= 45:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"

    explanation = "; ".join(factors[:3])  # Top 3 factors

    return (confidence_level, score, explanation)


def _project_price(current_price: float, cagr: float, months: int) -> float:
    """Projects price forward using historical CAGR."""
    years = months / 12.0
    return current_price * ((1 + cagr) ** years)


def _default_workers() -> int:
    return min(8, os.cpu_count() or 4)


def _normalize_tickers(values: Iterable[Any]) -> list[str]:
    seen = set()
    ordered = []
    for raw in values:
        if raw is None:
            continue
        ticker = str(raw).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        ordered.append(ticker)
    return ordered


def _history_map_from_bulk(df: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    if df is None or df.empty:
        return {}
    out: Dict[str, pd.DataFrame] = {}
    work = df.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    for symbol, grp in work.groupby("symbol", sort=False):
        cols = [c for c in grp.columns if c != "symbol"]
        out[symbol] = grp[cols].sort_values("date").reset_index(drop=True)
    return out


def _financial_map_from_bulk(df: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    if df is None or df.empty:
        return {}
    out: Dict[str, pd.DataFrame] = {}
    work = df.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    for symbol, grp in work.groupby("symbol", sort=False):
        out[symbol] = grp.pivot(index="metric", columns="period", values="value")
    return out


def _prepare_prefetched_maps(
    tickers: list[str], prefetched: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    prefetched = prefetched or {}

    history_by_symbol = prefetched.get("history_by_symbol")
    if not isinstance(history_by_symbol, dict):
        history_bulk = prefetched.get("history")
        if history_bulk is None:
            history_bulk = get_ticker_history_bulk(tickers)
        history_by_symbol = _history_map_from_bulk(history_bulk)
    history_by_symbol = {
        str(k).upper(): v
        for k, v in history_by_symbol.items()
        if isinstance(v, pd.DataFrame)
    }

    cashflow_by_symbol = prefetched.get("cash_flow_by_symbol")
    if not isinstance(cashflow_by_symbol, dict):
        cashflow_bulk = prefetched.get("cash_flow")
        if cashflow_bulk is None:
            cashflow_bulk = get_financial_data_bulk(tickers, "cash_flow")
        cashflow_by_symbol = _financial_map_from_bulk(cashflow_bulk)
    cashflow_by_symbol = {
        str(k).upper(): v
        for k, v in cashflow_by_symbol.items()
        if isinstance(v, pd.DataFrame)
    }

    bs_by_symbol = prefetched.get("balance_sheet_by_symbol")
    if not isinstance(bs_by_symbol, dict):
        bs_bulk = prefetched.get("balance_sheet")
        if bs_bulk is None:
            bs_bulk = get_financial_data_bulk(tickers, "balance_sheet")
        bs_by_symbol = _financial_map_from_bulk(bs_bulk)
    bs_by_symbol = {
        str(k).upper(): v for k, v in bs_by_symbol.items() if isinstance(v, pd.DataFrame)
    }

    inc_by_symbol = prefetched.get("income_statement_by_symbol")
    if not isinstance(inc_by_symbol, dict):
        inc_bulk = prefetched.get("income_statement")
        if inc_bulk is None:
            inc_bulk = get_financial_data_bulk(tickers, "income_statement")
        inc_by_symbol = _financial_map_from_bulk(inc_bulk)
    inc_by_symbol = {
        str(k).upper(): v for k, v in inc_by_symbol.items() if isinstance(v, pd.DataFrame)
    }

    return history_by_symbol, cashflow_by_symbol, bs_by_symbol, inc_by_symbol


def _get_latest_metric(df: pd.DataFrame, metric_name: str) -> Optional[float]:
    """Get the most recent value for a metric from financial data."""
    if df.empty or metric_name not in df.index:
        return None

    # Get the row for this metric
    row = df.loc[metric_name]

    # Find the most recent non-null value (columns are periods/dates)
    # Sort columns (dates) in descending order
    sorted_cols = sorted(row.index, reverse=True)
    for col in sorted_cols:
        val = row[col]
        if pd.notna(val):
            return float(val)
    return None


def _analyze_ticker_valuation(
    ticker: str,
    strategy: Optional[str] = None,
    verbose: bool = False,
    prefetched_history: Optional[pd.DataFrame] = None,
    prefetched_cash_flow: Optional[pd.DataFrame] = None,
    prefetched_balance_sheet: Optional[pd.DataFrame] = None,
    prefetched_income_statement: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """
    Perform valuation analysis on a single ticker with scenario bands.

    Now includes:
    - Strategy-conditioned valuation parameters
    - Bear/Base/Bull scenario valuations
    - Probability-weighted fair value
    - Epistemic confidence scoring

    Args:
        ticker: Stock ticker symbol
        strategy: Optional strategy name (determines valuation parameters)
        verbose: Whether to print progress messages

    Returns:
        dict with valuation results, or None if data unavailable
    """
    try:
        # Get price history
        hist = prefetched_history if prefetched_history is not None else get_ticker_history(ticker)
        if hist.empty:
            if verbose:
                print(f"  No history data for {ticker}, skipping...")
            return None

        # Get financial data
        cash_flow = (
            prefetched_cash_flow
            if prefetched_cash_flow is not None
            else get_financial_data(ticker, "cash_flow")
        )
        balance_sheet = (
            prefetched_balance_sheet
            if prefetched_balance_sheet is not None
            else get_financial_data(ticker, "balance_sheet")
        )

        # Calculate inputs
        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist.sort_values("date")
        current_price = hist["close"].iloc[-1]

        # Get FCF and Shares Outstanding
        fcf = _get_latest_metric(cash_flow, "FreeCashFlow") or 0
        shares = _get_latest_metric(balance_sheet, "OrdinarySharesNumber")
        if shares is None or shares <= 0:
            shares = 1

        fcf_per_share = fcf / shares if shares > 0 else 0
        fcf_available = fcf > 0

        # Get Revenue/Sales for P/S fallback valuation
        income_stmt = (
            prefetched_income_statement
            if prefetched_income_statement is not None
            else get_financial_data(ticker, "income_statement")
        )
        total_revenue = _get_latest_metric(income_stmt, "TotalRevenue")

        # Try alternative revenue fields if TotalRevenue is missing
        if total_revenue is None or total_revenue <= 0:
            total_revenue = _get_latest_metric(income_stmt, "OperatingRevenue") or 0
        if total_revenue is None or total_revenue <= 0:
            total_revenue = _get_latest_metric(income_stmt, "Revenue") or 0

        sales_per_share = (
            total_revenue / shares if (shares > 0 and total_revenue > 0) else 0
        )
        revenue_available = total_revenue > 0

        # Get Book Value for P/B fallback (pre-revenue companies)
        stockholders_equity = _get_latest_metric(balance_sheet, "StockholdersEquity")
        if stockholders_equity is None or stockholders_equity <= 0:
            stockholders_equity = (
                _get_latest_metric(balance_sheet, "TotalEquityGrossMinorityInterest")
                or 0
            )
        book_value_per_share = (
            stockholders_equity / shares
            if (shares > 0 and stockholders_equity > 0)
            else 0
        )
        book_value_available = stockholders_equity > 0

        # Historical Growth (CAGR) from last 3 years
        lookback_days = 365 * 3
        data_years = min(len(hist) // 252, 5)  # Trading days to years
        if len(hist) > lookback_days:
            price_3y_ago = hist["close"].iloc[-lookback_days]
            cagr = (
                (current_price / price_3y_ago) ** (1 / 3) - 1
                if price_3y_ago > 0
                else 0.08
            )
        else:
            cagr = 0.08

        # Cap CAGR to be conservative
        safe_cagr = min(max(cagr, -0.10), 0.25)

        # ==================================================================
        # SCENARIO-BASED VALUATION (Bear / Base / Bull)
        # ==================================================================
        scenario_results = calculate_scenario_valuations(
            fcf_per_share=fcf_per_share,
            sales_per_share=sales_per_share,
            book_value_per_share=book_value_per_share,
            strategy=strategy,
        )

        bear_value = scenario_results["bear_value"]
        base_value = scenario_results["base_value"]
        bull_value = scenario_results["bull_value"]
        prob_weighted_value = scenario_results["prob_weighted_value"]
        valuation_method = scenario_results["valuation_method"]
        scenario_spread = scenario_results["scenario_spread"]
        strategy_params_desc = scenario_results["strategy_params"]

        if verbose:
            params = get_strategy_params(strategy)
            if valuation_method == "DCF":
                print(
                    f"  {ticker}: DCF via {strategy or 'default'} "
                    f"(g={params.growth_rate*100:.0f}%, r={params.discount_rate*100:.0f}%, "
                    f"mult={params.terminal_multiple}x)"
                )
            elif valuation_method == "P/S":
                print(f"  {ticker}: P/S fallback (Revenue=${total_revenue:,.0f})")
            elif valuation_method == "P/B":
                print(f"  {ticker}: P/B fallback (Book=${stockholders_equity:,.0f})")
            else:
                print(f"  {ticker}: Could not value - no FCF, Revenue, or Book Value")

        # ==================================================================
        # EPISTEMIC CONFIDENCE SCORING
        # ==================================================================
        confidence_level, confidence_score, confidence_explanation = (
            calculate_model_confidence(
                valuation_method=valuation_method,
                fcf_available=fcf_available,
                revenue_available=revenue_available,
                book_value_available=book_value_available,
                tail_risk=None,  # Will be updated in main.py after distribution fitting
                scenario_spread=scenario_spread,
                data_years=data_years,
            )
        )

        # Margin of Safety (use probability-weighted value as primary)
        margin_of_safety = (
            (prob_weighted_value - current_price) / current_price
            if current_price > 0 and prob_weighted_value > 0
            else 0
        )

        # ==================================================================
        # VALUATION SANITY GATE
        # ==================================================================
        sanity_passed, sanity_reason = check_valuation_sanity(
            fair_value=prob_weighted_value,
            current_price=current_price,
        )

        if not sanity_passed and verbose:
            print(f"  ⚠ {ticker}: Sanity check failed - {sanity_reason}")

        # ==================================================================
        # INVESTMENT SIGNAL CLASSIFICATION (Bear MOS-based)
        # ==================================================================
        # Note: distress_risk and tail_risk will be updated in main.py
        # after forensic and distribution analysis
        investment_signal, signal_reason = classify_investment_signal(
            bear_fv=bear_value,
            base_fv=base_value,
            current_price=current_price,
            distress_risk=None,  # Updated later in pipeline
            tail_risk=None,  # Updated later in pipeline
            sanity_passed=sanity_passed,
        )

        # Return comprehensive results
        return {
            "Ticker": ticker,
            "Strategy": strategy,
            "Current Price": round(current_price, 2),
            # Scenario-based fair values
            "Fair Value (Bear)": round(bear_value, 2),
            "Fair Value (Base)": round(base_value, 2),
            "Fair Value (Bull)": round(bull_value, 2),
            "Fair Value": round(prob_weighted_value, 2),  # Probability-weighted
            "Valuation Method": valuation_method,
            "Undervalued %": f"{margin_of_safety*100:.1f}%",
            "Scenario Spread %": f"{scenario_spread:.0f}%",
            # Sanity gate results
            "Valuation Sanity": "Passed" if sanity_passed else sanity_reason,
            # Investment signal (preliminary - updated after forensic/risk)
            "Investment Signal": investment_signal,
            "Signal Reason": signal_reason,
            # Confidence scoring
            "Model Confidence": confidence_level,
            "Confidence Score": confidence_score,
            "Confidence Reason": confidence_explanation,
            # Trend projections (CAGR extrapolations, NOT price targets)
            "Trend_Proj_6M": round(_project_price(current_price, safe_cagr, 6), 2),
            "Trend_Proj_1Y": round(_project_price(current_price, safe_cagr, 12), 2),
            "Trend_Proj_3Y": round(_project_price(current_price, safe_cagr, 36), 2),
            "Trend_CAGR": f"{safe_cagr*100:.1f}%",
            # Strategy context
            "Strategy Params": strategy_params_desc,
            # Raw valuation inputs (for auditability)
            "_Input_FCF": round(fcf, 2) if fcf else 0,
            "_Input_Revenue": round(total_revenue, 2) if total_revenue else 0,
            "_Input_BookValue": (
                round(stockholders_equity, 2) if stockholders_equity else 0
            ),
            "_Input_Shares": round(shares, 0) if shares else 0,
            "_Input_FCF_PerShare": round(fcf_per_share, 4) if fcf_per_share else 0,
        }

    except Exception as e:
        if verbose:
            print(f"  ✗ Could not value {ticker}: {e}")
        return None


def _print_report(val_df: pd.DataFrame) -> None:
    """Print the valuation report with scenario bands."""
    print("\n" + "=" * 90)
    print("  VALUATION & PRICE TARGETS (Scenario-Based)")
    print("=" * 90)

    # Display key columns for terminal output
    display_cols = [
        "Ticker",
        "Strategy",
        "Current Price",
        "Fair Value (Bear)",
        "Fair Value",
        "Fair Value (Bull)",
        "Model Confidence",
        "Undervalued %",
    ]
    display_cols = [c for c in display_cols if c in val_df.columns]

    if display_cols:
        print(val_df[display_cols].to_string(index=False))
    else:
        print(val_df.to_string(index=False))


def run_valuation_scan(
    buy_list_df: pd.DataFrame,
    verbose: bool = True,
    prefetched: Optional[Dict[str, Any]] = None,
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run valuation analysis on a buy list DataFrame.

    This is the primary function-based API for valuation scanning.
    Now includes strategy-conditioned parameters, scenario bands, and confidence scoring.

    Args:
        buy_list_df: DataFrame with at least a 'Ticker' column.
                     Optional 'Strategy' column determines valuation parameters.
        verbose: Whether to print progress/results

    Returns:
        DataFrame with original columns plus valuation results:
        - Current Price: Latest stock price
        - Fair Value (Bear/Base/Bull): Scenario intrinsic values
        - Fair Value: Probability-weighted intrinsic value
        - Valuation Method: Which method was used (DCF, P/S, P/B, or N/A)
        - Undervalued %: Margin of safety percentage
        - Scenario Spread %: Bull-Bear spread as uncertainty measure
        - Valuation Sanity: Whether valuation passed sanity checks
        - Investment Signal: Strong Buy/Speculative Buy/Hold/Caution/Avoid/Needs Review
        - Signal Reason: Explanation for investment signal
        - Model Confidence: High/Medium/Low epistemic confidence
        - Confidence Score: Numeric confidence (0-100)
        - Confidence Reason: Explanation for confidence level
        - Trend_Proj_6M/1Y/3Y: CAGR-based trend projections (NOT price targets)
        - Trend_CAGR: Historical growth rate used for trend projections
        - Strategy Params: Description of valuation parameters used
        - _Input_*: Raw valuation inputs for auditability
    """
    if buy_list_df.empty:
        if verbose:
            print("Empty buy list provided.")
        return buy_list_df.copy()

    if "Ticker" not in buy_list_df.columns:
        raise ValueError("buy_list_df must contain a 'Ticker' column")

    if verbose:
        print(f"Running scenario-based valuation scan on {len(buy_list_df)} stocks...")

    tickers = _normalize_tickers(buy_list_df["Ticker"].tolist())
    history_map, cashflow_map, bs_map, inc_map = _prepare_prefetched_maps(
        tickers, prefetched
    )

    workers = max_workers if max_workers is not None else _default_workers()
    workers = max(1, int(workers))

    results = []
    rows = list(buy_list_df.iterrows())

    if workers == 1 or len(rows) <= 1:
        for order, (_, row) in enumerate(rows):
            ticker = str(row["Ticker"]).strip().upper()
            strategy = row.get("Strategy")
            result = _analyze_ticker_valuation(
                ticker,
                strategy,
                verbose=verbose,
                prefetched_history=history_map.get(ticker),
                prefetched_cash_flow=cashflow_map.get(ticker),
                prefetched_balance_sheet=bs_map.get(ticker),
                prefetched_income_statement=inc_map.get(ticker),
            )
            if result:
                result["_order"] = order
                results.append(result)
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for order, (_, row) in enumerate(rows):
                ticker = str(row["Ticker"]).strip().upper()
                strategy = row.get("Strategy")
                future = executor.submit(
                    _analyze_ticker_valuation,
                    ticker,
                    strategy,
                    verbose,
                    history_map.get(ticker),
                    cashflow_map.get(ticker),
                    bs_map.get(ticker),
                    inc_map.get(ticker),
                )
                futures[future] = order

            for future in as_completed(futures):
                result = future.result()
                if result:
                    result["_order"] = futures[future]
                    results.append(result)

    if not results:
        if verbose:
            print("No valid valuations generated.")
        return buy_list_df.copy()

    val_df = pd.DataFrame(results).sort_values("_order").drop(columns=["_order"])

    if verbose:
        _print_report(val_df)

    # Merge ALL valuation results back to original DataFrame
    val_cols = [
        "Ticker",
        "Current Price",
        "Fair Value (Bear)",
        "Fair Value (Base)",
        "Fair Value (Bull)",
        "Fair Value",
        "Valuation Method",
        "Undervalued %",
        "Scenario Spread %",
        "Valuation Sanity",
        "Investment Signal",
        "Signal Reason",
        "Model Confidence",
        "Confidence Score",
        "Confidence Reason",
        "Trend_Proj_6M",
        "Trend_Proj_1Y",
        "Trend_Proj_3Y",
        "Trend_CAGR",
        "Strategy Params",
        # Raw inputs for auditability
        "_Input_FCF",
        "_Input_Revenue",
        "_Input_BookValue",
        "_Input_Shares",
        "_Input_FCF_PerShare",
    ]
    val_subset = val_df[[c for c in val_cols if c in val_df.columns]]
    result_df = buy_list_df.merge(val_subset, on="Ticker", how="left")

    return result_df
