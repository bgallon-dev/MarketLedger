"""
Forensic Scan Module - Bankruptcy risk analysis using Altman Z-Score.

This module provides forensic accounting analysis to identify stocks
with potential financial distress. It calculates Altman Z-Scores and
flags companies at risk of bankruptcy.

Usage:
------
    from forensic.forensic_scan import run_forensic_scan

    # Pass a DataFrame with tickers to analyze
    buy_list_df = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT"],
        "Strategy": ["magic_formula", "moat"]
    })
    result_df = run_forensic_scan(buy_list_df)

    # Result includes: Altman Z-Score, Distress Risk (SAFE/GREY ZONE/DISTRESS), Price
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add pyfinancial to path for database imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "pyfinancial"))
from pyfinancial.database.database import (
    get_ticker_history,
    get_financial_data,
    get_ticker_history_bulk,
    get_financial_data_bulk,
)


def _get_latest_price_from_db(ticker: str, verbose: bool = False) -> float:
    """Get the latest closing price for a ticker from the database."""
    try:
        hist = get_ticker_history(ticker)
        if hist is not None and not hist.empty and "close" in hist.columns:
            return hist["close"].iloc[-1]
    except Exception as e:
        if verbose:
            print(f"  Error getting price for {ticker}: {e}")
    return 0


def _get_financial_data_for_ticker(
    ticker: str, table_name: str, verbose: bool = False
) -> Optional[pd.DataFrame]:
    """Load financial data for a ticker from the database."""
    try:
        df = get_financial_data(ticker, table_name)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        if verbose:
            print(f"  Error loading {table_name} for {ticker}: {e}")
    return None


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
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
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

    return history_by_symbol, bs_by_symbol, inc_by_symbol


def _get_latest_value(df: Optional[pd.DataFrame], metric_name: str) -> Optional[float]:
    """Get the latest value for a metric from a financial DataFrame."""
    if df is None or metric_name not in df.index:
        return None
    row = df.loc[metric_name]
    # Get the first (most recent) column value
    if isinstance(row, pd.Series):
        valid_values = pd.to_numeric(row, errors="coerce").dropna()
        if len(valid_values) > 0:
            return valid_values.iloc[0]
    return None


def _calculate_altman_z(row: dict) -> float:
    """
    Calculates Altman Z-Score for Manufacturing/General Firms.
    Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
    """
    try:
        # 1. Working Capital / Total Assets
        A = row["WorkingCapital"] / row["TotalAssets"]

        # 2. Retained Earnings / Total Assets
        B = row["RetainedEarnings"] / row["TotalAssets"]

        # 3. EBIT / Total Assets
        C = row["EBIT"] / row["TotalAssets"]

        # 4. Market Value of Equity / Total Liabilities
        # MV = Market Cap
        D = row["MarketCap"] / row["TotalLiabilitiesNetMinorityInterest"]

        # 5. Sales / Total Assets
        E = row["TotalRevenue"] / row["TotalAssets"]

        z_score = (1.2 * A) + (1.4 * B) + (3.3 * C) + (0.6 * D) + (1.0 * E)
        return z_score
    except Exception:
        return np.nan


def _analyze_ticker(
    ticker: str,
    strategy: Optional[str] = None,
    verbose: bool = False,
    prefetched_history: Optional[pd.DataFrame] = None,
    prefetched_balance_sheet: Optional[pd.DataFrame] = None,
    prefetched_income_statement: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """
    Perform forensic analysis on a single ticker.

    Args:
        ticker: Stock ticker symbol
        strategy: Optional strategy name
        verbose: Whether to print error messages

    Returns:
        dict with analysis results, or None if data unavailable
    """
    bs = (
        prefetched_balance_sheet
        if prefetched_balance_sheet is not None
        else _get_financial_data_for_ticker(ticker, "balance_sheet", verbose)
    )
    inc = (
        prefetched_income_statement
        if prefetched_income_statement is not None
        else _get_financial_data_for_ticker(ticker, "income_statement", verbose)
    )

    if bs is None or inc is None:
        return None

    calc_row = {
        "TotalAssets": _get_latest_value(bs, "TotalAssets") or 0,
        "TotalLiabilitiesNetMinorityInterest": _get_latest_value(
            bs, "TotalLiabilitiesNetMinorityInterest"
        )
        or 0,
        "WorkingCapital": _get_latest_value(bs, "WorkingCapital") or 0,
        "RetainedEarnings": _get_latest_value(bs, "RetainedEarnings") or 0,
        "OrdinarySharesNumber": _get_latest_value(bs, "OrdinarySharesNumber") or 0,
        "EBIT": _get_latest_value(inc, "EBIT") or 0,
        "TotalRevenue": _get_latest_value(inc, "TotalRevenue") or 0,
    }

    if prefetched_history is not None and not prefetched_history.empty:
        try:
            current_price = float(prefetched_history["close"].iloc[-1])
        except Exception:
            current_price = _get_latest_price_from_db(ticker, verbose)
    else:
        current_price = _get_latest_price_from_db(ticker, verbose)
    shares = calc_row.get("OrdinarySharesNumber", 0)
    calc_row["MarketCap"] = current_price * shares if shares and current_price else 0

    z_score = _calculate_altman_z(calc_row)

    z_verdict = "SAFE"
    if pd.isna(z_score):
        z_verdict = "N/A"
    elif z_score < 1.81:
        z_verdict = "DISTRESS (Risk)"
    elif z_score < 2.99:
        z_verdict = "GREY ZONE"

    return {
        "Ticker": ticker,
        "Strategy": strategy,
        "Altman Z-Score": round(z_score, 2) if not pd.isna(z_score) else "N/A",
        "Distress Risk": z_verdict,
        "Price": round(current_price, 2),
    }


def _print_report(forensic_df: pd.DataFrame) -> None:
    """Print the forensic analysis report."""
    safe_picks = forensic_df[forensic_df["Distress Risk"] == "SAFE"]

    print("\n" + "=" * 80)
    print("  FORENSIC ACCOUNTING REPORT")
    print("=" * 80)
    print(forensic_df.to_string(index=False))
    print(
        f"\n[INSIGHT] Found {len(safe_picks)} 'Fortress' Balance Sheets out of {len(forensic_df)}."
    )

    risks = forensic_df[forensic_df["Distress Risk"] == "DISTRESS (Risk)"]
    if not risks.empty:
        print("\n[WARNING] The following stocks flagged for Bankruptcy Risk:")
        print(risks[["Ticker", "Altman Z-Score"]])


def run_forensic_scan(
    buy_list_df: pd.DataFrame,
    verbose: bool = True,
    prefetched: Optional[Dict[str, Any]] = None,
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run forensic analysis on a buy list DataFrame.

    This is the primary function-based API for forensic scanning.

    Args:
        buy_list_df: DataFrame with at least a 'Ticker' column.
                     Optional 'Strategy' column will be preserved.
        verbose: Whether to print progress/results

    Returns:
        DataFrame with original columns plus forensic analysis results:
        - Altman Z-Score: The calculated Z-Score
        - Distress Risk: SAFE, GREY ZONE, DISTRESS (Risk), or N/A
        - Price: Current stock price
    """
    if buy_list_df.empty:
        if verbose:
            print("Empty buy list provided.")
        return buy_list_df.copy()

    if "Ticker" not in buy_list_df.columns:
        raise ValueError("buy_list_df must contain a 'Ticker' column")

    if verbose:
        print(f"Running forensic scan on {len(buy_list_df)} stocks...")

    tickers = _normalize_tickers(buy_list_df["Ticker"].tolist())
    history_map, bs_map, inc_map = _prepare_prefetched_maps(tickers, prefetched)

    workers = max_workers if max_workers is not None else _default_workers()
    workers = max(1, int(workers))

    results = []
    rows = list(buy_list_df.iterrows())

    if workers == 1 or len(rows) <= 1:
        for order, (_, row) in enumerate(rows):
            ticker = str(row["Ticker"]).strip().upper()
            strategy = row.get("Strategy")
            result = _analyze_ticker(
                ticker,
                strategy,
                verbose=verbose,
                prefetched_history=history_map.get(ticker),
                prefetched_balance_sheet=bs_map.get(ticker),
                prefetched_income_statement=inc_map.get(ticker),
            )
            if result:
                result["_order"] = order
                results.append(result)
            elif verbose:
                print(f"  Skipping {ticker}: Missing financial data")
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for order, (_, row) in enumerate(rows):
                ticker = str(row["Ticker"]).strip().upper()
                strategy = row.get("Strategy")
                future = executor.submit(
                    _analyze_ticker,
                    ticker,
                    strategy,
                    verbose,
                    history_map.get(ticker),
                    bs_map.get(ticker),
                    inc_map.get(ticker),
                )
                futures[future] = (order, ticker)

            for future in as_completed(futures):
                order, ticker = futures[future]
                result = future.result()
                if result:
                    result["_order"] = order
                    results.append(result)
                elif verbose:
                    print(f"  Skipping {ticker}: Missing financial data")

    if not results:
        if verbose:
            print("No forensic results generated.")
        return buy_list_df.copy()

    forensic_df = pd.DataFrame(results).sort_values("_order").drop(columns=["_order"])

    if verbose:
        _print_report(forensic_df)

    # Merge forensic results back to original DataFrame
    merge_cols = ["Ticker", "Altman Z-Score", "Distress Risk", "Price"]
    forensic_subset = forensic_df[[c for c in merge_cols if c in forensic_df.columns]]
    result_df = buy_list_df.merge(forensic_subset, on="Ticker", how="left")

    return result_df
