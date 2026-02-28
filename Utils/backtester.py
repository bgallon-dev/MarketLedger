"""
Point-in-Time Backtester - Avoids Look-Ahead Bias

The key insight: You cannot use 2023 Annual Numbers to buy a stock in Jan 2023,
because those numbers weren't released yet.

Logic:
1. Pick a Trade Date (e.g., 2023-01-01)
2. Apply a "Reporting Lag" (e.g., 90 days) - financial data dated 2022-09-30 is visible on 2023-01-01
3. Filter: Only look at data older than the reporting lag
4. Rank & Pick: Run screening logic on that historical snapshot
5. Calculate Return: Check price change from Trade Date to Exit Date
"""

import pandas as pd
import numpy as np
import os
import time
from numbers import Real
from datetime import timedelta, datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, cast
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from database.database import get_connection, query_database, ensure_indices
from Utils.scoring import robust_score
from risk.risk_vector import RiskVectorConfig, attach_risk_vectors
from valuation.valuation_projector import (
    calculate_scenario_valuations,
    check_valuation_sanity,
)

# Define metrics required for all strategies to optimize loading
REQUIRED_METRICS = [
    "EBIT",
    "InvestedCapital",
    "TotalDebt",
    "CashAndCashEquivalents",
    "OrdinarySharesNumber",
    "TotalRevenue",
    "OperatingIncome",
    "ResearchAndDevelopment",
    "OperatingCashFlow",
    "CapitalExpenditure",
    # Added for super_margins strategy
    "NetIncome",
    "StockholdersEquity",
    "GrossProfit",
    # Added for cannibal, piotroski_f_score, rule_of_40 strategies
    "TotalAssets",
    "CurrentAssets",
    "CurrentLiabilities",
    "RepurchaseOfCapitalStock",
    # Added for risk-vector distress calculations
    "WorkingCapital",
    "RetainedEarnings",
    "TotalLiabilitiesNetMinorityInterest",
]


def _require_real(value: Any, field_name: str) -> float:
    """
    Validate that a value is a non-missing real number and convert it to float.

    Raises:
        TypeError: If the value is missing or not a real numeric type.
    """
    if value is None:
        raise TypeError(f"{field_name} must be a real number, got None")
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a real number, got bool")
    try:
        missing = pd.isna(value)
    except Exception:
        # Some objects may not support pandas null checks.
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        raise TypeError(f"{field_name} must be a real number, got missing value")
    if not isinstance(value, Real):
        raise TypeError(
            f"{field_name} must be a real number, got {type(value).__name__}"
        )
    return float(value)


class VectorBacktester:
    """
    Point-in-time backtester that avoids look-ahead bias.

    Usage:
    ------
    >>> bt = VectorBacktester()
    >>> bt.load_data()
    >>> results = bt.run_backtest('2023-01-01', '2024-01-01', strategy='magic_formula')
    """

    def __init__(self, reporting_lag_days: int = 90):
        """
        Initialize the backtester.

        Parameters:
        -----------
        reporting_lag_days : int
            Days between period end and when data becomes available.
            Default 90 days (companies report ~45-90 days after quarter end)
        """
        self.reporting_lag = reporting_lag_days
        self.price_matrix: Optional[pd.DataFrame] = None
        self.financial_data: Optional[pd.DataFrame] = None
        self.benchmark_symbol = "QQQ"  # Nasdaq ETF for comparison
        self._last_research_outputs: Optional[Dict[str, pd.DataFrame]] = None
        self._regime_cache: Dict[Tuple[str, int, int], str] = {}
        self._tail_state_cache: Dict[Tuple[str, str, int, int], Dict[str, Any]] = {}
        self._momentum_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        self._snapshot_cache: Dict[Tuple[str, int], pd.DataFrame] = {}

    def load_data(self):
        """Load all required data from database."""
        print("Loading data from database...")
        ensure_indices()
        conn = get_connection()

        # Load price history (Long format)
        print("  ...loading price history")
        df_prices = pd.read_sql_query(
            """
            SELECT t.symbol, h.date, h.close
            FROM history h
            JOIN tickers t ON h.ticker_id = t.id
            ORDER BY h.date
        """,
            conn,
        )
        # Force conversion to UTC then strip timezone to ensure naive datetime64[ns]
        df_prices["date"] = pd.to_datetime(
            df_prices["date"], utc=True, errors="coerce"
        ).dt.tz_localize(None)

        print("  ...creating price matrix")
        self.price_matrix = df_prices.pivot(
            index="date", columns="symbol", values="close"
        )
        self.price_matrix = self.price_matrix.sort_index()
        # Fill gaps (weekends/holidays) so 'exact date' lookups work better
        self.price_matrix = self.price_matrix.ffill().bfill()

        print(
            f"      Loaded price history for {len(self.price_matrix.columns)} symbols"
        )

        # Close main connection before starting threads (threads open their own)
        conn.close()

        # Load financial tables in parallel
        tables = ["income_statement", "balance_sheet", "cash_flow", "financials"]
        metrics_str = "', '".join(REQUIRED_METRICS)

        dfs = []
        print("  ...loading financials")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for table in tables:
                sql = f"""
                    SELECT t.symbol, f.metric, f.period, f.value
                    FROM {table} f
                    JOIN tickers t ON f.ticker_id = t.id
                    WHERE f.metric IN ('{metrics_str}')
                """
                futures.append(executor.submit(query_database, sql))

            for future in as_completed(futures):
                try:
                    df = future.result()
                    if not df.empty:
                        df["period"] = pd.to_datetime(
                            df["period"], utc=True, errors="coerce"
                        ).dt.tz_localize(None)
                        # Pivot immediately: index=(symbol, period), columns=metric
                        pivoted = df.pivot_table(
                            index=["symbol", "period"],
                            columns="metric",
                            values="value",
                            aggfunc="last",
                        )
                        dfs.append(pivoted)
                except Exception as e:
                    print(f"Error loading table: {e}")

        if dfs:
            print("  ...merging financial data")
            # Merge all pivoted DFs.
            self.financial_data = pd.concat(dfs, axis=1)
            # Deduplicate columns (if same metric in multiple tables)
            # Pandas 3 removed DataFrame.groupby(axis=1); transpose to group columns.
            self.financial_data = self.financial_data.T.groupby(level=0).last().T
            self.financial_data = self.financial_data.sort_index()
        else:
            self.financial_data = pd.DataFrame()

        # Invalidate derived caches after reload.
        self._regime_cache.clear()
        self._tail_state_cache.clear()
        self._momentum_cache.clear()
        self._snapshot_cache.clear()

        print("Data loading complete.\n")

    def get_valid_snapshot(
        self, trade_date: str, lag_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get financial data snapshot as it would have been visible on the trade date.

        Parameters:
        -----------
        trade_date : str
            The date we're "trading" on (e.g., '2023-01-01')
        lag_days : int, optional
            Override default reporting lag

        Returns:
        --------
        pd.DataFrame
            Wide-format DataFrame with symbol as index, metrics as columns
        """
        if lag_days is None:
            lag_days = self.reporting_lag

        cache_key = (str(trade_date), int(lag_days))
        if cache_key in self._snapshot_cache:
            return self._snapshot_cache[cache_key].copy()

        trade_dt = pd.to_datetime(trade_date)
        cutoff_dt = trade_dt - timedelta(days=lag_days)

        if self.financial_data is None or self.financial_data.empty:
            return pd.DataFrame()

        # Filter by period <= cutoff
        # self.financial_data index is (symbol, period)
        valid_mask = self.financial_data.index.get_level_values("period") <= cutoff_dt
        valid_data = self.financial_data[valid_mask]

        if valid_data.empty:
            print(f"WARNING: No financial data available before {cutoff_dt}")
            return pd.DataFrame()

        # Get latest for each symbol
        # Group by symbol and take last (since we sorted by index (symbol, period))
        snapshot = valid_data.groupby(level="symbol").last()
        self._snapshot_cache[cache_key] = snapshot

        return snapshot.copy()

    def enrich_with_yoy_data(
        self, current_df: pd.DataFrame, trade_date: str
    ) -> pd.DataFrame:
        """
        Fetches data from 1 year ago and merges it with the current snapshot
        to allow Year-Over-Year (YoY) calculations.
        """
        # 1. Calculate the date 1 year ago
        curr_dt = pd.to_datetime(trade_date)
        prev_dt = curr_dt - pd.Timedelta(days=365)
        prev_date_str = prev_dt.strftime("%Y-%m-%d")

        # 2. Fetch the snapshot for that previous date
        # We reuse your existing get_valid_snapshot method
        print(f"    (Data) Fetching YoY comparison data from {prev_date_str}...")
        try:
            prev_df = self.get_valid_snapshot(prev_date_str)
        except Exception:
            print(
                "    (Warning) Could not fetch previous year data. YoY metrics will be NaN."
            )
            prev_df = pd.DataFrame()

        # 3. If no previous data, create empty _Prev columns so strategies don't crash
        if prev_df.empty:
            print(
                "    (Warning) No previous year data available. Adding empty _Prev columns."
            )
            for col in current_df.columns:
                current_df[f"{col}_Prev"] = np.nan
            return current_df

        # 4. Rename columns to '_Prev' (e.g., 'TotalRevenue' -> 'TotalRevenue_Prev')
        prev_df = prev_df.add_suffix("_Prev")

        # 5. Merge with current data
        # Left join ensures we keep our current universe of stocks
        combined_df = current_df.join(prev_df, how="left")

        return combined_df

    def get_price(self, symbol: str, date: str, tolerance_days: int = 7) -> float:
        """Get closing price with fixed types for Pylance."""
        if self.price_matrix is None:
            return np.nan

        target_dt = pd.to_datetime(date)
        if hasattr(target_dt, "tzinfo") and target_dt.tzinfo is not None:
            target_dt = target_dt.tz_localize(None)

        if symbol not in self.price_matrix.columns:
            return np.nan

        # Check exact date
        if target_dt in self.price_matrix.index:
            val = self.price_matrix.at[target_dt, symbol]
            return cast(float, val) if pd.notna(val) else np.nan

        # Search within tolerance
        start_win = target_dt - timedelta(days=tolerance_days)
        end_win = target_dt + timedelta(days=tolerance_days)

        try:
            window = self.price_matrix.loc[start_win:end_win]
        except KeyError:
            return np.nan

        if window.empty:
            return np.nan

        # FIX: Convert index to Series to allow subtraction and abs() with Timestamp
        idx_series = window.index.to_series()
        distances = (idx_series - target_dt).abs()

        # Find closest date with valid data for this symbol
        # FIX: Convert to numpy array explicitly for np.argsort compatibility
        sorted_dates = window.index[np.argsort(np.asarray(distances.values))]

        for d in sorted_dates:
            val = self.price_matrix.at[d, symbol]
            if pd.notna(val):
                return cast(float, val)

        return np.nan

    def get_prices_bulk(self, symbols: list, date: str) -> pd.Series:
        """Get prices for multiple symbols with explicit casting."""
        if self.price_matrix is None:
            return pd.Series(np.nan, index=symbols)

        target_dt = pd.to_datetime(date)

        if target_dt in self.price_matrix.index:
            closest_date = target_dt
        else:
            start_win = target_dt - timedelta(days=7)
            end_win = target_dt + timedelta(days=7)
            window = self.price_matrix.loc[start_win:end_win]

            if window.empty:
                return pd.Series(np.nan, index=symbols)

            # Fix: Convert argmin result to int for index compatibility
            distances = (window.index.to_series() - target_dt).abs()
            closest_date = window.index[int(np.asarray(distances.values).argmin())]

        # FIX: Ensure 'prices' is treated as a Series and reindex returns a Series
        prices = cast(pd.Series, self.price_matrix.loc[closest_date])
        result = prices.reindex(symbols)

        return cast(pd.Series, result)

    def filter_universe(
        self, df: pd.DataFrame, trade_date: str, min_mkt_cap_mm: int = 200
    ) -> pd.DataFrame:
        """
        Filters out warrants, illiquid stocks, and micro-caps.

        Parameters:
        -----------
        df : pd.DataFrame
            The financial snapshot
        trade_date : str
            Date to check prices for Market Cap calculation
        min_mkt_cap_mm : int
            Minimum Market Cap in Millions (default $200M)
        """
        # 1. Filter Warrants and Derivatives by Symbol Suffix
        mask_warrants = df.index.str.match(r"^[A-Z]{4}[WRU]$")
        df = df[~mask_warrants]

        # 2. Market Cap Filter
        if "OrdinarySharesNumber" in df.columns:
            # Optimization: Drop rows with missing shares BEFORE price lookup
            df = df.dropna(subset=["OrdinarySharesNumber"])

            # Get price for remaining symbols
            current_prices = self.get_prices_bulk(df.index.tolist(), trade_date)

            # Align indices
            df = df.copy()
            df["_CurrentPrice"] = current_prices

            # Calculate Market Cap (Value in Millions usually, assuming shares are raw count)
            df["MarketCap"] = (
                df["OrdinarySharesNumber"] * df["_CurrentPrice"]
            ) / 1_000_000

            # Filter
            original_count = len(df)
            df = df[df["MarketCap"] > min_mkt_cap_mm]
            print(
                f"    (Universe Filter) Dropped {original_count - len(df)} stocks < ${min_mkt_cap_mm}M Market Cap"
            )

        elif "TotalRevenue" in df.columns:
            # Fallback: Use Revenue as a proxy for size if Shares/Price missing
            print(
                "    (Universe Filter) 'OrdinarySharesNumber' missing. Using Revenue proxy."
            )
            df = df[
                df["TotalRevenue"] > (min_mkt_cap_mm * 1_000_000 * 0.5)
            ]  # Approx proxy

        return df

    def apply_strategy(
        self,
        snapshot: pd.DataFrame,
        strategy: str,
        top_n: int = 10,
        trade_date: Optional[str] = None,
    ) -> list:
        """
        Apply a stock selection strategy to the snapshot.

        Parameters:
        -----------
        snapshot : pd.DataFrame
            Point-in-time financial data
        strategy : str
            Strategy name: 'magic_formula', 'moat', 'fcf_yield', 'quality'
        top_n : int
            Number of stocks to select
        trade_date : str
            Date for universe filtering

        Returns:
        --------
        list
            Selected stock symbols
        """
        # --- NEW: Apply Universe Filter First ---
        if trade_date:
            snapshot = self.filter_universe(snapshot, trade_date, min_mkt_cap_mm=250)
        # ----------------------------------------

        df = snapshot.copy()

        # Ensure MarketCap exists for tie-breaking (fill with 0 if missing)
        if "MarketCap" not in df.columns:
            df["MarketCap"] = 0

        if strategy == "magic_formula":
            # Greenblatt's Magic Formula: High ROIC + High Earnings Yield

            # Calculate EnterpriseValue if missing (EV = Market Cap + Debt - Cash)
            if "EnterpriseValue" not in df.columns:
                ev_cols = ["TotalDebt", "CashAndCashEquivalents", "MarketCap"]
                if all(c in df.columns for c in ev_cols):
                    # Ensure MarketCap is valid for calculation
                    df = df[df["MarketCap"] > 0]
                    # MarketCap is in Millions, convert to raw
                    df["EnterpriseValue"] = (
                        (df["MarketCap"] * 1_000_000)
                        + df["TotalDebt"]
                        - df["CashAndCashEquivalents"]
                    )

            required = ["EBIT", "InvestedCapital", "EnterpriseValue"]
            if not all(col in df.columns for col in required):
                print(
                    f"  Missing columns for magic_formula. Have: {df.columns.tolist()}"
                )
                return []

            df = df.dropna(subset=required)
            df = df[df["InvestedCapital"] > 0]
            df = df[df["EnterpriseValue"] > 0]

            # 1. Calculate the Ratios
            df["ROIC"] = df["EBIT"] / df["InvestedCapital"]
            df["EarningsYield"] = df["EBIT"] / df["EnterpriseValue"]

            # 2. Rank both factors (higher is better)
            df["Quality_Rank"] = robust_score(df["ROIC"], higher_is_better=True)
            df["Value_Rank"] = robust_score(df["EarningsYield"], higher_is_better=True)

            # 3. Combine Ranks
            df["Magic_Score"] = df["Quality_Rank"] + df["Value_Rank"]

            # 4. Pick top N, using MarketCap as a tie-breaker to avoid alphabetical bias
            top_picks = df.nlargest(top_n, ["Magic_Score", "MarketCap"]).index.tolist()

        elif strategy == "moat":
            # High Operating Margin + R&D Intensity
            required = ["OperatingIncome", "TotalRevenue"]
            if not all(col in df.columns for col in required):
                print(f"  Missing columns for moat. Have: {df.columns.tolist()}")
                return []

            df = df.dropna(subset=required)
            df = df[df["TotalRevenue"] > 100_000_000]  # > $100M revenue

            df["Operating_Margin"] = df["OperatingIncome"] / df["TotalRevenue"]

            # R&D is optional
            if "ResearchAndDevelopment" in df.columns:
                df["RD_Intensity"] = (
                    df["ResearchAndDevelopment"].fillna(0) / df["TotalRevenue"]
                )
                df["RD_Score"] = robust_score(df["RD_Intensity"], higher_is_better=True)
            else:
                df["RD_Score"] = 50  # Neutral

            df["Margin_Score"] = robust_score(
                df["Operating_Margin"], higher_is_better=True
            )
            df["Moat_Score"] = (df["Margin_Score"] + df["RD_Score"]) / 2

            top_picks = df.nlargest(top_n, ["Moat_Score", "MarketCap"]).index.tolist()

        elif strategy == "fcf_yield":
            # Free Cash Flow Yield
            required = ["OperatingCashFlow", "CapitalExpenditure"]
            if not all(col in df.columns for col in required):
                print(f"  Missing columns for fcf_yield. Have: {df.columns.tolist()}")
                return []

            df = df.dropna(subset=required)
            df["FreeCashFlow"] = (
                df["OperatingCashFlow"] - df["CapitalExpenditure"].abs()
            )
            df = df[df["FreeCashFlow"] > 0]  # Positive FCF only

            df["FCF_Score"] = robust_score(df["FreeCashFlow"], higher_is_better=True)
            top_picks = df.nlargest(top_n, ["FCF_Score", "MarketCap"]).index.tolist()

        elif strategy == "quality":
            # Simple quality: High ROIC but stricter
            required = ["EBIT", "InvestedCapital"]
            if not all(col in df.columns for col in required):
                return []

            df = df.dropna(subset=required)
            df = df[df["InvestedCapital"] > 0]
            df["ROIC"] = df["EBIT"] / df["InvestedCapital"]

            # Filter for realistic ROIC (avoid data errors > 100%)
            df = df[(df["ROIC"] > 0.05) & (df["ROIC"] < 1.0)]

            df["Quality_Score"] = robust_score(df["ROIC"], higher_is_better=True)
            top_picks = df.nlargest(
                top_n, ["Quality_Score", "MarketCap"]
            ).index.tolist()

        elif strategy in ("super_margins", "diamonds_in_dirt"):
            # -----------------------------------------------------------
            # Strategy: "Diamond in the Dirt"
            # Find high-quality companies (great margins) at cheap prices
            # -----------------------------------------------------------

            required = [
                "NetIncome",
                "StockholdersEquity",
                "TotalRevenue",
                "GrossProfit",
                "OperatingIncome",
                "TotalDebt",
            ]

            # Check availability
            if not all(col in df.columns for col in required):
                print(
                    f"    [Warning] Missing data for strategy: {[c for c in required if c not in df.columns]}"
                )
                return []

            # Clean Zeros
            df = df[(df["TotalRevenue"] > 0) & (df["StockholdersEquity"] > 0)].copy()

            # A. Calculate Metrics
            df["ROE"] = df["NetIncome"] / df["StockholdersEquity"]
            df["Gross_Margin"] = df["GrossProfit"] / df["TotalRevenue"]
            df["Op_Margin"] = df["OperatingIncome"] / df["TotalRevenue"]
            df["Debt_to_Rev"] = df["TotalDebt"] / df["TotalRevenue"]

            # B. The "Diamond" Screen (High Quality, but REALISTIC)
            #    - Gross Margin > 40% (Keep this, it's good for Moats)
            #    - Op Margin > 15% (Lowered from 40% to catch real businesses like AAPL/MSFT)
            #    - ROE > 15% (Increased to ensure efficiency)
            #    - Debt < 3x Revenue (Safety)
            mask_quality = (
                (df["Gross_Margin"] > 0.40)
                & (df["Op_Margin"] > 0.15)  # <--- CRITICAL FIX (Was 0.40)
                & (df["ROE"] > 0.15)
                & (df["Debt_to_Rev"] < 3.0)
            )
            passed = df[mask_quality].copy()

            # Debugging: See how many survived
            if len(passed) < top_n and len(passed) > 0:
                print(
                    f"    (Note) Only {len(passed)} companies met the 'Diamond' criteria."
                )
            elif len(passed) == 0:
                print("    (Note) No companies met the 'Diamond' criteria.")
                return []

            # C. The "Dirt" Sort (Find the CHEAPEST of the Great Companies)
            #    If we just pick High ROE, we buy expensive stocks.
            #    We must sort by EARNINGS YIELD (Operating Income / Market Cap).
            if "MarketCap" in passed.columns and (passed["MarketCap"] > 0).any():
                # Higher Yield = Cheaper Stock
                passed["Earnings_Yield"] = passed["OperatingIncome"] / (
                    passed["MarketCap"] * 1_000_000
                )
                # Filter out negative yields (loss makers that slipped through)
                passed = passed[passed["Earnings_Yield"] > 0]
                top_picks = passed.nlargest(top_n, "Earnings_Yield").index.tolist()
            else:
                # Fallback if no Market Cap data (Rank by ROE)
                top_picks = passed.nlargest(top_n, "ROE").index.tolist()

        # ==============================================================================
        # STRATEGY: THE CANNIBAL (Aggressive Buybacks)
        # Logic: Buy companies reducing share count by >3% AND paying with Free Cash Flow
        # ==============================================================================
        elif strategy == "cannibal":
            # 1. Get YoY Data (Need 'OrdinarySharesNumber_Prev')
            if trade_date is None:
                print("  trade_date is required for cannibal strategy.")
                return []
            df = self.enrich_with_yoy_data(df, trade_date)

            # Calculate FreeCashFlow if not present
            if "FreeCashFlow" not in df.columns:
                if (
                    "OperatingCashFlow" in df.columns
                    and "CapitalExpenditure" in df.columns
                ):
                    df["FreeCashFlow"] = (
                        df["OperatingCashFlow"] - df["CapitalExpenditure"].abs()
                    )
                else:
                    print("  Missing columns for FreeCashFlow calculation.")
                    return []

            # 2. Required Columns - check if YoY data is available
            req = [
                "OrdinarySharesNumber",
                "OrdinarySharesNumber_Prev",
                "FreeCashFlow",
                "MarketCap",
            ]
            if not all(c in df.columns for c in req):
                print(f"  Missing columns for cannibal. Have: {df.columns.tolist()}")
                return []

            # Drop rows where we don't have YoY comparison data
            df = df.dropna(subset=["OrdinarySharesNumber", "OrdinarySharesNumber_Prev"])
            if df.empty:
                print("  No stocks with valid YoY share data for cannibal strategy.")
                return []

            # 3. Calculate Metrics
            # Share Change: (Current - Prev) / Prev. Negative number = Buyback.
            df["Share_Change_Pct"] = (
                df["OrdinarySharesNumber"] - df["OrdinarySharesNumber_Prev"]
            ) / df["OrdinarySharesNumber_Prev"]

            # Buyback Yield: RepurchaseOfCapitalStock (usually negative in CF) / MarketCap
            if "RepurchaseOfCapitalStock" in df.columns:
                df["Buyback_Yield"] = df["RepurchaseOfCapitalStock"].abs() / (
                    df["MarketCap"] * 1_000_000
                )
            else:
                df["Buyback_Yield"] = 0

            # 4. The "Cannibal" Screen
            mask = (
                (df["Share_Change_Pct"] < -0.03)  # Shares reduced by at least 3%
                & (df["FreeCashFlow"] > 0)  # Must be FCF positive
                & (df["MarketCap"] > 200)  # Liquidity Filter ($200M+)
            )
            passed = df[mask].copy()

            # 5. Rank by "Buyback Yield" (Who is eating themselves the fastest?)
            # We want the highest yield (biggest buyback relative to price)
            top_picks = passed.nlargest(top_n, "Buyback_Yield").index.tolist()

        # ==============================================================================
        # STRATEGY: PIOTROSKI F-SCORE (Fundamental Health)
        # Logic: 0-9 Point Score based on Profitability, Leverage, and Efficiency
        # ==============================================================================
        elif strategy == "piotroski_f_score":
            # 1. Get YoY Data for comparisons
            if trade_date is None:
                print("  trade_date is required for piotroski_f_score strategy.")
                return []
            df = self.enrich_with_yoy_data(df, trade_date)

            # Check if we have any valid YoY data
            prev_cols = [c for c in df.columns if c.endswith("_Prev")]
            if not prev_cols or df[prev_cols].isna().all().all():
                print(
                    "  No YoY data available for piotroski_f_score. Skipping YoY criteria."
                )

            # 2. Define Score Components (Start at 0)
            df["F_Score"] = 0

            # --- Profitability (4 Points) ---
            # 1. Positive Net Income
            if "NetIncome" in df.columns:
                df.loc[df["NetIncome"] > 0, "F_Score"] += 1
            # 2. Positive Operating Cash Flow
            if "OperatingCashFlow" in df.columns:
                df.loc[df["OperatingCashFlow"] > 0, "F_Score"] += 1
            # 3. ROA Increasing (Current ROA > Prev ROA)
            # Need Average Assets for strict ROA, but simplified: NetIncome/TotalAssets
            if all(
                c in df.columns
                for c in [
                    "NetIncome",
                    "TotalAssets",
                    "NetIncome_Prev",
                    "TotalAssets_Prev",
                ]
            ):
                df["ROA"] = df["NetIncome"] / df["TotalAssets"]
                df["ROA_Prev"] = df["NetIncome_Prev"] / df["TotalAssets_Prev"]
                df.loc[df["ROA"] > df["ROA_Prev"], "F_Score"] += 1
            # 4. Quality of Earnings (Cash Flow > Net Income)
            if all(c in df.columns for c in ["OperatingCashFlow", "NetIncome"]):
                df.loc[df["OperatingCashFlow"] > df["NetIncome"], "F_Score"] += 1

            # --- Leverage & Liquidity (3 Points) ---
            # 5. Long Term Debt Decreasing (or stable)
            if all(c in df.columns for c in ["TotalDebt", "TotalDebt_Prev"]):
                df.loc[df["TotalDebt"] <= df["TotalDebt_Prev"], "F_Score"] += 1
            # 6. Current Ratio Increasing (Liquidity improving)
            if all(
                c in df.columns
                for c in [
                    "CurrentAssets",
                    "CurrentLiabilities",
                    "CurrentAssets_Prev",
                    "CurrentLiabilities_Prev",
                ]
            ):
                df["Current_Ratio"] = df["CurrentAssets"] / df["CurrentLiabilities"]
                df["Current_Ratio_Prev"] = (
                    df["CurrentAssets_Prev"] / df["CurrentLiabilities_Prev"]
                )
                df.loc[df["Current_Ratio"] > df["Current_Ratio_Prev"], "F_Score"] += 1
            # 7. No Dilution (Shares <= Prev Shares)
            if all(
                c in df.columns
                for c in ["OrdinarySharesNumber", "OrdinarySharesNumber_Prev"]
            ):
                df.loc[
                    df["OrdinarySharesNumber"] <= df["OrdinarySharesNumber_Prev"],
                    "F_Score",
                ] += 1

            # --- Operating Efficiency (2 Points) ---
            # 8. Gross Margin Increasing
            if all(
                c in df.columns
                for c in [
                    "GrossProfit",
                    "TotalRevenue",
                    "GrossProfit_Prev",
                    "TotalRevenue_Prev",
                ]
            ):
                df["Gross_Margin"] = df["GrossProfit"] / df["TotalRevenue"]
                df["Gross_Margin_Prev"] = (
                    df["GrossProfit_Prev"] / df["TotalRevenue_Prev"]
                )
                df.loc[df["Gross_Margin"] > df["Gross_Margin_Prev"], "F_Score"] += 1
            # 9. Asset Turnover Increasing (Revenue / Assets)
            if all(
                c in df.columns
                for c in [
                    "TotalRevenue",
                    "TotalAssets",
                    "TotalRevenue_Prev",
                    "TotalAssets_Prev",
                ]
            ):
                df["Asset_Turnover"] = df["TotalRevenue"] / df["TotalAssets"]
                df["Asset_Turnover_Prev"] = (
                    df["TotalRevenue_Prev"] / df["TotalAssets_Prev"]
                )
                df.loc[df["Asset_Turnover"] > df["Asset_Turnover_Prev"], "F_Score"] += 1

            # 3. Filter: Only Elite Stocks (Score 7, 8, or 9)
            passed = df[df["F_Score"] >= 7].copy()

            # 4. Rank: Break ties with Cheapest Valuation (Earnings Yield)
            if "MarketCap" in passed.columns and "EBIT" in passed.columns:
                passed["Earnings_Yield"] = passed["EBIT"] / (
                    passed["MarketCap"] * 1_000_000
                )
                top_picks = passed.nlargest(top_n, "Earnings_Yield").index.tolist()
            else:
                top_picks = passed.nlargest(top_n, "F_Score").index.tolist()

        # ==============================================================================
        # STRATEGY: RULE OF 40 (SaaS / High Growth)
        # Logic: Revenue Growth % + FCF Margin % > 40
        # ==============================================================================
        elif strategy == "rule_of_40":
            # 1. Get YoY Data
            if trade_date is None:
                print("  trade_date is required for rule_of_40 strategy.")
                return []
            df = self.enrich_with_yoy_data(df, trade_date)

            # Calculate FreeCashFlow if not present
            if "FreeCashFlow" not in df.columns:
                if (
                    "OperatingCashFlow" in df.columns
                    and "CapitalExpenditure" in df.columns
                ):
                    df["FreeCashFlow"] = (
                        df["OperatingCashFlow"] - df["CapitalExpenditure"].abs()
                    )
                else:
                    print("  Missing columns for FreeCashFlow calculation.")
                    return []

            # 2. Calculate Growth & Margin
            # Rev Growth = (Current - Prev) / Prev
            if not all(c in df.columns for c in ["TotalRevenue", "TotalRevenue_Prev"]):
                print("  Missing TotalRevenue columns for rule_of_40.")
                return []

            # Drop rows without YoY revenue data
            df = df.dropna(subset=["TotalRevenue", "TotalRevenue_Prev"])
            if df.empty:
                print(
                    "  No stocks with valid YoY revenue data for rule_of_40 strategy."
                )
                return []

            df["Rev_Growth"] = (df["TotalRevenue"] - df["TotalRevenue_Prev"]) / df[
                "TotalRevenue_Prev"
            ]

            # FCF Margin = FCF / TotalRevenue
            df["FCF_Margin"] = df["FreeCashFlow"] / df["TotalRevenue"]

            # 3. The "Rule of 40" Score
            # Multiply by 100 to get the integer (e.g., 0.20 + 0.20 = 40)
            df["R40_Score"] = (df["Rev_Growth"] + df["FCF_Margin"]) * 100

            # 4. Filter
            # Standard rule is > 40. We can be stricter (> 50) for top picks.
            mask = (df["R40_Score"] > 40) & (
                df["TotalRevenue"] > 50_000_000
            )  # Minimum $50M Revenue to avoid tiny startups
            passed = df[mask].copy()

            # 5. Rank by Score (Highest combo of Growth + Cash Flow)
            top_picks = passed.nlargest(top_n, "R40_Score").index.tolist()

        else:
            # Fallback for existing strategies
            # Ensure we return at least something if specific logic isn't changed
            return snapshot.head(top_n).index.tolist()

        return top_picks

    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        strategy: str = "magic_formula",
        top_n: int = 10,
        verbose: bool = True,
    ) -> dict:
        """
        Run a single backtest period.

        Parameters:
        -----------
        start_date : str
            Entry date (e.g., '2023-01-01')
        end_date : str
            Exit date (e.g., '2024-01-01')
        strategy : str
            Strategy to use
        top_n : int
            Number of stocks to hold
        verbose : bool
            Print detailed output

        Returns:
        --------
        dict
            Results including picks, returns, and benchmark comparison
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  BACKTEST: {start_date} → {end_date}")
            print(f"  Strategy: {strategy.upper()} | Top {top_n} picks")
            print(f"{'='*60}")

        # 1. Get point-in-time snapshot
        snapshot = self.get_valid_snapshot(start_date)
        if snapshot.empty:
            return {"error": "No data available for this period"}

        if verbose:
            print(f"\n  Data snapshot: {len(snapshot)} companies available")

        # 2. Apply strategy to select stocks
        picks = self.apply_strategy(snapshot, strategy, top_n, trade_date=start_date)

        if not picks:
            return {"error": "No stocks selected by strategy"}

        if verbose:
            print(f"  Selected: {picks}")

        # 3. Get entry and exit prices
        entry_prices = self.get_prices_bulk(picks, start_date)
        exit_prices = self.get_prices_bulk(picks, end_date)

        # 4. Calculate returns
        # FIX: Align prices to symbols using index to avoid mismatch
        results_df = pd.DataFrame(index=picks)
        results_df.index.name = "symbol"
        results_df["entry_price"] = entry_prices
        results_df["exit_price"] = exit_prices
        results_df = results_df.reset_index()

        results_df["return"] = (
            results_df["exit_price"] - results_df["entry_price"]
        ) / results_df["entry_price"]

        # Remove stocks with missing prices
        valid_results = results_df.dropna()

        if valid_results.empty:
            return {"error": "No valid price data for selected stocks"}

        # Portfolio return (equal weighted)
        portfolio_return = valid_results["return"].mean()

        # 5. Benchmark comparison (QQQ or SPY)
        benchmark_entry = self.get_price(self.benchmark_symbol, start_date)
        benchmark_exit = self.get_price(self.benchmark_symbol, end_date)

        if pd.notna(benchmark_entry) and pd.notna(benchmark_exit):
            benchmark_return = (benchmark_exit - benchmark_entry) / benchmark_entry
        else:
            benchmark_return = np.nan

        # 6. Alpha (excess return vs benchmark)
        alpha = (
            portfolio_return - benchmark_return
            if pd.notna(benchmark_return)
            else np.nan
        )

        if verbose:
            print(f"\n  RESULTS:")
            print(f"  {'-'*40}")
            print(valid_results.to_string(index=False))
            print(f"  {'-'*40}")
            print(f"  Portfolio Return: {portfolio_return*100:+.2f}%")
            print(
                f"  Benchmark ({self.benchmark_symbol}): {benchmark_return*100:+.2f}%"
                if pd.notna(benchmark_return)
                else f"  Benchmark: N/A"
            )
            print(f"  Alpha: {alpha*100:+.2f}%" if pd.notna(alpha) else "  Alpha: N/A")

        return {
            "start_date": start_date,
            "end_date": end_date,
            "strategy": strategy,
            "picks": picks,
            "results_df": valid_results,
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return,
            "alpha": alpha,
            "num_stocks": len(valid_results),
        }

    def run_multi_period_backtest(
        self,
        periods: list,
        strategy: str = "magic_formula",
        top_n: int = 10,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Run backtest across multiple periods.

        Parameters:
        -----------
        periods : list of tuples
            List of (start_date, end_date) tuples
        strategy : str
            Strategy to use
        top_n : int
            Number of stocks to hold

        Returns:
        --------
        pd.DataFrame
            Summary of all period results
        """
        all_results = []

        loop = tqdm(
            periods,
            desc=f"Backtest {strategy}",
            unit="period",
            disable=not show_progress,
        )
        for start, end in loop:
            result = self.run_backtest(start, end, strategy, top_n, verbose=True)

            if "error" not in result:
                all_results.append(
                    {
                        "Period": f"{start} → {end}",
                        "Strategy_Return": result["portfolio_return"],
                        "Benchmark_Return": result["benchmark_return"],
                        "Alpha": result["alpha"],
                        "Num_Stocks": result["num_stocks"],
                    }
                )
            loop.set_postfix({"valid": len(all_results)})

        if not all_results:
            print("No valid results from any period.")
            return pd.DataFrame()

        summary = pd.DataFrame(all_results)

        # Calculate cumulative/average performance
        print(f"\n{'='*60}")
        print("  MULTI-PERIOD SUMMARY")
        print(f"{'='*60}")
        print(summary.to_string(index=False))

        avg_return = summary["Strategy_Return"].mean()
        avg_benchmark = summary["Benchmark_Return"].mean()
        avg_alpha = summary["Alpha"].mean()

        print(f"\n  Average Strategy Return: {avg_return*100:+.2f}%")
        print(f"  Average Benchmark Return: {avg_benchmark*100:+.2f}%")
        print(f"  Average Alpha: {avg_alpha*100:+.2f}%")

        return summary

    def calculate_portfolio_return(
        self, symbols: list, start_date: str, end_date: str
    ) -> dict:
        """
        Calculate portfolio return for a list of symbols (equal-weighted).

        Parameters:
        -----------
        symbols : list
            List of stock symbols
        start_date : str
            Entry date
        end_date : str
            Exit date

        Returns:
        --------
        dict
            Dictionary with portfolio_return and individual results
        """
        if not symbols:
            return {"portfolio_return": np.nan, "results_df": pd.DataFrame()}

        entry_prices = self.get_prices_bulk(symbols, start_date)
        exit_prices = self.get_prices_bulk(symbols, end_date)

        results_df = pd.DataFrame(index=symbols)
        results_df.index.name = "symbol"
        results_df["entry_price"] = entry_prices
        results_df["exit_price"] = exit_prices
        results_df = results_df.reset_index()

        results_df["return"] = (
            results_df["exit_price"] - results_df["entry_price"]
        ) / results_df["entry_price"]

        valid_results = results_df.dropna()

        if valid_results.empty:
            return {"portfolio_return": np.nan, "results_df": valid_results}

        portfolio_return = valid_results["return"].mean()

        return {"portfolio_return": portfolio_return, "results_df": valid_results}

    def run_combo_backtest(
        self,
        start_date: str,
        end_date: str,
        strategies: list,
        top_n_per_strat: int = 5,
        verbose: bool = True,
    ) -> dict:
        """
        Runs a backtest combining multiple strategies (e.g., 5 picks from Moat + 5 from Magic Formula).
        """
        # 1. Get Snapshot
        snapshot = self.get_valid_snapshot(start_date)

        # 2. Run Universe Filter ONCE (Efficiency)
        snapshot = self.filter_universe(snapshot, start_date, min_mkt_cap_mm=250)

        # 3. Collect Picks from each Strategy
        combined_picks = set()  # Use a set to handle duplicates automatically

        if verbose:
            print(f"\n  [Combo] Testing Mix: {strategies} on {start_date}")

        per_strategy_pick_counts: Dict[str, int] = {}
        for strat in strategies:
            # We assume 'snapshot' is already filtered, but apply_strategy might re-filter.
            # That's fine, it's fast enough.
            picks = self.apply_strategy(
                snapshot, strat, top_n=top_n_per_strat, trade_date=start_date
            )
            per_strategy_pick_counts[strat] = len(picks)
            if verbose:
                print(f"    -> {strat} contributed: {picks}")
            combined_picks.update(picks)

        final_portfolio = list(combined_picks)

        # 4. Calculate Returns
        # Note: If a stock is picked by BOTH strategies, it is just held once in this simple model.
        result = self.calculate_portfolio_return(final_portfolio, start_date, end_date)
        valid_results = cast(pd.DataFrame, result["results_df"])
        selected_count = int(len(final_portfolio))
        priced_count = int(len(valid_results))

        if selected_count == 0:
            status = "no_selections"
        elif priced_count == 0:
            status = "no_valid_prices"
        else:
            status = "ok"

        return {
            "period": f"{start_date} -> {end_date}",
            "strategy": " + ".join(strategies),
            "return": result["portfolio_return"],
            "picks": final_portfolio,
            "selected_count": selected_count,
            "picks_count": priced_count,
            "status": status,
            "per_strategy_pick_counts": per_strategy_pick_counts,
        }

    def _generate_monthly_periods(
        self, start_date: str, end_date: str, holding_months: int = 1
    ) -> List[Tuple[str, str]]:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        if start >= end:
            return []

        entry_dates = pd.date_range(start=start, end=end, freq="MS")
        periods: List[Tuple[str, str]] = []
        for entry_dt in entry_dates:
            exit_dt = entry_dt + pd.DateOffset(months=holding_months)
            if exit_dt > end:
                break
            periods.append(
                (entry_dt.strftime("%Y-%m-%d"), exit_dt.strftime("%Y-%m-%d"))
            )

        if not periods:
            exit_dt = min(start + pd.DateOffset(months=holding_months), end)
            if exit_dt > start:
                periods.append(
                    (start.strftime("%Y-%m-%d"), exit_dt.strftime("%Y-%m-%d"))
                )
        return periods

    def _classify_regime(
        self,
        date: str,
        return_window_days: int = 252,
        vol_window_days: int = 63,
    ) -> str:
        cache_key = (str(date), int(return_window_days), int(vol_window_days))
        if cache_key in self._regime_cache:
            return self._regime_cache[cache_key]

        if self.price_matrix is None or self.benchmark_symbol not in self.price_matrix:
            self._regime_cache[cache_key] = "NEUTRAL"
            return "NEUTRAL"

        target_dt = pd.to_datetime(date)
        bench = self.price_matrix[self.benchmark_symbol].dropna()
        bench = bench[bench.index <= target_dt]
        if len(bench) < 20:
            self._regime_cache[cache_key] = "NEUTRAL"
            return "NEUTRAL"

        lookback = bench.iloc[-min(return_window_days, len(bench)) :]
        trailing_return = (
            (lookback.iloc[-1] / lookback.iloc[0] - 1) if len(lookback) > 1 else 0
        )
        drawdown = (lookback.iloc[-1] / lookback.max() - 1) if len(lookback) > 0 else 0

        recent_rets = bench.pct_change().dropna().iloc[-min(vol_window_days, len(bench)) :]
        realized_vol = (
            float(recent_rets.std() * np.sqrt(252)) if len(recent_rets) > 1 else 0.0
        )

        if drawdown <= -0.20 or trailing_return <= -0.15:
            regime = "BEAR"
        elif drawdown <= -0.10 or realized_vol >= 0.35:
            regime = "STRESS"
        elif trailing_return >= 0.15 and drawdown > -0.10:
            regime = "BULL"
        else:
            regime = "NEUTRAL"

        self._regime_cache[cache_key] = regime
        return regime

    def _compute_momentum_state(
        self,
        symbol: str,
        date: str,
        ma_period: int = 200,
        precomputed: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if precomputed is not None:
            pre_key = (symbol, str(date))
            if pre_key in precomputed:
                return dict(precomputed[pre_key])

        cache_key = (str(symbol), str(date), int(ma_period))
        if cache_key in self._momentum_cache:
            return dict(self._momentum_cache[cache_key])

        if self.price_matrix is None or symbol not in self.price_matrix.columns:
            result = {"above_ma": None, "ma_gap_pct": None, "price": np.nan, "ma": np.nan}
            self._momentum_cache[cache_key] = result
            return dict(result)

        target_dt = pd.to_datetime(date)
        series = self.price_matrix[symbol].dropna()
        series = series[series.index <= target_dt]
        if series.empty:
            result = {"above_ma": None, "ma_gap_pct": None, "price": np.nan, "ma": np.nan}
            self._momentum_cache[cache_key] = result
            return dict(result)

        price = float(series.iloc[-1])
        if len(series) < ma_period:
            result = {"above_ma": None, "ma_gap_pct": None, "price": price, "ma": np.nan}
            self._momentum_cache[cache_key] = result
            return dict(result)

        ma = float(series.iloc[-ma_period:].mean())
        if ma == 0:
            result = {"above_ma": None, "ma_gap_pct": None, "price": price, "ma": ma}
            self._momentum_cache[cache_key] = result
            return dict(result)
        gap = ((price - ma) / ma) * 100.0
        result = {"above_ma": price > ma, "ma_gap_pct": gap, "price": price, "ma": ma}
        self._momentum_cache[cache_key] = result
        return dict(result)

    def _compute_tail_state(
        self,
        symbol: str,
        date: str,
        lookback_days: int = 252,
        min_obs: int = 80,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        cache_key = (str(symbol), str(date), int(lookback_days), int(min_obs))
        if use_cache and cache_key in self._tail_state_cache:
            return dict(self._tail_state_cache[cache_key])

        if self.price_matrix is None or symbol not in self.price_matrix.columns:
            result = {"tail_bucket": "Unknown", "tail_a": np.nan, "tail_b": np.nan}
            if use_cache:
                self._tail_state_cache[cache_key] = result
            return dict(result)

        target_dt = pd.to_datetime(date)
        series = self.price_matrix[symbol].dropna()
        series = series[series.index <= target_dt]
        if len(series) < min_obs:
            result = {"tail_bucket": "Unknown", "tail_a": np.nan, "tail_b": np.nan}
            if use_cache:
                self._tail_state_cache[cache_key] = result
            return dict(result)

        returns = series.pct_change().dropna().iloc[-lookback_days:]
        if len(returns) < min_obs:
            result = {"tail_bucket": "Unknown", "tail_a": np.nan, "tail_b": np.nan}
            if use_cache:
                self._tail_state_cache[cache_key] = result
            return dict(result)

        kurt_fisher = _require_real(returns.kurtosis(), "kurtosis")
        skewness = _require_real(returns.skew(), "skewness")

        if kurt_fisher > 2.0:
            bucket = "Heavy"
        elif kurt_fisher < 0.0:
            bucket = "Thin"
        else:
            bucket = "Normal"

        # Lightweight shape proxies for reporting continuity.
        tail_a = 1.0 - np.clip(skewness, -1.5, 1.5) * 0.2
        tail_b = 1.0 + np.clip(skewness, -1.5, 1.5) * 0.2
        result = {"tail_bucket": bucket, "tail_a": tail_a, "tail_b": tail_b}
        if use_cache:
            self._tail_state_cache[cache_key] = result
        return dict(result)

    def _precompute_regimes(
        self, period_starts: List[str], use_precomputed_regimes: bool = True
    ) -> Dict[str, str]:
        if not use_precomputed_regimes:
            return {}
        return {start: self._classify_regime(start) for start in period_starts}

    def _precompute_momentum_states(
        self,
        symbols: List[str],
        period_starts: List[str],
        ma_period: int = 200,
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        table: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if self.price_matrix is None or not symbols or not period_starts:
            return table

        cols = [s for s in symbols if s in self.price_matrix.columns]
        if not cols:
            return table

        prices = self.price_matrix[cols].sort_index()
        ma = prices.rolling(ma_period, min_periods=ma_period).mean()
        index = prices.index

        for date_str in period_starts:
            target = pd.to_datetime(date_str)
            pos = index.searchsorted(target, side="right") - 1
            if pos < 0:
                continue
            px_row = prices.iloc[pos]
            ma_row = ma.iloc[pos]
            for symbol in cols:
                px = px_row.get(symbol, np.nan)
                ma_val = ma_row.get(symbol, np.nan)
                if pd.isna(px):
                    state = {
                        "above_ma": None,
                        "ma_gap_pct": None,
                        "price": np.nan,
                        "ma": np.nan,
                    }
                elif pd.isna(ma_val) or ma_val == 0:
                    state = {
                        "above_ma": None,
                        "ma_gap_pct": None,
                        "price": float(px),
                        "ma": np.nan if pd.isna(ma_val) else float(ma_val),
                    }
                else:
                    gap = ((float(px) - float(ma_val)) / float(ma_val)) * 100.0
                    state = {
                        "above_ma": bool(float(px) > float(ma_val)),
                        "ma_gap_pct": gap,
                        "price": float(px),
                        "ma": float(ma_val),
                    }
                table[(symbol, date_str)] = state
        return table

    def _compute_altman_state(
        self, row: pd.Series, current_price: float
    ) -> Tuple[float, str]:
        total_assets = row.get("TotalAssets")
        total_liab = row.get("TotalLiabilitiesNetMinorityInterest")
        working_capital = row.get("WorkingCapital")
        retained_earnings = row.get("RetainedEarnings")
        ebit = row.get("EBIT")
        revenue = row.get("TotalRevenue")
        shares = row.get("OrdinarySharesNumber")

        fields = [
            total_assets,
            total_liab,
            working_capital,
            retained_earnings,
            ebit,
            revenue,
            shares,
        ]
        if any(pd.isna(v) for v in fields):
            return (np.nan, "Unknown")
        total_assets_f = _require_real(total_assets, "TotalAssets")
        total_liab_f = _require_real(
            total_liab, "TotalLiabilitiesNetMinorityInterest"
        )
        working_capital_f = _require_real(working_capital, "WorkingCapital")
        retained_earnings_f = _require_real(retained_earnings, "RetainedEarnings")
        ebit_f = _require_real(ebit, "EBIT")
        revenue_f = _require_real(revenue, "TotalRevenue")
        shares_f = _require_real(shares, "OrdinarySharesNumber")

        if total_assets_f <= 0 or total_liab_f <= 0:
            return (np.nan, "Unknown")
        if pd.isna(current_price) or current_price <= 0:
            return (np.nan, "Unknown")

        market_cap = float(current_price) * shares_f
        z_score = (
            1.2 * (working_capital_f / total_assets_f)
            + 1.4 * (retained_earnings_f / total_assets_f)
            + 3.3 * (ebit_f / total_assets_f)
            + 0.6 * (market_cap / total_liab_f)
            + 1.0 * (revenue_f / total_assets_f)
        )
        if z_score < 1.81:
            bucket = "DISTRESS (Risk)"
        elif z_score < 2.99:
            bucket = "GREY ZONE"
        else:
            bucket = "SAFE"
        return (z_score, bucket)

    def _compute_valuation_state(
        self, row: pd.Series, strategy: str, current_price: float
    ) -> Dict[str, Any]:
        shares = row.get("OrdinarySharesNumber")
        if pd.isna(current_price) or current_price <= 0 or pd.isna(shares) or shares <= 0:
            return {
                "Fair Value (Bear)": np.nan,
                "Fair Value (Base)": np.nan,
                "Fair Value": np.nan,
                "Valuation Sanity": "Unknown",
            }

        operating_cf = row.get("OperatingCashFlow")
        capex = row.get("CapitalExpenditure")
        revenue = row.get("TotalRevenue")
        equity = row.get("StockholdersEquity")

        fcf = 0.0
        if pd.notna(operating_cf) and pd.notna(capex):
            fcf = float(operating_cf) - abs(float(capex))

        sales = float(revenue) if pd.notna(revenue) else 0.0
        book = float(equity) if pd.notna(equity) else 0.0

        fcf_per_share = fcf / float(shares)
        sales_per_share = sales / float(shares) if sales > 0 else 0.0
        book_per_share = book / float(shares) if book > 0 else 0.0

        scenario = calculate_scenario_valuations(
            fcf_per_share=fcf_per_share,
            sales_per_share=sales_per_share,
            book_value_per_share=book_per_share,
            strategy=strategy,
        )
        fair_value = float(scenario["prob_weighted_value"])
        sanity_passed, sanity_reason = check_valuation_sanity(
            fair_value=fair_value,
            current_price=float(current_price),
        )

        return {
            "Fair Value (Bear)": float(scenario["bear_value"]),
            "Fair Value (Base)": float(scenario["base_value"]),
            "Fair Value": fair_value,
            "Valuation Sanity": "Passed" if sanity_passed else sanity_reason,
        }

    def _build_asset_panel_for_period(
        self,
        start_date: str,
        end_date: str,
        strategy: str,
        top_n: int,
        regime: str,
        risk_config: RiskVectorConfig,
        snapshot: Optional[pd.DataFrame] = None,
        momentum_table: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
        use_cached_tail: bool = True,
    ) -> pd.DataFrame:
        if snapshot is None:
            snapshot = self.get_valid_snapshot(start_date)
        if snapshot.empty:
            return pd.DataFrame()

        picks = self.apply_strategy(snapshot, strategy, top_n, trade_date=start_date)
        if not picks:
            return pd.DataFrame()

        entry_prices = self.get_prices_bulk(picks, start_date)
        exit_prices = self.get_prices_bulk(picks, end_date)
        rows: List[Dict[str, Any]] = []

        for symbol in picks:
            if symbol not in snapshot.index:
                continue
            row = snapshot.loc[symbol]
            entry_px = entry_prices.get(symbol, np.nan)
            exit_px = exit_prices.get(symbol, np.nan)
            ret = (
                (exit_px - entry_px) / entry_px
                if pd.notna(entry_px) and pd.notna(exit_px) and entry_px > 0
                else np.nan
            )

            momentum = self._compute_momentum_state(
                symbol, start_date, precomputed=momentum_table
            )
            tail = self._compute_tail_state(
                symbol, start_date, use_cache=use_cached_tail
            )
            altman_z, distress = self._compute_altman_state(row, float(entry_px))
            valuation = self._compute_valuation_state(row, strategy, float(entry_px))

            rows.append(
                {
                    "symbol": symbol,
                    "strategy": strategy,
                    "start_date": start_date,
                    "end_date": end_date,
                    "regime": regime,
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "return": ret,
                    "Altman Z-Score": altman_z,
                    "Distress Risk": distress,
                    "Tail_Risk": tail["tail_bucket"],
                    "Tail_a": tail["tail_a"],
                    "Tail_b": tail["tail_b"],
                    "Momentum_Above_MA": momentum["above_ma"],
                    "Momentum_MA_Gap_Pct": momentum["ma_gap_pct"],
                    "Momentum_Price": momentum["price"],
                    "Momentum_MA_200": momentum["ma"],
                    "Current Price": entry_px,
                    "Fair Value (Bear)": valuation["Fair Value (Bear)"],
                    "Fair Value (Base)": valuation["Fair Value (Base)"],
                    "Fair Value": valuation["Fair Value"],
                    "Valuation Sanity": valuation["Valuation Sanity"],
                }
            )

        if not rows:
            return pd.DataFrame()

        panel = pd.DataFrame(rows)
        panel = attach_risk_vectors(panel, config=risk_config, include_signal=True)
        return panel

    def run_research_backtest(
        self,
        start_date: str,
        end_date: str,
        strategy: str = "magic_formula",
        top_n: int = 10,
        holding_months: int = 1,
        risk_config: Optional[RiskVectorConfig] = None,
        verbose: bool = True,
        show_progress: bool = True,
        max_workers: Optional[int] = None,
        use_cached_tail: bool = True,
        use_precomputed_regimes: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run monthly rolling research backtests and return query-ready panels.
        """
        if self.price_matrix is None or self.financial_data is None:
            self.load_data()

        periods = self._generate_monthly_periods(start_date, end_date, holding_months)
        if not periods:
            empty = pd.DataFrame()
            self._last_research_outputs = {
                "asset_panel": empty,
                "portfolio_panel": empty,
                "equity_panel": empty,
            }
            return self._last_research_outputs

        cfg = risk_config or RiskVectorConfig()
        workers = (
            int(max_workers)
            if max_workers is not None and int(max_workers) > 0
            else min(8, os.cpu_count() or 4)
        )
        asset_frames: List[pd.DataFrame] = []
        portfolio_rows: List[Dict[str, Any]] = []
        equity_rows: List[Dict[str, Any]] = []
        stage_times: Dict[str, float] = {}
        run_start_ts = time.perf_counter()

        period_starts = [start for start, _ in periods]
        all_symbols = (
            self.price_matrix.columns.tolist() if self.price_matrix is not None else []
        )

        if verbose:
            print(
                f"[research] periods={len(periods)} workers={workers} cached_tail={use_cached_tail} precomputed_regimes={use_precomputed_regimes}"
            )
            if workers >= 8:
                print(
                    "[research] warning: high worker counts can increase SQLite contention."
                )

        t0 = time.perf_counter()
        snapshot_map: Dict[str, pd.DataFrame] = {}
        if workers > 1 and len(period_starts) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(period_starts))) as executor:
                futures = {
                    executor.submit(self.get_valid_snapshot, start): start
                    for start in period_starts
                }
                for future in as_completed(futures):
                    snapshot_map[futures[future]] = future.result()
        else:
            for start in period_starts:
                snapshot_map[start] = self.get_valid_snapshot(start)
        stage_times["snapshot_build"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        regime_map = self._precompute_regimes(
            period_starts, use_precomputed_regimes=use_precomputed_regimes
        )
        stage_times["regime_precompute"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        momentum_table = self._precompute_momentum_states(all_symbols, period_starts)
        stage_times["momentum_precompute"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        benchmark_return_map: Dict[Tuple[str, str], float] = {}
        for period_start, period_end in periods:
            benchmark_entry = self.get_price(self.benchmark_symbol, period_start)
            benchmark_exit = self.get_price(self.benchmark_symbol, period_end)
            if (
                pd.notna(benchmark_entry)
                and pd.notna(benchmark_exit)
                and benchmark_entry > 0
            ):
                benchmark_return_map[(period_start, period_end)] = (
                    benchmark_exit - benchmark_entry
                ) / benchmark_entry
            else:
                benchmark_return_map[(period_start, period_end)] = np.nan
        stage_times["benchmark_returns"] = time.perf_counter() - t0

        def _build_period_panel(period: Tuple[str, str]) -> Tuple[str, str, str, pd.DataFrame]:
            p_start, p_end = period
            regime = regime_map.get(p_start) or self._classify_regime(p_start)
            snap = snapshot_map.get(p_start, pd.DataFrame())
            panel = self._build_asset_panel_for_period(
                start_date=p_start,
                end_date=p_end,
                strategy=strategy,
                top_n=top_n,
                regime=regime,
                risk_config=cfg,
                snapshot=snap,
                momentum_table=momentum_table,
                use_cached_tail=use_cached_tail,
            )
            return (p_start, p_end, regime, panel)

        t0 = time.perf_counter()
        period_panel_map: Dict[Tuple[str, str], Tuple[str, pd.DataFrame]] = {}
        if workers > 1 and len(periods) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(periods))) as executor:
                futures = {executor.submit(_build_period_panel, p): p for p in periods}
                for future in as_completed(futures):
                    p_start, p_end, regime, panel = future.result()
                    period_panel_map[(p_start, p_end)] = (regime, panel)
        else:
            for period in periods:
                p_start, p_end, regime, panel = _build_period_panel(period)
                period_panel_map[(p_start, p_end)] = (regime, panel)
        stage_times["asset_panel_build"] = time.perf_counter() - t0

        progress = tqdm(
            total=len(periods) * 2,
            desc=f"Research {strategy}",
            unit="period",
            disable=not show_progress,
        )

        for track in ("ungated", "gated"):
            equity = 1.0
            peak = 1.0
            for period_start, period_end in periods:
                regime, base_panel = period_panel_map.get(
                    (period_start, period_end), ("NEUTRAL", pd.DataFrame())
                )

                if not base_panel.empty:
                    asset_panel = base_panel.copy()
                    asset_panel["track"] = track
                    if track == "gated":
                        asset_panel["SelectedForPortfolio"] = (
                            asset_panel["RV_Gate_Count"] == 0
                        )
                    else:
                        asset_panel["SelectedForPortfolio"] = True
                    asset_frames.append(asset_panel)

                    selected = asset_panel[asset_panel["SelectedForPortfolio"] == True]
                    portfolio_return = (
                        selected["return"].mean() if not selected.empty else np.nan
                    )
                    selected_count = int(selected["symbol"].nunique())
                    candidate_count = int(asset_panel["symbol"].nunique())
                    gate_trip_rate = float((asset_panel["RV_Gate_Count"] > 0).mean())
                else:
                    portfolio_return = np.nan
                    selected_count = 0
                    candidate_count = 0
                    gate_trip_rate = np.nan

                benchmark_return = benchmark_return_map.get(
                    (period_start, period_end), np.nan
                )
                alpha = (
                    portfolio_return - benchmark_return
                    if pd.notna(portfolio_return) and pd.notna(benchmark_return)
                    else np.nan
                )

                portfolio_rows.append(
                    {
                        "track": track,
                        "strategy": strategy,
                        "period_start": period_start,
                        "period_end": period_end,
                        "regime": regime,
                        "candidate_count": candidate_count,
                        "selected_count": selected_count,
                        "portfolio_return": portfolio_return,
                        "benchmark_return": benchmark_return,
                        "alpha": alpha,
                        "gate_trip_rate": gate_trip_rate,
                    }
                )

                if pd.notna(portfolio_return):
                    equity *= 1.0 + float(portfolio_return)
                peak = max(peak, equity)
                drawdown = (equity / peak) - 1 if peak > 0 else 0.0
                equity_rows.append(
                    {
                        "track": track,
                        "strategy": strategy,
                        "period_start": period_start,
                        "period_end": period_end,
                        "regime": regime,
                        "equity": equity,
                        "drawdown": drawdown,
                        "portfolio_return": portfolio_return,
                    }
                )

                if verbose:
                    print(
                        f"[research] {track} {period_start}->{period_end}: return={portfolio_return:+.2%}"
                        if pd.notna(portfolio_return)
                        else f"[research] {track} {period_start}->{period_end}: return=N/A"
                    )
                progress.set_postfix({"track": track, "regime": regime})
                progress.update(1)

        progress.close()

        asset_panel_df = (
            pd.concat(asset_frames, ignore_index=True) if asset_frames else pd.DataFrame()
        )
        portfolio_panel_df = pd.DataFrame(portfolio_rows)
        equity_panel_df = pd.DataFrame(equity_rows)

        outputs = {
            "asset_panel": asset_panel_df,
            "portfolio_panel": portfolio_panel_df,
            "equity_panel": equity_panel_df,
        }
        self._last_research_outputs = outputs

        stage_times["total"] = time.perf_counter() - run_start_ts
        if verbose:
            stage_msg = ", ".join(f"{k}={v:.2f}s" for k, v in stage_times.items())
            print(f"[research] timings: {stage_msg}")

        return outputs

    def query_signal_stability(
        self, asset_panel: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Aggregate signal performance across regimes."""
        if asset_panel is None:
            asset_panel = (
                self._last_research_outputs.get("asset_panel")
                if self._last_research_outputs
                else None
            )
        if asset_panel is None or asset_panel.empty:
            return pd.DataFrame()

        panel = asset_panel.dropna(subset=["return"]).copy()
        if panel.empty:
            return pd.DataFrame()

        summary = (
            panel.groupby(["track", "strategy", "regime", "Investment Signal"])
            .agg(
                observations=("symbol", "count"),
                mean_return=("return", "mean"),
                median_return=("return", "median"),
                std_return=("return", "std"),
                hit_rate=("return", lambda x: float((x > 0).mean())),
                gate_trip_rate=("RV_Gate_Count", lambda x: float((x > 0).mean())),
            )
            .reset_index()
            .sort_values(["track", "strategy", "regime", "observations"], ascending=[True, True, True, False])
        )
        return summary

    def query_gate_trip_rates(
        self, asset_panel: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Report gate trip counts/rates by track, strategy, and regime."""
        if asset_panel is None:
            asset_panel = (
                self._last_research_outputs.get("asset_panel")
                if self._last_research_outputs
                else None
            )
        if asset_panel is None or asset_panel.empty:
            return pd.DataFrame()

        gate_cols = [
            "RV_Gate_Distress",
            "RV_Gate_Tail",
            "RV_Gate_Momentum",
            "RV_Gate_Valuation",
        ]
        rows: List[Dict[str, Any]] = []
        grouped = asset_panel.groupby(["track", "strategy", "regime"])
        for (track, strategy, regime), grp in grouped:
            total = len(grp)
            if total == 0:
                continue
            for gate_col in gate_cols:
                trips = int(grp[gate_col].fillna(False).astype(bool).sum())
                rows.append(
                    {
                        "track": track,
                        "strategy": strategy,
                        "regime": regime,
                        "gate": gate_col.replace("RV_Gate_", "").lower(),
                        "trips": trips,
                        "candidates": total,
                        "trip_rate": trips / total,
                    }
                )
        return pd.DataFrame(rows)

    def query_drawdown_failures(
        self,
        equity_panel: Optional[pd.DataFrame] = None,
        threshold: float = -0.15,
    ) -> pd.DataFrame:
        """Return drawdown episodes where drawdown breaches the threshold."""
        if equity_panel is None:
            equity_panel = (
                self._last_research_outputs.get("equity_panel")
                if self._last_research_outputs
                else None
            )
        if equity_panel is None or equity_panel.empty:
            return pd.DataFrame()

        events: List[Dict[str, Any]] = []
        for (track, strategy), grp in equity_panel.groupby(["track", "strategy"]):
            grp = grp.sort_values("period_end").reset_index(drop=True)
            in_event = False
            start_idx = -1
            trough_idx = -1
            trough_dd = 0.0

            for idx_i in range(len(grp)):
                row = grp.iloc[idx_i]
                dd = float(row["drawdown"]) if pd.notna(row["drawdown"]) else 0.0
                if not in_event and dd <= threshold:
                    in_event = True
                    start_idx = idx_i
                    trough_idx = idx_i
                    trough_dd = dd
                if in_event and dd < trough_dd:
                    trough_dd = dd
                    trough_idx = idx_i
                if in_event and dd > threshold:
                    start_row = grp.iloc[start_idx]
                    trough_row = grp.iloc[trough_idx]
                    recovery_date = row["period_end"]
                    end_date = (
                        grp.iloc[idx_i - 1]["period_end"]
                        if idx_i > 0
                        else row["period_end"]
                    )
                    events.append(
                        {
                            "track": track,
                            "strategy": strategy,
                            "start_date": start_row["period_start"],
                            "regime_at_start": start_row["regime"],
                            "trough_date": trough_row["period_end"],
                            "min_drawdown": trough_dd,
                            "end_date": end_date,
                            "recovery_date": recovery_date,
                            "duration_periods": int(idx_i - start_idx),
                        }
                    )
                    in_event = False

            if in_event:
                start_row = grp.iloc[start_idx]
                trough_row = grp.iloc[trough_idx]
                events.append(
                    {
                        "track": track,
                        "strategy": strategy,
                        "start_date": start_row["period_start"],
                        "regime_at_start": start_row["regime"],
                        "trough_date": trough_row["period_end"],
                        "min_drawdown": trough_dd,
                        "end_date": grp.iloc[-1]["period_end"],
                        "recovery_date": None,
                        "duration_periods": int(len(grp) - start_idx),
                    }
                )

        return pd.DataFrame(events)

    def export_research_results(
        self, results: Dict[str, pd.DataFrame], output_dir: str
    ) -> Dict[str, str]:
        """Export research result panels to CSV files."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        exported: Dict[str, str] = {}
        for name, df in results.items():
            file_path = out_dir / f"{name}.csv"
            df.to_csv(file_path, index=False)
            exported[name] = str(file_path)
        return exported


def _get_period_benchmark_return(
    bt: VectorBacktester, symbol: str, start_date: str, end_date: str
) -> float:
    entry = bt.get_price(symbol, start_date)
    exit_ = bt.get_price(symbol, end_date)
    if pd.notna(entry) and pd.notna(exit_) and entry > 0:
        return float((exit_ - entry) / entry)
    return np.nan


def _format_pct_or_na(value: float, width: int = 10) -> str:
    if pd.isna(value):
        return f"{'N/A':>{width}}"
    return f"{value * 100:>{width}.2f}%"


def _format_pick_count(value: Optional[int], width: int = 5) -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{int(value):>{width}d}"


def _nanmean(values: List[float]) -> float:
    valid = [float(v) for v in values if pd.notna(v)]
    if not valid:
        return np.nan
    return float(np.mean(valid))


def _format_tournament_row(
    name: str,
    returns: List[float],
    picks: List[Optional[int]],
    avg_return: Optional[float] = None,
) -> str:
    avg_val = _nanmean(returns) if avg_return is None else avg_return
    return (
        f"{name:<30} | {_format_pct_or_na(returns[0])} | {_format_pick_count(picks[0])} | "
        f"{_format_pct_or_na(returns[1])} | {_format_pick_count(picks[1])} | "
        f"{_format_pct_or_na(returns[2])} | {_format_pick_count(picks[2])} | "
        f"{_format_pct_or_na(avg_val)}"
    )


def _investigate_magic_formula_weakness(
    bt: VectorBacktester,
    start_date: str,
    end_date: str,
    top_n: int = 10,
) -> None:
    """Print focused diagnostics for Magic Formula in a specific period."""
    print(f"\n{'='*110}")
    print(f"  MAGIC FORMULA DIAGNOSTIC ({start_date} -> {end_date})")
    print(f"{'='*110}")

    snapshot = bt.get_valid_snapshot(start_date)
    if snapshot.empty:
        print("  No point-in-time snapshot available for the diagnostic window.")
        return

    initial_count = int(len(snapshot))
    universe_df = bt.filter_universe(snapshot.copy(), start_date, min_mkt_cap_mm=250)
    universe_count = int(len(universe_df))

    df = universe_df.copy()
    if "MarketCap" not in df.columns:
        df["MarketCap"] = 0.0

    market_cap_positive_count = int((df["MarketCap"] > 0).sum())

    if "EnterpriseValue" not in df.columns:
        ev_cols = ["TotalDebt", "CashAndCashEquivalents", "MarketCap"]
        if all(c in df.columns for c in ev_cols):
            df = df[df["MarketCap"] > 0].copy()
            df["EnterpriseValue"] = (
                (df["MarketCap"] * 1_000_000)
                + df["TotalDebt"]
                - df["CashAndCashEquivalents"]
            )

    required = ["EBIT", "InvestedCapital", "EnterpriseValue"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        print(
            f"  Missing required columns for Magic Formula in this window: {missing_required}"
        )
        print(
            f"  Attrition so far: snapshot={initial_count}, post_universe_filter={universe_count}"
        )
        return

    eligible = df.dropna(subset=required).copy()
    post_dropna_count = int(len(eligible))

    eligible = eligible[eligible["InvestedCapital"] > 0].copy()
    post_invested_capital_count = int(len(eligible))

    eligible = eligible[eligible["EnterpriseValue"] > 0].copy()
    post_enterprise_value_count = int(len(eligible))

    if eligible.empty:
        print("  No eligible names remained after Magic Formula filters.")
        print(
            "  Counts: "
            f"snapshot={initial_count}, "
            f"post_universe_filter={universe_count}, "
            f"marketcap_gt0={market_cap_positive_count}, "
            f"post_required_non_null={post_dropna_count}, "
            f"post_invested_capital_filter={post_invested_capital_count}, "
            f"post_enterprise_value_filter={post_enterprise_value_count}"
        )
        return

    eligible["ROIC"] = eligible["EBIT"] / eligible["InvestedCapital"]
    eligible["EarningsYield"] = eligible["EBIT"] / eligible["EnterpriseValue"]
    eligible["Quality_Rank"] = robust_score(eligible["ROIC"], higher_is_better=True)
    eligible["Value_Rank"] = robust_score(
        eligible["EarningsYield"], higher_is_better=True
    )
    eligible["Magic_Score"] = eligible["Quality_Rank"] + eligible["Value_Rank"]

    picks = eligible.nlargest(top_n, ["Magic_Score", "MarketCap"]).index.tolist()
    pick_returns = bt.calculate_portfolio_return(picks, start_date, end_date)
    pick_results = cast(pd.DataFrame, pick_returns["results_df"])
    priced_count = int(len(pick_results))
    if len(picks) == 0:
        status = "no_selections"
    elif priced_count == 0:
        status = "no_valid_prices"
    else:
        status = "ok"
    portfolio_return = pick_returns["portfolio_return"]
    qqq_return = _get_period_benchmark_return(bt, "QQQ", start_date, end_date)
    spy_return = _get_period_benchmark_return(bt, "SPY", start_date, end_date)

    print("  Universe attrition:")
    print(
        "    "
        f"snapshot={initial_count}, "
        f"post_universe_filter={universe_count}, "
        f"marketcap_gt0={market_cap_positive_count}, "
        f"post_required_non_null={post_dropna_count}, "
        f"post_invested_capital_filter={post_invested_capital_count}, "
        f"post_enterprise_value_filter={post_enterprise_value_count}"
    )
    print(
        "  Selection summary: "
        f"picked={len(picks)} | priced={priced_count} | status={status}"
    )
    print(f"  Picks: {picks if picks else 'None'}")
    print(
        "  Returns: "
        f"Magic={_format_pct_or_na(portfolio_return, width=8).strip()} | "
        f"QQQ={_format_pct_or_na(qqq_return, width=8).strip()} | "
        f"SPY={_format_pct_or_na(spy_return, width=8).strip()}"
    )
    if pd.notna(portfolio_return) and pd.notna(qqq_return):
        print(f"  Alpha vs QQQ: {(portfolio_return - qqq_return) * 100:+.2f}%")
    if pd.notna(portfolio_return) and pd.notna(spy_return):
        print(f"  Alpha vs SPY: {(portfolio_return - spy_return) * 100:+.2f}%")

    if not pick_results.empty:
        ranked = pick_results.sort_values("return")
        print("  Per-pick returns (worst -> best):")
        for _, row in ranked.iterrows():
            symbol = row.get("symbol", "")
            ret = row.get("return", np.nan)
            print(f"    {symbol:<8} {_format_pct_or_na(ret, width=8).strip()}")
    else:
        print("  No valid per-pick price pairs available for diagnostic ranking.")


def run_strategy_tournament(show_progress: bool = True, verbose_combo: bool = False):
    """Run a tournament between pure strategies and hybrid strategies."""
    bt = VectorBacktester(reporting_lag_days=90)
    bt.load_data()

    # Define the 3 distinct market periods
    periods = [
        ("2022-01-03", "2023-01-03"),  # Bear Market
        ("2023-01-03", "2024-01-02"),  # Recovery
        ("2024-01-02", "2024-12-20"),  # Bull Market
    ]

    # Define the Contenders
    contenders = [
        # --- PURE STRATEGIES (Top 10) ---
        {"name": "Pure Magic Formula", "strategies": ["magic_formula"], "n": 10},
        {"name": "Pure Moat", "strategies": ["moat"], "n": 10},
        {"name": "Pure Diamonds", "strategies": ["diamonds_in_dirt"], "n": 10},
        {"name": "Pure FCF Yield", "strategies": ["fcf_yield"], "n": 10},
        {"name": "Pure Quality", "strategies": ["quality"], "n": 10},
        {"name": "Pure Cannibal", "strategies": ["cannibal"], "n": 10},
        {
            "name": "Pure Piotroski F-Score",
            "strategies": ["piotroski_f_score"],
            "n": 10,
        },
        {"name": "Pure Rule of 40", "strategies": ["rule_of_40"], "n": 10},
        # --- HYBRID STRATEGIES (5 + 5) ---
        # "The Fortress": High Moats + Undervalued Diamonds
        {
            "name": "Fortress (Moat + Diamond)",
            "strategies": ["moat", "diamonds_in_dirt"],
            "n": 5,
        },
        # "Greenblatt Plus": Magic Formula + High Moat Stability
        {
            "name": "Greenblatt+ (Magic + Moat)",
            "strategies": ["magic_formula", "moat"],
            "n": 5,
        },
        # "Deep Value": Magic Formula (Cheap) + Diamonds (Quality)
        {
            "name": "Deep Value (Magic + Diamond)",
            "strategies": ["magic_formula", "diamonds_in_dirt"],
            "n": 5,
        },
        # "Cash Cow": FCF Yield + Cannibal (Buybacks funded by FCF)
        {
            "name": "Cash Cow (FCF + Cannibal)",
            "strategies": ["fcf_yield", "cannibal"],
            "n": 5,
        },
        # "Growth + Quality": Rule of 40 + Piotroski
        {
            "name": "Growth Quality (R40 + Piotroski)",
            "strategies": ["rule_of_40", "piotroski_f_score"],
            "n": 5,
        },
    ]

    benchmark_symbols = ["QQQ", "SPY"]
    benchmark_rows: List[Tuple[str, List[float]]] = []
    for symbol in benchmark_symbols:
        symbol_returns: List[float] = []
        for start, end in periods:
            symbol_returns.append(_get_period_benchmark_return(bt, symbol, start, end))
        if any(pd.notna(r) for r in symbol_returns):
            benchmark_rows.append((symbol, symbol_returns))

    print(f"\n{'='*110}")
    print(f"  STRATEGY TOURNAMENT")
    print(f"{'='*110}")
    print(
        f"{'STRATEGY':<30} | {'2022 Ret':>10} | {'Picks':>5} | "
        f"{'2023 Ret':>10} | {'Picks':>5} | "
        f"{'2024 Ret':>10} | {'Picks':>5} | {'AVG':>10}"
    )
    print("-" * 110)

    for symbol, symbol_returns in benchmark_rows:
        benchmark_row = _format_tournament_row(
            f"Benchmark {symbol}",
            symbol_returns,
            [None, None, None],
        )
        print(benchmark_row)
    if benchmark_rows:
        print("-" * 110)

    contender_loop = tqdm(
        contenders,
        desc="Tournament",
        unit="strategy",
        disable=not show_progress,
    )

    for contender in contender_loop:
        returns: List[float] = []
        picks_by_period: List[int] = []
        for start, end in periods:
            # We use the combo runner for everything (even pure strategies just pass a list of 1)
            res = bt.run_combo_backtest(
                start,
                end,
                contender["strategies"],
                top_n_per_strat=contender["n"],
                verbose=verbose_combo,
            )
            returns.append(res["return"] if pd.notna(res["return"]) else np.nan)
            picks_by_period.append(int(res.get("picks_count", 0)))

        avg_ret = _nanmean(returns)

        # Print Row
        row_text = _format_tournament_row(
            contender["name"],
            returns,
            [picks_by_period[0], picks_by_period[1], picks_by_period[2]],
            avg_return=avg_ret,
        )
        if show_progress:
            tqdm.write(row_text)
            contender_loop.set_postfix(
                {"last_avg": _format_pct_or_na(avg_ret, width=8).strip()}
            )
        else:
            print(row_text)

    print("-" * 110)

    # Focused diagnostic for the suspicious 2024 Magic Formula result.
    _investigate_magic_formula_weakness(
        bt,
        start_date="2024-01-02",
        end_date="2024-12-20",
        top_n=10,
    )


if __name__ == "__main__":
    run_strategy_tournament()
