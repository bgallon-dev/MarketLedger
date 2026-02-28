"""
Strategy Module - Modular stock screening and portfolio construction.

This module provides a flexible framework for:
- Defining and running stock screening strategies
- Building multi-strategy portfolios
- Backtesting with point-in-time data (avoiding look-ahead bias)

Usage:
------
    from strat.strat import StrategyEngine, Portfolio

    # Quick usage with defaults
    engine = StrategyEngine()
    engine.load_data()
    picks = engine.run_strategy("magic_formula", trade_date="2025-01-01")

    # Build a multi-strategy portfolio
    portfolio = Portfolio(engine)
    buy_list = portfolio.generate_buy_list("2025-01-01")

    # Custom strategy
    engine.register_strategy("custom", MyCustomStrategy())
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Protocol

import database.database as db


# ==============================================================================
# 1. CONFIGURATION & DATA CLASSES
# ==============================================================================


@dataclass
class StrategyConfig:
    """Configuration for a screening strategy."""

    name: str
    description: str
    required_metrics: List[str]
    default_top_n: int = 10
    min_market_cap_mm: float = 250.0


@dataclass
class ScreeningResult:
    """Result from running a strategy screen."""

    strategy_name: str
    trade_date: str
    tickers: List[str]
    scores: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UniverseFilter:
    """Configuration for universe filtering."""

    min_market_cap_mm: float = 250.0
    exclude_warrants: bool = True
    exclude_adrs: bool = False
    custom_filter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None


@dataclass
class MomentumFilter:
    """Configuration for momentum/technical filtering to avoid value traps."""

    require_above_ma: bool = True
    ma_period: int = 200
    min_days_above_ma: int = 1  # Must be above MA for at least N days


# ==============================================================================
# 2. STRATEGY BASE CLASS & IMPLEMENTATIONS
# ==============================================================================


class BaseStrategy(ABC):
    """
    Abstract base class for stock screening strategies.

    To create a custom strategy:
    1. Inherit from BaseStrategy
    2. Define required_metrics
    3. Implement the screen() method
    """

    name: str = "base"
    description: str = "Base strategy"
    required_metrics: List[str] = []

    @abstractmethod
    def screen(self, df: pd.DataFrame, top_n: int = 10) -> List[str]:
        """
        Screen stocks and return top picks.

        Parameters:
        -----------
        df : pd.DataFrame
            Financial snapshot with Ticker as index
        top_n : int
            Number of stocks to select

        Returns:
        --------
        list
            List of selected ticker symbols
        """
        pass

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data by ensuring required columns exist and are numeric."""
        df = df.copy()
        for col in self.required_metrics:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df


class MagicFormulaStrategy(BaseStrategy):
    """
    Joel Greenblatt's Magic Formula: Combines Quality (ROIC) + Value (Earnings Yield).

    Ranks stocks by:
    1. Return on Invested Capital (ROIC) = Operating Income / Invested Capital
    2. Earnings Yield = Operating Income / Enterprise Value

    Stocks are ranked on both metrics, and combined rank determines selection.
    """

    name = "magic_formula"
    description = "Greenblatt Magic Formula (ROIC + Earnings Yield)"
    required_metrics = [
        "OperatingIncome",
        "StockholdersEquity",
        "TotalDebt",
        "MarketCap",
    ]

    def screen(self, df: pd.DataFrame, top_n: int = 10) -> List[str]:
        df = self.prepare_data(df)

        if "MarketCap" not in df.columns or df["MarketCap"].isna().all():
            return []

        # ROIC = Operating Income / Invested Capital
        df["InvestedCapital"] = df["StockholdersEquity"] + df["TotalDebt"]
        df["ROIC"] = df["OperatingIncome"] / df["InvestedCapital"].replace(0, np.nan)

        # Earnings Yield = Operating Income / Enterprise Value
        df["EV"] = (df["MarketCap"] * 1e6) + df["TotalDebt"]
        df["Earnings_Yield"] = df["OperatingIncome"] / df["EV"].replace(0, np.nan)

        # Combined ranking (lower is better)
        df["Rank_ROIC"] = df["ROIC"].rank(ascending=False)
        df["Rank_EY"] = df["Earnings_Yield"].rank(ascending=False)
        df["Combined_Rank"] = df["Rank_ROIC"] + df["Rank_EY"]

        return df.nsmallest(top_n, "Combined_Rank").index.tolist()


class MoatStrategy(BaseStrategy):
    """
    Wide Moat Strategy: Identifies companies with durable competitive advantages.

    Filters for companies with Gross Margin > 60%, indicating strong pricing power.
    Ranks by gross margin to find the most defensible businesses.
    """

    name = "moat"
    description = "Wide Moat (High Gross Margin > 60%)"
    required_metrics = ["GrossProfit", "TotalRevenue"]

    def __init__(self, min_gross_margin: float = 0.60):
        self.min_gross_margin = min_gross_margin

    def screen(self, df: pd.DataFrame, top_n: int = 10) -> List[str]:
        df = self.prepare_data(df)

        df["Gross_Margin"] = df["GrossProfit"] / df["TotalRevenue"].replace(0, np.nan)

        # Filter for high gross margin
        passed = df[df["Gross_Margin"] > self.min_gross_margin].copy()

        return passed.nlargest(top_n, "Gross_Margin").index.tolist()


class FCFYieldStrategy(BaseStrategy):
    """
    Free Cash Flow Yield Strategy: Finds cash-generating machines trading cheaply.

    FCF Yield = Free Cash Flow / Market Cap
    Higher yield indicates more cash generation per dollar of market value.
    """

    name = "fcf_yield"
    description = "Free Cash Flow Yield (Cash Cows)"
    required_metrics = ["FreeCashFlow", "MarketCap"]

    def __init__(self, min_fcf: float = 0):
        self.min_fcf = min_fcf

    def screen(self, df: pd.DataFrame, top_n: int = 10) -> List[str]:
        df = self.prepare_data(df)

        if "MarketCap" not in df.columns or df["MarketCap"].isna().all():
            return []

        df["FCF_Yield"] = df["FreeCashFlow"] / (df["MarketCap"] * 1e6).replace(
            0, np.nan
        )

        # Filter for positive FCF if specified
        if self.min_fcf > 0:
            df = df[df["FreeCashFlow"] > self.min_fcf]

        return df.nlargest(top_n, "FCF_Yield").index.tolist()


class CannibalStrategy(BaseStrategy):
    """
    Cannibal Strategy: Finds companies aggressively buying back shares.

    Criteria:
    1. Share count decreased by > 3% YoY (aggressive buybacks)
    2. Positive Free Cash Flow (buybacks funded by operations, not debt)

    These companies are "eating themselves" - returning capital to shareholders.
    """

    name = "cannibal"
    description = "Share Cannibals (Aggressive Buybacks)"
    required_metrics = ["OrdinarySharesNumber", "FreeCashFlow"]

    def __init__(
        self, min_buyback_pct: float = 0.03, require_positive_fcf: bool = True
    ):
        self.min_buyback_pct = min_buyback_pct
        self.require_positive_fcf = require_positive_fcf

    def screen(self, df: pd.DataFrame, top_n: int = 10) -> List[str]:
        df = self.prepare_data(df)

        # Calculate share change if YoY data available
        if "Share_Change_Pct" not in df.columns:
            if "OrdinarySharesNumber_Prev" in df.columns:
                df["OrdinarySharesNumber_Prev"] = pd.to_numeric(
                    df["OrdinarySharesNumber_Prev"], errors="coerce"
                )
                df["Share_Change_Pct"] = (
                    df["OrdinarySharesNumber"] - df["OrdinarySharesNumber_Prev"]
                ) / df["OrdinarySharesNumber_Prev"].replace(0, np.nan)
            else:
                return []  # Cannot run without YoY data

        # Filter: Significant buybacks + positive FCF
        mask = df["Share_Change_Pct"] < -self.min_buyback_pct
        if self.require_positive_fcf:
            mask = mask & (df["FreeCashFlow"] > 0)

        passed = df[mask].copy()

        if passed.empty:
            return []

        # Rank by most aggressive buybacks (most negative = best)
        return passed.nsmallest(top_n, "Share_Change_Pct").index.tolist()


class QualityStrategy(BaseStrategy):
    """
    Quality Strategy: Combines multiple quality metrics.

    Looks for:
    - High ROE (Return on Equity)
    - High Operating Margin
    - Positive and growing Free Cash Flow
    """

    name = "quality"
    description = "Quality Composite (ROE + Margins + FCF)"
    required_metrics = [
        "NetIncome",
        "StockholdersEquity",
        "OperatingIncome",
        "TotalRevenue",
        "FreeCashFlow",
    ]

    def screen(self, df: pd.DataFrame, top_n: int = 10) -> List[str]:
        df = self.prepare_data(df)

        # ROE
        df["ROE"] = df["NetIncome"] / df["StockholdersEquity"].replace(0, np.nan)

        # Operating Margin
        df["Op_Margin"] = df["OperatingIncome"] / df["TotalRevenue"].replace(0, np.nan)

        # FCF Margin
        df["FCF_Margin"] = df["FreeCashFlow"] / df["TotalRevenue"].replace(0, np.nan)

        # Simple composite: rank each metric, sum ranks
        df["Rank_ROE"] = df["ROE"].rank(ascending=False)
        df["Rank_OpM"] = df["Op_Margin"].rank(ascending=False)
        df["Rank_FCF"] = df["FCF_Margin"].rank(ascending=False)
        df["Quality_Rank"] = df["Rank_ROE"] + df["Rank_OpM"] + df["Rank_FCF"]

        return df.nsmallest(top_n, "Quality_Rank").index.tolist()


# ==============================================================================
# 3. STRATEGY REGISTRY
# ==============================================================================


class StrategyRegistry:
    """
    Registry for managing available strategies.

    Provides a central place to register, retrieve, and list strategies.
    """

    def __init__(self):
        self._strategies: Dict[str, BaseStrategy] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register built-in strategies."""
        self.register(MagicFormulaStrategy())
        self.register(MoatStrategy())
        self.register(FCFYieldStrategy())
        self.register(CannibalStrategy())
        self.register(QualityStrategy())

    def register(self, strategy: BaseStrategy) -> None:
        """Register a strategy instance."""
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> Optional[BaseStrategy]:
        """Get a strategy by name."""
        return self._strategies.get(name)

    def list_strategies(self) -> List[str]:
        """List all registered strategy names."""
        return list(self._strategies.keys())

    def get_info(self) -> pd.DataFrame:
        """Get info about all registered strategies."""
        data = []
        for name, strat in self._strategies.items():
            data.append(
                {
                    "Name": name,
                    "Description": strat.description,
                    "Required Metrics": ", ".join(strat.required_metrics[:3]) + "...",
                }
            )
        return pd.DataFrame(data)


# ==============================================================================
# 4. DATA PROVIDER (Abstraction over database)
# ==============================================================================


class DataProvider:
    """
    Abstraction layer for financial data access.

    Handles:
    - Loading ticker lists
    - Fetching financial snapshots with point-in-time logic
    - Price lookups with caching
    """

    # Default metrics to fetch from each financial table
    DEFAULT_METRICS = {
        "income_statement": [
            "NetIncome",
            "TotalRevenue",
            "GrossProfit",
            "OperatingIncome",
            "EBIT",
        ],
        "balance_sheet": [
            "TotalDebt",
            "StockholdersEquity",
            "OrdinarySharesNumber",
            "TotalAssets",
            "CurrentAssets",
            "CurrentLiabilities",
        ],
        "cash_flow": ["FreeCashFlow", "OperatingCashFlow"],
    }

    # Metrics that need YoY comparison
    YOY_METRICS = ["OrdinarySharesNumber", "TotalRevenue"]

    def __init__(
        self,
        reporting_lag_days: int = 90,
        exchange: Optional[str] = None,
        verbose: bool = True,
    ):
        self.reporting_lag_days = reporting_lag_days
        self.exchange = exchange
        self.verbose = verbose
        self.tickers_df: Optional[pd.DataFrame] = None
        self._price_cache: Dict[str, pd.Series] = {}

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def load_tickers(self) -> pd.DataFrame:
        """Load ticker list from database."""
        self._log("Loading tickers from database...")
        try:
            self.tickers_df = db.get_all_tickers(exchange=self.exchange)
            self._log(f"Loaded {len(self.tickers_df)} tickers.")
            return self.tickers_df
        except Exception as e:
            self._log(f"Error loading tickers: {e}")
            return pd.DataFrame()

    def get_snapshot(
        self, trade_date: str, metrics: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Get point-in-time financial snapshot.

        Parameters:
        -----------
        trade_date : str
            The date to get snapshot for (YYYY-MM-DD)
        metrics : dict, optional
            Custom metrics to fetch. Defaults to DEFAULT_METRICS.

        Returns:
        --------
        pd.DataFrame
            Financial snapshot with Ticker as index
        """
        trade_dt = pd.to_datetime(trade_date)
        cutoff_dt = trade_dt - timedelta(days=self.reporting_lag_days)
        cutoff_str = cutoff_dt.strftime("%Y-%m-%d")

        self._log(f"Building snapshot (trade: {trade_date}, cutoff: {cutoff_str})...")

        if self.tickers_df is None or self.tickers_df.empty:
            self._log("No tickers loaded. Call load_tickers() first.")
            return pd.DataFrame()

        metrics = metrics or self.DEFAULT_METRICS
        all_data = []

        for _, row in self.tickers_df.iterrows():
            symbol = row["symbol"]
            ticker_data = {"Ticker": symbol}

            for table_name, metric_list in metrics.items():
                try:
                    fin_df = db.get_financial_data(symbol, table_name)
                    if fin_df.empty:
                        continue

                    # Get periods before cutoff
                    valid_periods = [p for p in fin_df.columns if str(p) <= cutoff_str]
                    if not valid_periods:
                        valid_periods = list(fin_df.columns)

                    if valid_periods:
                        valid_periods = sorted(valid_periods, reverse=True)
                        latest = valid_periods[0]
                        prev = valid_periods[1] if len(valid_periods) > 1 else None

                        for metric in metric_list:
                            if metric in fin_df.index:
                                ticker_data[metric] = fin_df.loc[metric, latest]
                                # YoY data for specific metrics
                                if prev and metric in self.YOY_METRICS:
                                    ticker_data[f"{metric}_Prev"] = fin_df.loc[
                                        metric, prev
                                    ]

                except Exception:
                    continue

            if len(ticker_data) > 1:
                all_data.append(ticker_data)

        if not all_data:
            self._log("No financial data found.")
            return pd.DataFrame()

        snapshot = pd.DataFrame(all_data).set_index("Ticker")
        self._log(f"Built snapshot with {len(snapshot)} tickers.")
        return snapshot

    def get_prices(self, tickers: List[str], trade_date: str) -> pd.Series:
        """Get closing prices for multiple tickers."""
        trade_date_str = pd.to_datetime(trade_date).strftime("%Y-%m-%d")
        prices = {}

        for ticker in tickers:
            # Check cache
            if ticker in self._price_cache:
                cached = self._price_cache[ticker]
                if trade_date_str in cached.index:
                    prices[ticker] = cached.loc[trade_date_str]
                    continue

            try:
                start = (pd.to_datetime(trade_date) - timedelta(days=10)).strftime(
                    "%Y-%m-%d"
                )
                hist = db.get_ticker_history(
                    ticker, start_date=start, end_date=trade_date_str
                )

                if not hist.empty:
                    hist = hist.set_index("date")["close"]
                    self._price_cache[ticker] = hist
                    valid = hist[hist.index <= trade_date_str]
                    if not valid.empty:
                        prices[ticker] = valid.iloc[-1]
            except Exception:
                continue

        return pd.Series(prices)

    def get_momentum_check(
        self, tickers: List[str], trade_date: str, ma_period: int = 200
    ) -> pd.DataFrame:
        """
        Check if stocks are trading above their moving average.

        Returns DataFrame with columns: Ticker, Price, MA_200, Above_MA
        """
        trade_dt = pd.to_datetime(trade_date)
        # Need enough history for MA calculation
        # Use ~1.5x calendar days to account for weekends/holidays (5 trading days per 7 calendar days)
        start_dt = trade_dt - timedelta(days=int(ma_period * 1.5) + 50)
        start_str = start_dt.strftime("%Y-%m-%d")
        trade_date_str = trade_dt.strftime("%Y-%m-%d")

        results = []

        for ticker in tickers:
            try:
                hist = db.get_ticker_history(
                    ticker, start_date=start_str, end_date=trade_date_str
                )

                if hist.empty or len(hist) < ma_period:
                    # Not enough data - exclude by default
                    results.append(
                        {
                            "Ticker": ticker,
                            "Price": np.nan,
                            f"MA_{ma_period}": np.nan,
                            "Above_MA": False,
                        }
                    )
                    continue

                hist = hist.sort_values("date")
                hist[f"MA_{ma_period}"] = hist["close"].rolling(window=ma_period).mean()

                # Get latest values
                latest = hist.iloc[-1]
                current_price = latest["close"]
                ma_value = latest[f"MA_{ma_period}"]

                results.append(
                    {
                        "Ticker": ticker,
                        "Price": current_price,
                        f"MA_{ma_period}": ma_value,
                        "Above_MA": (
                            current_price > ma_value if pd.notna(ma_value) else False
                        ),
                    }
                )

            except Exception:
                results.append(
                    {
                        "Ticker": ticker,
                        "Price": np.nan,
                        f"MA_{ma_period}": np.nan,
                        "Above_MA": False,
                    }
                )

        return pd.DataFrame(results).set_index("Ticker")

    def clear_cache(self) -> None:
        """Clear price cache."""
        self._price_cache.clear()


# ==============================================================================
# 5. STRATEGY ENGINE
# ==============================================================================


class StrategyEngine:
    """
    Main engine for running stock screening strategies.

    Combines data loading, universe filtering, and strategy execution.

    Examples:
    ---------
    >>> engine = StrategyEngine()
    >>> engine.load_data()
    >>> picks = engine.run_strategy("magic_formula", trade_date="2025-01-01")
    >>> print(picks)
    """

    def __init__(
        self,
        reporting_lag_days: int = 90,
        exchange: Optional[str] = None,
        universe_filter: Optional[UniverseFilter] = None,
        momentum_filter: Optional[MomentumFilter] = None,
        verbose: bool = True,
    ):
        self.data_provider = DataProvider(
            reporting_lag_days=reporting_lag_days, exchange=exchange, verbose=verbose
        )
        self.registry = StrategyRegistry()
        self.universe_filter = universe_filter or UniverseFilter()
        self.momentum_filter = momentum_filter or MomentumFilter()
        self.verbose = verbose
        self._last_snapshot: Optional[pd.DataFrame] = None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def load_data(self) -> None:
        """Load data from database."""
        self.data_provider.load_tickers()

    def register_strategy(self, name: str, strategy: BaseStrategy) -> None:
        """Register a custom strategy."""
        strategy.name = name
        self.registry.register(strategy)

    def list_strategies(self) -> List[str]:
        """List available strategies."""
        return self.registry.list_strategies()

    def _filter_universe(self, df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        """Apply universe filters."""
        config = self.universe_filter

        # Filter warrants
        if config.exclude_warrants:
            mask = df.index.str.match(r"^[A-Z]{4}[WRU]$", na=False)
            df = df[~mask]

        # Market cap filter
        if config.min_market_cap_mm > 0 and "OrdinarySharesNumber" in df.columns:
            prices = self.data_provider.get_prices(df.index.tolist(), trade_date)
            df = df.copy()
            df["_CurrentPrice"] = prices.reindex(df.index)
            df["MarketCap"] = (
                df["OrdinarySharesNumber"] * df["_CurrentPrice"]
            ) / 1_000_000
            df = df.dropna(subset=["MarketCap"])
            df = df[df["MarketCap"] > config.min_market_cap_mm]

        # Custom filter
        if config.custom_filter:
            df = config.custom_filter(df)

        return df

    def _apply_momentum_filter(self, tickers: List[str], trade_date: str) -> List[str]:
        """
        Filter tickers to only those trading above their moving average.

        This helps avoid "value traps" - stocks that are cheap because they're dying.
        Rule: Only buy if Price > 200-Day Moving Average.
        """
        if not self.momentum_filter.require_above_ma:
            return tickers

        if not tickers:
            return tickers

        self._log(
            f"Applying momentum filter (Price > {self.momentum_filter.ma_period}-day MA)..."
        )

        momentum_data = self.data_provider.get_momentum_check(
            tickers, trade_date, self.momentum_filter.ma_period
        )

        # Filter to stocks above MA
        passed = momentum_data[momentum_data["Above_MA"] == True]
        filtered_tickers = passed.index.tolist()

        rejected_count = len(tickers) - len(filtered_tickers)
        if rejected_count > 0:
            self._log(
                f"  Momentum filter removed {rejected_count} stocks below {self.momentum_filter.ma_period}-day MA"
            )

        return filtered_tickers

    def run_strategy(
        self,
        strategy_name: str,
        trade_date: str,
        top_n: int = 10,
        apply_universe_filter: bool = True,
        apply_momentum_filter: bool = True,
        capture_diagnostics: bool = False,
    ) -> ScreeningResult:
        """
        Run a single strategy.

        Parameters:
        -----------
        strategy_name : str
            Name of registered strategy
        trade_date : str
            Date to run screen (YYYY-MM-DD)
        top_n : int
            Number of stocks to select
        apply_universe_filter : bool
            Whether to apply universe filters
        apply_momentum_filter : bool
            Whether to apply momentum filter (Price > 200-day MA)
        capture_diagnostics : bool
            Whether to include pre-momentum candidates and rejects in metadata

        Returns:
        --------
        ScreeningResult
            Result containing selected tickers
        """
        strategy = self.registry.get(strategy_name)
        if strategy is None:
            self._log(f"Unknown strategy: {strategy_name}")
            self._log(f"Available: {self.list_strategies()}")
            return ScreeningResult(
                strategy_name,
                trade_date,
                [],
                metadata={
                    "pre_momentum_candidates": [],
                    "momentum_rejected": [],
                    "momentum_diagnostics": [],
                },
            )

        # Get snapshot
        snapshot = self.data_provider.get_snapshot(trade_date)
        if snapshot.empty:
            return ScreeningResult(strategy_name, trade_date, [])

        # Apply universe filter
        if apply_universe_filter:
            snapshot = self._filter_universe(snapshot, trade_date)

        if snapshot.empty:
            return ScreeningResult(strategy_name, trade_date, [])

        self._last_snapshot = snapshot

        # Run strategy to get candidates (request more if momentum filter will be applied)
        tickers = strategy.screen(
            snapshot, top_n * 2 if apply_momentum_filter else top_n
        )
        pre_momentum_candidates = list(tickers)
        momentum_rejected: List[str] = []
        momentum_diagnostics: List[Dict[str, Any]] = []

        # Apply momentum filter (avoid value traps)
        if apply_momentum_filter and tickers:
            if capture_diagnostics:
                momentum_data = self.data_provider.get_momentum_check(
                    tickers, trade_date, self.momentum_filter.ma_period
                )
                passed = momentum_data[momentum_data["Above_MA"] == True]
                tickers = passed.index.tolist()
                rejected = momentum_data[momentum_data["Above_MA"] != True]
                momentum_rejected = rejected.index.tolist()
                diagnostics_records = (
                    momentum_data.reset_index()
                    .rename(
                        columns={
                            f"MA_{self.momentum_filter.ma_period}": "MA_Value",
                        }
                    )
                    .to_dict("records")
                )
                momentum_diagnostics = [
                    {str(key): value for key, value in record.items()}
                    for record in diagnostics_records
                ]
            else:
                tickers = self._apply_momentum_filter(tickers, trade_date)
            tickers = tickers[:top_n]  # Trim to requested count

        return ScreeningResult(
            strategy_name=strategy_name,
            trade_date=trade_date,
            tickers=tickers,
            metadata={
                "universe_size": len(snapshot),
                "momentum_filter_applied": apply_momentum_filter,
                "pre_momentum_candidates": pre_momentum_candidates,
                "momentum_rejected": momentum_rejected,
                "momentum_diagnostics": momentum_diagnostics,
            },
        )

    def run_multiple(
        self,
        strategy_names: List[str],
        trade_date: str,
        top_n_per_strategy: int = 10,
        capture_diagnostics: bool = False,
    ) -> Dict[str, ScreeningResult]:
        """Run multiple strategies and return all results."""
        results = {}
        for name in strategy_names:
            results[name] = self.run_strategy(
                name,
                trade_date,
                top_n_per_strategy,
                capture_diagnostics=capture_diagnostics,
            )
        return results


# ==============================================================================
# 6. PORTFOLIO BUILDER
# ==============================================================================


class Portfolio:
    """
    Builds portfolios from multiple strategy outputs.

    Combines results from different strategies into a consolidated buy list.

    Examples:
    ---------
    >>> portfolio = Portfolio(engine)
    >>> buy_list = portfolio.generate_buy_list("2025-01-01")
    >>> portfolio.save("my_picks.csv")
    """

    # Default strategy combinations
    DEFAULT_COMBOS = {
        "Greenblatt+": ["magic_formula", "moat"],
        "Cash Cow": ["fcf_yield", "cannibal"],
    }

    def __init__(self, engine: StrategyEngine):
        self.engine = engine
        self.results: Dict[str, ScreeningResult] = {}
        self.buy_list: Optional[pd.DataFrame] = None

    def generate_buy_list(
        self,
        trade_date: str,
        combos: Optional[Dict[str, List[str]]] = None,
        top_n_per_strategy: int = 10,
        capture_diagnostics: bool = False,
    ) -> pd.DataFrame:
        """
        Generate buy list from strategy combinations.

        Parameters:
        -----------
        trade_date : str
            Trade date (YYYY-MM-DD)
        combos : dict, optional
            Strategy combinations. Keys are combo names, values are strategy lists.
        top_n_per_strategy : int
            Picks per strategy

        Returns:
        --------
        pd.DataFrame
            Consolidated buy list with Ticker and Strategy columns
        """
        combos = combos or self.DEFAULT_COMBOS
        print(f"Generating Buy List for {trade_date}...")

        all_picks = []

        for combo_name, strategies in combos.items():
            for strat_name in strategies:
                result = self.engine.run_strategy(
                    strat_name,
                    trade_date,
                    top_n_per_strategy,
                    capture_diagnostics=capture_diagnostics,
                )
                self.results[strat_name] = result

                label = f"{combo_name} ({strat_name})"
                for ticker in result.tickers:
                    all_picks.append({"Ticker": ticker, "Strategy": label})

        if not all_picks:
            print("No stocks found matching criteria.")
            self.buy_list = pd.DataFrame()
            return self.buy_list

        # Consolidate and dedupe
        df = pd.DataFrame(all_picks)
        df = (
            df.groupby("Ticker")["Strategy"]
            .apply(lambda x: " + ".join(sorted(set(x))))
            .reset_index()
        )

        self.buy_list = df
        print(f"\nFound {len(df)} unique stocks.")
        return df

    def save(self, filename: str = "final_buy_list.csv") -> str:
        """Save buy list to CSV."""
        if self.buy_list is None or self.buy_list.empty:
            print("No buy list to save. Run generate_buy_list() first.")
            return ""

        self.buy_list.to_csv(filename, index=False)
        print(f"Saved buy list to {filename}")
        return filename

    def summary(self) -> None:
        """Print summary of current portfolio."""
        if self.buy_list is None or self.buy_list.empty:
            print("No buy list generated yet.")
            return

        print("\n" + "=" * 60)
        print("  PORTFOLIO SUMMARY")
        print("=" * 60)
        print(f"  Total Picks: {len(self.buy_list)}")
        print("\n  By Strategy:")
        for _, row in self.buy_list.iterrows():
            print(f"    {row['Ticker']:<8} {row['Strategy']}")


# ==============================================================================
# 7. LEGACY COMPATIBILITY WRAPPER
# ==============================================================================


class VectorBacktester:
    """
    Legacy wrapper for backward compatibility.

    Use StrategyEngine and Portfolio for new code.
    """

    def __init__(self, reporting_lag_days: int = 90, exchange: Optional[str] = None):
        self.engine = StrategyEngine(
            reporting_lag_days=reporting_lag_days, exchange=exchange
        )
        self.portfolio = Portfolio(self.engine)
        # Expose for legacy access
        self.reporting_lag_days = reporting_lag_days
        self.exchange = exchange
        self.tickers_df = None
        self.price_cache = {}

    def load_data(self) -> None:
        """Load data from database."""
        self.engine.load_data()
        self.tickers_df = self.engine.data_provider.tickers_df

    def get_valid_snapshot(self, trade_date_str: str) -> pd.DataFrame:
        """Get financial snapshot."""
        return self.engine.data_provider.get_snapshot(trade_date_str)

    def get_prices_bulk(self, tickers: List[str], trade_date: str) -> pd.Series:
        """Get prices for multiple tickers."""
        return self.engine.data_provider.get_prices(tickers, trade_date)

    def filter_universe(
        self, df: pd.DataFrame, trade_date: str, min_mkt_cap_mm: int = 250
    ) -> pd.DataFrame:
        """Filter universe."""
        self.engine.universe_filter.min_market_cap_mm = min_mkt_cap_mm
        return self.engine._filter_universe(df, trade_date)

    def enrich_with_yoy_data(
        self, current_df: pd.DataFrame, trade_date: str
    ) -> pd.DataFrame:
        """YoY data is included in snapshot via _Prev columns."""
        return current_df

    def apply_strategy(
        self,
        snapshot: pd.DataFrame,
        strategy: str,
        top_n: int = 10,
        trade_date: Optional[str] = None,
    ) -> List[str]:
        """Apply a strategy to snapshot."""
        if snapshot.empty:
            return []

        if trade_date:
            snapshot = self.filter_universe(snapshot, trade_date)

        if snapshot.empty:
            return []

        strat = self.engine.registry.get(strategy)
        if strat is None:
            return []

        return strat.screen(snapshot, top_n)

    def run_multi_strategy(self, trade_date: str) -> pd.DataFrame:
        """Run multi-strategy portfolio generation."""
        return self.portfolio.generate_buy_list(trade_date)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STRATEGY ENGINE DEMO")
    print("=" * 60)

    # Initialize
    engine = StrategyEngine(verbose=True)
    engine.load_data()

    # Show available strategies
    print("\nAvailable Strategies:")
    for name in engine.list_strategies():
        strat = engine.registry.get(name)
        if strat:
            print(f"  - {name}: {strat.description}")

    # Set trade date
    trade_date = "2025-12-24"

    # Build portfolio
    portfolio = Portfolio(engine)
    buy_list = portfolio.generate_buy_list(trade_date)

    # Save and display
    if not buy_list.empty:
        portfolio.save("final_buy_list.csv")
        portfolio.summary()

        print("\n--- TOP PICKS ---")
        print(buy_list.to_string(index=False))
