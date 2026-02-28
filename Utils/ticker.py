"""
Ticker Management Module - Fetches and manages stock ticker lists from various sources.

This module provides a modular, reusable interface for:
- Fetching ticker lists from S&P 500, NASDAQ, NYSE, and AMEX exchanges
- Caching tickers locally for offline use
- Filtering and cleaning ticker symbols
- Loading custom ticker lists

Usage:
------
    from Utils.ticker import TickerManager

    # Initialize with default settings
    tm = TickerManager()

    # Fetch and cache S&P 500 tickers
    sp500 = tm.get_sp500()

    # Get all US tickers (combines NASDAQ, NYSE, AMEX)
    all_us = tm.get_all_us()

    # Load from cached file
    cached = tm.load("sp500")

    # Custom directory for caching
    tm_custom = TickerManager(cache_dir="/path/to/cache")
"""

import pandas as pd
import os
from typing import List, Optional, Set, Callable
from dataclasses import dataclass, field


@dataclass
class TickerSource:
    """Configuration for a ticker data source."""

    name: str
    url: str
    parser: str = "csv"  # 'csv' or 'txt'
    symbol_column: Optional[str] = None  # Column name for CSV files


class TickerManager:
    """
    Manages stock ticker lists from various sources with caching support.

    Attributes:
    -----------
    cache_dir : str
        Directory for storing cached ticker files
    verbose : bool
        Whether to print status messages

    Examples:
    ---------
    >>> tm = TickerManager()
    >>> sp500 = tm.get_sp500()
    >>> nasdaq = tm.get_nasdaq()
    >>> all_tickers = tm.get_all_us()
    """

    # Default data sources
    SOURCES = {
        "sp500": TickerSource(
            name="S&P 500",
            url="https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
            parser="csv",
            symbol_column="Symbol",
        ),
        "nasdaq": TickerSource(
            name="NASDAQ",
            url="https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.txt",
            parser="txt",
        ),
        "nyse": TickerSource(
            name="NYSE",
            url="https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_tickers.txt",
            parser="txt",
        ),
        "amex": TickerSource(
            name="AMEX",
            url="https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/amex/amex_tickers.txt",
            parser="txt",
        ),
    }

    # Characters that yfinance can't handle
    INVALID_CHARS = {"^", "/", "$", "-"}

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        verbose: bool = True,
        auto_cache: bool = True,
    ):
        """
        Initialize the TickerManager.

        Parameters:
        -----------
        cache_dir : str, optional
            Directory to store cached ticker files. Defaults to module directory.
        verbose : bool
            Print status messages (default True)
        auto_cache : bool
            Automatically save fetched tickers to cache (default True)
        """
        self.cache_dir = cache_dir or os.path.dirname(__file__)
        self.verbose = verbose
        self.auto_cache = auto_cache
        self._cache: dict = {}  # In-memory cache

    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _clean_tickers(
        self, tickers: List, exclude_chars: Optional[Set[str]] = None
    ) -> List[str]:
        """
        Clean and validate ticker symbols.

        Parameters:
        -----------
        tickers : list
            Raw ticker list (may contain NaN, floats, invalid chars)
        exclude_chars : set, optional
            Characters that invalidate a ticker. Defaults to INVALID_CHARS.

        Returns:
        --------
        list
            Cleaned list of valid ticker symbols
        """
        exclude = exclude_chars or self.INVALID_CHARS
        cleaned = []

        for t in tickers:
            # Handle NaN and non-string values
            if pd.isna(t) or not isinstance(t, str):
                continue

            t = t.strip().upper()

            # Skip empty or invalid tickers
            if not t or any(char in t for char in exclude):
                continue

            cleaned.append(t)

        return cleaned

    def _deduplicate(self, tickers: List[str]) -> List[str]:
        """Remove duplicates while preserving order."""
        seen: Set[str] = set()
        unique = []
        for t in tickers:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

    def _fetch_from_source(self, source: TickerSource) -> Optional[List[str]]:
        """
        Fetch tickers from a data source.

        Parameters:
        -----------
        source : TickerSource
            The source configuration

        Returns:
        --------
        list or None
            List of tickers, or None if fetch failed
        """
        try:
            if source.parser == "csv":
                df = pd.read_csv(source.url)
                if source.symbol_column:
                    tickers = df[source.symbol_column].tolist()
                else:
                    tickers = df.iloc[:, 0].tolist()
            else:  # txt
                tickers = pd.read_csv(source.url, header=None)[0].tolist()

            return self._clean_tickers(tickers)

        except Exception as e:
            self._log(f"Error fetching {source.name} tickers: {e}")
            return None

    def _save_to_cache(self, tickers: List[str], name: str) -> str:
        """Save tickers to cache file."""
        filename = f"{name}_tickers.txt"
        filepath = os.path.join(self.cache_dir, filename)

        with open(filepath, "w") as f:
            for ticker in tickers:
                f.write(ticker + "\n")

        return filepath

    def _get_cache_path(self, name: str) -> str:
        """Get the cache file path for a source."""
        return os.path.join(self.cache_dir, f"{name}_tickers.txt")

    # -------------------------------------------------------------------------
    # Public API - Fetch Methods
    # -------------------------------------------------------------------------

    def fetch(
        self, source_name: str, save: Optional[bool] = None
    ) -> Optional[List[str]]:
        """
        Fetch tickers from a named source.

        Parameters:
        -----------
        source_name : str
            Name of the source ('sp500', 'nasdaq', 'nyse', 'amex')
        save : bool, optional
            Override auto_cache setting for this call

        Returns:
        --------
        list or None
            List of tickers, or None if fetch failed
        """
        if source_name not in self.SOURCES:
            self._log(
                f"Unknown source: {source_name}. Available: {list(self.SOURCES.keys())}"
            )
            return None

        source = self.SOURCES[source_name]
        self._log(f"Fetching {source.name} tickers...")

        tickers = self._fetch_from_source(source)

        if tickers:
            self._cache[source_name] = tickers
            should_save = save if save is not None else self.auto_cache

            if should_save:
                filepath = self._save_to_cache(tickers, source_name)
                self._log(f"Saved {len(tickers)} {source.name} tickers to {filepath}")
            else:
                self._log(f"Fetched {len(tickers)} {source.name} tickers")

        return tickers

    def get_sp500(self, save: Optional[bool] = None) -> Optional[List[str]]:
        """Fetch S&P 500 tickers."""
        return self.fetch("sp500", save)

    def get_nasdaq(self, save: Optional[bool] = None) -> Optional[List[str]]:
        """Fetch NASDAQ tickers."""
        return self.fetch("nasdaq", save)

    def get_nyse(self, save: Optional[bool] = None) -> Optional[List[str]]:
        """Fetch NYSE tickers."""
        return self.fetch("nyse", save)

    def get_amex(self, save: Optional[bool] = None) -> Optional[List[str]]:
        """Fetch AMEX tickers."""
        return self.fetch("amex", save)

    def get_all_us(self, save: Optional[bool] = None) -> Optional[List[str]]:
        """
        Fetch all US stock tickers (NASDAQ, NYSE, AMEX combined).

        Returns:
        --------
        list or None
            Deduplicated list of all US tickers
        """
        self._log("Fetching all US tickers...")
        all_tickers = []

        for exchange in ["nasdaq", "nyse", "amex"]:
            tickers = self.fetch(exchange, save=False)  # Don't save individual
            if tickers:
                all_tickers.extend(tickers)
                self._log(f"  Loaded {len(tickers)} from {exchange.upper()}")

        if not all_tickers:
            return None

        unique_tickers = self._deduplicate(all_tickers)
        self._cache["all_us"] = unique_tickers

        should_save = save if save is not None else self.auto_cache
        if should_save:
            filepath = self._save_to_cache(unique_tickers, "all_us")
            self._log(f"Saved {len(unique_tickers)} total US tickers to {filepath}")

        return unique_tickers

    # -------------------------------------------------------------------------
    # Public API - Load Methods
    # -------------------------------------------------------------------------

    def load(self, name: str) -> List[str]:
        """
        Load tickers from cache file.

        Parameters:
        -----------
        name : str
            Source name ('sp500', 'nasdaq', 'nyse', 'amex', 'all_us')
            or custom filename (without _tickers.txt suffix)

        Returns:
        --------
        list
            List of tickers (empty if file not found)
        """
        filepath = self._get_cache_path(name)

        if not os.path.exists(filepath):
            self._log(f"Cache file not found: {filepath}")
            return []

        with open(filepath, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]

        self._cache[name] = tickers
        return tickers

    def load_custom(self, filepath: str) -> List[str]:
        """
        Load tickers from a custom file path.

        Parameters:
        -----------
        filepath : str
            Full path to the ticker file

        Returns:
        --------
        list
            List of tickers (empty if file not found)
        """
        if not os.path.exists(filepath):
            self._log(f"File not found: {filepath}")
            return []

        with open(filepath, "r") as f:
            return [line.strip() for line in f if line.strip()]

    # -------------------------------------------------------------------------
    # Public API - Utility Methods
    # -------------------------------------------------------------------------

    def filter(self, tickers: List[str], predicate: Callable[[str], bool]) -> List[str]:
        """
        Filter tickers using a custom predicate function.

        Parameters:
        -----------
        tickers : list
            List of tickers to filter
        predicate : callable
            Function that returns True for tickers to keep

        Returns:
        --------
        list
            Filtered list of tickers

        Examples:
        ---------
        >>> # Keep only tickers starting with 'A'
        >>> filtered = tm.filter(tickers, lambda t: t.startswith('A'))
        """
        return [t for t in tickers if predicate(t)]

    def exclude_warrants(self, tickers: List[str]) -> List[str]:
        """Remove warrant symbols (typically end in W, WS, or have + suffix)."""
        return self.filter(
            tickers, lambda t: not (t.endswith("W") or t.endswith("WS") or "+" in t)
        )

    def get_cached(self, name: str) -> Optional[List[str]]:
        """Get tickers from in-memory cache without fetching or loading."""
        return self._cache.get(name)

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()

    def add_source(self, name: str, source: TickerSource) -> None:
        """
        Add a custom ticker source.

        Parameters:
        -----------
        name : str
            Name for the source
        source : TickerSource
            Source configuration

        Examples:
        ---------
        >>> tm.add_source("russell2000", TickerSource(
        ...     name="Russell 2000",
        ...     url="https://example.com/russell2000.csv",
        ...     parser="csv",
        ...     symbol_column="Ticker"
        ... ))
        """
        self.SOURCES[name] = source


# -------------------------------------------------------------------------
# Convenience Functions (Backward Compatibility)
# -------------------------------------------------------------------------

_default_manager: Optional[TickerManager] = None


def _get_default_manager() -> TickerManager:
    """Get or create the default TickerManager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = TickerManager()
    return _default_manager


def get_sp500_tickers() -> Optional[List[str]]:
    """Fetch S&P 500 tickers (backward compatible)."""
    return _get_default_manager().get_sp500()


def get_nasdaq_tickers() -> Optional[List[str]]:
    """Fetch NASDAQ tickers (backward compatible)."""
    return _get_default_manager().get_nasdaq()


def get_nyse_tickers() -> Optional[List[str]]:
    """Fetch NYSE tickers (backward compatible)."""
    return _get_default_manager().get_nyse()


def get_all_us_tickers() -> Optional[List[str]]:
    """Fetch all US tickers (backward compatible)."""
    return _get_default_manager().get_all_us()


def load_tickers_from_file(filename: str) -> List[str]:
    """Load tickers from file (backward compatible)."""
    # Handle both "name" and "name_tickers.txt" formats
    name = filename.replace("_tickers.txt", "").replace(".txt", "")
    return _get_default_manager().load(name)


# -------------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # Demo usage
    tm = TickerManager(verbose=True)

    print("\n" + "=" * 60)
    print("  TICKER MANAGER DEMO")
    print("=" * 60)

    print("\n1. Fetching NASDAQ tickers...")
    nasdaq = tm.get_nasdaq()
    if nasdaq:
        print(f"   Total: {len(nasdaq)}")
        print(f"   Sample: {nasdaq[:10]}")

    print("\n2. Fetching NYSE tickers...")
    nyse = tm.get_nyse()
    if nyse:
        print(f"   Total: {len(nyse)}")
        print(f"   Sample: {nyse[:10]}")

    print("\n3. Fetching S&P 500 tickers...")
    sp500 = tm.get_sp500()
    if sp500:
        print(f"   Total: {len(sp500)}")
        print(f"   Sample: {sp500[:10]}")

    print("\n4. Loading from cache...")
    cached = tm.load("sp500")
    if cached:
        print(f"   Loaded {len(cached)} tickers from cache")

    print("\n5. Using filter...")
    if sp500:
        tech_like = tm.filter(sp500, lambda t: t.startswith(("A", "M", "G")))
        print(f"   Tickers starting with A, M, or G: {len(tech_like)}")

    print("\n" + "=" * 60)
