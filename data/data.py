import os
import sys
import argparse
import warnings
import yfinance as yf
from tqdm import tqdm
from pathlib import Path

# Get the script directory and set up paths
SCRIPT_DIR = Path(__file__).parent
PACKAGE_DIR = SCRIPT_DIR.parent
DEFAULT_EXCHANGE = "CUSTOM"

from pyfinancial.database.database import init_database, save_financial_data


warnings.filterwarnings(
    "ignore",
    message=".*Timestamp.utcnow is deprecated and will be removed in a future version.*",
    module=r"yfinance\..*",
)


def _normalize_and_dedupe_tickers(tickers):
    """Normalize ticker symbols and remove duplicates while preserving order."""
    normalized = [str(t).strip().upper() for t in tickers if str(t).strip()]
    return list(dict.fromkeys(normalized))


def parse_args():
    """Parse command-line arguments for ticker flags."""
    parser = argparse.ArgumentParser(
        description="Fetch financial data and save to database.",
        epilog="Example: python data.py --QQQ --SPY --AAPL",
    )
    parser.add_argument(
        "tickers", nargs="*", help="Specific tickers to fetch (e.g., QQQ SPY AAPL)"
    )
    parser.add_argument(
        "--exchange",
        default=DEFAULT_EXCHANGE,
        help=f"Exchange name for the tickers (default: {DEFAULT_EXCHANGE})",
    )

    # Parse known args and collect any --TICKER style flags
    args, unknown = parser.parse_known_args()

    # Extract tickers from --TICKER style flags (e.g., --QQQ, --SPY)
    flag_tickers = []
    for arg in unknown:
        if arg.startswith("--"):
            ticker = arg[2:].upper()
            if ticker:
                flag_tickers.append(ticker)
        elif arg.startswith("-") and len(arg) > 1:
            # Single dash flags like -QQQ
            ticker = arg[1:].upper()
            if ticker:
                flag_tickers.append(ticker)

    # Combine positional tickers and flag-style tickers
    all_tickers = [t.upper() for t in args.tickers] + flag_tickers

    return all_tickers, args.exchange


def fetch_single_ticker(symbol, exchange="CUSTOM", verbose=True):
    """
    Fetch and save all financial data for a single ticker.

    Args:
        symbol: Stock ticker symbol
        exchange: Exchange name for categorization
        verbose: Whether to print progress messages

    Returns:
        bool: True if successful, False otherwise
    """
    if verbose:
        print(f"Fetching data for {symbol}...")

    try:
        t = yf.Ticker(symbol)
        save_financial_data(symbol, "balance_sheet", t.get_balance_sheet(), exchange)
        save_financial_data(symbol, "income_statement", t.get_income_stmt(), exchange)
        save_financial_data(symbol, "financials", t.get_financials(), exchange)
        save_financial_data(symbol, "cash_flow", t.get_cash_flow(), exchange)
        save_financial_data(symbol, "history", t.history(period="max"), exchange)

        if verbose:
            print(f"  ✓ Completed {symbol}")
        return True
    except Exception as e:
        if verbose:
            print(f"  ✗ Error processing {symbol}: {e}")
        return False


def fetch_tickers(tickers, exchange="CUSTOM", verbose=True, show_progress=True):
    """
    Fetch financial data for a list of tickers.

    Args:
        tickers: List of ticker symbols
        exchange: Exchange name for categorization
        verbose: Whether to print status messages
        show_progress: Whether to show progress bar

    Returns:
        dict: Results with 'success' and 'failed' ticker lists
    """
    unique_tickers = _normalize_and_dedupe_tickers(tickers)
    duplicate_count = len(tickers) - len(unique_tickers)

    if verbose:
        print(f"Fetching data for {len(unique_tickers)} ticker(s)...")
        if duplicate_count > 0:
            print(f"Removed {duplicate_count} duplicate ticker entries.")

    results = {"success": [], "failed": []}

    iterator = (
        tqdm(unique_tickers, desc=f"Fetching {exchange}", unit="ticker")
        if show_progress
        else unique_tickers
    )

    for symbol in iterator:
        success = fetch_single_ticker(symbol, exchange, verbose=False)
        if success:
            results["success"].append(symbol)
        else:
            results["failed"].append(symbol)

    if verbose:
        print(
            f"\nCompleted: {len(results['success'])} succeeded, {len(results['failed'])} failed."
        )

    return results


def fetch_data_from_file(
    ticker_file, exchange="SP500", verbose=True, show_progress=True
):
    """
    Fetch financial data for tickers listed in a file.

    Args:
        ticker_file: Path to file containing ticker symbols (one per line)
        exchange: Exchange name for categorization
        verbose: Whether to print status messages
        show_progress: Whether to show progress bar

    Returns:
        dict: Results with 'success' and 'failed' ticker lists, or None if file not found
    """
    ticker_path = Path(ticker_file)
    if ticker_path.is_absolute():
        txt_file = ticker_path
    else:
        # Primary path: relative to package root (pyfinancial/Utils/*.txt)
        package_relative = PACKAGE_DIR / ticker_path
        # Backward-compatible fallback: relative to this module dir
        data_relative = SCRIPT_DIR / ticker_path
        txt_file = package_relative if package_relative.exists() else data_relative

    if not txt_file.exists():
        if verbose:
            print(f"Error: {txt_file} not found. Please run ticker.py first.")
        return None

    with open(txt_file, "r") as f:
        raw_tickers = [line.strip() for line in f if line.strip()]
    tickers = _normalize_and_dedupe_tickers(raw_tickers)

    if verbose:
        duplicate_count = len(raw_tickers) - len(tickers)
        print(f"Found {len(tickers)} unique tickers in {ticker_file}")
        if duplicate_count > 0:
            print(f"Removed {duplicate_count} duplicate ticker entries from file.")

    return fetch_tickers(
        tickers, exchange=exchange, verbose=verbose, show_progress=show_progress
    )


def fetch_sp500_data(verbose=True, show_progress=True):
    """Fetch data for all S&P 500 tickers."""
    return fetch_data_from_file(
        "Utils/sp500_tickers.txt",
        exchange="SP500",
        verbose=verbose,
        show_progress=show_progress,
    )


def fetch_nasdaq_data(verbose=True, show_progress=True):
    """Fetch data for all NASDAQ tickers."""
    return fetch_data_from_file(
        "Utils/nasdaq_tickers.txt",
        exchange="NASDAQ",
        verbose=verbose,
        show_progress=show_progress,
    )


def fetch_nyse_data(verbose=True, show_progress=True):
    """Fetch data for all NYSE tickers."""
    return fetch_data_from_file(
        "Utils/nyse_tickers.txt",
        exchange="NYSE",
        verbose=verbose,
        show_progress=show_progress,
    )


def run_data_fetch(tickers=None, exchange=None, verbose=True):
    """
    Main entry point for fetching ticker data.

    Args:
        tickers: List of ticker symbols (if None, fetches NYSE by default)
        exchange: Exchange name (defaults based on context)
        verbose: Whether to print progress messages

    Returns:
        dict: Results with 'success' and 'failed' ticker lists
    """
    init_database()

    if tickers:
        ex = exchange or DEFAULT_EXCHANGE
        if verbose:
            print(f"Fetching data for specified tickers: {', '.join(tickers)}")
        return fetch_tickers(tickers, exchange=ex, verbose=verbose)
    else:
        # Default behavior: fetch all NYSE tickers
        return fetch_nyse_data(verbose=verbose)


# Legacy CSV functions (kept for backward compatibility)
def save_dataframe_csv(ticker_symbol, name, df, subfolder="ticker_data"):
    """Save DataFrame to CSV file (legacy method)."""
    if df is None or getattr(df, "empty", False):
        return
    ticker_dir = SCRIPT_DIR / subfolder / ticker_symbol
    ticker_dir.mkdir(parents=True, exist_ok=True)
    out_path = ticker_dir / f"{name}.csv"
    df.to_csv(out_path)


def main():
    """CLI entry point - parses arguments and fetches data."""
    tickers, exchange = parse_args()
    run_data_fetch(tickers=tickers if tickers else None, exchange=exchange)


if __name__ == "__main__":
    main()
