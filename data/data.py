import argparse
import warnings
from pathlib import Path

import yfinance as yf
from tqdm import tqdm

from database.database import init_database, save_financial_data


# Get the script directory and set up paths
SCRIPT_DIR = Path(__file__).parent
PACKAGE_DIR = SCRIPT_DIR.parent
DEFAULT_EXCHANGE = "CUSTOM"
UNIVERSE_PRESETS = {
    "nyse": {"file": "Utils/nyse_tickers.txt", "exchange": "NYSE"},
    "nasdaq": {"file": "Utils/nasdaq_tickers.txt", "exchange": "NASDAQ"},
    "sp500": {"file": "Utils/sp500_tickers.txt", "exchange": "SP500"},
}


warnings.filterwarnings(
    "ignore",
    message=".*Timestamp.utcnow is deprecated and will be removed in a future version.*",
    module=r"yfinance\..*",
)


def _normalize_and_dedupe_tickers(tickers):
    """Normalize ticker symbols and remove duplicates while preserving order."""
    normalized = [str(t).strip().upper() for t in tickers if str(t).strip()]
    return list(dict.fromkeys(normalized))


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    pass


def build_parser():
    """Build command-line parser for data ingestion."""
    parser = argparse.ArgumentParser(
        description="Fetch financial data and save to database.",
        formatter_class=_HelpFormatter,
        epilog="""
Examples:
  py -m data.data AAPL MSFT --exchange CUSTOM
  py -m data.data --ticker AAPL --ticker-file .\\my_tickers.txt --exchange CUSTOM
  py -m data.data --universe nyse --universe nasdaq
  py -m data.data --quiet --no-progress --ticker AAPL
        """,
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Positional tickers to fetch (for example: QQQ SPY AAPL).",
    )
    parser.add_argument(
        "--ticker",
        action="append",
        default=[],
        help="Repeatable explicit ticker argument.",
    )
    parser.add_argument(
        "--ticker-file",
        action="append",
        default=[],
        help="Repeatable path to a text file with one ticker per line.",
    )
    parser.add_argument(
        "--universe",
        action="append",
        choices=sorted(UNIVERSE_PRESETS.keys()),
        default=[],
        help="Repeatable built-in ticker universe preset.",
    )
    parser.add_argument(
        "--exchange",
        default=DEFAULT_EXCHANGE,
        help=(
            "Exchange label for positional/--ticker/--ticker-file symbols. "
            "Universe presets always use canonical labels."
        ),
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress status messages.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser


def parse_args(argv=None):
    """Parse command-line arguments for data ingestion."""
    return build_parser().parse_args(argv)


def _resolve_ticker_file_path(ticker_file):
    ticker_path = Path(ticker_file)
    if ticker_path.is_absolute():
        return ticker_path

    # Primary path: relative to project root (for example Utils/*.txt)
    package_relative = PACKAGE_DIR / ticker_path
    # Backward-compatible fallback: relative to this module dir
    data_relative = SCRIPT_DIR / ticker_path
    return package_relative if package_relative.exists() else data_relative


def _load_tickers_from_file(ticker_file, verbose=True, strict=False):
    """Load and normalize tickers from a file."""
    txt_file = _resolve_ticker_file_path(ticker_file)
    if not txt_file.exists():
        msg = f"{txt_file} not found. Please run ticker.py first."
        if strict:
            raise FileNotFoundError(msg)
        if verbose:
            print(f"Error: {msg}")
        return None

    with open(txt_file, "r", encoding="utf-8") as f:
        raw_tickers = [line.strip() for line in f if line.strip()]
    tickers = _normalize_and_dedupe_tickers(raw_tickers)

    if verbose:
        duplicate_count = len(raw_tickers) - len(tickers)
        print(f"Found {len(tickers)} unique tickers in {ticker_file}")
        if duplicate_count > 0:
            print(f"Removed {duplicate_count} duplicate ticker entries from file.")

    return tickers


def resolve_ticker_exchange_map(args):
    """
    Resolve CLI inputs into a deterministic ticker->exchange mapping.

    Resolution order:
    1) positional tickers
    2) --ticker entries
    3) --ticker-file entries
    4) --universe presets
    """
    symbol_exchange = {}

    def _add_symbols(symbols, exchange):
        for symbol in _normalize_and_dedupe_tickers(symbols):
            if symbol not in symbol_exchange:
                symbol_exchange[symbol] = exchange

    _add_symbols(args.tickers, args.exchange)
    _add_symbols(args.ticker, args.exchange)

    for ticker_file in args.ticker_file:
        file_symbols = _load_tickers_from_file(
            ticker_file,
            verbose=not args.quiet,
            strict=True,
        )
        _add_symbols(file_symbols or [], args.exchange)

    provided_sources = bool(args.tickers or args.ticker or args.ticker_file or args.universe)
    universe_sequence = list(args.universe)
    if not provided_sources:
        universe_sequence = ["nyse"]

    for universe_name in universe_sequence:
        universe_spec = UNIVERSE_PRESETS[universe_name]
        universe_symbols = _load_tickers_from_file(
            universe_spec["file"],
            verbose=not args.quiet,
            strict=True,
        )
        _add_symbols(universe_symbols or [], universe_spec["exchange"])

    return symbol_exchange


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
        ticker_obj = yf.Ticker(symbol)
        save_financial_data(symbol, "balance_sheet", ticker_obj.get_balance_sheet(), exchange)
        save_financial_data(symbol, "income_statement", ticker_obj.get_income_stmt(), exchange)
        save_financial_data(symbol, "financials", ticker_obj.get_financials(), exchange)
        save_financial_data(symbol, "cash_flow", ticker_obj.get_cash_flow(), exchange)
        save_financial_data(symbol, "history", ticker_obj.history(period="max"), exchange)

        if verbose:
            print(f"  [OK] Completed {symbol}")
        return True
    except Exception as exc:
        if verbose:
            print(f"  [ERR] Error processing {symbol}: {exc}")
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
            f"\nCompleted: {len(results['success'])} succeeded, "
            f"{len(results['failed'])} failed."
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
    tickers = _load_tickers_from_file(ticker_file, verbose=verbose, strict=False)
    if tickers is None:
        return None

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


def run_data_fetch(
    tickers=None,
    exchange=None,
    verbose=True,
    show_progress=True,
    ticker_exchange_map=None,
):
    """
    Main entry point for fetching ticker data.

    Args:
        tickers: List of ticker symbols (if None, fetches NYSE by default)
        exchange: Exchange name (defaults based on context)
        verbose: Whether to print progress messages
        show_progress: Whether to show progress bar
        ticker_exchange_map: Explicit ticker->exchange mapping

    Returns:
        dict: Results with 'success' and 'failed' ticker lists
    """
    init_database()

    if ticker_exchange_map:
        grouped = {}
        for symbol, symbol_exchange in ticker_exchange_map.items():
            grouped.setdefault(symbol_exchange, []).append(symbol)

        merged_results = {"success": [], "failed": []}
        for grouped_exchange, grouped_tickers in grouped.items():
            if verbose:
                print(
                    f"Fetching data for exchange group {grouped_exchange}: "
                    f"{len(grouped_tickers)} ticker(s)"
                )
            group_results = fetch_tickers(
                grouped_tickers,
                exchange=grouped_exchange,
                verbose=verbose,
                show_progress=show_progress,
            )
            merged_results["success"].extend(group_results.get("success", []))
            merged_results["failed"].extend(group_results.get("failed", []))
        return merged_results

    if tickers:
        ex = exchange or DEFAULT_EXCHANGE
        if verbose:
            print(f"Fetching data for specified tickers: {', '.join(tickers)}")
        return fetch_tickers(
            tickers,
            exchange=ex,
            verbose=verbose,
            show_progress=show_progress,
        )

    # Default behavior: fetch all NYSE tickers
    return fetch_nyse_data(verbose=verbose, show_progress=show_progress)


# Legacy CSV functions (kept for backward compatibility)
def save_dataframe_csv(ticker_symbol, name, df, subfolder="ticker_data"):
    """Save DataFrame to CSV file (legacy method)."""
    if df is None or getattr(df, "empty", False):
        return
    ticker_dir = SCRIPT_DIR / subfolder / ticker_symbol
    ticker_dir.mkdir(parents=True, exist_ok=True)
    out_path = ticker_dir / f"{name}.csv"
    df.to_csv(out_path)


def main(argv=None):
    """CLI entry point - parses arguments and fetches data."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        ticker_exchange_map = resolve_ticker_exchange_map(args)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    run_data_fetch(
        tickers=None,
        exchange=args.exchange,
        verbose=not args.quiet,
        show_progress=not args.no_progress,
        ticker_exchange_map=ticker_exchange_map,
    )


if __name__ == "__main__":
    main()
