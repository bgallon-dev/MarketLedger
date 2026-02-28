import sqlite3
import os
import sys
from typing import Optional, Sequence, List
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "financial_data.db")


def query_database(sql, params=None):
    """Execute a SQL query and return results as a DataFrame."""
    base_dir = os.path.dirname(__file__)
    db_path = os.path.join(base_dir, "financial_data.db")  # adjust if different

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def get_connection():
    """Get a database connection."""
    return sqlite3.connect(DB_PATH)


def ensure_indices():
    """Ensure all necessary indices exist for performance."""
    conn = get_connection()
    cursor = conn.cursor()

    # History indices
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_history_ticker_date ON history(ticker_id, date)"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_date ON history(date)")

    # Financials indices
    tables = [
        "balance_sheet",
        "income_statement",
        "cash_flow",
        "financials",
        "earnings",
    ]
    for table in tables:
        # Index for filtering by ticker and metric
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_ticker_metric ON {table}(ticker_id, metric)"
        )
        # Index for filtering by period
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_period ON {table}(period)"
        )

    conn.commit()
    conn.close()
    print("Indices verified.")


def init_database():
    """Initialize the database with required tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # Tickers table - stores all stock symbols
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS tickers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE NOT NULL,
            exchange TEXT,
            name TEXT,
            sector TEXT,
            industry TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Create index on symbol for faster lookups
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tickers_symbol ON tickers(symbol)
    """
    )

    # Create index on exchange for filtering
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tickers_exchange ON tickers(exchange)
    """
    )

    # Financial data tables (normalized structure)
    for table in [
        "balance_sheet",
        "income_statement",
        "financials",
        "cash_flow",
        "earnings",
    ]:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker_id INTEGER NOT NULL,
                metric TEXT NOT NULL,
                period TEXT NOT NULL,
                value REAL,
                FOREIGN KEY (ticker_id) REFERENCES tickers(id),
                UNIQUE(ticker_id, metric, period)
            )
        """
        )
        # Create index for faster queries
        cursor.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{table}_ticker ON {table}(ticker_id)
        """
        )

    # Price history table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            dividends REAL,
            stock_splits REAL,
            FOREIGN KEY (ticker_id) REFERENCES tickers(id),
            UNIQUE(ticker_id, date)
        )
    """
    )

    # Create index for history queries
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_history_ticker_date ON history(ticker_id, date)
    """
    )

    conn.commit()
    conn.close()
    print("Database initialized successfully.")


def get_or_create_ticker(
    symbol, exchange="SP500", name=None, sector=None, industry=None
):
    """Get ticker_id or create new ticker entry."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM tickers WHERE symbol = ?", (symbol,))
    result = cursor.fetchone()

    if result:
        ticker_id = result[0]
        # Update exchange if different
        cursor.execute(
            "UPDATE tickers SET exchange = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND exchange != ?",
            (exchange, ticker_id, exchange),
        )
    else:
        cursor.execute(
            "INSERT INTO tickers (symbol, exchange, name, sector, industry) VALUES (?, ?, ?, ?, ?)",
            (symbol, exchange, name, sector, industry),
        )
        ticker_id = cursor.lastrowid

    conn.commit()
    conn.close()
    return ticker_id


def save_financial_data(ticker_symbol, table_name, df, exchange="SP500"):
    """Save financial DataFrame to database."""
    if df is None or df.empty:
        print(f"  Skipping {table_name} for {ticker_symbol} - no data")
        return

    conn = get_connection()
    ticker_id = get_or_create_ticker(ticker_symbol, exchange)

    try:
        if table_name == "history":
            _save_history_data(conn, ticker_id, df)
        else:
            _save_financial_table(conn, ticker_id, table_name, df)
        print(f"  Saved {table_name} for {ticker_symbol}")
    except Exception as e:
        print(f"  Error saving {table_name} for {ticker_symbol}: {e}")

    conn.close()


def _save_history_data(conn, ticker_id, df):
    """Save historical price data."""
    cursor = conn.cursor()
    df_reset = df.reset_index()

    for _, row in df_reset.iterrows():
        cursor.execute(
            """
            INSERT OR REPLACE INTO history 
            (ticker_id, date, open, high, low, close, volume, dividends, stock_splits)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                ticker_id,
                str(row.get("Date", row.get("index", ""))),
                row.get("Open"),
                row.get("High"),
                row.get("Low"),
                row.get("Close"),
                row.get("Volume"),
                row.get("Dividends", 0),
                row.get("Stock Splits", 0),
            ),
        )

    conn.commit()


def _save_financial_table(conn, ticker_id, table_name, df):
    """Save financial data (balance sheet, income statement, etc.)."""
    cursor = conn.cursor()

    # Financial data has metrics as rows and periods as columns
    for metric in df.index:
        for period in df.columns:
            value = df.loc[metric, period]
            # Convert to float, handle NaN
            if pd.notna(value):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = None
            else:
                value = None

            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {table_name} 
                (ticker_id, metric, period, value)
                VALUES (?, ?, ?, ?)
            """,
                (ticker_id, str(metric), str(period), value),
            )

    conn.commit()


# ============== Query Functions ==============


def get_all_tickers(exchange=None):
    """Get all tickers, optionally filtered by exchange."""
    conn = get_connection()
    if exchange:
        df = pd.read_sql_query(
            "SELECT * FROM tickers WHERE exchange = ?", conn, params=(exchange,)
        )
    else:
        df = pd.read_sql_query("SELECT * FROM tickers", conn)
    conn.close()
    return df


def get_ticker_history(symbol, start_date=None, end_date=None):
    """Get historical price data for a ticker."""
    conn = get_connection()
    query = """
        SELECT h.date, h.open, h.high, h.low, h.close, h.volume, h.dividends, h.stock_splits
        FROM history h
        JOIN tickers t ON h.ticker_id = t.id
        WHERE t.symbol = ?
    """
    params = [symbol]

    if start_date:
        query += " AND h.date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND h.date <= ?"
        params.append(end_date)

    query += " ORDER BY h.date"

    df = pd.read_sql_query(query, conn, params=tuple(params))
    conn.close()
    return df


def _normalize_symbol_list(symbols: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for raw in symbols:
        if raw is None:
            continue
        symbol = str(raw).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        ordered.append(symbol)
    return ordered


def _chunked(values: Sequence[str], chunk_size: int = 500) -> List[List[str]]:
    return [list(values[i : i + chunk_size]) for i in range(0, len(values), chunk_size)]


def get_ticker_history_bulk(symbols, start_date=None, end_date=None) -> pd.DataFrame:
    """Get historical price data for many tickers in one query."""
    normalized = _normalize_symbol_list(symbols or [])
    if not normalized:
        return pd.DataFrame(
            columns=[
                "symbol",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "dividends",
                "stock_splits",
            ]
        )

    conn = get_connection()
    frames = []
    try:
        for chunk in _chunked(normalized):
            placeholders = ",".join(["?"] * len(chunk))
            query = f"""
                SELECT t.symbol, h.date, h.open, h.high, h.low, h.close, h.volume, h.dividends, h.stock_splits
                FROM history h
                JOIN tickers t ON h.ticker_id = t.id
                WHERE t.symbol IN ({placeholders})
            """
            params = list(chunk)
            if start_date:
                query += " AND h.date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND h.date <= ?"
                params.append(end_date)
            query += " ORDER BY t.symbol, h.date"
            frames.append(pd.read_sql_query(query, conn, params=tuple(params)))
    finally:
        conn.close()

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_financial_data(symbol, table_name):
    """Get financial data for a ticker (balance_sheet, income_statement, etc.)."""
    conn = get_connection()
    query = f"""
        SELECT f.metric, f.period, f.value
        FROM {table_name} f
        JOIN tickers t ON f.ticker_id = t.id
        WHERE t.symbol = ?
    """
    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()

    if df.empty:
        return df

    # Pivot to get original format (metrics as rows, periods as columns)
    return df.pivot(index="metric", columns="period", values="value")


def get_financial_data_bulk(
    symbols, table_name: str, metrics: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """
    Get long-format financial data for many tickers from one table.

    Returns columns: symbol, metric, period, value
    """
    valid_tables = {
        "balance_sheet",
        "income_statement",
        "financials",
        "cash_flow",
        "earnings",
    }
    if table_name not in valid_tables:
        raise ValueError(f"Invalid table_name '{table_name}'. Must be one of {sorted(valid_tables)}")

    normalized = _normalize_symbol_list(symbols or [])
    if not normalized:
        return pd.DataFrame(columns=["symbol", "metric", "period", "value"])

    metric_values = []
    if metrics:
        metric_values = [str(m).strip() for m in metrics if str(m).strip()]

    conn = get_connection()
    frames = []
    try:
        for chunk in _chunked(normalized):
            symbol_placeholders = ",".join(["?"] * len(chunk))
            query = f"""
                SELECT t.symbol, f.metric, f.period, f.value
                FROM {table_name} f
                JOIN tickers t ON f.ticker_id = t.id
                WHERE t.symbol IN ({symbol_placeholders})
            """
            params: List[str] = list(chunk)
            if metric_values:
                metric_placeholders = ",".join(["?"] * len(metric_values))
                query += f" AND f.metric IN ({metric_placeholders})"
                params.extend(metric_values)
            query += " ORDER BY t.symbol, f.metric, f.period"
            frames.append(pd.read_sql_query(query, conn, params=tuple(params)))
    finally:
        conn.close()

    if not frames:
        return pd.DataFrame(columns=["symbol", "metric", "period", "value"])
    return pd.concat(frames, ignore_index=True)


def get_database_stats():
    """Get statistics about the database."""
    conn = get_connection()
    cursor = conn.cursor()

    stats = {}

    # Count tickers by exchange
    cursor.execute("SELECT exchange, COUNT(*) FROM tickers GROUP BY exchange")
    stats["tickers_by_exchange"] = dict(cursor.fetchall())

    # Total records in each table
    for table in [
        "tickers",
        "balance_sheet",
        "income_statement",
        "financials",
        "cash_flow",
        "earnings",
        "history",
    ]:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        stats[f"{table}_count"] = cursor.fetchone()[0]

    conn.close()
    return stats


# ============== Search & Analysis Functions ==============


def search_tickers(query, exchange=None):
    """Search tickers by symbol or name pattern."""
    conn = get_connection()
    sql = "SELECT * FROM tickers WHERE symbol LIKE ? OR name LIKE ?"
    params = [f"%{query}%", f"%{query}%"]

    if exchange:
        sql += " AND exchange = ?"
        params.append(exchange)

    df = pd.read_sql_query(sql, conn, params=tuple(params))
    conn.close()
    return df


def get_metric_across_tickers(table_name, metric, exchange=None):
    """Get a specific metric for all tickers (e.g., 'TotalRevenue' across all stocks)."""
    valid_tables = [
        "balance_sheet",
        "income_statement",
        "financials",
        "cash_flow",
        "earnings",
    ]
    if table_name not in valid_tables:
        raise ValueError(f"Invalid table name. Must be one of: {valid_tables}")

    conn = get_connection()
    query = f"""
        SELECT t.symbol, f.period, f.value
        FROM {table_name} f
        JOIN tickers t ON f.ticker_id = t.id
        WHERE f.metric = ?
    """
    params = [metric]

    if exchange:
        query += " AND t.exchange = ?"
        params.append(exchange)

    df = pd.read_sql_query(query, conn, params=tuple(params))
    conn.close()

    if df.empty:
        return df

    return df.pivot(index="symbol", columns="period", values="value")


def compare_tickers(symbols, table_name, metrics=None):
    """Compare multiple tickers on selected metrics."""
    valid_tables = [
        "balance_sheet",
        "income_statement",
        "financials",
        "cash_flow",
        "earnings",
    ]
    if table_name not in valid_tables:
        raise ValueError(f"Invalid table name. Must be one of: {valid_tables}")

    conn = get_connection()
    placeholders = ",".join("?" * len(symbols))

    query = f"""
        SELECT t.symbol, f.metric, f.period, f.value
        FROM {table_name} f
        JOIN tickers t ON f.ticker_id = t.id
        WHERE t.symbol IN ({placeholders})
    """
    params = list(symbols)

    if metrics:
        metric_placeholders = ",".join("?" * len(metrics))
        query += f" AND f.metric IN ({metric_placeholders})"
        params.extend(metrics)

    df = pd.read_sql_query(query, conn, params=tuple(params))
    conn.close()
    return df


def get_available_metrics(table_name):
    """List all available metrics in a financial table."""
    valid_tables = [
        "balance_sheet",
        "income_statement",
        "financials",
        "cash_flow",
        "earnings",
    ]
    if table_name not in valid_tables:
        raise ValueError(f"Invalid table name. Must be one of: {valid_tables}")

    conn = get_connection()
    query = f"SELECT DISTINCT metric FROM {table_name} ORDER BY metric"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df["metric"].tolist()


def get_price_summary(symbol):
    """Get price summary statistics for a ticker."""
    conn = get_connection()
    query = """
        SELECT 
            MIN(date) as first_date,
            MAX(date) as last_date,
            MIN(close) as min_price,
            MAX(close) as max_price,
            AVG(close) as avg_price,
            SUM(volume) as total_volume,
            COUNT(*) as trading_days
        FROM history h
        JOIN tickers t ON h.ticker_id = t.id
        WHERE t.symbol = ?
    """
    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()

    if df.empty or df.iloc[0]["first_date"] is None:
        return None

    return df.iloc[0].to_dict()


def get_top_performers(metric, table_name, period=None, limit=10, ascending=False):
    """Get top performing tickers by a specific metric."""
    valid_tables = [
        "balance_sheet",
        "income_statement",
        "financials",
        "cash_flow",
        "earnings",
    ]
    if table_name not in valid_tables:
        raise ValueError(f"Invalid table name. Must be one of: {valid_tables}")

    conn = get_connection()
    order = "ASC" if ascending else "DESC"

    query = f"""
        SELECT t.symbol, f.period, f.value
        FROM {table_name} f
        JOIN tickers t ON f.ticker_id = t.id
        WHERE f.metric = ?
    """
    params = [metric]

    if period:
        query += " AND f.period = ?"
        params.append(period)

    query += f" ORDER BY f.value {order} LIMIT ?"
    params.append(limit)

    df = pd.read_sql_query(query, conn, params=tuple(params))
    conn.close()
    return df


def interactive_query():
    """Simple interactive query interface for data analysis."""
    print("\n" + "=" * 50)
    print("       Financial Data Analysis Interface")
    print("=" * 50)
    print("\nCommands:")
    print("  search <query>           - Search tickers by symbol/name")
    print("  history <symbol>         - Get price history (last 20 days)")
    print("  summary <symbol>         - Get price summary statistics")
    print("  financials <symbol>      - Get income statement data")
    print("  metrics <table>          - List available metrics")
    print("  compare <sym1,sym2,...>  - Compare tickers")
    print("  top <metric> <table>     - Get top performers by metric")
    print("  sql <query>              - Run custom SQL query")
    print("  stats                    - Show database statistics")
    print("  help                     - Show this help message")
    print("  quit                     - Exit")
    print()

    while True:
        try:
            cmd = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        try:
            if action in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            elif action == "help":
                print(
                    "\nAvailable tables: balance_sheet, income_statement, financials, cash_flow, earnings"
                )
                print("Example: metrics income_statement")
                print("Example: compare AAPL,MSFT,GOOGL")
                print("Example: top TotalRevenue income_statement\n")
            elif action == "search":
                result = search_tickers(arg)
                print(result if not result.empty else "No tickers found.")
            elif action == "history":
                result = get_ticker_history(arg)
                print(result.tail(20) if not result.empty else f"No history for {arg}")
            elif action == "summary":
                result = get_price_summary(arg)
                if result:
                    for k, v in result.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"No data for {arg}")
            elif action == "financials":
                result = get_financial_data(arg, "income_statement")
                print(result if not result.empty else f"No financial data for {arg}")
            elif action == "metrics":
                table = arg or "income_statement"
                result = get_available_metrics(table)
                print(f"\nMetrics in {table}:")
                for m in result:
                    print(f"  - {m}")
            elif action == "compare":
                symbols = [s.strip().upper() for s in arg.split(",")]
                result = compare_tickers(symbols, "income_statement")
                print(result if not result.empty else "No data found.")
            elif action == "top":
                args = arg.split()
                if len(args) >= 2:
                    metric, table = args[0], args[1]
                    result = get_top_performers(metric, table)
                    print(result if not result.empty else "No data found.")
                else:
                    print("Usage: top <metric> <table>")
            elif action == "sql":
                result = query_database(arg)
                print(result)
            elif action == "stats":
                stats = get_database_stats()
                print("\nDatabase Statistics:")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
            else:
                print(f"Unknown command: {action}. Type 'help' for available commands.")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    init_database()

    # Check for command-line arguments
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        arg = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""

        try:
            if action == "metrics":
                table = arg or "income_statement"
                result = get_available_metrics(table)
                print(f"\nMetrics in {table}:")
                for m in result:
                    print(f"  - {m}")
            elif action == "search":
                result = search_tickers(arg)
                print(result if not result.empty else "No tickers found.")
            elif action == "history":
                result = get_ticker_history(arg)
                print(result.tail(20) if not result.empty else f"No history for {arg}")
            elif action == "summary":
                result = get_price_summary(arg)
                if result:
                    for k, v in result.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"No data for {arg}")
            elif action == "financials":
                result = get_financial_data(arg, "income_statement")
                print(result if not result.empty else f"No financial data for {arg}")
            elif action == "compare":
                symbols = [s.strip().upper() for s in arg.split(",")]
                result = compare_tickers(symbols, "income_statement")
                print(result if not result.empty else "No data found.")
            elif action == "top":
                args = arg.split()
                if len(args) >= 2:
                    metric, table = args[0], args[1]
                    result = get_top_performers(metric, table)
                    print(result if not result.empty else "No data found.")
                else:
                    print("Usage: top <metric> <table>")
            elif action == "stats":
                stats = get_database_stats()
                print("\nDatabase Statistics:")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
            elif action == "interactive":
                interactive_query()
            else:
                print(f"Unknown command: {action}")
                print(
                    "Available commands: metrics, search, history, summary, financials, compare, top, stats, interactive"
                )
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("\nDatabase stats:", get_database_stats())
        print(
            "\nRun with 'interactive' argument for interactive mode, or use commands directly:"
        )
        print("  py database.py metrics <table>")
        print("  py database.py search <query>")
        print("  py database.py history <symbol>")
        print("  py database.py stats")
