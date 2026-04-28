import sqlite3
from pathlib import Path

_MARKET_LEDGER = Path(__file__).parent.parent.parent


def get_financial_db() -> sqlite3.Connection:
    path = _MARKET_LEDGER / "database" / "financial_data.db"
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def get_paper_db() -> sqlite3.Connection:
    path = _MARKET_LEDGER / "database" / "paper_trading.db"
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn
