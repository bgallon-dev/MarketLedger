# database/

SQLite interface for all financial and price data. The DB file lives at `database/financial_data.db`.

## Schema

| Table | Key Columns |
|-------|------------|
| `tickers` | symbol (UNIQUE), exchange, sector, industry |
| `history` | ticker_id, date, open, high, low, close, volume, dividends, stock_splits |
| `balance_sheet`, `income_statement`, `cash_flow`, `financials`, `earnings` | ticker_id, metric, period, value |

All metric tables use **long format** — pivot on demand. Indices on `(ticker_id, metric)` and `period` for fast filtering.

## Key Functions

| Function | Purpose |
|----------|---------|
| `get_ticker_history(ticker)` | Price/OHLCV for one ticker |
| `get_ticker_history_bulk(tickers)` | Bulk price fetch → `{ticker: df}` |
| `get_financial_data(ticker, table)` | One financial table for one ticker |
| `get_financial_data_bulk(tickers, table)` | Bulk financial fetch → `{ticker: df}` |
| `get_metric_across_tickers(metric)` | One metric compared across all stocks |
| `compare_tickers(tickers)` | Side-by-side multi-company comparison |
| `search_tickers(query)` | Full-text search on symbol/name |
| `interactive_query()` | CLI ad-hoc query interface |

## CLI

```bash
py -m database.database stats
py -m database.database history AAPL
py -m database.database interactive
```

## Conventions

- Bulk queries chunk in batches of 500 symbols to avoid SQLite parameter limits.
- Use `get_ticker_history_bulk` / `get_financial_data_bulk` in pipeline nodes to avoid N+1 DB hits — pass the result as `prefetched=` to downstream functions.
