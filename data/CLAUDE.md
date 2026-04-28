# data/

Fetches OHLCV + financials from Yahoo Finance and writes them to SQLite (`database/financial_data.db`).

## Entry Point

```bash
py -m data.data                                    # default NYSE universe
py -m data.data AAPL MSFT --exchange CUSTOM        # specific tickers
py -m data.data --universe sp500 --universe nasdaq # combined universes
```

## Key Functions

| Function | Purpose |
|----------|---------|
| `fetch_single_ticker(ticker, exchange, db)` | Downloads all tables (history, balance_sheet, income_statement, cash_flow, financials) for one ticker |
| `fetch_tickers(ticker_map, db)` | Batch fetch with tqdm progress bar |
| `resolve_ticker_exchange_map(args)` | Parses CLI arguments into `{ticker: exchange}` dict |
| `run_data_fetch()` | Main CLI entry point |

## Conventions

- Tickers are normalized to uppercase and deduplicated before fetching.
- Falls back to NYSE exchange label if no `--exchange` is specified.
- Universe presets (`nyse`, `nasdaq`, `sp500`) read from `Utils/*.txt` files.
- Supports repeatable `--ticker-file` arguments for batch ingestion.
- Financial tables are stored in long format: `(ticker_id, metric, period, value)`.
