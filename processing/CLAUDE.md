# processing/

Standalone fundamental screener. Not part of the main pipeline DAG — used for ad-hoc S&P 500 screening.

## TickerScreener

Screens individual stocks against hard thresholds:

| Metric | Threshold |
|--------|-----------|
| P/E Ratio (TTM) | < 25 |
| Return on Equity (TTM) | > 15% |
| Gross Margin (TTM) | > 40% |
| Free Cash Flow Growth (YoY) | > 0% |
| Net Income Growth (YoY) | > 0% |
| Operating Margin (MRQ) | > 20% |

## Key Methods

| Method | Purpose |
|--------|---------|
| `get_ttm_value(df, metric)` | Sum of last 4 quarters; forward-fills if <4 available |
| `get_most_recent_value(df, metric)` | First non-NaN from period columns (newest first) |
| `calculate_pe_ratio()` | TTM EPS vs current price |
| `calculate_roe()` | Net income / avg equity |
| `calculate_gross_margin()` | TTM gross profit / revenue |
| `screen_ticker(ticker)` | Run all criteria, return dict of metric values + pass/fail flags |

## Conventions

- TTM = sum of 4 most recent quarterly periods.
- Duplicate metric names are de-duped by keeping the first occurrence.
- Returns a dict with both raw metric values and boolean quality flags per criterion.
- Not wired into `main.py` — run directly for one-off screening.
