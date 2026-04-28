# Utils/

Shared utilities: point-in-time backtester, MAD-based scoring, and ticker list management.

---

## backtester.py

Point-in-time backtester — avoids look-ahead bias by only using financials dated before `(trade_date − reporting_lag)`.

### Key Classes

| Class | Purpose |
|-------|---------|
| `BacktesterProfile` | Profile name, strategy list, top_n, gates to enable |
| `BacktesterConfig` | Global settings (profiles, universe, default profile) |
| `BacktestResult` | trade_date, tickers, entry/exit prices, return_pct, signals |
| `VectorBacktester` | Main engine — `run_backtest()`, `tournament()` |

### Constants

| Constant | Value |
|----------|-------|
| `DEFAULT_REPORTING_LAG_DAYS` | 90 |
| `DEFAULT_PROXY_SYMBOLS` | SPY, QQQ, IWD, IWF, ^IRX |
| Supported strategies | magic_formula, moat, fcf_yield, quality, diamonds_in_dirt, cannibal, piotroski_f_score, rule_of_40, super_margins |

### Regime Detection

`detect_current_regime(research_dir, lookback_days=365)` — standalone function, can be called from `main.py`:
- Fetches SPY/IWD/IWF/^IRX trailing prices from DB
- Derives thresholds from `research_outputs/regime_labels.csv` quantiles
- Returns dict with `Market_Regime`, `Value_Rotation`, `Rate_Regime`, `regime_label`

### Configuration

Profiles defined in `Utils/backtester_config.json`. Load with `load_profile(name)`.

### Research Outputs

Backtester writes to `research_outputs/`:
- `strategy_risk_metrics.csv` — Sharpe ratios per strategy (drives `top_n` weighting in main pipeline)
- `regime_winners.csv` — best strategy per regime bucket
- `magic_gate_comparison.csv` — gate CAGR delta analysis

---

## scoring.py

Outlier-resistant scoring via **Median Absolute Deviation (MAD)**. Standard Z-score breaks with outliers; MAD doesn't.

### `robust_score(series, higher_is_better=True)`

1. Compute median + MAD
2. Z-score: `z = 0.6745 × (x − median) / MAD`
3. Clip to ±3
4. Normalize to 0–100
5. Flip if `higher_is_better=False`

### `combined_robust_score(df, metric_weights)`

Weighted composite of multiple `robust_score()` calls.

### Conventions

- Use `higher_is_better=False` for P/E ratio (lower is better).
- Clipping at ±3 MAD prevents one outlier from squashing the whole distribution.

---

## ticker.py

Fetches and caches ticker lists from multiple sources.

### `TickerManager`

| Method | Source |
|--------|--------|
| `get_sp500()` | Wikipedia S&P 500 list |
| `get_nasdaq()` | NASDAQ exchange list |
| `get_nyse()` | NYSE exchange list |
| `get_all_us()` | Union of all above |

- Caches results to `Utils/*.txt` files.
- Filters invalid characters: `^`, `/`, `$`, `-`.

### Universe Files (checked in)

- `nyse_tickers.txt` — default pipeline universe
- `nasdaq_tickers.txt`
- `sp500_tickers.txt`
