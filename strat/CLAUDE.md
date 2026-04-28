# strat/

Stock screening strategies. `StrategyEngine` runs one or more strategies against the financial DB and returns ranked candidate lists.

## Key Classes

| Class | Purpose |
|-------|---------|
| `BaseStrategy` | Abstract base — all strategies implement `screen(df, top_n)` |
| `StrategyConfig` | Name, description, required metrics, min market cap |
| `ScreeningResult` | Dataclass: strategy_name, trade_date, tickers, scores |
| `UniverseFilter` | Min market cap filter; excludes warrants/ADRs |
| `MomentumFilter` | 200-day MA filter (configurable `days` parameter) |
| `StrategyEngine` | Orchestrates multi-strategy runs; exposes `_build_strategy_decisions()` |

## Strategies

| Strategy | Key Signal | Filter Criteria |
|----------|-----------|-----------------|
| `MagicFormulaStrategy` | ROIC + Earnings Yield rank | None (rank-based) |
| `MoatStrategy` | Gross Margin rank | Gross Margin > 60% |
| `FCFYieldStrategy` | FCF / Market Cap | FCF > 0 |
| `CannibalStrategy` | Share buyback intensity | Shares ↓ 3%+ YoY + FCF > 0 |
| `QualityStrategy` | Composite ROE + Op Margin + FCF Margin | None (rank-based) |
| `PiotroskiStrategy` | 9-point F-Score | F-Score ≥ 7 |
| `DiamondsInDirtStrategy` | Value + quality turnaround | Low P/B + improving fundamentals |

## DEFAULT_COMBOS (multi-strategy portfolios)

Defined in `Portfolio.DEFAULT_COMBOS` — includes: `Greenblatt+`, `Cash Cow`, `Fortress`, `Piotroski`.

## Backtester Feedback Integration

`_build_strategy_decisions()` accepts `top_n_per_strategy: int | dict[str, int]` to allow Sharpe-weighted allocation per strategy (loaded from `research_outputs/strategy_risk_metrics.csv` in `main.py`).

## Conventions

- All strategies call `prepare_data()` to coerce numeric columns before screening.
- `top_n` selection uses `.nsmallest()` on combined rank or `.nlargest()` on score, depending on strategy.
- Strategy name on the candidate row drives DCF assumptions in `valuation/valuation_projector.py` — keep names consistent with `STRATEGY_VALUATION_PARAMS` keys.
