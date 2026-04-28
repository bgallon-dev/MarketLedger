# distro/

Fits stock return distributions to a **Kumaraswamy-Laplace** model for tail risk classification. Output feeds `RiskVector.tail_bucket` in `risk/risk_vector.py`.

## Key Classes

### `KumaraswamyLaplace`
Four-parameter heavy-tail distribution: μ (location), σ (scale), `a` (right tail shape), `b` (left tail shape).

| Method | Purpose |
|--------|---------|
| `fit(returns)` | MLE parameter estimation |
| `pdf()` / `cdf()` | Density and cumulative density |
| `interpret_params()` | Returns tail classification + AIC/BIC |

**Tail interpretation:**
- `a < 0.8` or `b < 0.8` → **Heavy** tails (increase VaR 1.5×–2×)
- `a > 1.5` and `b > 1.5` → **Thin** tails
- Otherwise → **Normal** tails

### `NormalDistribution` / `LaplaceDistribution`
Baseline comparisons for AIC/BIC model selection.

## Key Functions

| Function | Purpose |
|----------|---------|
| `fit_distribution_for_ticker(ticker, returns)` | Fit Kum-Laplace to one ticker's 252-day returns |
| `process_tickers_from_buy_list(df)` | Batch fit with progress reporting |
| `compare_distributions(returns)` | Compare Normal vs Laplace vs Kum-Laplace fits |

## Output Columns

`ticker, mu, sigma, a_shape, b_shape, aic, bic, n_observations, tail_bucket`

## Conventions

- Uses 252 trading days (1 year) of daily returns as input.
- `tail_bucket` values map directly to `RiskVector.tail_bucket`: `"Light"`, `"Moderate"`, `"Heavy"`.
- Heavy tail classification triggers the tail gate in `gate_stage_and_reason()`.
