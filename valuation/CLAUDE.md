# valuation/

Strategy-conditioned DCF valuation with Bear/Base/Bull scenario analysis, plus a sanity gate to catch bad valuations before they reach the risk vector.

## valuation_projector.py

### Strategy-Conditioned Assumptions (`STRATEGY_VALUATION_PARAMS`)

| Strategy | Growth | Discount | Terminal Multiple |
|----------|--------|----------|-------------------|
| `moat` | 5% | 8% | 15× |
| `cannibal` | 5% | 10% | 18× |
| `magic_formula` | 8% | 12% | 12× |
| `fcf_yield` | 6% | 10% | 12× |
| `quality` | 10% | 10% | 14× |

The `Strategy` column on each candidate row selects these parameters.

### Scenario Adjustments (`ScenarioConfig`)

| Scenario | Growth Δ | Discount Δ | Multiple Δ | Weight |
|----------|----------|------------|------------|--------|
| Bear | −3% | +2% | ×0.75 | 25% |
| Base | — | — | — | 50% |
| Bull | +3% | −1% | ×1.25 | 25% |

### Valuation Waterfall

`calculate_intrinsic_value()` tries in order:
1. **DCF** — if FCF > 0
2. **P/S** — if revenue > 0
3. **P/B** — if equity > 0
4. **N/A** — if all fail

### Investment Signal Classification

| Signal | Condition |
|--------|-----------|
| `Strong Buy` | Bear FV ≥ price AND Base MOS ≥ 20% |
| `Speculative Buy` | Base MOS ≥ 20% but Bear FV < price |
| `Hold` | 0% < Base MOS < 20% |
| `Overvalued` | Base MOS ≤ 0% |

### Key Functions

| Function | Purpose |
|----------|---------|
| `calculate_intrinsic_value(ticker, params, prefetched)` | Single-point valuation |
| `calculate_scenario_valuations(ticker, strategy, prefetched)` | Bear/Base/Bull → weighted FV |
| `classify_investment_signal(bear_fv, base_fv, price)` | Signal from scenario outputs |
| `run_valuation_scan(df, prefetched)` | Batch API — adds FV columns to DataFrame |

---

## valuation_sanity_gate.py

Multi-check gate applied before signals reach `risk_vector.py`. Returns a `SanityResult` with a typed failure code.

### Failure Codes & Behaviors

| Code | Behavior |
|------|----------|
| `NEGATIVE_FCF_DCF` | `SUPPRESS_SIGNAL` |
| `ZERO_PRICE` / `ZERO_FAIR_VALUE` | `EXCLUDE` |
| `SECTOR_EXCLUDED` (Financials/REITs) | `SUPPRESS_SIGNAL` |
| `FV_TOO_LOW` / `FV_TOO_HIGH` (outside 20%–500% of price) | `SUPPRESS_SIGNAL` |
| `FV_EXTREME_HIGH` (>1000%) | `EXCLUDE` |
| `CYCLICAL_PEAK/TROUGH_RISK` | `WARN` (soft flag only) |

### `SanityContext` Input

Pass: `ticker, fair_value, current_price, valuation_method, fcf_latest, fcf_series, total_revenue, stockholders_equity, operating_income, sector, shares_outstanding, periods_in_series`

## Conventions

- Always pass `prefetched=` to avoid repeated DB hits in batch runs.
- The sanity gate runs before the valuation gate in risk vector gate precedence.
- P/S fallback is blocked if `operating_margin < 0` (would produce a nonsense valuation).
