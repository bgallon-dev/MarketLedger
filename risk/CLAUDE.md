# risk/

Unified risk profile system. Merges signals from forensic, distribution, valuation, and momentum modules into a single `RiskVector` per asset and applies gate logic.

## Key Classes

### `RiskVectorConfig`
Gate on/off flags and bucket definitions.

| Field | Default |
|-------|---------|
| `enable_distress_gate` | `True` |
| `enable_tail_gate` | `True` |
| `enable_momentum_gate` | `True` |
| `enable_valuation_gate` | `True` |
| `distress_safe_bucket` | `"SAFE"` |
| `tail_heavy_bucket` | `"Heavy"` |

### `RiskVector`
Per-asset risk profile. Key fields:

| Field | Source |
|-------|--------|
| `altman_z_score`, `distress_bucket` | `forensic/` |
| `tail_bucket`, `tail_a`, `tail_b` | `distro/` |
| `momentum_above_ma200`, `momentum_ma_gap_pct`, `momentum_regime` | price history |
| `valuation_sanity_passed`, `valuation_sanity_ratio`, `valuation_sanity_reason` | `valuation/valuation_sanity_gate.py` |
| `gate_distress`, `gate_tail`, `gate_momentum`, `gate_valuation` | derived |
| `gate_count` | sum of failed gates |

## Key Functions

| Function | Purpose |
|----------|---------|
| `build_risk_vector(row)` | Create `RiskVector` from a DataFrame row |
| `attach_risk_vectors(df, config)` | Append `RV_*` columns to DataFrame |
| `risk_vector_to_columns(rv)` | Flatten `RiskVector` to dict for DataFrame insertion |
| `gate_stage_and_reason(rv, config)` | Returns `(stage, reason)` for first failing gate |
| `classify_investment_signal(rv, price, fair_value)` | Maps risk gates + MOS → investment signal |

## Gate Precedence

Gates are evaluated in this order — first failure is the `DecisionStage`:
1. `distress` — blocks if bucket is `DISTRESS (Risk)`
2. `valuation_sanity` — blocks if sanity check failed
3. `tail_risk` — blocks if bucket is `Heavy`
4. `momentum` — blocks if not above MA200

## Investment Signals

| Signal | Condition |
|--------|-----------|
| `Strong Buy` | SAFE + Bear FV ≥ price + Base MOS ≥ 20% |
| `Speculative Buy` | Base MOS ≥ 20%, but Bear FV < price |
| `Hold` | 0% < Base MOS < 20% |
| `Overvalued` | Base MOS ≤ 0% |
| `Caution` / `Avoid` / `Needs Review` | Various partial-signal states |

## Conventions

- `_normalize_distress_bucket()` / `_normalize_tail_bucket()` are case-insensitive — bucket strings from different sources are safe to pass directly.
- `momentum_regime`: BULLISH = above MA200 with gap ≥ 3%; BEARISH = below with gap ≤ -3%; NEUTRAL otherwise.
- DataFrame output columns are prefixed `RV_` (e.g., `RV_Gate_Count`, `RV_Distress_Bucket`).
