# tests/

Unit and integration tests for the pipeline modules.

## Running Tests

```bash
py -m pytest -q                              # all tests
py -m pytest tests/test_risk_vector.py -v   # single file
```

> **Note:** pytest may fail to start due to a Werkzeug/Dash version conflict (pre-existing issue unrelated to test logic). As a workaround, use:
> ```bash
> py -c "import py_compile; py_compile.compile('module/file.py')"  # syntax check
> py -c "from module import ClassName"                              # import check
> ```

## Test Files

| File | Covers |
|------|--------|
| `test_risk_vector.py` | `RiskVector` construction, gate logic, signal classification |
| `test_valuation.py` | DCF scenarios, waterfall fallback, signal thresholds |
| `test_forensic.py` | Altman Z-Score calculation, bucket classification |
| `test_strat.py` | Strategy screening, ranking, filter criteria |
| `test_backtester.py` | Point-in-time data isolation, return calculation |
| `test_distributions.py` | Kum-Laplace fitting, tail bucket assignment |
| `test_scoring.py` | MAD scoring, outlier clipping, composite scores |

## Conventions

- Tests use synthetic DataFrames — no live DB or network calls.
- Gate logic tests cover all four gates independently and in combination.
- Valuation tests cover all three fallback methods (DCF, P/S, P/B) and edge cases (zero FCF, negative equity).
