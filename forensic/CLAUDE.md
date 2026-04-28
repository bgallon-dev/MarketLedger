# forensic/

Altman Z-Score bankruptcy risk detection. Output feeds `RiskVector.distress_bucket` and the distress gate.

## Formula

```
Z = 1.2×(Working Capital / Total Assets)
  + 1.4×(Retained Earnings / Total Assets)
  + 3.3×(EBIT / Total Assets)
  + 0.6×(Market Cap / Total Liabilities)
  + 1.0×(Sales / Total Assets)
```

## Classification

| Z-Score | Bucket |
|---------|--------|
| Z > 2.99 | `SAFE` |
| 1.81 < Z < 2.99 | `GREY ZONE` |
| Z < 1.81 | `DISTRESS (Risk)` |
| Missing data | `N/A` |

## Key Functions

| Function | Purpose |
|----------|---------|
| `run_forensic_scan(buy_list_df, prefetched=None)` | Main API — adds `Altman Z-Score` and `Distress Risk` columns |
| `_analyze_ticker(ticker, prefetched)` | Single-stock calculation |
| `_prepare_prefetched_maps(prefetched)` | Caches bulk-loaded data into per-ticker dicts |

## Conventions

- Pass `prefetched=` dict (from `prefetch_data_node`) to avoid repeated DB calls.
- Parallel workers default to `min(8, cpu_count())`.
- Input DataFrame must have a `Ticker` column.
- Only the distress gate uses this output — `GREY ZONE` passes the gate; only `DISTRESS (Risk)` trips it (when `enable_distress_gate=True`).
