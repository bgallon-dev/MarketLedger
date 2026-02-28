import numpy as np
import pandas as pd

from Utils.backtester import VectorBacktester


def test_get_price_returns_nan_when_symbol_starts_trading_after_target_date():
    bt = VectorBacktester(price_staleness_days=7)
    bt.price_matrix = pd.DataFrame(
        {"FUTR": [30.0, 31.0]},
        index=pd.to_datetime(["2025-04-17", "2025-04-18"]),
    )

    px = bt.get_price("FUTR", "2024-04-01")
    bulk = bt.get_prices_bulk(["FUTR"], "2024-04-01")

    assert pd.isna(px)
    assert pd.isna(bulk.iloc[0])


def test_get_price_returns_nan_when_last_observation_is_staler_than_cap():
    bt = VectorBacktester(price_staleness_days=7)
    bt.price_matrix = pd.DataFrame(
        {"STALE": [10.0]},
        index=pd.to_datetime(["2024-03-20"]),
    )

    px = bt.get_price("STALE", "2024-04-01")
    assert pd.isna(px)


def test_asof_pricing_never_uses_future_observation_within_nearest_window():
    bt = VectorBacktester(price_staleness_days=7)
    bt.price_matrix = pd.DataFrame(
        {
            "FUTR_ONLY": [np.nan, 50.0],
            "HAS_PAST": [40.0, 41.0],
        },
        index=pd.to_datetime(["2024-03-29", "2024-04-03"]),
    )

    futr_px = bt.get_price("FUTR_ONLY", "2024-04-01")
    past_px = bt.get_price("HAS_PAST", "2024-04-01")
    bulk = bt.get_prices_bulk(["FUTR_ONLY", "HAS_PAST"], "2024-04-01")

    assert pd.isna(futr_px)
    assert past_px == 40.0
    assert pd.isna(bulk.loc["FUTR_ONLY"])
    assert bulk.loc["HAS_PAST"] == 40.0
