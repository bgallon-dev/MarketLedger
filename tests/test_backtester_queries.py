import numpy as np
import pandas as pd
import pytest

from Utils.backtester import VectorBacktester


def test_query_signal_stability_and_gate_trip_rates():
    bt = VectorBacktester()
    asset_panel = pd.DataFrame(
        [
            {
                "track": "ungated",
                "strategy": "magic_formula",
                "regime": "BULL",
                "symbol": "AAA",
                "return": 0.10,
                "Investment Signal": "Strong Buy",
                "RV_Gate_Count": 0,
                "RV_Gate_Distress": False,
                "RV_Gate_Tail": False,
                "RV_Gate_Momentum": False,
                "RV_Gate_Valuation": False,
            },
            {
                "track": "gated",
                "strategy": "magic_formula",
                "regime": "BEAR",
                "symbol": "BBB",
                "return": -0.08,
                "Investment Signal": "Caution",
                "RV_Gate_Count": 2,
                "RV_Gate_Distress": True,
                "RV_Gate_Tail": True,
                "RV_Gate_Momentum": False,
                "RV_Gate_Valuation": False,
            },
        ]
    )

    stability = bt.query_signal_stability(asset_panel)
    gate_rates = bt.query_gate_trip_rates(asset_panel)

    assert not stability.empty
    assert {"track", "strategy", "regime", "Investment Signal"}.issubset(stability.columns)
    assert not gate_rates.empty
    assert {"gate", "trip_rate", "candidates"}.issubset(gate_rates.columns)


def test_query_drawdown_failures_detects_event():
    bt = VectorBacktester()
    equity_panel = pd.DataFrame(
        [
            {
                "track": "gated",
                "strategy": "magic_formula",
                "period_start": "2024-01-01",
                "period_end": "2024-02-01",
                "regime": "BULL",
                "equity": 1.00,
                "drawdown": 0.00,
            },
            {
                "track": "gated",
                "strategy": "magic_formula",
                "period_start": "2024-02-01",
                "period_end": "2024-03-01",
                "regime": "STRESS",
                "equity": 0.82,
                "drawdown": -0.18,
            },
            {
                "track": "gated",
                "strategy": "magic_formula",
                "period_start": "2024-03-01",
                "period_end": "2024-04-01",
                "regime": "BEAR",
                "equity": 0.78,
                "drawdown": -0.22,
            },
            {
                "track": "gated",
                "strategy": "magic_formula",
                "period_start": "2024-04-01",
                "period_end": "2024-05-01",
                "regime": "NEUTRAL",
                "equity": 1.01,
                "drawdown": 0.00,
            },
        ]
    )

    failures = bt.query_drawdown_failures(equity_panel, threshold=-0.15)
    assert not failures.empty
    assert float(failures.iloc[0]["min_drawdown"]) <= -0.18


def test_compute_altman_state_returns_unknown_on_missing_fields():
    bt = VectorBacktester()
    row = pd.Series(
        {
            "TotalAssets": 1_000_000.0,
            "TotalLiabilitiesNetMinorityInterest": np.nan,
            "WorkingCapital": 200_000.0,
            "RetainedEarnings": 250_000.0,
            "EBIT": 120_000.0,
            "TotalRevenue": 900_000.0,
            "OrdinarySharesNumber": 10_000.0,
        }
    )

    score, bucket = bt._compute_altman_state(row, current_price=50.0)
    assert pd.isna(score)
    assert bucket == "Unknown"


def test_compute_altman_state_raises_on_non_numeric_required_field():
    bt = VectorBacktester()
    row = pd.Series(
        {
            "TotalAssets": 1_000_000.0,
            "TotalLiabilitiesNetMinorityInterest": "bad",
            "WorkingCapital": 200_000.0,
            "RetainedEarnings": 250_000.0,
            "EBIT": 120_000.0,
            "TotalRevenue": 900_000.0,
            "OrdinarySharesNumber": 10_000.0,
        }
    )

    with pytest.raises(TypeError):
        bt._compute_altman_state(row, current_price=50.0)


def test_compute_tail_state_converts_real_kurtosis_and_skew():
    bt = VectorBacktester()
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    bt.price_matrix = pd.DataFrame(
        {"AAA": np.linspace(100.0, 130.0, len(dates))},
        index=dates,
    )

    result = bt._compute_tail_state(
        symbol="AAA",
        date=str(dates[-1].date()),
        lookback_days=100,
        min_obs=80,
        use_cache=False,
    )

    assert result["tail_bucket"] in {"Heavy", "Normal", "Thin"}
    assert isinstance(result["tail_a"], float)
    assert isinstance(result["tail_b"], float)
