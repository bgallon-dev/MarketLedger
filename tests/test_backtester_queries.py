import pandas as pd

from pyfinancial.Utils.backtester import VectorBacktester


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

