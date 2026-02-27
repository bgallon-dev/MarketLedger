import pandas as pd

from pyfinancial.risk.risk_vector import (
    RiskVectorConfig,
    attach_risk_vectors,
    build_risk_vector,
    gate_stage_and_reason,
)


def test_build_risk_vector_maps_signals_and_gates():
    row = pd.Series(
        {
            "Altman Z-Score": 1.2,
            "Distress Risk": "DISTRESS (Risk)",
            "Tail_Risk": "Heavy",
            "Momentum_Above_MA": False,
            "Momentum_MA_Gap_Pct": -8.0,
            "Valuation Sanity": "Passed",
            "Current Price": 100.0,
            "Fair Value": 120.0,
        }
    )
    rv = build_risk_vector(row, RiskVectorConfig())

    assert rv.distress_bucket == "DISTRESS (Risk)"
    assert rv.tail_bucket == "Heavy"
    assert rv.momentum_regime == "BEARISH"
    assert rv.gate_distress is True
    assert rv.gate_tail is True
    assert rv.gate_momentum is True
    assert rv.gate_valuation is False
    assert rv.gate_count == 3


def test_gate_stage_precedence():
    row = pd.Series(
        {
            "Distress Risk": "SAFE",
            "Tail_Risk": "Normal",
            "Momentum_Above_MA": True,
            "Valuation Sanity": "Passed",
            "Current Price": 50.0,
            "Fair Value": 60.0,
        }
    )
    rv = build_risk_vector(row, RiskVectorConfig())
    stage, reason = gate_stage_and_reason(rv)
    assert stage == "selected"
    assert reason == ""


def test_attach_risk_vectors_adds_rv_columns_and_signal():
    df = pd.DataFrame(
        [
            {
                "Ticker": "ABC",
                "Distress Risk": "SAFE",
                "Tail_Risk": "Normal",
                "Momentum_Above_MA": True,
                "Momentum_MA_Gap_Pct": 5.0,
                "Valuation Sanity": "Passed",
                "Current Price": 100.0,
                "Fair Value (Bear)": 110.0,
                "Fair Value (Base)": 130.0,
                "Fair Value": 125.0,
            }
        ]
    )
    out = attach_risk_vectors(df, RiskVectorConfig(), include_signal=True)

    assert "RV_Version" in out.columns
    assert "RV_Gate_Count" in out.columns
    assert out.loc[0, "RV_Gate_Count"] == 0
    assert out.loc[0, "Investment Signal"] in {"Strong Buy", "Speculative Buy", "Hold"}

