from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Utils.backtester import (
    DEFAULT_MAGIC_GATE_VARIANTS,
    VectorBacktester,
)


def test_label_yearly_regimes_assigns_expected_buckets():
    bt = VectorBacktester()
    features = pd.DataFrame(
        [
            {
                "Year_Bucket": "2022",
                "Calendar_Year": 2022,
                "Trade_Start": "2022-01-03",
                "Trade_End": "2022-12-30",
                "spy_return": 0.03,
                "qqq_return": -0.01,
                "iwd_return": 0.08,
                "iwf_return": -0.04,
                "growth_minus_value": -0.12,
                "value_minus_growth": 0.12,
                "irx_mean_annual": 0.01,
            },
            {
                "Year_Bucket": "2023",
                "Calendar_Year": 2023,
                "Trade_Start": "2023-01-03",
                "Trade_End": "2023-12-29",
                "spy_return": -0.14,
                "qqq_return": -0.10,
                "iwd_return": -0.02,
                "iwf_return": -0.04,
                "growth_minus_value": -0.02,
                "value_minus_growth": 0.02,
                "irx_mean_annual": 0.03,
            },
            {
                "Year_Bucket": "2024",
                "Calendar_Year": 2024,
                "Trade_Start": "2024-01-02",
                "Trade_End": "2024-12-31",
                "spy_return": 0.28,
                "qqq_return": 0.34,
                "iwd_return": 0.10,
                "iwf_return": 0.25,
                "growth_minus_value": 0.15,
                "value_minus_growth": -0.15,
                "irx_mean_annual": 0.05,
            },
            {
                "Year_Bucket": "2025_YTD",
                "Calendar_Year": 2025,
                "Trade_Start": "2025-01-02",
                "Trade_End": "2025-06-30",
                "spy_return": 0.10,
                "qqq_return": 0.12,
                "iwd_return": 0.06,
                "iwf_return": 0.08,
                "growth_minus_value": 0.02,
                "value_minus_growth": -0.02,
                "irx_mean_annual": 0.02,
            },
        ]
    )

    out = bt._label_yearly_regimes(features, quantile_low=0.33, quantile_high=0.67)
    by_year = out.set_index("Year_Bucket")

    assert by_year.loc["2023", "Market_Regime"] == "Risk-off"
    assert by_year.loc["2024", "Market_Regime"] == "Bull growth"
    assert bool(by_year.loc["2022", "Value_Rotation"]) is True
    assert by_year.loc["2024", "Rate_Regime"] == "High rate"
    assert by_year.loc["2022", "Rate_Regime"] == "Low rate"


def test_validate_required_proxies_fails_fast_with_guidance():
    bt = VectorBacktester()
    bt.price_matrix = pd.DataFrame(
        {"SPY": [100.0, 101.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    with pytest.raises(ValueError) as exc:
        bt._validate_required_proxies(["SPY", "QQQ", "IWD", "IWF", "^IRX"])

    msg = str(exc.value)
    assert "Missing required proxy symbols" in msg
    assert 'py -m data.data SPY QQQ IWD IWF "^IRX" --exchange CUSTOM' in msg


def test_magic_gate_variants_apply_masks_and_cash_fill(monkeypatch):
    bt = VectorBacktester()
    snapshot = pd.DataFrame({"EBIT": [1.0]}, index=["AAA"])
    panel = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "return": 0.10,
                "RV_Gate_Distress": False,
                "RV_Gate_Tail": False,
                "RV_Gate_Momentum": False,
                "RV_Gate_Count": 1,
            },
            {
                "symbol": "BBB",
                "return": -0.20,
                "RV_Gate_Distress": True,
                "RV_Gate_Tail": False,
                "RV_Gate_Momentum": False,
                "RV_Gate_Count": 1,
            },
            {
                "symbol": "CCC",
                "return": 0.00,
                "RV_Gate_Distress": False,
                "RV_Gate_Tail": True,
                "RV_Gate_Momentum": False,
                "RV_Gate_Count": 1,
            },
            {
                "symbol": "DDD",
                "return": np.nan,
                "RV_Gate_Distress": False,
                "RV_Gate_Tail": False,
                "RV_Gate_Momentum": True,
                "RV_Gate_Count": 1,
            },
        ]
    )

    monkeypatch.setattr(
        bt,
        "_build_asset_panel_for_period",
        lambda *args, **kwargs: panel.copy(),
    )

    out = bt._evaluate_magic_gate_variants_for_period(
        start_date="2024-01-01",
        end_date="2024-02-01",
        regime="Bull growth",
        snapshot=snapshot,
        top_n=10,
    )

    assert out["Magic (raw)"]["monthly_return"] == pytest.approx((-0.1 / 3.0), rel=1e-6)
    assert out["Magic + Distress Gate"]["monthly_return"] == pytest.approx(0.05, rel=1e-6)
    assert out["Magic + Tail Gate"]["monthly_return"] == pytest.approx(-0.05, rel=1e-6)
    assert out["Magic + Momentum Gate"]["monthly_return"] == pytest.approx((-0.1 / 3.0), rel=1e-6)
    assert pd.isna(out["Magic + Full Risk Vector"]["monthly_return"])
    assert out["Magic + Full Risk Vector"]["is_cash_fill"] is True


def test_compute_strategy_risk_metrics_matches_expected_math():
    bt = VectorBacktester()
    returns = np.array([0.10, 0.00, -0.05, 0.02], dtype=float)
    rf = np.array([0.001, 0.001, 0.001, 0.001], dtype=float)

    monthly_df = pd.DataFrame(
        {
            "Strategy": ["S1"] * 4,
            "Period_Start": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
            "Period_End": ["2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"],
            "Year_Bucket": ["2024"] * 4,
            "Monthly_Return": returns,
            "Monthly_RF": rf,
        }
    )

    metrics = bt.compute_strategy_risk_metrics(monthly_df, rf_symbol="^IRX")
    row = metrics.loc[metrics["Strategy"] == "S1"].iloc[0]

    equity_curve = np.cumprod(1.0 + returns)
    expected_total = equity_curve[-1] - 1.0
    expected_cagr = equity_curve[-1] ** (12.0 / len(returns)) - 1.0
    expected_vol = np.std(returns, ddof=1) * np.sqrt(12.0)
    expected_sharpe = ((returns - rf).mean() / np.std(returns, ddof=1)) * np.sqrt(12.0)
    running_peak = np.maximum.accumulate(equity_curve)
    expected_max_dd = np.min((equity_curve / running_peak) - 1.0)
    expected_worst_year = expected_total

    assert row["Total_Return"] == pytest.approx(expected_total, rel=1e-6)
    assert row["CAGR"] == pytest.approx(expected_cagr, rel=1e-6)
    assert row["Volatility"] == pytest.approx(expected_vol, rel=1e-6)
    assert row["Sharpe"] == pytest.approx(expected_sharpe, rel=1e-6)
    assert row["Max_Drawdown"] == pytest.approx(expected_max_dd, rel=1e-6)
    assert row["Worst_Month"] == pytest.approx(-0.05, rel=1e-6)
    assert row["Worst_Year"] == pytest.approx(expected_worst_year, rel=1e-6)


def test_compute_regime_strategy_metrics_uses_tie_breakers():
    bt = VectorBacktester()
    rows = []
    returns = [0.02, 0.01, 0.03]
    for strategy in ("Alpha", "Beta"):
        for idx, ret in enumerate(returns, start=1):
            rows.append(
                {
                    "Strategy": strategy,
                    "Period_Start": f"2024-0{idx}-01",
                    "Period_End": f"2024-0{idx+1}-01",
                    "Year_Bucket": "2024",
                    "Monthly_Return": ret,
                    "Monthly_RF": 0.0,
                    "Market_Regime": "Bull growth",
                    "Value_Rotation": True,
                    "Rate_Regime": "High rate",
                }
            )
    monthly_df = pd.DataFrame(rows)

    all_metrics, winners = bt.compute_regime_strategy_metrics(
        monthly_df,
        regime_labels_df=pd.DataFrame(),
        min_months=3,
    )

    bull = all_metrics[all_metrics["Regime_Bucket"] == "Bull growth"]
    assert set(bull["Strategy"]) == {"Alpha", "Beta"}
    assert bull["Score"].notna().all()

    bull_winner = winners[winners["Regime_Bucket"] == "Bull growth"].iloc[0]
    assert bull_winner["Strategy"] == "Alpha"


def test_run_regime_tournament_exports_csv_contract(monkeypatch, tmp_path):
    bt = VectorBacktester()
    dates = pd.date_range("2022-01-01", "2025-06-30", freq="MS")
    bt.price_matrix = pd.DataFrame(
        {
            "SPY": np.linspace(100.0, 140.0, len(dates)),
            "QQQ": np.linspace(100.0, 150.0, len(dates)),
            "IWD": np.linspace(100.0, 130.0, len(dates)),
            "IWF": np.linspace(100.0, 145.0, len(dates)),
            "^IRX": np.linspace(2.0, 5.0, len(dates)),
        },
        index=dates,
    )
    bt.financial_data = pd.DataFrame(
        {"EBIT": [1.0]},
        index=pd.MultiIndex.from_tuples(
            [("AAA", pd.Timestamp("2021-12-31"))],
            names=["symbol", "period"],
        ),
    )

    def fake_combo(start_date, end_date, strategies, top_n_per_strat=5, verbose=True):
        out_ret = 0.01 if len(strategies) == 1 else 0.015
        if str(start_date).endswith("-03-01"):
            out_ret = np.nan
        return {
            "period": f"{start_date}->{end_date}",
            "return": out_ret,
            "selected_count": 10,
            "picks_count": 10 if pd.notna(out_ret) else 0,
            "status": "ok" if pd.notna(out_ret) else "no_valid_prices",
        }

    def fake_gate(*args, **kwargs):
        out = {}
        for name in DEFAULT_MAGIC_GATE_VARIANTS:
            out[name] = {
                "monthly_return": 0.012 if name == "Magic (raw)" else 0.010,
                "selected_count": 8,
                "priced_count": 8,
                "candidate_count": 10,
                "status": "ok",
                "is_cash_fill": False,
            }
        return out

    monkeypatch.setattr(bt, "run_combo_backtest", fake_combo)
    monkeypatch.setattr(bt, "_evaluate_magic_gate_variants_for_period", fake_gate)
    monkeypatch.setattr(
        bt,
        "get_valid_snapshot",
        lambda *args, **kwargs: pd.DataFrame({"EBIT": [1.0]}, index=["AAA"]),
    )

    outputs = bt.run_regime_tournament(
        start_date="2022-01-01",
        end_date="2025-06-30",
        output_dir=str(tmp_path),
        csv_only=True,
        show_progress=False,
    )

    expected_keys = {
        "regime_labels",
        "tournament_monthly_returns",
        "strategy_risk_metrics",
        "regime_strategy_metrics",
        "regime_winners",
        "magic_gate_comparison",
    }
    assert expected_keys.issubset(outputs.keys())

    assert {"Year_Bucket", "Market_Regime", "Value_Rotation", "Rate_Regime"}.issubset(
        outputs["regime_labels"].columns
    )
    assert {"Strategy", "Monthly_Return", "Year_Bucket"}.issubset(
        outputs["tournament_monthly_returns"].columns
    )
    assert {"Strategy", "CAGR", "Volatility", "Sharpe", "Max_Drawdown"}.issubset(
        outputs["strategy_risk_metrics"].columns
    )
    assert {"Regime_Bucket", "Strategy", "Score"}.issubset(
        outputs["regime_strategy_metrics"].columns
    )
    assert {"Strategy", "delta_cagr", "delta_max_drawdown", "publishable_flag"}.issubset(
        outputs["magic_gate_comparison"].columns
    )

    expected_files = [
        "regime_labels.csv",
        "tournament_monthly_returns.csv",
        "strategy_risk_metrics.csv",
        "regime_strategy_metrics.csv",
        "regime_winners.csv",
        "magic_gate_comparison.csv",
    ]
    for filename in expected_files:
        assert (Path(tmp_path) / filename).exists()

    march_rows = outputs["tournament_monthly_returns"][
        outputs["tournament_monthly_returns"]["Period_Start"].str.endswith("-03-01")
    ]
    assert not march_rows.empty
    assert bool(march_rows["Is_Cash_Fill"].any())
    cash_rows = march_rows[march_rows["Is_Cash_Fill"] == True]
    assert not cash_rows.empty
    expected_cash = cash_rows["Monthly_RF"] - 0.0005
    assert np.allclose(cash_rows["Monthly_Return"], expected_cash)
    assert bool(cash_rows["Raw_Strategy_Return"].isna().all())


def test_run_regime_tournament_applies_cash_fill_for_magic_gate_missing_returns(
    monkeypatch, tmp_path
):
    bt = VectorBacktester()
    dates = pd.date_range("2022-01-01", "2022-04-01", freq="MS")
    bt.price_matrix = pd.DataFrame(
        {
            "SPY": np.linspace(100.0, 104.0, len(dates)),
            "QQQ": np.linspace(100.0, 106.0, len(dates)),
            "IWD": np.linspace(100.0, 103.0, len(dates)),
            "IWF": np.linspace(100.0, 105.0, len(dates)),
            "^IRX": np.linspace(2.0, 2.5, len(dates)),
        },
        index=dates,
    )
    bt.financial_data = pd.DataFrame(
        {"EBIT": [1.0]},
        index=pd.MultiIndex.from_tuples(
            [("AAA", pd.Timestamp("2021-12-31"))],
            names=["symbol", "period"],
        ),
    )

    monkeypatch.setattr(
        bt,
        "run_combo_backtest",
        lambda *args, **kwargs: {
            "period": "x",
            "return": 0.01,
            "selected_count": 10,
            "picks_count": 10,
            "status": "ok",
        },
    )
    monkeypatch.setattr(
        bt,
        "_evaluate_magic_gate_variants_for_period",
        lambda *args, **kwargs: {
            name: {
                "monthly_return": np.nan,
                "selected_count": 0,
                "priced_count": 0,
                "candidate_count": 10,
                "status": "no_survivors",
                "is_cash_fill": True,
            }
            for name in DEFAULT_MAGIC_GATE_VARIANTS
        },
    )
    monkeypatch.setattr(
        bt,
        "get_valid_snapshot",
        lambda *args, **kwargs: pd.DataFrame({"EBIT": [1.0]}, index=["AAA"]),
    )

    outputs = bt.run_regime_tournament(
        start_date="2022-01-01",
        end_date="2022-04-01",
        output_dir=str(tmp_path),
        csv_only=True,
        show_progress=False,
    )

    gate_rows = outputs["tournament_monthly_returns"][
        outputs["tournament_monthly_returns"]["Source"] == "magic_gate"
    ]
    assert not gate_rows.empty
    assert bool(gate_rows["Is_Cash_Fill"].all())
    assert bool(gate_rows["Raw_Strategy_Return"].isna().all())
    expected_cash = gate_rows["Monthly_RF"] - 0.0005
    assert np.allclose(gate_rows["Monthly_Return"], expected_cash)


def test_run_regime_tournament_future_only_symbol_does_not_create_flat_invested_return(
    monkeypatch, tmp_path
):
    bt = VectorBacktester()
    idx = pd.to_datetime(
        [
            "2022-01-01",
            "2022-02-01",
            "2022-03-01",
            "2022-04-01",
            "2025-04-17",
            "2025-05-01",
        ]
    )
    bt.price_matrix = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0, 103.0, 150.0, 151.0],
            "QQQ": [100.0, 102.0, 103.0, 104.0, 170.0, 171.0],
            "IWD": [100.0, 100.5, 101.0, 101.5, 140.0, 141.0],
            "IWF": [100.0, 101.0, 102.0, 103.0, 160.0, 161.0],
            "^IRX": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            "FUTR": [np.nan, np.nan, np.nan, np.nan, 30.0, 31.0],
        },
        index=idx,
    )
    bt.financial_data = pd.DataFrame(
        {"EBIT": [1.0]},
        index=pd.MultiIndex.from_tuples(
            [("FUTR", pd.Timestamp("2021-12-31"))],
            names=["symbol", "period"],
        ),
    )

    snapshot = pd.DataFrame(
        {
            "EBIT": [1.0],
            "OrdinarySharesNumber": [10_000.0],
            "TotalAssets": [1_000_000.0],
            "TotalLiabilitiesNetMinorityInterest": [300_000.0],
            "WorkingCapital": [100_000.0],
            "RetainedEarnings": [50_000.0],
            "TotalRevenue": [200_000.0],
            "OperatingCashFlow": [10_000.0],
            "CapitalExpenditure": [-2_000.0],
            "StockholdersEquity": [700_000.0],
        },
        index=["FUTR"],
    )

    monkeypatch.setattr(
        bt,
        "run_combo_backtest",
        lambda *args, **kwargs: {
            "period": "x",
            "return": 0.01,
            "selected_count": 10,
            "picks_count": 10,
            "status": "ok",
        },
    )
    monkeypatch.setattr(bt, "get_valid_snapshot", lambda *args, **kwargs: snapshot.copy())
    monkeypatch.setattr(
        bt,
        "apply_strategy",
        lambda snapshot, strategy, top_n=10, trade_date=None: ["FUTR"]
        if strategy == "magic_formula"
        else [],
    )

    outputs = bt.run_regime_tournament(
        start_date="2022-01-01",
        end_date="2022-04-01",
        output_dir=str(tmp_path),
        csv_only=True,
        show_progress=False,
    )

    tail_rows = outputs["tournament_monthly_returns"][
        outputs["tournament_monthly_returns"]["Strategy"] == "Magic + Tail Gate"
    ]
    assert not tail_rows.empty
    assert bool(tail_rows["Is_Cash_Fill"].all())
    assert bool(tail_rows["Raw_Strategy_Return"].isna().all())
    expected_cash = tail_rows["Monthly_RF"] - 0.0005
    assert np.allclose(tail_rows["Monthly_Return"], expected_cash)


def test_run_regime_tournament_allows_missing_optional_proxies(monkeypatch, tmp_path):
    bt = VectorBacktester()
    dates = pd.date_range("2022-01-01", "2025-06-30", freq="MS")
    bt.price_matrix = pd.DataFrame(
        {
            "SPY": np.linspace(100.0, 140.0, len(dates)),
            "QQQ": np.linspace(100.0, 150.0, len(dates)),
        },
        index=dates,
    )
    bt.financial_data = pd.DataFrame(
        {"EBIT": [1.0]},
        index=pd.MultiIndex.from_tuples(
            [("AAA", pd.Timestamp("2021-12-31"))],
            names=["symbol", "period"],
        ),
    )

    monkeypatch.setattr(
        bt,
        "run_combo_backtest",
        lambda *args, **kwargs: {
            "period": "x",
            "return": 0.01,
            "selected_count": 10,
            "picks_count": 10,
            "status": "ok",
        },
    )
    monkeypatch.setattr(
        bt,
        "_evaluate_magic_gate_variants_for_period",
        lambda *args, **kwargs: {
            name: {
                "monthly_return": 0.009,
                "selected_count": 9,
                "priced_count": 9,
                "candidate_count": 10,
                "status": "ok",
                "is_cash_fill": False,
            }
            for name in DEFAULT_MAGIC_GATE_VARIANTS
        },
    )
    monkeypatch.setattr(
        bt,
        "get_valid_snapshot",
        lambda *args, **kwargs: pd.DataFrame({"EBIT": [1.0]}, index=["AAA"]),
    )

    outputs = bt.run_regime_tournament(
        start_date="2022-01-01",
        end_date="2025-06-30",
        output_dir=str(tmp_path),
        csv_only=True,
        show_progress=False,
    )

    assert not outputs["regime_labels"].empty
    assert bool((outputs["regime_labels"]["Value_Rotation"] == False).all())
    assert bool((outputs["regime_labels"]["Rate_Regime"] == "Mid rate").all())
    assert bool((outputs["tournament_monthly_returns"]["Monthly_RF"] == 0.0).all())
