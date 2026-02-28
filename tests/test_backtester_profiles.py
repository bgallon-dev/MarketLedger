import json

import pandas as pd
import pytest

from Utils.backtester import DEFAULT_MAGIC_GATE_VARIANTS, VectorBacktester


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_profile_load_success_and_default_merge(tmp_path):
    config_path = _write_json(
        tmp_path / "profiles.json",
        {
            "version": 1,
            "profiles": {
                "default": {
                    "universe": {"strategy_min_market_cap_mm": 300},
                }
            },
        },
    )

    bt = VectorBacktester(profile_name="default", config_path=config_path)
    profile = bt.get_active_profile()

    assert profile["universe"]["strategy_min_market_cap_mm"] == 300.0
    assert profile["strategies"]["moat"]["min_total_revenue"] == 100_000_000
    assert profile["tournament"]["contenders"]


def test_profile_fail_fast_on_missing_malformed_unknown(tmp_path):
    missing_path = str(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="does not exist"):
        VectorBacktester(profile_name="default", config_path=missing_path)

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        VectorBacktester(profile_name="default", config_path=str(malformed_path))

    unknown_profile_path = _write_json(
        tmp_path / "unknown_profile.json",
        {"version": 1, "profiles": {"foo": {}}},
    )
    with pytest.raises(ValueError, match="Profile 'default' not found"):
        VectorBacktester(profile_name="default", config_path=unknown_profile_path)

    unknown_key_path = _write_json(
        tmp_path / "unknown_key.json",
        {
            "version": 1,
            "profiles": {"default": {"universe": {"bad_key": 1}}},
        },
    )
    with pytest.raises(ValueError, match="Unknown universe keys"):
        VectorBacktester(profile_name="default", config_path=unknown_key_path)


def test_global_and_strategy_override_precedence_for_cannibal(tmp_path, monkeypatch):
    config_path = _write_json(
        tmp_path / "profiles.json",
        {
            "version": 1,
            "profiles": {
                "custom": {
                    "universe": {"strategy_min_market_cap_mm": 250},
                    "strategies": {"cannibal": {"min_market_cap_mm": 450}},
                }
            },
        },
    )

    bt = VectorBacktester(profile_name="custom", config_path=config_path)
    snapshot = pd.DataFrame(
        {
            "OrdinarySharesNumber": [1_000_000.0, 1_000_000.0],
            "OrdinarySharesNumber_Prev": [1_050_000.0, 1_050_000.0],
            "OperatingCashFlow": [100.0, 100.0],
            "CapitalExpenditure": [-10.0, -10.0],
            "RepurchaseOfCapitalStock": [-5.0, -5.0],
        },
        index=["AAA", "BBB"],
    )

    monkeypatch.setattr(
        bt,
        "get_prices_bulk",
        lambda symbols, date: pd.Series({"AAA": 300.0, "BBB": 500.0}).reindex(symbols),
    )
    monkeypatch.setattr(bt, "enrich_with_yoy_data", lambda df, trade_date: df)

    picks = bt.apply_strategy(
        snapshot=snapshot,
        strategy="cannibal",
        top_n=10,
        trade_date="2024-01-01",
    )

    assert picks == ["BBB"]


def test_strategy_threshold_change_with_profile_is_deterministic(tmp_path):
    config_path = _write_json(
        tmp_path / "profiles.json",
        {
            "version": 1,
            "profiles": {
                "custom_low_rev": {
                    "strategies": {"moat": {"min_total_revenue": 50_000_000}}
                }
            },
        },
    )

    bt = VectorBacktester()
    snapshot = pd.DataFrame(
        {
            "OperatingIncome": [10.0, 20.0],
            "TotalRevenue": [80_000_000.0, 120_000_000.0],
            "MarketCap": [1000.0, 1200.0],
            "ResearchAndDevelopment": [5_000_000.0, 5_000_000.0],
        },
        index=["AAA", "BBB"],
    )

    low_rev_picks = bt.apply_strategy(
        snapshot=snapshot,
        strategy="moat",
        top_n=10,
        profile_name="custom_low_rev",
        config_path=config_path,
    )
    default_picks = bt.apply_strategy(snapshot=snapshot, strategy="moat", top_n=10)

    assert set(low_rev_picks) == {"AAA", "BBB"}
    assert default_picks == ["BBB"]


def test_run_regime_tournament_uses_profile_contenders_and_gate_top_n(tmp_path, monkeypatch):
    config_path = _write_json(
        tmp_path / "profiles.json",
        {
            "version": 1,
            "profiles": {
                "custom": {
                    "tournament": {
                        "magic_gate_top_n": 4,
                        "contenders": [
                            {
                                "name": "Only Magic",
                                "strategies": ["magic_formula"],
                                "n": 3,
                            }
                        ],
                    }
                }
            },
        },
    )

    bt = VectorBacktester()
    bt.price_matrix = pd.DataFrame(
        {"SPY": [100.0], "QQQ": [101.0]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    bt.financial_data = pd.DataFrame(
        {"EBIT": [1.0]},
        index=pd.MultiIndex.from_tuples(
            [("AAA", pd.Timestamp("2023-12-31"))],
            names=["symbol", "period"],
        ),
    )

    calls = {"combo": None, "gate_top_n": None}
    monkeypatch.setattr(
        bt,
        "_resolve_tournament_proxies",
        lambda: {"rate": "^IRX", "value": None, "growth_style": None},
    )
    monkeypatch.setattr(
        bt,
        "build_yearly_regime_labels",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "Year_Bucket": "2024",
                    "Market_Regime": "Neutral",
                    "Value_Rotation": False,
                    "Rate_Regime": "Mid rate",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        bt,
        "_generate_monthly_periods",
        lambda *args, **kwargs: [("2024-01-01", "2024-02-01")],
    )
    monkeypatch.setattr(
        bt,
        "_compute_monthly_risk_free_map",
        lambda *args, **kwargs: {("2024-01-01", "2024-02-01"): 0.0},
    )
    monkeypatch.setattr(bt, "_resolve_year_bucket", lambda *args, **kwargs: "2024")
    monkeypatch.setattr(
        bt,
        "get_valid_snapshot",
        lambda *args, **kwargs: pd.DataFrame({"EBIT": [1.0]}, index=["AAA"]),
    )

    def fake_combo(
        start_date, end_date, strategies, top_n_per_strat=5, verbose=True, **kwargs
    ):
        calls["combo"] = (list(strategies), int(top_n_per_strat))
        return {
            "period": f"{start_date}->{end_date}",
            "return": 0.01,
            "selected_count": 3,
            "picks_count": 3,
            "status": "ok",
        }

    def fake_gate(*args, **kwargs):
        calls["gate_top_n"] = int(kwargs.get("top_n", 0))
        return {
            name: {
                "monthly_return": 0.01,
                "selected_count": 3,
                "priced_count": 3,
                "candidate_count": 3,
                "status": "ok",
                "is_cash_fill": False,
            }
            for name in DEFAULT_MAGIC_GATE_VARIANTS
        }

    monkeypatch.setattr(bt, "run_combo_backtest", fake_combo)
    monkeypatch.setattr(bt, "_evaluate_magic_gate_variants_for_period", fake_gate)

    outputs = bt.run_regime_tournament(
        start_date="2024-01-01",
        end_date="2024-02-01",
        output_dir=str(tmp_path),
        csv_only=True,
        show_progress=False,
        profile_name="custom",
        config_path=config_path,
    )

    assert calls["combo"] == (["magic_formula"], 3)
    assert calls["gate_top_n"] == 4
    assert "Only Magic" in set(outputs["tournament_monthly_returns"]["Strategy"])


def test_method_level_profile_override_does_not_mutate_active_profile(tmp_path):
    config_path = _write_json(
        tmp_path / "profiles.json",
        {
            "version": 1,
            "profiles": {
                "custom_low_rev": {
                    "strategies": {"moat": {"min_total_revenue": 50_000_000}}
                }
            },
        },
    )

    bt = VectorBacktester()
    snapshot = pd.DataFrame(
        {
            "OperatingIncome": [10.0, 20.0],
            "TotalRevenue": [80_000_000.0, 120_000_000.0],
            "MarketCap": [1000.0, 1200.0],
            "ResearchAndDevelopment": [1_000_000.0, 1_000_000.0],
        },
        index=["AAA", "BBB"],
    )

    picks_override = bt.apply_strategy(
        snapshot=snapshot,
        strategy="moat",
        top_n=10,
        profile_name="custom_low_rev",
        config_path=config_path,
    )
    picks_default = bt.apply_strategy(snapshot=snapshot, strategy="moat", top_n=10)
    active = bt.get_active_profile()

    assert set(picks_override) == {"AAA", "BBB"}
    assert picks_default == ["BBB"]
    assert active["strategies"]["moat"]["min_total_revenue"] == 100_000_000
