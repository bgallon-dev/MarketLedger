import pandas as pd

from pyfinancial.forensic import forensic_scan
from pyfinancial.valuation import valuation_projector
from pyfinancial.Utils.backtester import VectorBacktester


def _sample_balance_sheet() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "2024-12-31": {
                "TotalAssets": 1_000_000.0,
                "TotalLiabilitiesNetMinorityInterest": 300_000.0,
                "WorkingCapital": 200_000.0,
                "RetainedEarnings": 250_000.0,
                "OrdinarySharesNumber": 10_000.0,
                "StockholdersEquity": 700_000.0,
            }
        }
    )


def _sample_income_statement() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "2024-12-31": {
                "EBIT": 120_000.0,
                "TotalRevenue": 900_000.0,
            }
        }
    )


def _sample_cash_flow() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "2024-12-31": {
                "FreeCashFlow": 80_000.0,
                "OperatingCashFlow": 110_000.0,
                "CapitalExpenditure": -30_000.0,
            }
        }
    )


def _sample_history() -> pd.DataFrame:
    close = [90.0 + i * 0.5 for i in range(300)]
    dates = pd.date_range("2023-01-01", periods=len(close), freq="D")
    return pd.DataFrame({"date": dates, "close": close})


def test_forensic_scan_uses_prefetched_maps(monkeypatch):
    buy_list = pd.DataFrame([{"Ticker": "AAA", "Strategy": "magic_formula"}])
    prefetched = {
        "history_by_symbol": {"AAA": _sample_history()},
        "balance_sheet_by_symbol": {"AAA": _sample_balance_sheet()},
        "income_statement_by_symbol": {"AAA": _sample_income_statement()},
    }

    monkeypatch.setattr(
        forensic_scan,
        "get_ticker_history",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Should not call per-ticker history API when prefetched")
        ),
    )
    monkeypatch.setattr(
        forensic_scan,
        "get_financial_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Should not call per-ticker financial API when prefetched")
        ),
    )

    out = forensic_scan.run_forensic_scan(
        buy_list,
        verbose=False,
        prefetched=prefetched,
        max_workers=1,
    )
    assert not out.empty
    assert {"Ticker", "Altman Z-Score", "Distress Risk", "Price"}.issubset(out.columns)
    assert out.loc[0, "Ticker"] == "AAA"


def test_valuation_scan_uses_prefetched_maps(monkeypatch):
    buy_list = pd.DataFrame([{"Ticker": "AAA", "Strategy": "magic_formula"}])
    prefetched = {
        "history_by_symbol": {"AAA": _sample_history()},
        "cash_flow_by_symbol": {"AAA": _sample_cash_flow()},
        "balance_sheet_by_symbol": {"AAA": _sample_balance_sheet()},
        "income_statement_by_symbol": {"AAA": _sample_income_statement()},
    }

    monkeypatch.setattr(
        valuation_projector,
        "get_ticker_history",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Should not call per-ticker history API when prefetched")
        ),
    )
    monkeypatch.setattr(
        valuation_projector,
        "get_financial_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Should not call per-ticker financial API when prefetched")
        ),
    )

    out = valuation_projector.run_valuation_scan(
        buy_list,
        verbose=False,
        prefetched=prefetched,
        max_workers=1,
    )
    assert not out.empty
    assert {"Ticker", "Current Price", "Fair Value", "Valuation Sanity"}.issubset(out.columns)
    assert out.loc[0, "Ticker"] == "AAA"


def test_research_backtest_builds_base_panel_once_per_period(monkeypatch):
    bt = VectorBacktester()
    bt.price_matrix = pd.DataFrame(
        {
            "QQQ": [100.0, 102.0, 101.0, 104.0],
            "AAA": [10.0, 10.5, 10.2, 11.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
    )
    bt.financial_data = pd.DataFrame(
        {"EBIT": [1.0]},
        index=pd.MultiIndex.from_tuples([("AAA", pd.Timestamp("2023-12-31"))], names=["symbol", "period"]),
    )

    periods = [("2024-01-01", "2024-02-01"), ("2024-02-01", "2024-03-01")]
    monkeypatch.setattr(bt, "_generate_monthly_periods", lambda *_args, **_kwargs: periods)
    monkeypatch.setattr(bt, "get_valid_snapshot", lambda *_args, **_kwargs: pd.DataFrame({"x": [1]}, index=["AAA"]))
    monkeypatch.setattr(bt, "_precompute_momentum_states", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(bt, "get_price", lambda *_args, **_kwargs: 100.0)

    calls = {"count": 0}

    def _fake_panel(*_args, **kwargs):
        calls["count"] += 1
        p_start = kwargs["start_date"]
        p_end = kwargs["end_date"]
        regime = kwargs["regime"]
        return pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "strategy": "magic_formula",
                    "start_date": p_start,
                    "end_date": p_end,
                    "regime": regime,
                    "return": 0.02,
                    "RV_Gate_Count": 0,
                }
            ]
        )

    monkeypatch.setattr(bt, "_build_asset_panel_for_period", _fake_panel)

    outputs = bt.run_research_backtest(
        start_date="2024-01-01",
        end_date="2024-04-01",
        strategy="magic_formula",
        top_n=1,
        holding_months=1,
        verbose=False,
        show_progress=False,
        max_workers=2,
        use_cached_tail=True,
        use_precomputed_regimes=True,
    )

    assert calls["count"] == len(periods)
    assert not outputs["asset_panel"].empty
    assert set(outputs["asset_panel"]["track"].unique()) == {"gated", "ungated"}
