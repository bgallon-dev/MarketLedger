import pandas as pd
import pytest

import data.data as data_cli
import main as main_cli


def test_data_help_includes_explicit_options():
    help_text = data_cli.build_parser().format_help()
    assert "--ticker" in help_text
    assert "--ticker-file" in help_text
    assert "--universe" in help_text
    assert "--exchange" in help_text
    assert "--quiet" in help_text
    assert "--no-progress" in help_text


def test_data_rejects_unknown_ticker_flags():
    parser = data_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--AAPL"])


def test_data_source_union_dedupe_and_first_source_exchange(monkeypatch):
    def fake_load(ticker_file, verbose=True, strict=False):
        mapping = {
            "custom.txt": ["MSFT", "TSLA"],
            "Utils/nyse_tickers.txt": ["IBM", "AAPL"],
        }
        return mapping[ticker_file]

    monkeypatch.setattr(data_cli, "_load_tickers_from_file", fake_load)
    args = data_cli.parse_args(
        [
            "AAPL",
            "MSFT",
            "--ticker",
            "GOOG",
            "--ticker",
            "AAPL",
            "--ticker-file",
            "custom.txt",
            "--universe",
            "nyse",
            "--exchange",
            "CUSTOM",
            "--quiet",
        ]
    )

    ticker_exchange = data_cli.resolve_ticker_exchange_map(args)

    assert list(ticker_exchange.items()) == [
        ("AAPL", "CUSTOM"),
        ("MSFT", "CUSTOM"),
        ("GOOG", "CUSTOM"),
        ("TSLA", "CUSTOM"),
        ("IBM", "NYSE"),
    ]


def test_data_no_input_defaults_to_nyse_universe(monkeypatch):
    monkeypatch.setattr(
        data_cli,
        "_load_tickers_from_file",
        lambda ticker_file, verbose=True, strict=False: ["IBM", "GE"]
        if ticker_file == "Utils/nyse_tickers.txt"
        else [],
    )
    args = data_cli.parse_args([])
    ticker_exchange = data_cli.resolve_ticker_exchange_map(args)
    assert list(ticker_exchange.items()) == [("IBM", "NYSE"), ("GE", "NYSE")]


def test_data_main_passes_quiet_and_progress_flags(monkeypatch):
    captured = {}

    def fake_run_data_fetch(*args, **kwargs):
        captured.update(kwargs)
        return {"success": [], "failed": []}

    monkeypatch.setattr(data_cli, "run_data_fetch", fake_run_data_fetch)
    monkeypatch.setattr(
        data_cli,
        "resolve_ticker_exchange_map",
        lambda _args: {"AAPL": "CUSTOM"},
    )
    data_cli.main(["--quiet", "--no-progress", "--ticker", "AAPL"])

    assert captured["verbose"] is False
    assert captured["show_progress"] is False
    assert captured["ticker_exchange_map"] == {"AAPL": "CUSTOM"}


def test_main_help_includes_advanced_flags():
    help_text = main_cli.build_parser().format_help()
    assert "--no-distress-gate" in help_text
    assert "--no-tail-gate" in help_text
    assert "--no-valuation-gate" in help_text
    assert "--no-parallel-forensic" in help_text
    assert "--no-parallel-valuation" in help_text
    assert "--no-parallel-tail" in help_text


def test_main_no_parallel_alias_disables_all_parallel_stages():
    args = main_cli.parse_args(["--no-parallel"])
    exec_cfg = main_cli.build_execution_config(args)
    assert exec_cfg.enable_parallel_forensic is False
    assert exec_cfg.enable_parallel_valuation is False
    assert exec_cfg.enable_parallel_tail is False


def test_main_per_stage_parallel_flags_are_granular():
    args = main_cli.parse_args(["--no-parallel-tail"])
    exec_cfg = main_cli.build_execution_config(args)
    assert exec_cfg.enable_parallel_forensic is True
    assert exec_cfg.enable_parallel_valuation is True
    assert exec_cfg.enable_parallel_tail is False


def test_main_risk_flags_map_to_risk_config():
    args = main_cli.parse_args(
        ["--no-distress-gate", "--no-tail-gate", "--no-valuation-gate"]
    )
    risk_cfg = main_cli.build_risk_config(args)
    assert risk_cfg.enable_distress_gate is False
    assert risk_cfg.enable_tail_gate is False
    assert risk_cfg.enable_valuation_gate is False
    assert risk_cfg.enable_momentum_gate is True


def test_main_legacy_pipeline_rejects_advanced_flags():
    with pytest.raises(SystemExit):
        main_cli.parse_args(["--legacy-pipeline", "--no-distress-gate"])


def test_run_pipeline_distress_gate_controls_prefilter(monkeypatch, tmp_path):
    class DummyEngine:
        def __init__(self, *args, **kwargs):
            self.momentum_filter = type("MomentumFilterState", (), {"ma_period": 200})()

        def load_data(self):
            return None

    monkeypatch.setattr(main_cli, "StrategyEngine", DummyEngine)
    monkeypatch.setattr(
        main_cli,
        "_build_strategy_decisions",
        lambda *args, **kwargs: (
            pd.DataFrame(
                [
                    {"Ticker": "SAFE1", "Strategy": "test_strategy"},
                    {"Ticker": "RISK1", "Strategy": "test_strategy"},
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "Ticker": "SAFE1",
                        "Strategy": "test_strategy",
                        "Decision": "selected",
                        "DecisionStage": "strategy",
                        "RejectedReason": "",
                    },
                    {
                        "Ticker": "RISK1",
                        "Strategy": "test_strategy",
                        "Decision": "selected",
                        "DecisionStage": "strategy",
                        "RejectedReason": "",
                    },
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "Ticker": "SAFE1",
                        "Momentum_Above_MA": True,
                        "Momentum_MA_Gap_Pct": 3.0,
                    },
                    {
                        "Ticker": "RISK1",
                        "Momentum_Above_MA": True,
                        "Momentum_MA_Gap_Pct": 1.0,
                    },
                ]
            ),
        ),
    )
    monkeypatch.setattr(
        main_cli,
        "_prefetch_pipeline_data",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        main_cli,
        "run_forensic_scan",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "Ticker": "SAFE1",
                    "Altman Z-Score": 3.5,
                    "Distress Risk": "SAFE",
                    "Price": 100.0,
                },
                {
                    "Ticker": "RISK1",
                    "Altman Z-Score": 1.0,
                    "Distress Risk": "DISTRESS (Risk)",
                    "Price": 90.0,
                },
            ]
        ),
    )
    monkeypatch.setattr(
        main_cli,
        "_compute_tail_risk_frame",
        lambda tickers, **kwargs: pd.DataFrame(
            [
                {"Ticker": ticker, "Tail_Risk": "Normal", "Tail_a": 1.0, "Tail_b": 1.0}
                for ticker in tickers
            ]
        ),
    )
    monkeypatch.setattr(main_cli, "generate_html_report", lambda *args, **kwargs: None)

    valuation_inputs = []

    def fake_run_valuation_scan(df, *args, **kwargs):
        symbols = df["Ticker"].tolist() if "Ticker" in df.columns else []
        valuation_inputs.append(symbols)
        return pd.DataFrame(
            [
                {
                    "Ticker": ticker,
                    "Current Price": 100.0,
                    "Fair Value": 125.0,
                    "Fair Value (Bear)": 110.0,
                    "Fair Value (Base)": 125.0,
                    "Valuation Sanity": "Passed",
                }
                for ticker in symbols
            ]
        )

    monkeypatch.setattr(main_cli, "run_valuation_scan", fake_run_valuation_scan)

    exec_cfg = main_cli.ExecutionConfig(
        max_workers=1,
        enable_bulk_prefetch=False,
        enable_tail_cache=False,
        enable_parallel_forensic=False,
        enable_parallel_valuation=False,
        enable_parallel_tail=False,
    )

    main_cli.run_pipeline(
        update_data=False,
        exchange="CUSTOM",
        apply_momentum_filter=False,
        output_file=str(tmp_path / "with_distress_gate.csv"),
        verbose=False,
        strict_contracts=False,
        include_rejected=True,
        export_decision_log=False,
        risk_config=main_cli.RiskVectorConfig(
            enable_distress_gate=True,
            enable_tail_gate=False,
            enable_momentum_gate=False,
            enable_valuation_gate=False,
        ),
        execution_config=exec_cfg,
    )
    main_cli.run_pipeline(
        update_data=False,
        exchange="CUSTOM",
        apply_momentum_filter=False,
        output_file=str(tmp_path / "without_distress_gate.csv"),
        verbose=False,
        strict_contracts=False,
        include_rejected=True,
        export_decision_log=False,
        risk_config=main_cli.RiskVectorConfig(
            enable_distress_gate=False,
            enable_tail_gate=False,
            enable_momentum_gate=False,
            enable_valuation_gate=False,
        ),
        execution_config=exec_cfg,
    )

    assert valuation_inputs[0] == ["SAFE1"]
    assert valuation_inputs[1] == ["SAFE1", "RISK1"]
