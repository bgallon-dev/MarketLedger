"""
Main Pipeline Orchestrator - Automated Investment Analysis

This script ties together all components of the investment pipeline:
1. Data Layer - Fetch/update market data
2. Strategy Layer - Screen stocks using multiple strategies
3. Forensic Layer - Filter out distressed companies (Altman Z-Score)
4. Valuation Layer - Calculate intrinsic values and price targets

Usage:
------
    # Run full pipeline with current data
    python main.py

    # Update data before running
    python main.py --update

    # Specify exchange/universe
    python main.py --exchange SP500

    # Disable momentum filter (include stocks below 200-day MA)
    python main.py --no-momentum

    # Custom output file
    python main.py --output my_picks.csv
"""

import argparse
import os
import sys
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Set, Tuple

import pandas as pd


# ==============================================================================
# DATA CONTRACTS - Column validation between pipeline steps
# ==============================================================================

# Define required columns for each pipeline step
DATA_CONTRACTS = {
    "strategy_output": {
        "required": ["Ticker"],
        "optional": ["Strategy", "Score", "Rank"],
        "description": "Output from strategy screening layer",
    },
    "forensic_input": {
        "required": ["Ticker"],
        "optional": ["Strategy"],
        "description": "Input for forensic scan (Altman Z-Score)",
    },
    "forensic_output": {
        "required": ["Ticker", "Altman Z-Score", "Distress Risk"],
        "optional": ["Strategy", "Price"],
        "description": "Output from forensic analysis",
    },
    "valuation_input": {
        "required": ["Ticker"],
        "optional": ["Strategy", "Distress Risk", "Altman Z-Score"],
        "description": "Input for valuation layer (DCF/P-S/P-B analysis)",
    },
    "valuation_output": {
        "required": ["Ticker", "Current Price", "Fair Value"],
        "optional": [
            "Strategy",
            "Valuation Method",
            "Undervalued %",
            "Proj 6 Months",
            "Proj 1 Year",
        ],
        "description": "Final output with valuations (method: DCF, P/S, or P/B)",
    },
    "risk_vector_output": {
        "required": ["Ticker", "RV_Version", "RV_Gate_Count"],
        "optional": [
            "Decision",
            "DecisionStage",
            "RejectedReason",
            "RV_Distress_Bucket",
            "RV_Tail_Bucket",
            "RV_Momentum_Regime",
        ],
        "description": "Output with unified risk vectors and gate decisions",
    },
}


@dataclass
class ExecutionConfig:
    """Execution controls for pipeline performance tuning."""

    max_workers: Optional[int] = None
    enable_bulk_prefetch: bool = True
    enable_tail_cache: bool = True
    enable_parallel_forensic: bool = True
    enable_parallel_valuation: bool = True
    enable_parallel_tail: bool = True

    def resolved_workers(self) -> int:
        if self.max_workers is not None and int(self.max_workers) > 0:
            return int(self.max_workers)
        return min(8, os.cpu_count() or 4)


@dataclass
class StageTimer:
    """Simple per-stage timer with deterministic elapsed recording."""

    starts: Dict[str, float] = field(default_factory=dict)
    elapsed: Dict[str, float] = field(default_factory=dict)

    def start(self, stage: str) -> None:
        self.starts[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        start = self.starts.pop(stage, None)
        if start is None:
            return 0.0
        duration = time.perf_counter() - start
        self.elapsed[stage] = self.elapsed.get(stage, 0.0) + duration
        return duration

    def format_summary(self) -> str:
        if not self.elapsed:
            return "No stage timings recorded."
        parts = [f"{k}={v:.2f}s" for k, v in self.elapsed.items()]
        return ", ".join(parts)


class DataContractError(Exception):
    """Raised when a data contract validation fails."""

    pass


def validate_columns(
    df: pd.DataFrame,
    contract_name: str,
    strict: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Validate that a DataFrame meets the data contract requirements.

    Args:
        df: DataFrame to validate
        contract_name: Name of the contract from DATA_CONTRACTS
        strict: If True, raise exception on failure; if False, just warn
        verbose: Whether to print validation messages

    Returns:
        True if validation passes

    Raises:
        DataContractError: If strict=True and validation fails
        KeyError: If contract_name is not found
    """
    if contract_name not in DATA_CONTRACTS:
        raise KeyError(f"Unknown data contract: {contract_name}")

    contract = DATA_CONTRACTS[contract_name]
    required_cols = contract["required"]
    description = contract["description"]

    # Check for missing required columns
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        error_msg = (
            f"Data Contract Violation [{contract_name}]: "
            f"Missing required columns: {missing}\n"
            f"  Contract: {description}\n"
            f"  Required: {required_cols}\n"
            f"  Present: {list(df.columns)}"
        )
        if strict:
            raise DataContractError(error_msg)
        elif verbose:
            print(f"  [WARNING] {error_msg}")
        return False

    if verbose:
        print(f"  [✓] Data contract '{contract_name}' validated successfully")

    return True


def get_contract_info(contract_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about data contracts.

    Args:
        contract_name: Specific contract to get info for, or None for all

    Returns:
        Contract definition(s)
    """
    if contract_name:
        return DATA_CONTRACTS.get(contract_name, {})
    return DATA_CONTRACTS


# Ensure pyfinancial is in path
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT_DIR))

# Import pipeline modules
import pyfinancial.data.data as data
import pyfinancial.strat.strat as strat
from pyfinancial.strat.strat import StrategyEngine, Portfolio, MomentumFilter
from pyfinancial.database.database import (
    get_ticker_history,
    get_ticker_history_bulk,
    get_financial_data_bulk,
)

# Import analysis modules from parent directory
sys.path.insert(0, str(ROOT_DIR))
from forensic.forensic_scan import run_forensic_scan
from valuation.valuation_projector import (
    run_valuation_scan,
)
from pyfinancial.risk.risk_vector import (
    RiskVectorConfig,
    attach_risk_vectors,
    build_risk_vector,
    gate_stage_and_reason,
)


def generate_html_report(df: pd.DataFrame, filename: str = "report.html") -> None:
    """
    Generate an HTML report with Bootstrap styling for better visualization.

    Highlights stocks based on their Investment Signal and risk profile.

    Args:
        df: DataFrame with portfolio recommendations
        filename: Output HTML filename
    """
    if df.empty:
        print("Cannot generate report: DataFrame is empty")
        return

    def highlight_row(row):
        """Apply conditional formatting to rows based on Investment Signal."""
        styles = [""] * len(row)

        # Get Investment Signal (primary classification)
        signal = row.get("Investment Signal", "")
        sanity = row.get("Valuation Sanity", "Passed")

        # Color code based on Investment Signal
        if signal == "Strong Buy":
            styles = ["background-color: #d4edda"] * len(row)  # Green
        elif signal == "Speculative Buy":
            styles = ["background-color: #e8f5e9"] * len(row)  # Light green
        elif signal == "Hold":
            styles = ["background-color: #f8f9fa"] * len(row)  # Light gray
        elif signal == "Caution":
            styles = ["background-color: #fff3cd"] * len(row)  # Yellow
        elif signal in ["Avoid", "Overvalued"]:
            styles = ["background-color: #f8d7da"] * len(row)  # Red
        elif signal == "Needs Review" or sanity != "Passed":
            styles = ["background-color: #fff8e1"] * len(row)  # Amber

        return styles

    # Apply styling to dataframe (requires optional dependency: jinja2)
    try:
        styled_df = df.style.apply(highlight_row, axis=1)
        html_table = styled_df.to_html(
            classes="table table-striped table-hover", index=False
        )
    except (AttributeError, ImportError) as e:
        if "jinja2" not in str(e).lower():
            raise
        print(
            "Warning: jinja2 is not installed. "
            "Generating HTML report without conditional row highlighting."
        )
        html_table = df.to_html(classes="table table-striped table-hover", index=False)

    # Calculate signal counts
    signal_counts = (
        df["Investment Signal"].value_counts()
        if "Investment Signal" in df.columns
        else {}
    )

    # Build signal summary HTML
    signal_order = [
        "Strong Buy",
        "Speculative Buy",
        "Hold",
        "Caution",
        "Avoid",
        "Overvalued",
        "Needs Review",
    ]
    signal_emoji = {
        "Strong Buy": "🟢",
        "Speculative Buy": "🟡",
        "Hold": "⚪",
        "Caution": "🟠",
        "Avoid": "🔴",
        "Overvalued": "🔴",
        "Needs Review": "⚠️",
    }
    signal_items = "".join(
        [
            f"<li>{signal_emoji.get(s, '')} {s}: {signal_counts.get(s, 0)}</li>"
            for s in signal_order
            if signal_counts.get(s, 0) > 0
        ]
    )

    # Build full HTML document with Bootstrap
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Investment Pipeline Results</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body {{ padding: 20px; }}
        .table {{ font-size: 0.85rem; }}
        .legend {{ margin-bottom: 20px; }}
        .legend span {{ 
            display: inline-block; 
            padding: 5px 10px; 
            margin-right: 10px; 
            border-radius: 4px;
            margin-bottom: 5px;
        }}
        .legend-green {{ background-color: #d4edda; }}
        .legend-light-green {{ background-color: #e8f5e9; }}
        .legend-gray {{ background-color: #f8f9fa; }}
        .legend-yellow {{ background-color: #fff3cd; }}
        .legend-amber {{ background-color: #fff8e1; }}
        .legend-red {{ background-color: #f8d7da; }}
        h2 {{ color: #333; margin-bottom: 5px; }}
        .subtitle {{ color: #666; margin-bottom: 20px; }}
        .stats-card {{ 
            background: #f8f9fa; 
            padding: 15px; 
            border-radius: 8px; 
            margin-bottom: 20px;
        }}
        .stats-card h5 {{ margin-bottom: 10px; }}
    </style>
</head>
<body class="container-fluid">
    <h2>📊 Investment Pipeline Results</h2>
    <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Bear MOS Investment Signal Classification</p>
    
    <div class="row">
        <div class="col-md-8">
            <div class="legend">
                <strong>Investment Signal Legend:</strong><br>
                <span class="legend-green">🟢 Strong Buy (Bear-protected, ≥20% MOS)</span>
                <span class="legend-light-green">🟡 Speculative Buy (≥20% MOS, bear downside)</span>
                <span class="legend-gray">⚪ Hold (Modest upside)</span><br>
                <span class="legend-yellow">🟠 Caution (Heavy tail risk)</span>
                <span class="legend-red">🔴 Avoid/Overvalued (Distress or negative MOS)</span>
                <span class="legend-amber">⚠️ Needs Review (Sanity check failed)</span>
            </div>
        </div>
        <div class="col-md-4">
            <div class="stats-card">
                <h5>📈 Investment Signals</h5>
                <ul class="list-unstyled mb-0">
                    {signal_items}
                </ul>
            </div>
        </div>
    </div>
    
    <div class="table-responsive">
        {html_table}
    </div>
    
    <hr>
    <p class="text-muted">
        <strong>Total Stocks:</strong> {len(df)} | 
        <strong>Analysis includes:</strong> Bear/Base/Bull Scenario DCF, Altman Z-Score, 
        Kumaraswamy-Laplace Tail Risk, Valuation Sanity Gate
    </p>
    <p class="text-muted small">
        <strong>Investment Signal Logic:</strong><br>
        • <strong>Strong Buy:</strong> SAFE + Bear Fair Value ≥ Current Price + Base MOS ≥ 20%<br>
        • <strong>Speculative Buy:</strong> SAFE + Base MOS ≥ 20% but Bear Fair Value &lt; Current Price<br>
        • <strong>Caution:</strong> Heavy tail risk (high volatility)<br>
        • <strong>Avoid:</strong> Distress risk or significantly overvalued
    </p>
    <p class="text-muted small">
        <strong>Note:</strong> Trend projections (Trend_Proj_*) are CAGR extrapolations, NOT price targets.
        Raw valuation inputs (_Input_*) are included for auditability.
    </p>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML Report saved to: {filename}")


def _legacy_run_pipeline(
    update_data: bool = False,
    exchange: Optional[str] = None,
    apply_momentum_filter: bool = True,
    output_file: Optional[str] = None,
    verbose: bool = True,
    strict_contracts: bool = False,
) -> pd.DataFrame:
    """
    Run the complete investment analysis pipeline.

    Args:
        update_data: Whether to fetch fresh data before analysis
        exchange: Target universe (SP500, NASDAQ, etc.)
        apply_momentum_filter: Whether to filter stocks below 200-day MA
        output_file: Path to save final results (optional)
        verbose: Whether to print progress messages
        strict_contracts: If True, raise exceptions on contract violations

    Returns:
        DataFrame with final portfolio recommendations

    Raises:
        DataContractError: If strict_contracts=True and a contract is violated
    """
    trade_date = datetime.now().strftime("%Y-%m-%d")

    if verbose:
        print(f"{'='*60}")
        print(f"  AUTOMATED INVESTMENT PIPELINE")
        print(f"  Date: {trade_date}")
        print(f"{'='*60}")

    # -------------------------------------------------------------------------
    # 1. DATA LAYER
    # -------------------------------------------------------------------------
    if update_data:
        if verbose:
            print("\n[1/4] Updating Data...")
        try:
            data.run_data_fetch(exchange=exchange, verbose=verbose)
        except Exception as e:
            print(f"  Warning: Data update failed: {e}")
    else:
        if verbose:
            print("\n[1/4] Using cached data (use --update to refresh)")

    # -------------------------------------------------------------------------
    # 2. STRATEGY LAYER
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[2/4] Running Strategy Screen...")

    # Configure momentum filter
    momentum_filter = MomentumFilter(require_above_ma=apply_momentum_filter)

    # Initialize the strategy engine
    engine = StrategyEngine(
        exchange=exchange,
        momentum_filter=momentum_filter,
        verbose=verbose,
    )
    engine.load_data()

    # Build portfolio using multiple strategies
    portfolio = Portfolio(engine)
    candidates_df = portfolio.generate_buy_list(trade_date)

    if candidates_df.empty:
        if verbose:
            print("\nNo candidates found matching criteria. Exiting.")
        return pd.DataFrame()

    if verbose:
        print(f"  -> Found {len(candidates_df)} candidate stocks")

    # Validate strategy output contract
    validate_columns(
        candidates_df,
        "strategy_output",
        strict=strict_contracts,
        verbose=verbose,
    )

    # -------------------------------------------------------------------------
    # 3. FORENSIC LAYER (Filter out distressed companies)
    # -------------------------------------------------------------------------
    if verbose:
        print(
            f"\n[3/4] Performing Forensic Audit on {len(candidates_df)} candidates..."
        )

    # Validate forensic input contract
    validate_columns(
        candidates_df,
        "forensic_input",
        strict=strict_contracts,
        verbose=verbose,
    )

    audited_df = run_forensic_scan(candidates_df, verbose=verbose)

    # Filter to only "SAFE" stocks (exclude DISTRESS and GREY ZONE)
    if "Distress Risk" in audited_df.columns:
        safe_picks = audited_df[audited_df["Distress Risk"] == "SAFE"].copy()
        filtered_count = len(candidates_df) - len(safe_picks)
        if verbose and filtered_count > 0:
            print(f"  -> Removed {filtered_count} stocks with distress risk")
            print(f"  -> {len(safe_picks)} stocks passed the Altman Z-Score test")

        # Validate forensic output contract
        validate_columns(
            audited_df,
            "forensic_output",
            strict=strict_contracts,
            verbose=verbose,
        )
    else:
        safe_picks = audited_df
        if verbose:
            print("  [WARNING] Forensic scan did not produce 'Distress Risk' column")

    if safe_picks.empty:
        if verbose:
            print("\nNo stocks passed forensic screening. Exiting.")
        return audited_df

    # -------------------------------------------------------------------------
    # 4. VALUATION LAYER (Calculate intrinsic values)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[4/4] Calculating Intrinsic Values for {len(safe_picks)} stocks...")

    # Validate valuation input contract
    validate_columns(
        safe_picks,
        "valuation_input",
        strict=strict_contracts,
        verbose=verbose,
    )

    final_portfolio = run_valuation_scan(safe_picks, verbose=verbose)

    # Validate final output contract
    if not final_portfolio.empty:
        validate_columns(
            final_portfolio,
            "valuation_output",
            strict=strict_contracts,
            verbose=verbose,
        )

    # -------------------------------------------------------------------------
    # 5. RISK LAYER (Distributions) - Tail Risk Analysis & Confidence Update
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[5/5] Analyzing Tail Risk & Updating Model Confidence...")

    try:
        from distro.distributions import KumaraswamyLaplace

        risk_results = []
        for ticker in final_portfolio["Ticker"]:
            # Fetch history from DB using imported function
            hist = get_ticker_history(ticker)
            if hist is not None and not hist.empty:
                # Calculate returns
                if "close" in hist.columns:
                    returns = hist["close"].pct_change().dropna()
                elif "Close" in hist.columns:
                    returns = hist["Close"].pct_change().dropna()
                else:
                    continue

                if len(returns) < 50:  # Need minimum data for fitting
                    continue

                # Fit the Kumaraswamy-Laplace model
                model = KumaraswamyLaplace()
                try:
                    model.fit(returns)
                    interp = model.interpret_params()
                    risk_results.append(
                        {
                            "Ticker": ticker,
                            "Tail_Risk": interp["tail_weight"],
                            "Skew_Direction": interp["skew_direction"],
                            "VaR_Adjustment": interp["var_adjustment"],
                            "Risk_Summary": interp["risk_summary"],
                        }
                    )
                    if verbose:
                        print(f"  ✓ {ticker}: {interp['tail_weight']} tail risk")
                except Exception as fit_error:
                    if verbose:
                        print(f"  ✗ {ticker}: Could not fit distribution - {fit_error}")

        # Merge Risk Data back into Final DF
        if risk_results:
            risk_df = pd.DataFrame(risk_results)
            final_portfolio = final_portfolio.merge(risk_df, on="Ticker", how="left")
            if verbose:
                print(f"  -> Risk analysis completed for {len(risk_results)} stocks")

            # ----------------------------------------------------------------
            # UPDATE MODEL CONFIDENCE WITH TAIL RISK DATA
            # ----------------------------------------------------------------
            if verbose:
                print(f"  -> Updating Model Confidence with tail risk data...")

            updated_confidences = []
            for _, row in final_portfolio.iterrows():
                ticker = row["Ticker"]
                tail_risk = row.get("Tail_Risk")
                valuation_method = row.get("Valuation Method", "N/A")
                scenario_spread_str = row.get("Scenario Spread %", "0%")

                # Parse scenario spread
                try:
                    scenario_spread = float(str(scenario_spread_str).replace("%", ""))
                except (ValueError, TypeError):
                    scenario_spread = 100.0

                # Determine data availability from existing columns
                fair_value = row.get("Fair Value", 0)
                fcf_available = valuation_method == "DCF"
                revenue_available = valuation_method in ["DCF", "P/S"]
                book_value_available = valuation_method in ["DCF", "P/S", "P/B"]

                # Recalculate confidence with tail risk
                confidence_level, confidence_score, confidence_reason = (
                    calculate_model_confidence(
                        valuation_method=valuation_method,
                        fcf_available=fcf_available,
                        revenue_available=revenue_available,
                        book_value_available=book_value_available,
                        tail_risk=tail_risk,
                        scenario_spread=scenario_spread,
                        data_years=3,  # Assume 3 years as default
                    )
                )

                updated_confidences.append(
                    {
                        "Ticker": ticker,
                        "Model Confidence": confidence_level,
                        "Confidence Score": confidence_score,
                        "Confidence Reason": confidence_reason,
                    }
                )

            # Update confidence columns
            if updated_confidences:
                confidence_df = pd.DataFrame(updated_confidences)
                # Drop old confidence columns and merge updated ones
                cols_to_drop = [
                    "Model Confidence",
                    "Confidence Score",
                    "Confidence Reason",
                ]
                for col in cols_to_drop:
                    if col in final_portfolio.columns:
                        final_portfolio = final_portfolio.drop(columns=[col])
                final_portfolio = final_portfolio.merge(
                    confidence_df, on="Ticker", how="left"
                )

                if verbose:
                    print(f"  -> Model confidence updated with tail risk integration")

            # ----------------------------------------------------------------
            # UPDATE INVESTMENT SIGNALS WITH COMPLETE DATA
            # Now we have distress_risk and tail_risk to refine signals
            # ----------------------------------------------------------------
            if verbose:
                print(f"  -> Updating Investment Signals with risk data...")

            updated_signals = []
            for _, row in final_portfolio.iterrows():
                ticker = row["Ticker"]
                bear_fv = row.get("Fair Value (Bear)", 0)
                base_fv = row.get("Fair Value (Base)", 0)
                current_price = row.get("Current Price", 0)
                distress_risk = row.get("Distress Risk")
                tail_risk = row.get("Tail_Risk")
                sanity_status = row.get("Valuation Sanity", "Passed")
                sanity_passed = sanity_status == "Passed"

                signal, reason = classify_investment_signal(
                    bear_fv=bear_fv,
                    base_fv=base_fv,
                    current_price=current_price,
                    distress_risk=distress_risk,
                    tail_risk=tail_risk,
                    sanity_passed=sanity_passed,
                )

                updated_signals.append(
                    {
                        "Ticker": ticker,
                        "Investment Signal": signal,
                        "Signal Reason": reason,
                    }
                )

            # Update signal columns
            if updated_signals:
                signals_df = pd.DataFrame(updated_signals)
                signal_cols_to_drop = ["Investment Signal", "Signal Reason"]
                for col in signal_cols_to_drop:
                    if col in final_portfolio.columns:
                        final_portfolio = final_portfolio.drop(columns=[col])
                final_portfolio = final_portfolio.merge(
                    signals_df, on="Ticker", how="left"
                )

                if verbose:
                    print(f"  -> Investment signals updated with risk integration")

    except Exception as e:
        if verbose:
            print(f"  Warning: Risk analysis failed - {e}")

    # -------------------------------------------------------------------------
    # 6. REPORTING
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 80)
        print("  FINAL PORTFOLIO RECOMMENDATIONS")
        print("  (Strategy-Conditioned Valuation with Bear MOS Investment Signals)")
        print("=" * 80)

        # Display summary (including new Investment Signal column)
        display_cols = [
            "Ticker",
            "Strategy",
            "Current Price",
            "Fair Value (Bear)",
            "Fair Value",
            "Investment Signal",
            "Undervalued %",
            "Model Confidence",
            "Valuation Sanity",
        ]
        display_cols = [c for c in display_cols if c in final_portfolio.columns]

        if not final_portfolio.empty:
            print(final_portfolio[display_cols].to_string(index=False))
            print(f"\nTotal Recommendations: {len(final_portfolio)}")

            # Print Investment Signal summary
            if "Investment Signal" in final_portfolio.columns:
                signal_counts = final_portfolio["Investment Signal"].value_counts()
                print(f"\n📊 Investment Signal Distribution:")
                signal_order = [
                    "Strong Buy",
                    "Speculative Buy",
                    "Hold",
                    "Caution",
                    "Avoid",
                    "Overvalued",
                    "Needs Review",
                ]
                for signal in signal_order:
                    count = signal_counts.get(signal, 0)
                    if count > 0:
                        emoji = {
                            "Strong Buy": "🟢",
                            "Speculative Buy": "🟡",
                            "Hold": "⚪",
                            "Caution": "🟠",
                            "Avoid": "🔴",
                            "Overvalued": "🔴",
                            "Needs Review": "⚠️",
                        }.get(signal, "")
                        print(f"  {emoji} {signal}: {count} stocks")

            # Print confidence summary
            if "Model Confidence" in final_portfolio.columns:
                conf_counts = final_portfolio["Model Confidence"].value_counts()
                print(f"\n📈 Model Confidence Distribution:")
                for level in ["High", "Medium", "Low"]:
                    count = conf_counts.get(level, 0)
                    print(f"  {level}: {count} stocks")

            # Print sanity check summary
            if "Valuation Sanity" in final_portfolio.columns:
                failed_sanity = final_portfolio[
                    final_portfolio["Valuation Sanity"] != "Passed"
                ]
                if len(failed_sanity) > 0:
                    print(f"\n⚠️ Valuation Sanity Warnings: {len(failed_sanity)} stocks")
                    for _, row in failed_sanity.iterrows():
                        print(f"  - {row['Ticker']}: {row['Valuation Sanity']}")

    # Save results
    if output_file:
        output_path = Path(output_file)
        final_portfolio.to_csv(output_path, index=False)
        if verbose:
            print(f"\nResults saved to: {output_path}")
        # Also generate HTML report
        html_path = output_path.with_suffix(".html")
        generate_html_report(final_portfolio, str(html_path))
    else:
        # Default output location
        default_output = ROOT_DIR / "buy_list_with_projections.csv"
        final_portfolio.to_csv(default_output, index=False)
        if verbose:
            print(f"\nResults saved to: {default_output}")
        # Also generate HTML report
        html_output = ROOT_DIR / "buy_list_with_projections.html"
        generate_html_report(final_portfolio, str(html_output))

    if verbose:
        print("\n" + "=" * 60)
        print(f"  Pipeline Complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

    return final_portfolio


def _concat_strategy_labels(labels: pd.Series) -> str:
    values = sorted(
        {
            str(v).strip()
            for v in labels
            if pd.notna(v) and str(v).strip() != ""
        }
    )
    return " + ".join(values)


def _left_merge_new_columns(
    base_df: pd.DataFrame, add_df: pd.DataFrame, key: str = "Ticker"
) -> pd.DataFrame:
    if base_df.empty or add_df.empty:
        return base_df
    new_cols = [c for c in add_df.columns if c == key or c not in base_df.columns]
    return base_df.merge(add_df[new_cols], on=key, how="left")


def _build_strategy_decisions(
    engine: StrategyEngine,
    trade_date: str,
    top_n_per_strategy: int,
    apply_momentum_filter: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_rows: List[Dict[str, str]] = []
    momentum_reject_rows: List[Dict[str, str]] = []
    momentum_diag_rows: List[Dict[str, Any]] = []
    ma_period = engine.momentum_filter.ma_period

    for combo_name, strategies in Portfolio.DEFAULT_COMBOS.items():
        for strat_name in strategies:
            label = f"{combo_name} ({strat_name})"
            result = engine.run_strategy(
                strat_name,
                trade_date,
                top_n=top_n_per_strategy,
                apply_momentum_filter=apply_momentum_filter,
                capture_diagnostics=True,
            )

            for ticker in result.tickers:
                selected_rows.append({"Ticker": ticker, "Strategy": label})

            for ticker in result.metadata.get("momentum_rejected", []):
                momentum_reject_rows.append(
                    {
                        "Ticker": ticker,
                        "Strategy": label,
                        "Decision": "rejected",
                        "DecisionStage": "momentum",
                        "RejectedReason": f"Failed {ma_period}-day MA momentum filter",
                    }
                )

            for diag in result.metadata.get("momentum_diagnostics", []):
                ticker = diag.get("Ticker")
                if ticker is None:
                    continue
                px = diag.get("Price")
                ma_val = diag.get("MA_Value")
                gap_pct = None
                try:
                    if px is not None and ma_val not in (None, 0):
                        gap_pct = (float(px) - float(ma_val)) / float(ma_val) * 100
                except Exception:
                    gap_pct = None
                momentum_diag_rows.append(
                    {
                        "Ticker": ticker,
                        "Momentum_Price": px,
                        "Momentum_MA_200": ma_val,
                        "Momentum_Above_MA": diag.get("Above_MA"),
                        "Momentum_MA_Gap_Pct": gap_pct,
                    }
                )

    selected_df = pd.DataFrame(selected_rows)
    if not selected_df.empty:
        selected_df = selected_df.groupby("Ticker", as_index=False).agg(
            {"Strategy": _concat_strategy_labels}
        )
        selected_df["Decision"] = "selected"
        selected_df["DecisionStage"] = "strategy"
        selected_df["RejectedReason"] = ""
    else:
        selected_df = pd.DataFrame(
            columns=["Ticker", "Strategy", "Decision", "DecisionStage", "RejectedReason"]
        )

    momentum_reject_df = pd.DataFrame(momentum_reject_rows)
    if not momentum_reject_df.empty:
        momentum_reject_df = momentum_reject_df.groupby("Ticker", as_index=False).agg(
            {
                "Strategy": _concat_strategy_labels,
                "Decision": "first",
                "DecisionStage": "first",
                "RejectedReason": "first",
            }
        )
    else:
        momentum_reject_df = pd.DataFrame(
            columns=["Ticker", "Strategy", "Decision", "DecisionStage", "RejectedReason"]
        )

    decision_records: Dict[str, Dict[str, Any]] = {
        row["Ticker"]: row.to_dict() for _, row in momentum_reject_df.iterrows()
    }
    for _, row in selected_df.iterrows():
        decision_records[row["Ticker"]] = row.to_dict()

    decision_seed_df = pd.DataFrame(list(decision_records.values()))
    if decision_seed_df.empty:
        decision_seed_df = pd.DataFrame(
            columns=["Ticker", "Strategy", "Decision", "DecisionStage", "RejectedReason"]
        )
    else:
        decision_seed_df = decision_seed_df.sort_values("Ticker").reset_index(drop=True)

    momentum_df = pd.DataFrame(momentum_diag_rows)
    if not momentum_df.empty:
        momentum_df = momentum_df.dropna(subset=["Ticker"])
        momentum_df = (
            momentum_df.sort_values("Ticker")
            .drop_duplicates(subset=["Ticker"], keep="last")
            .reset_index(drop=True)
        )
    else:
        momentum_df = pd.DataFrame(
            columns=[
                "Ticker",
                "Momentum_Price",
                "Momentum_MA_200",
                "Momentum_Above_MA",
                "Momentum_MA_Gap_Pct",
            ]
        )

    candidates_df = (
        selected_df[["Ticker", "Strategy"]].copy()
        if not selected_df.empty
        else pd.DataFrame(columns=["Ticker", "Strategy"])
    )
    return candidates_df, decision_seed_df, momentum_df


def _normalize_tickers(tickers: List[Any]) -> List[str]:
    seen = set()
    ordered = []
    for raw in tickers:
        if raw is None:
            continue
        ticker = str(raw).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        ordered.append(ticker)
    return ordered


def _history_map_from_bulk(df: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    if df is None or df.empty:
        return {}
    out: Dict[str, pd.DataFrame] = {}
    work = df.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    for symbol, grp in work.groupby("symbol", sort=False):
        cols = [c for c in grp.columns if c != "symbol"]
        out[symbol] = grp[cols].sort_values("date").reset_index(drop=True)
    return out


def _financial_map_from_bulk(df: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    if df is None or df.empty:
        return {}
    out: Dict[str, pd.DataFrame] = {}
    work = df.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    for symbol, grp in work.groupby("symbol", sort=False):
        out[symbol] = grp.pivot(index="metric", columns="period", values="value")
    return out


def _prefetch_pipeline_data(
    tickers: List[str],
    execution_config: ExecutionConfig,
    verbose: bool = True,
) -> Dict[str, Any]:
    if not execution_config.enable_bulk_prefetch:
        return {}

    symbols = _normalize_tickers(tickers)
    if not symbols:
        return {}

    if verbose:
        print(f"  Prefetching history/financials for {len(symbols)} tickers...")

    history = get_ticker_history_bulk(symbols)
    balance_sheet = get_financial_data_bulk(symbols, "balance_sheet")
    income_statement = get_financial_data_bulk(symbols, "income_statement")
    cash_flow = get_financial_data_bulk(symbols, "cash_flow")

    return {
        "history": history,
        "balance_sheet": balance_sheet,
        "income_statement": income_statement,
        "cash_flow": cash_flow,
        "history_by_symbol": _history_map_from_bulk(history),
        "balance_sheet_by_symbol": _financial_map_from_bulk(balance_sheet),
        "income_statement_by_symbol": _financial_map_from_bulk(income_statement),
        "cash_flow_by_symbol": _financial_map_from_bulk(cash_flow),
    }


def _tail_row_from_history(
    ticker: str,
    history_df: Optional[pd.DataFrame],
    lookback_days: int = 252,
    min_points: int = 50,
    verbose: bool = True,
    cache_enabled: bool = True,
    tail_cache: Optional[Dict[Tuple[Any, ...], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    from distro.distributions import KumaraswamyLaplace

    row: Dict[str, Any] = {
        "Ticker": ticker,
        "Tail_Risk": "Unknown",
        "Skew_Direction": "Unknown",
        "VaR_Adjustment": None,
        "Risk_Summary": "Unknown",
        "Tail_a": None,
        "Tail_b": None,
    }
    if history_df is None or history_df.empty:
        return row

    close_col = "close" if "close" in history_df.columns else "Close"
    if close_col not in history_df.columns:
        return row

    returns = history_df[close_col].pct_change().dropna()
    if len(returns) < min_points:
        return row
    returns = returns.iloc[-lookback_days:]

    cache_key = None
    if cache_enabled:
        end_marker = None
        if "date" in history_df.columns and not history_df["date"].empty:
            end_marker = str(history_df["date"].iloc[-1])
        cache_key = (ticker, end_marker, lookback_days, min_points)
        if tail_cache is not None and cache_key in tail_cache:
            return dict(tail_cache[cache_key])

    try:
        model = KumaraswamyLaplace()
        model.fit(returns)
        interp = model.interpret_params()
        params = model.params or (None, None, None, None)

        row["Tail_Risk"] = interp.get("tail_weight", "Unknown")
        row["Skew_Direction"] = interp.get("skew_direction", "Unknown")
        row["VaR_Adjustment"] = interp.get("var_adjustment")
        row["Risk_Summary"] = interp.get("risk_summary", "Unknown")
        row["Tail_a"] = params[2]
        row["Tail_b"] = params[3]
        if verbose:
            print(f"  tail fit {ticker}: {row['Tail_Risk']}")
    except Exception as exc:
        if verbose:
            print(f"  tail fit failed {ticker}: {exc}")

    if cache_enabled and cache_key is not None and tail_cache is not None:
        tail_cache[cache_key] = dict(row)
    return row


def _compute_tail_risk_frame(
    tickers: List[str],
    verbose: bool = True,
    prefetched: Optional[Dict[str, Any]] = None,
    max_workers: Optional[int] = None,
    enable_parallel: bool = True,
    enable_cache: bool = True,
    tail_cache: Optional[Dict[Tuple[Any, ...], Dict[str, Any]]] = None,
) -> pd.DataFrame:
    symbols = _normalize_tickers(tickers)
    if not symbols:
        return pd.DataFrame()

    history_by_symbol: Dict[str, pd.DataFrame] = {}
    if prefetched and isinstance(prefetched.get("history_by_symbol"), dict):
        history_by_symbol = prefetched["history_by_symbol"]
    elif prefetched and isinstance(prefetched.get("history"), pd.DataFrame):
        history_by_symbol = _history_map_from_bulk(prefetched["history"])

    if not history_by_symbol:
        history_by_symbol = {ticker: get_ticker_history(ticker) for ticker in symbols}

    workers = max_workers if max_workers is not None else min(8, os.cpu_count() or 4)
    workers = max(1, int(workers))

    rows: List[Dict[str, Any]] = []
    if enable_parallel and workers > 1 and len(symbols) > 1:
        futures = {}
        with ThreadPoolExecutor(max_workers=min(workers, len(symbols))) as executor:
            for ticker in symbols:
                future = executor.submit(
                    _tail_row_from_history,
                    ticker,
                    history_by_symbol.get(ticker),
                    252,
                    50,
                    verbose,
                    enable_cache,
                    tail_cache,
                )
                futures[future] = ticker
            for future in as_completed(futures):
                rows.append(future.result())
    else:
        for ticker in symbols:
            rows.append(
                _tail_row_from_history(
                    ticker=ticker,
                    history_df=history_by_symbol.get(ticker),
                    verbose=verbose,
                    cache_enabled=enable_cache,
                    tail_cache=tail_cache,
                )
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)


@dataclass
class PipelineNode:
    name: str
    deps: Set[str]
    run: Callable[[Dict[str, Any]], Dict[str, Any]]
    io_contract: Dict[str, List[str]] = field(default_factory=dict)


def _merge_context_delta(context: Dict[str, Any], delta: Dict[str, Any]) -> None:
    for key, value in delta.items():
        if key in context and isinstance(context[key], dict) and isinstance(value, dict):
            merged = dict(context[key])
            merged.update(value)
            context[key] = merged
        else:
            context[key] = value


def _execute_pipeline_graph(
    nodes: List[PipelineNode],
    initial_context: Dict[str, Any],
) -> Dict[str, Any]:
    context = dict(initial_context)
    node_map = {n.name: n for n in nodes}
    pending = set(node_map.keys())
    completed: Set[str] = set()
    workers = 1
    exec_cfg = context.get("execution_config")
    if isinstance(exec_cfg, ExecutionConfig):
        workers = exec_cfg.resolved_workers()

    while pending:
        ready = sorted(
            name for name in pending if node_map[name].deps.issubset(completed)
        )
        if not ready:
            unresolved = ", ".join(sorted(pending))
            raise RuntimeError(f"Node graph deadlock; unresolved nodes: {unresolved}")

        if workers <= 1 or len(ready) == 1:
            for name in ready:
                delta = node_map[name].run(dict(context))
                if delta:
                    _merge_context_delta(context, delta)
                completed.add(name)
                pending.remove(name)
            continue

        futures = {}
        with ThreadPoolExecutor(max_workers=min(workers, len(ready))) as executor:
            for name in ready:
                futures[executor.submit(node_map[name].run, dict(context))] = name

            deltas: Dict[str, Dict[str, Any]] = {}
            for future in as_completed(futures):
                name = futures[future]
                deltas[name] = future.result()

        for name in sorted(deltas):
            delta = deltas[name]
            if delta:
                _merge_context_delta(context, delta)
            completed.add(name)
            pending.remove(name)

    return context


def run_pipeline(
    update_data: bool = False,
    exchange: Optional[str] = None,
    apply_momentum_filter: bool = True,
    output_file: Optional[str] = None,
    verbose: bool = True,
    strict_contracts: bool = False,
    include_rejected: bool = True,
    export_decision_log: bool = True,
    risk_config: Optional[RiskVectorConfig] = None,
    execution_config: Optional[ExecutionConfig] = None,
    use_legacy_debug: bool = False,
) -> pd.DataFrame:
    """Run the investment pipeline through a dependency-driven node graph."""
    if use_legacy_debug:
        print("Warning: _legacy_run_pipeline is deprecated and enabled via debug fallback.")
        return _legacy_run_pipeline(
            update_data=update_data,
            exchange=exchange,
            apply_momentum_filter=apply_momentum_filter,
            output_file=output_file,
            verbose=verbose,
            strict_contracts=strict_contracts,
        )

    trade_date = datetime.now().strftime("%Y-%m-%d")
    cfg = (
        RiskVectorConfig(**vars(risk_config))
        if risk_config is not None
        else RiskVectorConfig()
    )
    exec_cfg = (
        ExecutionConfig(**vars(execution_config))
        if execution_config is not None
        else ExecutionConfig()
    )
    if not apply_momentum_filter:
        cfg.enable_momentum_gate = False

    timer = StageTimer()
    tail_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    if verbose:
        print(f"{'='*60}")
        print("  AUTOMATED INVESTMENT PIPELINE")
        print(f"  Date: {trade_date}")
        print(f"{'='*60}")
        if exec_cfg.resolved_workers() >= 8:
            print(
                "  Warning: high worker counts can increase SQLite contention on local runs."
            )

    def _stop_stage(stage: str) -> None:
        duration = timer.stop(stage)
        if verbose:
            print(f"  [timing] {stage}: {duration:.2f}s")

    def data_update_node(_: Dict[str, Any]) -> Dict[str, Any]:
        timer.start("data_update")
        if update_data:
            if verbose:
                print("\n[1/10] Updating Data...")
            try:
                data.run_data_fetch(exchange=exchange, verbose=verbose)
            except Exception as exc:
                print(f"  Warning: Data update failed: {exc}")
        elif verbose:
            print("\n[1/10] Using cached data (use --update to refresh)")
        _stop_stage("data_update")
        return {}

    def strategy_candidates_node(_: Dict[str, Any]) -> Dict[str, Any]:
        timer.start("strategy")
        if verbose:
            print("\n[2/10] Running Strategy Screen...")

        momentum_filter = MomentumFilter(require_above_ma=apply_momentum_filter)
        engine = StrategyEngine(
            exchange=exchange,
            momentum_filter=momentum_filter,
            verbose=verbose,
        )
        engine.load_data()

        candidates_df, decision_log, momentum_df = _build_strategy_decisions(
            engine=engine,
            trade_date=trade_date,
            top_n_per_strategy=10,
            apply_momentum_filter=apply_momentum_filter,
        )

        if decision_log.empty:
            if verbose:
                print("\nNo candidates produced by strategy layer. Exiting.")
            _stop_stage("strategy")
            return {
                "pipeline_abort": True,
                "candidates_df": pd.DataFrame(columns=["Ticker", "Strategy"]),
                "decision_log": pd.DataFrame(columns=["Ticker"]),
                "momentum_df": pd.DataFrame(columns=["Ticker"]),
            }

        if not candidates_df.empty:
            validate_columns(
                candidates_df,
                "strategy_output",
                strict=strict_contracts,
                verbose=verbose,
            )
        if verbose:
            print(
                f"  -> Selected candidates: {len(candidates_df)} | Seed rejects: {len(decision_log) - len(candidates_df)}"
            )
        _stop_stage("strategy")
        return {
            "pipeline_abort": False,
            "candidates_df": candidates_df,
            "decision_log": decision_log,
            "momentum_df": momentum_df,
        }

    def prefetch_data_node(context: Dict[str, Any]) -> Dict[str, Any]:
        timer.start("prefetch")
        if context.get("pipeline_abort", False):
            _stop_stage("prefetch")
            return {"prefetched": {}}

        if verbose:
            print("\n[3/10] Prefetching Batch Data...")
        decision_log = context.get("decision_log", pd.DataFrame())
        tickers = decision_log["Ticker"].tolist() if not decision_log.empty else []
        prefetched = _prefetch_pipeline_data(tickers, exec_cfg, verbose=verbose)
        _stop_stage("prefetch")
        return {"prefetched": prefetched}

    def forensic_node(context: Dict[str, Any]) -> Dict[str, Any]:
        timer.start("forensic")
        if verbose:
            print("\n[4/10] Performing Forensic Audit...")
        if context.get("pipeline_abort", False):
            _stop_stage("forensic")
            return {
                "audited_df": pd.DataFrame(columns=["Ticker"]),
                "safe_picks": pd.DataFrame(columns=["Ticker"]),
            }

        candidates_df = context.get("candidates_df", pd.DataFrame(columns=["Ticker"]))
        decision_log = context.get("decision_log", pd.DataFrame(columns=["Ticker"])).copy()

        if not candidates_df.empty:
            validate_columns(
                candidates_df,
                "forensic_input",
                strict=strict_contracts,
                verbose=verbose,
            )
            forensic_workers = (
                exec_cfg.resolved_workers() if exec_cfg.enable_parallel_forensic else 1
            )
            audited_df = run_forensic_scan(
                candidates_df,
                verbose=verbose,
                prefetched=context.get("prefetched"),
                max_workers=forensic_workers,
            )
            if not audited_df.empty:
                validate_columns(
                    audited_df,
                    "forensic_output",
                    strict=strict_contracts,
                    verbose=verbose,
                )
        else:
            audited_df = pd.DataFrame(columns=["Ticker"])

        if not audited_df.empty:
            decision_log = _left_merge_new_columns(
                decision_log,
                audited_df[["Ticker", "Altman Z-Score", "Distress Risk", "Price"]],
            )
            if "Distress Risk" in audited_df.columns:
                safe_picks = audited_df[audited_df["Distress Risk"] == "SAFE"].copy()
            else:
                safe_picks = audited_df.copy()
        else:
            safe_picks = pd.DataFrame(columns=["Ticker"])

        if verbose:
            print(f"  -> SAFE after forensic: {len(safe_picks)}")
        _stop_stage("forensic")
        return {
            "decision_log": decision_log,
            "audited_df": audited_df,
            "safe_picks": safe_picks,
        }

    def safe_filter_node(context: Dict[str, Any]) -> Dict[str, Any]:
        timer.start("safe_filter")
        if context.get("pipeline_abort", False):
            _stop_stage("safe_filter")
            return {}

        audited_df = context.get("audited_df", pd.DataFrame(columns=["Ticker"]))
        if audited_df.empty:
            safe_picks = pd.DataFrame(columns=["Ticker"])
        elif "Distress Risk" in audited_df.columns:
            safe_picks = audited_df[audited_df["Distress Risk"] == "SAFE"].copy()
        else:
            safe_picks = audited_df.copy()

        _stop_stage("safe_filter")
        return {"safe_picks": safe_picks}

    def valuation_node(context: Dict[str, Any]) -> Dict[str, Any]:
        timer.start("valuation")
        if verbose:
            print("\n[5/10] Running Valuation Layer...")
        if context.get("pipeline_abort", False):
            _stop_stage("valuation")
            return {"valuation_df": pd.DataFrame(columns=["Ticker"])}

        safe_picks = context.get("safe_picks", pd.DataFrame(columns=["Ticker"]))
        decision_log = context.get("decision_log", pd.DataFrame(columns=["Ticker"])).copy()

        if not safe_picks.empty:
            validate_columns(
                safe_picks,
                "valuation_input",
                strict=strict_contracts,
                verbose=verbose,
            )
            valuation_workers = (
                exec_cfg.resolved_workers() if exec_cfg.enable_parallel_valuation else 1
            )
            valuation_df = run_valuation_scan(
                safe_picks,
                verbose=verbose,
                prefetched=context.get("prefetched"),
                max_workers=valuation_workers,
            )
            if not valuation_df.empty:
                validate_columns(
                    valuation_df,
                    "valuation_output",
                    strict=strict_contracts,
                    verbose=verbose,
                )
        else:
            valuation_df = pd.DataFrame(columns=["Ticker"])

        if not valuation_df.empty:
            decision_log = _left_merge_new_columns(decision_log, valuation_df)

        _stop_stage("valuation")
        return {"valuation_df": valuation_df, "decision_log": decision_log}

    def tail_risk_node(context: Dict[str, Any]) -> Dict[str, Any]:
        timer.start("tail_risk")
        if verbose:
            print("\n[6/10] Running Tail-Risk Fits...")
        if context.get("pipeline_abort", False):
            _stop_stage("tail_risk")
            return {"tail_df": pd.DataFrame(columns=["Ticker"])}

        decision_log = context.get("decision_log", pd.DataFrame(columns=["Ticker"]))
        tickers = decision_log["Ticker"].tolist() if not decision_log.empty else []
        tail_workers = exec_cfg.resolved_workers() if exec_cfg.enable_parallel_tail else 1
        tail_df = _compute_tail_risk_frame(
            tickers=tickers,
            verbose=verbose,
            prefetched=context.get("prefetched"),
            max_workers=tail_workers,
            enable_parallel=exec_cfg.enable_parallel_tail,
            enable_cache=exec_cfg.enable_tail_cache,
            tail_cache=tail_cache,
        )
        _stop_stage("tail_risk")
        return {"tail_df": tail_df}

    def risk_vector_node(context: Dict[str, Any]) -> Dict[str, Any]:
        timer.start("risk_vector")
        if verbose:
            print("\n[7/10] Building Unified Risk Vectors...")
        if context.get("pipeline_abort", False):
            _stop_stage("risk_vector")
            return {"decision_log": pd.DataFrame(columns=["Ticker"])}

        decision_log = context.get("decision_log", pd.DataFrame(columns=["Ticker"])).copy()
        if decision_log.empty:
            _stop_stage("risk_vector")
            return {"decision_log": decision_log}

        tail_df = context.get("tail_df", pd.DataFrame())
        if isinstance(tail_df, pd.DataFrame) and not tail_df.empty:
            decision_log = _left_merge_new_columns(decision_log, tail_df)

        momentum_df = context.get("momentum_df", pd.DataFrame())
        if isinstance(momentum_df, pd.DataFrame) and not momentum_df.empty:
            decision_log = _left_merge_new_columns(decision_log, momentum_df)

        decision_log = attach_risk_vectors(decision_log, config=cfg, include_signal=True)

        if "Decision" not in decision_log.columns:
            decision_log["Decision"] = "selected"
        if "DecisionStage" not in decision_log.columns:
            decision_log["DecisionStage"] = "strategy"
        if "RejectedReason" not in decision_log.columns:
            decision_log["RejectedReason"] = ""

        for idx, row in decision_log.iterrows():
            if str(row.get("Decision", "")).lower() == "rejected":
                continue
            rv = build_risk_vector(row, cfg)
            stage, reason = gate_stage_and_reason(rv)
            if stage == "selected":
                decision_log.at[idx, "Decision"] = "selected"
                decision_log.at[idx, "DecisionStage"] = "selected"
                decision_log.at[idx, "RejectedReason"] = ""
            else:
                decision_log.at[idx, "Decision"] = "rejected"
                decision_log.at[idx, "DecisionStage"] = stage
                decision_log.at[idx, "RejectedReason"] = reason

        decision_log = decision_log.sort_values("Ticker").reset_index(drop=True)
        validate_columns(
            decision_log,
            "risk_vector_output",
            strict=strict_contracts,
            verbose=verbose,
        )

        _stop_stage("risk_vector")
        return {"decision_log": decision_log}

    def decision_node(context: Dict[str, Any]) -> Dict[str, Any]:
        timer.start("decision")
        decision_log = context.get("decision_log", pd.DataFrame(columns=["Ticker"]))
        final_portfolio = (
            decision_log[decision_log["Decision"] == "selected"].copy()
            if not decision_log.empty and "Decision" in decision_log.columns
            else pd.DataFrame()
        )

        if verbose and not context.get("pipeline_abort", False):
            print("\n" + "=" * 80)
            print("  FINAL PORTFOLIO RECOMMENDATIONS")
            print("=" * 80)
            print(
                f"  Selected: {len(final_portfolio)} | Rejected: {len(decision_log) - len(final_portfolio)}"
                if not decision_log.empty
                else "  Selected: 0 | Rejected: 0"
            )
            display_cols = [
                "Ticker",
                "Strategy",
                "Current Price",
                "Fair Value (Bear)",
                "Fair Value",
                "Investment Signal",
                "RV_Gate_Count",
                "RV_Distress_Bucket",
                "RV_Tail_Bucket",
                "RV_Momentum_Regime",
            ]
            display_cols = [c for c in display_cols if c in final_portfolio.columns]
            if display_cols and not final_portfolio.empty:
                print(final_portfolio[display_cols].to_string(index=False))

        _stop_stage("decision")
        return {"final_portfolio": final_portfolio}

    def export_node(context: Dict[str, Any]) -> Dict[str, Any]:
        timer.start("export")
        if context.get("pipeline_abort", False):
            _stop_stage("export")
            return {"output_path": None}

        final_portfolio = context.get("final_portfolio", pd.DataFrame())
        decision_log = context.get("decision_log", pd.DataFrame())

        output_path = (
            Path(output_file)
            if output_file
            else ROOT_DIR / "buy_list_with_projections.csv"
        )
        final_portfolio.to_csv(output_path, index=False)
        if verbose:
            print(f"\nResults saved to: {output_path}")

        html_path = output_path.with_suffix(".html")
        generate_html_report(final_portfolio, str(html_path))

        if include_rejected and export_decision_log and not decision_log.empty:
            decision_export_df = decision_log
            decision_path = output_path.with_name(
                f"{output_path.stem}_decision_log.csv"
            )
            decision_export_df.to_csv(decision_path, index=False)
            if verbose:
                print(f"Decision log saved to: {decision_path}")

        _stop_stage("export")
        return {"output_path": str(output_path)}

    nodes = [
        PipelineNode("data_update_node", set(), data_update_node),
        PipelineNode("strategy_candidates_node", {"data_update_node"}, strategy_candidates_node),
        PipelineNode("prefetch_data_node", {"strategy_candidates_node"}, prefetch_data_node),
        PipelineNode(
            "forensic_node",
            {"strategy_candidates_node", "prefetch_data_node"},
            forensic_node,
        ),
        PipelineNode("safe_filter_node", {"forensic_node"}, safe_filter_node),
        PipelineNode(
            "valuation_node",
            {"safe_filter_node", "prefetch_data_node"},
            valuation_node,
        ),
        PipelineNode(
            "tail_risk_node",
            {"strategy_candidates_node", "prefetch_data_node"},
            tail_risk_node,
        ),
        PipelineNode(
            "risk_vector_node",
            {"valuation_node", "tail_risk_node", "strategy_candidates_node"},
            risk_vector_node,
        ),
        PipelineNode("decision_node", {"risk_vector_node"}, decision_node),
        PipelineNode("export_node", {"decision_node"}, export_node),
    ]

    final_context = _execute_pipeline_graph(
        nodes,
        {
            "trade_date": trade_date,
            "risk_config": cfg,
            "execution_config": exec_cfg,
            "tail_cache": tail_cache,
            "pipeline_abort": False,
        },
    )

    if verbose:
        print("\n" + "=" * 60)
        print(f"  Pipeline Complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Stage timings: {timer.format_summary()}")
        print("=" * 60)

    return final_context.get("final_portfolio", pd.DataFrame())


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Automated Investment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Run with cached data
  python main.py --update           Refresh data before running
  python main.py --exchange NASDAQ  Screen NASDAQ stocks
  python main.py --no-momentum      Include stocks below 200-day MA
  python main.py --strict           Fail on data contract violations
        """,
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Fetch fresh data before running analysis",
    )
    parser.add_argument(
        "--exchange",
        default=None,
        help="Target universe (SP500, NASDAQ, etc.)",
    )
    parser.add_argument(
        "--no-momentum",
        action="store_true",
        help="Disable momentum filter (include stocks below 200-day MA)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for results CSV",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress messages",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict data contract validation (fail on violations)",
    )
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help="Skip exporting rejected assets to the decision log",
    )
    parser.add_argument(
        "--no-decision-log",
        action="store_true",
        help="Disable decision log CSV export",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override worker count for parallel stages",
    )
    parser.add_argument(
        "--no-bulk-prefetch",
        action="store_true",
        help="Disable bulk DB prefetch optimization",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel forensic/valuation/tail execution",
    )
    parser.add_argument(
        "--no-tail-cache",
        action="store_true",
        help="Disable in-run tail fit caching",
    )
    parser.add_argument(
        "--legacy-pipeline",
        action="store_true",
        help="Use deprecated chronological pipeline for debugging",
    )

    args = parser.parse_args()
    execution_config = ExecutionConfig(
        max_workers=args.max_workers,
        enable_bulk_prefetch=not args.no_bulk_prefetch,
        enable_tail_cache=not args.no_tail_cache,
        enable_parallel_forensic=not args.no_parallel,
        enable_parallel_valuation=not args.no_parallel,
        enable_parallel_tail=not args.no_parallel,
    )

    # Run the pipeline
    result = run_pipeline(
        update_data=args.update,
        exchange=args.exchange,
        apply_momentum_filter=not args.no_momentum,
        output_file=args.output,
        verbose=not args.quiet,
        strict_contracts=args.strict,
        include_rejected=not args.selected_only,
        export_decision_log=not args.no_decision_log,
        execution_config=execution_config,
        use_legacy_debug=args.legacy_pipeline,
    )

    # Exit with appropriate code
    if result.empty:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
