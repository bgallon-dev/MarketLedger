"""
Financial Screening Script for S&P 500 Stocks

This script screens S&P 500 tickers based on the following criteria:
- P/E Ratio (TTM) < 25
- Return on Equity (TTM) > 15%
- Gross Margin (TTM) > 40%
- Free Cash Flow Growth (TTM YoY) > 0%
- Net Income Growth (TTM YoY) > 0%
- Operating Margin (FY) > 20%

Results are saved to a timestamped CSV file in the results/ directory.
"""

import pandas as pd
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Import centralized database module
sys.path.insert(0, str(Path(__file__).parent.parent))
from database.database import get_ticker_history, get_financial_data, get_all_tickers


class TickerScreener:
    """Screens tickers based on financial criteria."""

    def __init__(self, exchange: Optional[str] = "SP500"):
        """
        Initialize the screener.

        Args:
            exchange: Exchange filter for tickers (default: 'SP500')
        """
        self.exchange = exchange
        self.results = []

    def _get_data(self, ticker: str, table_name: str) -> Optional[pd.DataFrame]:
        """
        Load financial data for a ticker from the database.

        Args:
            ticker: Stock ticker symbol
            table_name: Name of table ('balance_sheet', 'income_statement', 'cash_flow', 'history')

        Returns:
            DataFrame or None if not found
        """
        try:
            if table_name == "history":
                df = get_ticker_history(ticker)
                if df is not None and not df.empty:
                    return df
            else:
                df = get_financial_data(ticker, table_name)
                if df is not None and not df.empty:
                    # Remove duplicate index rows to ensure .loc[] returns a Series, not a DataFrame
                    df = df[~df.index.duplicated(keep="first")]
                    return df
            return None
        except Exception as e:
            print(f"Error loading {table_name} for {ticker}: {e}")
            return None

    def get_ttm_value(self, df, metric_name, num_periods=4):
        """
        Calculate TTM (Trailing Twelve Months) value by summing last 4 quarters.

        Args:
            df: DataFrame with metrics as rows and dates as columns
            metric_name: Name of the metric row to sum
            num_periods: Number of periods to sum (default 4 for quarterly data)

        Returns:
            TTM value or None if insufficient data
        """
        if df is None or metric_name not in df.index:
            return None

        row = df.loc[metric_name]
        # Ensure we are working with numeric data
        valid_values = pd.to_numeric(row, errors="coerce").dropna()

        if len(valid_values) < num_periods:
            # Try with available data if we have at least 2 quarters
            if len(valid_values) >= 2:
                # FIX: Explicitly cast to float
                return float(valid_values.iloc[:num_periods].sum())
            return None

        # FIX: Explicitly cast to float
        return float(valid_values.iloc[:num_periods].sum())

    def get_most_recent_value(self, df, metric_name):
        """
        Get the most recent non-NaN value for a metric.

        Args:
            df: DataFrame with metrics as rows and dates as columns
            metric_name: Name of the metric row

        Returns:
            Most recent value or None if not available
        """
        if df is None or metric_name not in df.index:
            return None

        row = df.loc[metric_name]
        valid_values = pd.to_numeric(row, errors="coerce").dropna()

        if len(valid_values) > 0:
            # FIX: Explicitly cast to float
            return float(valid_values.iloc[0])
        return None

    def calculate_pe_ratio(self, ticker):
        """
        Calculate P/E ratio using most recent close price and TTM Basic EPS.

        Returns:
            P/E ratio or None
        """
        # Get most recent close price
        history = self._get_data(ticker, "history")
        if history is None or len(history) == 0:
            return None

        try:
            recent_price = float(history["close"].iloc[-1])
        except:
            return None

        # Get TTM Basic EPS
        income = self._get_data(ticker, "income_statement")
        ttm_eps = self.get_ttm_value(income, "BasicEPS")

        if ttm_eps is None or ttm_eps <= 0:
            return None

        return recent_price / ttm_eps

    def calculate_roe(self, ticker):
        """
        Calculate Return on Equity (TTM) = Net Income (TTM) / Stockholders Equity (most recent).

        Returns:
            ROE as percentage or None
        """
        income = self._get_data(ticker, "income_statement")
        balance = self._get_data(ticker, "balance_sheet")

        ttm_net_income = self.get_ttm_value(
            income, "NetIncomeFromContinuingAndDiscontinuedOperation"
        )
        if ttm_net_income is None:
            # Try alternative net income field
            ttm_net_income = self.get_ttm_value(income, "NetIncome")

        equity = self.get_most_recent_value(balance, "StockholdersEquity")

        if ttm_net_income is None or equity is None or equity <= 0:
            return None

        return (ttm_net_income / equity) * 100

    def calculate_gross_margin(self, ticker):
        """
        Calculate Gross Margin (TTM) = Gross Profit (TTM) / Total Revenue (TTM).

        Returns:
            Gross margin as percentage or None
        """
        income = self._get_data(ticker, "income_statement")

        ttm_gross_profit = self.get_ttm_value(income, "GrossProfit")
        ttm_revenue = self.get_ttm_value(income, "TotalRevenue")

        if ttm_gross_profit is None or ttm_revenue is None or ttm_revenue <= 0:
            return None

        return (ttm_gross_profit / ttm_revenue) * 100

    def calculate_fcf_growth(self, ticker):
        """
        Calculate Free Cash Flow Growth (YoY) comparing current TTM to prior TTM.

        Returns:
            FCF growth as percentage or None
        """
        cash_flow = self._get_data(ticker, "cash_flow")

        if cash_flow is None or "FreeCashFlow" not in cash_flow.index:
            return None

        fcf_row = cash_flow.loc["FreeCashFlow"]
        if not isinstance(fcf_row, pd.Series):
            return None

        valid_values = pd.to_numeric(fcf_row, errors="coerce").dropna()

        if len(valid_values) < 8:
            # Try with available data if we have at least 6 quarters
            if len(valid_values) < 6:
                return None

        # Current TTM: quarters 0-3
        current_ttm: float = float(valid_values.iloc[:4].sum())
        # Prior TTM: quarters 4-7
        prior_ttm: float = float(valid_values.iloc[4:8].sum())

        if prior_ttm == 0.0:
            return None

        return float(((current_ttm - prior_ttm) / abs(prior_ttm)) * 100)

    def calculate_net_income_growth(self, ticker):
        """
        Calculate Net Income Growth (YoY) comparing current TTM to prior TTM.

        Returns:
            Net income growth as percentage or None
        """
        income = self._get_data(ticker, "income_statement")

        if income is None:
            return None

        # Try primary net income field
        if "NetIncomeFromContinuingAndDiscontinuedOperation" in income.index:
            ni_row = income.loc["NetIncomeFromContinuingAndDiscontinuedOperation"]
        elif "NetIncome" in income.index:
            ni_row = income.loc["NetIncome"]
        else:
            return None

        if not isinstance(ni_row, pd.Series):
            return None

        valid_values = pd.to_numeric(ni_row, errors="coerce").dropna()

        if len(valid_values) < 8:
            if len(valid_values) < 6:
                return None

        # Current TTM: quarters 0-3
        current_ttm: float = float(valid_values.iloc[:4].sum())
        # Prior TTM: quarters 4-7
        prior_ttm: float = float(valid_values.iloc[4:8].sum())

        if prior_ttm == 0.0:
            return None

        return float(((current_ttm - prior_ttm) / abs(prior_ttm)) * 100)

    def calculate_operating_margin(self, ticker):
        """
        Calculate Operating Margin (most recent quarter) = Operating Income / Total Revenue.

        Returns:
            Operating margin as percentage or None
        """
        income = self._get_data(ticker, "income_statement")

        operating_income = self.get_most_recent_value(income, "OperatingIncome")
        if operating_income is None:
            operating_income = self.get_most_recent_value(
                income, "TotalOperatingIncomeAsReported"
            )

        revenue = self.get_most_recent_value(income, "TotalRevenue")

        if operating_income is None or revenue is None or revenue <= 0:
            return None

        return (operating_income / revenue) * 100

    def screen_ticker(self, ticker):
        """
        Screen a single ticker against all criteria.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with ticker results
        """
        result = {
            "ticker": ticker,
            "pe_ratio": None,
            "roe_pct": None,
            "gross_margin_pct": None,
            "fcf_growth_yoy_pct": None,
            "ni_growth_yoy_pct": None,
            "operating_margin_pct": None,
            "passes_all_criteria": False,
            "data_quality_flags": [],
        }

        # Calculate all metrics
        result["pe_ratio"] = self.calculate_pe_ratio(ticker)
        result["roe_pct"] = self.calculate_roe(ticker)
        result["gross_margin_pct"] = self.calculate_gross_margin(ticker)
        result["fcf_growth_yoy_pct"] = self.calculate_fcf_growth(ticker)
        result["ni_growth_yoy_pct"] = self.calculate_net_income_growth(ticker)
        result["operating_margin_pct"] = self.calculate_operating_margin(ticker)

        # Track missing data
        if result["pe_ratio"] is None:
            result["data_quality_flags"].append("Missing P/E")
        if result["roe_pct"] is None:
            result["data_quality_flags"].append("Missing ROE")
        if result["gross_margin_pct"] is None:
            result["data_quality_flags"].append("Missing Gross Margin")
        if result["fcf_growth_yoy_pct"] is None:
            result["data_quality_flags"].append("Missing FCF Growth")
        if result["ni_growth_yoy_pct"] is None:
            result["data_quality_flags"].append("Missing NI Growth")
        if result["operating_margin_pct"] is None:
            result["data_quality_flags"].append("Missing Op Margin")

        # Check if passes all criteria
        passes = True

        if result["pe_ratio"] is None or float(result["pe_ratio"]) >= 25:
            passes = False
        if result["roe_pct"] is None or float(result["roe_pct"]) <= 15:
            passes = False
        if (
            result["gross_margin_pct"] is None
            or float(result["gross_margin_pct"]) <= 40
        ):
            passes = False
        if (
            result["fcf_growth_yoy_pct"] is None
            or float(result["fcf_growth_yoy_pct"]) <= 0
        ):
            passes = False
        if (
            result["ni_growth_yoy_pct"] is None
            or float(result["ni_growth_yoy_pct"]) <= 0
        ):
            passes = False
        if (
            result["operating_margin_pct"] is None
            or float(result["operating_margin_pct"]) <= 20
        ):
            passes = False

        result["passes_all_criteria"] = passes
        result["data_quality_flags"] = (
            ", ".join(result["data_quality_flags"])
            if result["data_quality_flags"]
            else "Complete"
        )

        return result

    def run_screening(self, tickers: Optional[List[str]] = None):
        """
        Run screening on all tickers from database or provided list.

        Args:
            tickers: Optional list of ticker symbols. If None, uses database tickers.

        Returns:
            List of result dictionaries
        """
        # Get tickers from database if not provided
        if tickers is None:
            tickers_df = get_all_tickers(exchange=self.exchange)
            if tickers_df.empty:
                print("No tickers found in database.")
                return []
            tickers = tickers_df["symbol"].tolist()

        print(f"Screening {len(tickers)} tickers...")

        for i, ticker in enumerate(tickers, 1):
            if i % 50 == 0:
                print(f"Processed {i}/{len(tickers)} tickers...")

            result = self.screen_ticker(ticker)
            self.results.append(result)

        print(f"Screening complete! Processed {len(tickers)} tickers.")
        return self.results

    def save_results(self, output_dir="results"):
        """
        Save screening results to a timestamped CSV file.

        Args:
            output_dir: Directory to save results (will be created if doesn't exist)
        """
        # Create results directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screener_results_{timestamp}.csv"
        filepath = output_path / filename

        # Convert results to DataFrame
        df = pd.DataFrame(self.results)

        # Save to CSV
        df.to_csv(filepath, index=False, float_format="%.2f")

        # Print summary
        passing_count = int(df["passes_all_criteria"].sum())
        print(f"\n{'='*60}")
        print(f"Results saved to: {filepath}")
        print(f"{'='*60}")
        print(f"Total tickers screened: {len(df)}")
        print(f"Tickers passing all criteria: {passing_count}")
        print(f"Pass rate: {passing_count/len(df)*100:.1f}%")
        print(f"{'='*60}\n")

        # Show passing tickers
        if passing_count > 0:
            passing_tickers = df[df["passes_all_criteria"]]["ticker"].tolist()
            print(f"Tickers passing all criteria: {', '.join(passing_tickers)}")
        else:
            print("No tickers passed all criteria.")

        return filepath


def main():
    """Main execution function."""
    # Set up paths relative to script location
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "results"

    # Create screener using database
    screener = TickerScreener(exchange="SP500")

    # Run screening
    screener.run_screening()

    # Save results
    screener.save_results(str(output_dir))


if __name__ == "__main__":
    main()
