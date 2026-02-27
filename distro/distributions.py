"""
Kumaraswamy Laplace Distribution for Financial Returns Modeling.

This module implements the Kum-Laplace distribution using the
"Kumaraswamy generalized family" method applied to a standard Laplace distribution.

The Kum-Laplace PDF is defined as:
    f(x) = a * b * g(x) * G(x)^(a-1) * [1 - G(x)^a]^(b-1)

Where:
    - g(x) and G(x) are the PDF and CDF of the standard Laplace distribution
    - μ (mu) is the location parameter
    - σ (sigma) is the scale parameter
    - a and b are shape parameters controlling skewness and tail weight

Usage:
    model = KumaraswamyLaplace()
    params = model.fit(returns_data)
    aic, bic = model.get_aic_bic(returns_data)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import laplace, norm
import warnings
import os
import pandas as pd
from typing import Optional, List

# Import centralized database module
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from database.database import get_ticker_history, get_all_tickers

warnings.filterwarnings("ignore")


class KumaraswamyLaplace:
    """
    Implements the Kumaraswamy Laplace distribution for modeling
    financial returns with heavy tails and asymmetry.

    Attributes:
        params: Tuple of (mu, sigma, a, b) after fitting
        fitted: Boolean indicating if model has been fitted
    """

    def __init__(self):
        self.params = None
        self.fitted = False
        self._optimization_result = None

    def _laplace_pdf(self, x, mu, sigma):
        """
        Standard Laplace PDF g(x).

        Args:
            x: Input values
            mu: Location parameter
            sigma: Scale parameter

        Returns:
            PDF values at x
        """
        return laplace.pdf(x, loc=mu, scale=sigma)

    def _laplace_cdf(self, x, mu, sigma):
        """
        Standard Laplace CDF G(x).

        Args:
            x: Input values
            mu: Location parameter
            sigma: Scale parameter

        Returns:
            CDF values at x
        """
        return laplace.cdf(x, loc=mu, scale=sigma)

    def pdf(self, x, mu=None, sigma=None, a=None, b=None):
        """
        The Kumaraswamy Laplace PDF f(x).

        Formula: f(x) = a * b * g(x) * G(x)^(a-1) * [1 - G(x)^a]^(b-1)

        Args:
            x: Input values (array-like)
            mu: Location parameter (uses fitted if None)
            sigma: Scale parameter (uses fitted if None)
            a: Shape parameter 1 (uses fitted if None)
            b: Shape parameter 2 (uses fitted if None)

        Returns:
            PDF values at x
        """
        # Use fitted params if not provided
        if mu is None or sigma is None or a is None or b is None:
            if self.params is None:
                raise ValueError(
                    "Model must be fitted first or parameters must be provided."
                )
            mu, sigma, a, b = self.params

        x = np.asarray(x)

        # Ensure positive constraints for computation
        if sigma <= 0 or a <= 0 or b <= 0:
            return np.zeros_like(x, dtype=float)

        g_x = self._laplace_pdf(x, mu, sigma)
        G_x = self._laplace_cdf(x, mu, sigma)

        # Avoid numerical errors with log of zero by clipping G(x)
        G_x = np.clip(G_x, 1e-10, 1 - 1e-10)

        term1 = a * b * g_x
        term2 = np.power(G_x, a - 1)
        term3 = np.power(1 - np.power(G_x, a), b - 1)

        return term1 * term2 * term3

    def cdf(self, x, mu=None, sigma=None, a=None, b=None):
        """
        The Kumaraswamy Laplace CDF F(x).

        Formula: F(x) = 1 - [1 - G(x)^a]^b

        Args:
            x: Input values (array-like)
            mu: Location parameter (uses fitted if None)
            sigma: Scale parameter (uses fitted if None)
            a: Shape parameter 1 (uses fitted if None)
            b: Shape parameter 2 (uses fitted if None)

        Returns:
            CDF values at x
        """
        if mu is None or sigma is None or a is None or b is None:
            if self.params is None:
                raise ValueError(
                    "Model must be fitted first or parameters must be provided."
                )
            mu, sigma, a, b = self.params

        x = np.asarray(x)
        G_x = self._laplace_cdf(x, mu, sigma)
        G_x = np.clip(G_x, 1e-10, 1 - 1e-10)

        return 1 - np.power(1 - np.power(G_x, a), b)

    def neg_log_likelihood(self, params, data):
        """
        Calculates Negative Log-Likelihood (NLL) for optimization.

        Args:
            params: Tuple of (mu, sigma, a, b)
            data: Observed data array

        Returns:
            Negative log-likelihood value
        """
        mu, sigma, a, b = params

        # Enforce parameter constraints (sigma, a, b > 0)
        if sigma <= 0 or a <= 0 or b <= 0:
            return 1e10

        pdf_vals = self.pdf(data, mu, sigma, a, b)

        # Filter out zero probabilities to avoid log(0)
        pdf_vals = pdf_vals[pdf_vals > 0]

        if len(pdf_vals) == 0:
            return 1e10

        return -np.sum(np.log(pdf_vals))

    def fit(self, data, initial_guess=None):
        """
        Fits the distribution to data using Maximum Likelihood Estimation (MLE).

        Args:
            data: Array of observed values (e.g., log returns)
            initial_guess: Optional initial parameters [mu, sigma, a, b]

        Returns:
            Tuple of fitted parameters (mu, sigma, a, b)
        """
        data = np.asarray(data).flatten()
        data = data[np.isfinite(data)]  # Remove NaN/Inf values

        if len(data) < 10:
            raise ValueError("Insufficient data points for fitting (need at least 10).")

        # Initial Guesses:
        # mu = median of data (robust to outliers, Laplace property)
        # sigma = MAD (median absolute deviation) for robust scale estimate
        # a, b = 1.0 (defaults to standard Laplace at 1,1)
        if initial_guess is None:
            mu_init = np.median(data)
            sigma_init = (
                np.median(np.abs(data - mu_init)) * 1.4826
            )  # MAD to std conversion
            if sigma_init < 1e-6:
                sigma_init = np.std(data)
            initial_guess = [mu_init, max(sigma_init, 1e-5), 1.0, 1.0]

        # Bounds: mu (unbounded), sigma > 0, a > 0, b > 0
        bounds = [(None, None), (1e-6, None), (1e-6, 50), (1e-6, 50)]

        result = minimize(
            self.neg_log_likelihood,
            initial_guess,
            args=(data,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-8},
        )

        if result.x is not None and len(result.x) == 4:
            self.params = (
                float(result.x[0]),
                float(result.x[1]),
                float(result.x[2]),
                float(result.x[3]),
            )
            self.fitted = True
        else:
            raise RuntimeError("Optimization failed to produce valid parameters.")

        self._optimization_result = result

        return self.params

    def get_log_likelihood(self, data):
        """
        Calculate log-likelihood for fitted model.

        Args:
            data: Observed data array

        Returns:
            Log-likelihood value
        """
        if self.params is None:
            raise ValueError("Model must be fitted first.")
        return -self.neg_log_likelihood(self.params, data)

    def get_aic_bic(self, data):
        """
        Calculates AIC and BIC for model comparison.

        AIC = 2k - 2*log(L)
        BIC = k*log(n) - 2*log(L)

        Args:
            data: Observed data array

        Returns:
            Tuple of (AIC, BIC)
        """
        if self.params is None:
            raise ValueError("Model must be fitted first.")

        k = 4  # Number of parameters (mu, sigma, a, b)
        data = np.asarray(data).flatten()
        data = data[np.isfinite(data)]
        n = len(data)

        log_likelihood = self.get_log_likelihood(data)

        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        return aic, bic

    def get_params_dict(self):
        """
        Get fitted parameters as a dictionary.

        Returns:
            Dictionary with parameter names and values, or None if not fitted
        """
        if self.params is None:
            return None
        return {
            "mu": self.params[0],
            "sigma": self.params[1],
            "a": self.params[2],
            "b": self.params[3],
        }

    def interpret_params(self):
        """
        Interpret fitted parameters for risk assessment.

        Analyzes the shape parameters (a, b) to determine tail risk characteristics:
        - a < 1: Heavier right tail (larger upside surprises)
        - a > 1: Lighter right tail
        - b < 1: Heavier left tail (larger downside risk)
        - b > 1: Lighter left tail

        Returns:
            Dictionary with risk interpretation:
            - tail_weight: "Heavy", "Normal", or "Thin"
            - skew_direction: "Left", "Right", or "Symmetric"
            - var_adjustment: Recommended VaR adjustment factor
            - risk_summary: Human-readable risk summary
        """
        if self.params is None or not self.fitted:
            raise ValueError("Model must be fitted first. Call fit() first.")

        mu, sigma, a, b = self.params

        # Determine tail weight based on shape parameters
        # When both a and b are close to 1, it's standard Laplace (already fat-tailed)
        avg_shape = (a + b) / 2

        if a < 0.8 or b < 0.8:
            tail_weight = "Heavy"
            var_adjustment = 1.3  # Increase VaR by 30%
        elif a > 1.5 and b > 1.5:
            tail_weight = "Thin"
            var_adjustment = 0.9  # Decrease VaR by 10%
        else:
            tail_weight = "Normal"
            var_adjustment = 1.0

        # Determine skewness direction
        # Lower 'a' = heavier right tail (positive skew potential)
        # Lower 'b' = heavier left tail (negative skew / downside risk)
        skew_diff = a - b
        if skew_diff > 0.3:
            skew_direction = "Left"  # b < a means heavier left tail
        elif skew_diff < -0.3:
            skew_direction = "Right"  # a < b means heavier right tail
        else:
            skew_direction = "Symmetric"

        # Build risk summary
        if tail_weight == "Heavy" and b < 0.8:
            risk_summary = (
                "HIGH RISK: Heavy left tail indicates significant downside risk"
            )
        elif tail_weight == "Heavy":
            risk_summary = "ELEVATED RISK: Fat tails present, expect larger moves"
        elif tail_weight == "Thin":
            risk_summary = "LOW RISK: Thin tails suggest stable, predictable returns"
        else:
            risk_summary = "MODERATE RISK: Normal tail behavior"

        return {
            "tail_weight": tail_weight,
            "skew_direction": skew_direction,
            "var_adjustment": var_adjustment,
            "risk_summary": risk_summary,
            "a": a,
            "b": b,
            "sigma": sigma,
        }

    def summary(self, data):
        """
        Print a summary of the fitted model.

        Args:
            data: Observed data array used for fitting
        """
        if self.params is None or not self.fitted:
            print("Model not fitted yet. Call fit() first.")
            return

        mu, sigma, a, b = self.params
        aic, bic = self.get_aic_bic(data)
        log_lik = self.get_log_likelihood(data)

        print("=" * 50)
        print("Kumaraswamy Laplace Distribution - Fit Summary")
        print("=" * 50)
        print(f"\nEstimated Parameters:")
        print(f"  μ (Location):  {mu:12.6f}")
        print(f"  σ (Scale):     {sigma:12.6f}")
        print(f"  a (Shape 1):   {a:12.6f}")
        print(f"  b (Shape 2):   {b:12.6f}")
        print(f"\nGoodness of Fit:")
        print(f"  Log-Likelihood: {log_lik:12.4f}")
        print(f"  AIC:            {aic:12.4f}")
        print(f"  BIC:            {bic:12.4f}")
        print(f"\nInterpretation:")
        if abs(a - 1) < 0.1 and abs(b - 1) < 0.1:
            print("  → Returns behave like a standard Laplace distribution.")
        else:
            if a > 1:
                print(f"  → a={a:.2f} > 1: Right tail is lighter than Laplace.")
            elif a < 1:
                print(f"  → a={a:.2f} < 1: Right tail is heavier than Laplace.")
            if b > 1:
                print(f"  → b={b:.2f} > 1: Left tail is lighter than Laplace.")
            elif b < 1:
                print(f"  → b={b:.2f} < 1: Left tail is heavier than Laplace.")
        print("=" * 50)


class NormalDistribution:
    """
    Standard Normal distribution wrapper for comparison with Kum-Laplace.
    """

    def __init__(self):
        self.params = None
        self.fitted = False

    def fit(self, data):
        """Fit Normal distribution to data."""
        data = np.asarray(data).flatten()
        data = data[np.isfinite(data)]
        mu = np.mean(data)
        sigma = np.std(data, ddof=1)
        self.params = (mu, sigma)
        self.fitted = True
        return self.params

    def pdf(self, x, mu=None, sigma=None):
        """Normal PDF."""
        if mu is None or sigma is None:
            if self.params is None:
                raise ValueError("Model must be fitted first.")
            mu, sigma = self.params
        return norm.pdf(x, loc=mu, scale=sigma)

    def get_log_likelihood(self, data):
        """Calculate log-likelihood."""
        if self.params is None:
            raise ValueError("Model must be fitted first.")
        mu, sigma = self.params
        data = np.asarray(data).flatten()
        data = data[np.isfinite(data)]
        return np.sum(norm.logpdf(data, loc=mu, scale=sigma))

    def get_aic_bic(self, data):
        """Calculate AIC and BIC."""
        k = 2  # mu, sigma
        data = np.asarray(data).flatten()
        data = data[np.isfinite(data)]
        n = len(data)
        log_lik = self.get_log_likelihood(data)
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n) - 2 * log_lik
        return aic, bic


class LaplaceDistribution:
    """
    Standard Laplace distribution wrapper for comparison with Kum-Laplace.
    """

    def __init__(self):
        self.params = None
        self.fitted = False

    def fit(self, data):
        """Fit Laplace distribution to data (MLE: median for loc, MAD for scale)."""
        data = np.asarray(data).flatten()
        data = data[np.isfinite(data)]
        mu = np.median(data)
        sigma = np.mean(np.abs(data - mu))  # MAD is MLE for Laplace scale
        self.params = (mu, sigma)
        self.fitted = True
        return self.params

    def pdf(self, x, mu=None, sigma=None):
        """Laplace PDF."""
        if mu is None or sigma is None:
            if self.params is None:
                raise ValueError("Model must be fitted first.")
            mu, sigma = self.params
        return laplace.pdf(x, loc=mu, scale=sigma)

    def get_log_likelihood(self, data):
        """Calculate log-likelihood."""
        if self.params is None:
            raise ValueError("Model must be fitted first.")
        mu, sigma = self.params
        data = np.asarray(data).flatten()
        data = data[np.isfinite(data)]
        return np.sum(laplace.logpdf(data, loc=mu, scale=sigma))

    def get_aic_bic(self, data):
        """Calculate AIC and BIC."""
        k = 2  # mu, sigma
        data = np.asarray(data).flatten()
        data = data[np.isfinite(data)]
        n = len(data)
        log_lik = self.get_log_likelihood(data)
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n) - 2 * log_lik
        return aic, bic


def compare_distributions(returns, verbose=True):
    """
    Compare Normal, Laplace, and Kumaraswamy-Laplace distributions
    fitted to the same return data.

    Args:
        returns: Array of log returns
        verbose: Print comparison table

    Returns:
        Dictionary with fitted models and comparison metrics
    """
    returns = np.asarray(returns).flatten()
    returns = returns[np.isfinite(returns)]

    # Fit all three distributions
    normal = NormalDistribution()
    normal.fit(returns)
    normal_aic, normal_bic = normal.get_aic_bic(returns)

    laplace_dist = LaplaceDistribution()
    laplace_dist.fit(returns)
    laplace_aic, laplace_bic = laplace_dist.get_aic_bic(returns)

    kum_laplace = KumaraswamyLaplace()
    kum_laplace.fit(returns)
    kum_aic, kum_bic = kum_laplace.get_aic_bic(returns)

    results: dict[str, object] = {
        "normal": {
            "model": normal,
            "params": normal.params,
            "aic": normal_aic,
            "bic": normal_bic,
            "log_likelihood": normal.get_log_likelihood(returns),
        },
        "laplace": {
            "model": laplace_dist,
            "params": laplace_dist.params,
            "aic": laplace_aic,
            "bic": laplace_bic,
            "log_likelihood": laplace_dist.get_log_likelihood(returns),
        },
        "kumaraswamy_laplace": {
            "model": kum_laplace,
            "params": kum_laplace.params,
            "aic": kum_aic,
            "bic": kum_bic,
            "log_likelihood": kum_laplace.get_log_likelihood(returns),
        },
    }

    # Determine best model
    aic_values = {"Normal": normal_aic, "Laplace": laplace_aic, "Kum-Laplace": kum_aic}
    best_model = min(aic_values, key=lambda k: aic_values[k])
    results["best_model"] = best_model

    if verbose:
        print("\n" + "=" * 65)
        print("Distribution Comparison for Stock Returns")
        print("=" * 65)
        print(f"{'Model':<20} {'Log-Lik':>12} {'AIC':>12} {'BIC':>12}")
        print("-" * 65)
        print(
            f"{'Normal':<20} {normal.get_log_likelihood(returns):>12.2f} {normal_aic:>12.2f} {normal_bic:>12.2f}"
        )
        print(
            f"{'Laplace':<20} {laplace_dist.get_log_likelihood(returns):>12.2f} {laplace_aic:>12.2f} {laplace_bic:>12.2f}"
        )
        print(
            f"{'Kumaraswamy-Laplace':<20} {kum_laplace.get_log_likelihood(returns):>12.2f} {kum_aic:>12.2f} {kum_bic:>12.2f}"
        )
        print("-" * 65)
        print(f"Best Model (lowest AIC): {best_model}")
        print("=" * 65)

        # Print Kum-Laplace parameters
        if kum_laplace.params is not None:
            mu, sigma, a, b = kum_laplace.params
            print(f"\nKumaraswamy-Laplace Parameters:")
            print(f"  μ={mu:.6f}, σ={sigma:.6f}, a={a:.4f}, b={b:.4f}")

    return results


def calculate_log_returns(prices):
    """
    Calculate log returns from price series.

    Formula: r_t = ln(S_t / S_{t-1})

    Args:
        prices: Array or Series of prices

    Returns:
        Array of log returns
    """
    prices = np.asarray(prices).flatten()
    prices = prices[prices > 0]  # Filter non-positive prices
    return np.diff(np.log(prices))


# --- DATABASE-DRIVEN MODEL FITTING ---


def get_ticker_close_prices(symbol: str) -> Optional[pd.DataFrame]:
    """
    Get historical close prices for a ticker from the database.

    Args:
        symbol: Ticker symbol (e.g., 'AAPL')

    Returns:
        DataFrame with date and close columns, or None if not found
    """
    df = get_ticker_history(symbol)
    if df is None or df.empty:
        return None
    # Return only date and close columns for compatibility
    return df[["date", "close"]]


def fit_distribution_for_ticker(symbol, min_data_points=100):
    """
    Fit the Kumaraswamy-Laplace distribution to a ticker's returns.

    Args:
        symbol: Ticker symbol
        min_data_points: Minimum data points required for fitting

    Returns:
        Dictionary with results, or None if fitting failed
    """
    # Fetch price data
    df = get_ticker_close_prices(symbol)

    if df is None or len(df) < min_data_points:
        return None

    # Calculate log returns
    prices = np.asarray(df["close"].values)
    prices = prices[prices > 0]  # Filter non-positive prices

    if len(prices) < min_data_points:
        return None

    log_returns = np.diff(np.log(prices))

    # Remove zeros (days with no price change)
    clean_returns = log_returns[log_returns != 0]
    clean_returns = clean_returns[np.isfinite(clean_returns)]

    if len(clean_returns) < min_data_points:
        return None

    # Fit the model
    model = KumaraswamyLaplace()
    try:
        params = model.fit(clean_returns)
        mu, sigma, a, b = params
        aic, bic = model.get_aic_bic(clean_returns)

        return {
            "ticker": symbol,
            "mu": mu,
            "sigma": sigma,
            "a_shape": a,
            "b_shape": b,
            "aic": aic,
            "bic": bic,
            "n_observations": len(clean_returns),
        }
    except Exception as e:
        print(f"  Error fitting {symbol}: {e}")
        return None


def process_tickers_from_buy_list(
    buy_list_df: pd.DataFrame, min_data_points: int = 100, verbose: bool = True
) -> pd.DataFrame:
    """
    Process tickers from a buy list DataFrame.

    This is the primary function-based API for distribution fitting.

    Args:
        buy_list_df: DataFrame with at least a 'Ticker' column
        min_data_points: Minimum data points required for fitting
        verbose: Print progress information

    Returns:
        DataFrame with fitted parameters for all tickers
    """
    if buy_list_df.empty:
        if verbose:
            print("Empty buy list provided.")
        return pd.DataFrame()

    if "Ticker" not in buy_list_df.columns:
        raise ValueError("buy_list_df must contain a 'Ticker' column")

    tickers = buy_list_df["Ticker"].tolist()

    if verbose:
        print(f"Processing {len(tickers)} tickers...")
        print("=" * 60)

    results = []

    for i, symbol in enumerate(tickers, 1):
        if verbose:
            print(f"[{i}/{len(tickers)}] Processing {symbol}...", end=" ")

        result = fit_distribution_for_ticker(symbol, min_data_points)

        if result:
            results.append(result)
            if verbose:
                print(
                    f"✓ a={result['a_shape']:.4f}, b={result['b_shape']:.4f}, AIC={result['aic']:.2f}"
                )
        else:
            if verbose:
                print("✗ Skipped (insufficient data)")

    if not results:
        if verbose:
            print("No tickers were successfully fitted.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    if verbose:
        _print_distribution_report(results_df)

    return results_df


def process_all_database_tickers(
    min_data_points: int = 100, exchange: Optional[str] = None, verbose: bool = True
) -> pd.DataFrame:
    """
    Process all tickers from the database.

    Args:
        min_data_points: Minimum data points required for fitting
        exchange: Optional exchange filter (e.g., 'SP500')
        verbose: Print progress information

    Returns:
        DataFrame with fitted parameters for all tickers
    """
    tickers_df = get_all_tickers(exchange=exchange)

    if tickers_df.empty:
        if verbose:
            print("No tickers found in database.")
        return pd.DataFrame()

    # Create DataFrame with Ticker column for compatibility
    buy_list = pd.DataFrame({"Ticker": tickers_df["symbol"].tolist()})
    return process_tickers_from_buy_list(buy_list, min_data_points, verbose)


def _print_distribution_report(results_df: pd.DataFrame) -> None:
    """Print distribution fitting report."""
    print("\n" + "=" * 60)
    print("RISK PARAMETER SUMMARY")
    print("=" * 60)
    print(f"\nSuccessfully fitted: {len(results_df)} tickers")
    print(f"\nAll Results (sorted by AIC):")
    print(results_df.sort_values(by="aic").to_string(index=False))

    # Risk categorization
    print("\n" + "-" * 60)
    print("TAIL RISK ANALYSIS")
    print("-" * 60)

    heavy_tails = results_df[
        (results_df["a_shape"] < 0.8) | (results_df["b_shape"] < 0.8)
    ]
    if len(heavy_tails) > 0:
        print(f"\n⚠️  Heavy tail stocks ({len(heavy_tails)} tickers):")
        print("   These have higher probability of extreme events!")
        print(
            heavy_tails[["ticker", "a_shape", "b_shape", "aic"]].to_string(index=False)
        )

    right_skewed = results_df[results_df["a_shape"] < 0.9]
    left_skewed = results_df[results_df["b_shape"] < 0.9]

    print(f"\n📈 Positively skewed (a < 0.9): {len(right_skewed)} tickers")
    print(f"📉 Negatively skewed (b < 0.9): {len(left_skewed)} tickers")


def interpret_risk_params(a, b):
    """
    Interpret the shape parameters a and b for risk assessment.

    Args:
        a: Shape parameter a
        b: Shape parameter b

    Returns:
        Dictionary with interpretation
    """
    interpretation = {
        "a": a,
        "b": b,
        "right_tail": "normal" if abs(a - 1) < 0.2 else ("light" if a > 1 else "heavy"),
        "left_tail": "normal" if abs(b - 1) < 0.2 else ("light" if b > 1 else "heavy"),
        "skewness": (
            "symmetric" if abs(a - b) < 0.2 else ("positive" if a < b else "negative")
        ),
        "var_adjustment": "none",
    }

    # VaR adjustment recommendation
    if a < 0.8 or b < 0.8:
        interpretation["var_adjustment"] = "increase VaR limits significantly (1.5x-2x)"
    elif a < 0.95 or b < 0.95:
        interpretation["var_adjustment"] = "moderately increase VaR limits (1.2x-1.5x)"

    return interpretation


if __name__ == "__main__":
    # Demo with synthetic data
    print("Kumaraswamy Laplace Distribution - Demo")
    print("-" * 40)

    # Generate synthetic stock returns (Laplace-like with heavy tails)
    np.random.seed(42)
    synthetic_returns = np.random.laplace(loc=0.001, scale=0.02, size=1000)

    # Fit and summarize
    model = KumaraswamyLaplace()
    model.fit(synthetic_returns)
    model.summary(synthetic_returns)

    # Compare with other distributions
    print("\n")
    compare_distributions(synthetic_returns)

    # --- Process Tickers from Database ---
    print("\n" + "=" * 60)
    print("DATABASE DISTRIBUTION FITTING")
    print("=" * 60)

    # Get all tickers from database and process
    print("\nProcessing tickers from database...\n")
    results_df = process_all_database_tickers(min_data_points=100, exchange="SP500")

    # Save results to CSV
    if not results_df.empty:
        output_path = os.path.join(os.path.dirname(__file__), "distribution_params.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    else:
        print("\n⚠️  No tickers found in database or insufficient data for fitting.")
