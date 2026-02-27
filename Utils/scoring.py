"""
Robust Scoring Module - Uses Median Absolute Deviation (MAD) for outlier-resistant scoring.

Standard ranking (0–100) is flawed because it treats a "slightly better" stock the same as a
"massively better" one. Standard Deviation (Z-Score) is better, but it gets wrecked by outliers.

Solution: Use Median Absolute Deviation (MAD). It's like Standard Deviation, but it ignores
outliers. We then "clip" the scores at the extremes so one crazy number doesn't break the model.
"""

import pandas as pd
import numpy as np
from typing import Optional


def robust_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Converts raw metrics (like P/E or FCF Yield) into a 0-100 score.
    Uses Median Absolute Deviation (MAD) to handle outliers.

    Parameters:
    -----------
    series : pd.Series
        Raw metric values (e.g., ROIC, Earnings_Yield, Operating_Margin)
    higher_is_better : bool, default True
        If True, higher values get higher scores (e.g., ROIC, Yield)
        If False, lower values get higher scores (e.g., P/E ratio)

    Returns:
    --------
    pd.Series
        Scores normalized to 0-100 range, with outliers smoothed

    Examples:
    ---------
    >>> df['Quality_Score'] = robust_score(df['ROIC'])
    >>> df['Value_Score'] = robust_score(df['Earnings_Yield'])
    >>> df['PE_Score'] = robust_score(df['PE_Ratio'], higher_is_better=False)
    """
    # 1. Drop NaNs to calculate valid stats
    clean_series = series.dropna()
    if len(clean_series) < 3:
        return pd.Series(50, index=series.index)  # Default to neutral if no data

    # 2. Calculate Median and MAD (Robust Dispersion)
    median = clean_series.median()
    mad = (clean_series - median).abs().median()

    # 3. Calculate Z-Score (Modified)
    if mad == 0:
        # If all values are the same, return 50
        return pd.Series(50, index=series.index)

    # 0.6745 is a constant to make MAD comparable to Std Dev
    z_score = 0.6745 * (series - median) / mad

    # 4. CLIP Outliers (The "Smoothing" Magic)
    # We cap the score at +/- 3 sigmas.
    # This prevents a 5000% yield from squashing everyone else.
    z_clipped = z_score.clip(-3, 3)

    # 5. Normalize to 0-100 Range
    # -3 becomes 0, +3 becomes 100, 0 (Average) becomes 50
    final_score = ((z_clipped + 3) / 6) * 100

    # Flip if lower is better (e.g., P/E ratio)
    if not higher_is_better:
        final_score = 100 - final_score

    return final_score


def combined_robust_score(
    df: pd.DataFrame, metrics: list, weights: Optional[list] = None
) -> pd.Series:
    """
    Combine multiple metrics into a single robust score.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the metrics
    metrics : list of tuples
        Each tuple: (column_name, higher_is_better)
        e.g., [('ROIC', True), ('Earnings_Yield', True), ('PE_Ratio', False)]
    weights : list of floats, optional
        Weights for each metric. If None, equal weights are used.

    Returns:
    --------
    pd.Series
        Combined score (0-100)
    """
    if weights is None:
        weights = [1.0] * len(metrics)

    # Normalize weights to sum to 1
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    combined = pd.Series(0.0, index=df.index)

    for (col, higher_is_better), weight in zip(metrics, weights):
        score = robust_score(df[col], higher_is_better=higher_is_better)
        combined += score * weight

    return combined


def score_summary_stats(series: pd.Series, name: str = "Metric") -> dict:
    """
    Get summary statistics for a scored series - useful for debugging.

    Parameters:
    -----------
    series : pd.Series
        The raw metric values before scoring
    name : str
        Name of the metric for display

    Returns:
    --------
    dict
        Summary statistics including outlier info
    """
    clean = series.dropna()
    if len(clean) < 3:
        return {"name": name, "error": "Not enough data"}

    median = clean.median()
    mad = (clean - median).abs().median()

    # Count outliers (beyond 3 MAD)
    if mad > 0:
        z_scores = 0.6745 * (clean - median) / mad
        outliers_high = (z_scores > 3).sum()
        outliers_low = (z_scores < -3).sum()
    else:
        outliers_high = outliers_low = 0

    return {
        "name": name,
        "count": len(clean),
        "median": median,
        "mad": mad,
        "min": clean.min(),
        "max": clean.max(),
        "outliers_high": outliers_high,
        "outliers_low": outliers_low,
        "pct_outliers": (outliers_high + outliers_low) / len(clean) * 100,
    }


if __name__ == "__main__":
    # Quick test
    print("Testing robust_score function...")

    # Create test data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(10, 2, 100)  # Normal distribution
    outliers = [100, 200, -50]  # Extreme outliers
    test_data = pd.Series(list(normal_data) + outliers)

    scores = robust_score(test_data)

    print(f"\nTest Data Stats:")
    print(f"  Min: {test_data.min():.2f}, Max: {test_data.max():.2f}")
    print(f"  Mean: {test_data.mean():.2f}, Median: {test_data.median():.2f}")

    print(f"\nRobust Scores:")
    print(f"  Min Score: {scores.min():.2f}, Max Score: {scores.max():.2f}")
    print(f"  Mean Score: {scores.mean():.2f}")

    # Check that outliers are properly clipped
    print(f"\nOutlier handling:")
    print(f"  Value 100 gets score: {scores.iloc[-3]:.2f} (clipped at 100)")
    print(f"  Value 200 gets score: {scores.iloc[-2]:.2f} (clipped at 100)")
    print(f"  Value -50 gets score: {scores.iloc[-1]:.2f} (clipped at 0)")
