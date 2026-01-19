"""Lag and rolling window features for time-series forecasting"""

import numpy as np
import pandas as pd
from typing import List


def create_lag_features(
    df: pd.DataFrame,
    lag_periods: List[int] = [1, 7, 365],
    rolling_windows: List[int] = [7],
    group_cols: List[str] = ["store", "item"],
) -> pd.DataFrame:
    """
    Create lag and rolling window features for time-series forecasting.

    CRITICAL: All features use .shift() to prevent data leakage. Features
    only use historical data that would be available at prediction time.

    Args:
        df: DataFrame sorted by group_cols + ['date']
        lag_periods: List of lag periods to create (default: [1, 7, 365])
        rolling_windows: List of rolling window sizes (default: [7])
        group_cols: Columns to group by (default: ["store", "item"])

    Returns:
        DataFrame with added features:
        - sales_lag_{n}: Sales n days ago
        - roll_mean_{n}: n-day rolling average (with shift to prevent leakage)
        - wow_change: Week-over-week momentum
        - store_daily_avg_lag1: Store-level average from previous day

    Note:
        DataFrame must be sorted by group_cols + ['date'] before calling.
        Early periods will have NaN values due to insufficient history.

    Example:
        >>> df = df.sort_values(['store', 'item', 'date']).reset_index(drop=True)
        >>> df = create_lag_features(df)
        >>> print(f"Created lag features: {[c for c in df.columns if 'lag' in c]}")
    """
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("'date' column must be datetime type")

    # Verify sorting (critical for lag features)
    expected_sort = group_cols + ["date"]
    if not _is_sorted(df, expected_sort):
        raise ValueError(
            f"DataFrame must be sorted by {expected_sort} for lag features. "
            f"Call df.sort_values({expected_sort}).reset_index(drop=True) first."
        )

    # Create lag features per group
    for lag in lag_periods:
        df[f"sales_lag_{lag}"] = df.groupby(group_cols, observed=True)["sales"].shift(lag)

    # Week-over-week momentum (% change)
    if 1 in lag_periods and 7 in lag_periods:
        df["wow_change"] = (
            (df["sales_lag_1"] - df["sales_lag_7"]) / df["sales_lag_7"].replace(0, np.nan)
        )

    # Rolling mean features (with shift to prevent leakage)
    for window in rolling_windows:
        # CRITICAL: shift(1) ensures we only use historical data
        sales_shifted = df.groupby(group_cols, observed=True)["sales"].shift(1)
        roll_mean = (
            sales_shifted
            .groupby([df[col] for col in group_cols], observed=True)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=list(range(len(group_cols))), drop=True)
        )
        df[f"roll_mean_{window}"] = roll_mean

    # Store-level daily average (VECTORIZED - fast!)
    # This replaces the slow .apply() approach in the original code
    df = _create_store_daily_avg_vectorized(df)

    return df


def _is_sorted(df: pd.DataFrame, columns: List[str]) -> bool:
    """Check if DataFrame is sorted by specified columns."""
    # For large dataframes, just check a sample
    sample_size = min(10000, len(df))
    sample_indices = np.linspace(0, len(df) - 1, sample_size, dtype=int)
    sample = df.iloc[sample_indices]

    sorted_sample = sample.sort_values(columns)
    return sample.index.equals(sorted_sample.index)


def _create_store_daily_avg_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create store-level daily average from PREVIOUS day (vectorized).

    This is a MUCH faster replacement for the original .apply() approach.
    Original code took minutes on 913k rows, this takes seconds.

    Original (slow):
        store_daily = df.groupby(["store", "date"])["sales"].mean()
        df["store_daily_avg_lag1"] = df.apply(
            lambda r: store_daily.get((r["store"], r["date"] - pd.Timedelta(days=1)), np.nan),
            axis=1  # ⚠️ Extremely slow!
        )

    New approach (fast):
        1. Compute store-level daily averages
        2. Shift dates forward by 1 day
        3. Merge back to get "yesterday's" average
    """
    # Compute store-level daily average
    store_daily = df.groupby(["store", "date"], observed=True)["sales"].mean().reset_index()
    store_daily.columns = ["store", "date", "store_avg"]

    # Shift date forward by 1 day (so we get "yesterday's" average)
    store_daily["date"] = store_daily["date"] + pd.Timedelta(days=1)

    # Merge back to main dataframe
    df = df.merge(store_daily, on=["store", "date"], how="left")
    df.rename(columns={"store_avg": "store_daily_avg_lag1"}, inplace=True)

    return df
