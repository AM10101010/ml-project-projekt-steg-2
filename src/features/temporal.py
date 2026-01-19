"""Temporal feature engineering for time-series data"""

import numpy as np
import pandas as pd


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from datetime column.

    Generates both linear time features (day of week, month, etc.) and
    cyclical encodings using sin/cos transformations to capture the
    periodic nature of time (e.g., Sunday is close to Monday, December
    is close to January).

    Args:
        df: DataFrame with 'date' column (must be datetime type)

    Returns:
        DataFrame with added temporal features:
        - dow, month, day, quarter, weekofyear (linear features)
        - is_weekend, is_month_start, is_month_end (binary flags)
        - dow_sin, dow_cos, month_sin, month_cos (cyclical encodings)

    Note:
        Modifies DataFrame in-place and returns it for chaining.

    Example:
        >>> df['date'] = pd.to_datetime(df['date'])
        >>> df = create_temporal_features(df)
        >>> print(df[['date', 'dow', 'dow_sin', 'is_weekend']].head())
    """
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("'date' column must be datetime type. Use pd.to_datetime() first.")

    # Linear time features
    df["dow"] = df["date"].dt.dayofweek  # Monday=0, Sunday=6
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["quarter"] = df["date"].dt.quarter
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

    # Binary temporal flags
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # Cyclical encodings (captures periodic nature of time)
    # This ensures Sunday (6) is numerically close to Monday (0)
    # and December (12) is close to January (1) in the feature space
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df
