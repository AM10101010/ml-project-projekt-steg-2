"""Time-series data splitting utilities"""

import pandas as pd
from typing import Tuple


def temporal_train_val_split(
    df: pd.DataFrame, val_days: int = 90
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time-series data into train and validation sets temporally.

    Uses a temporal cutoff to ensure validation data is strictly AFTER
    training data in time, preventing data leakage. This is critical for
    time-series forecasting where future information must not leak into
    training.

    Args:
        df: DataFrame with 'date' column (must be datetime type)
        val_days: Number of days to use for validation set (default: 90)

    Returns:
        Tuple of (train_df, val_df) where:
        - train_df: All data up to and including cutoff date
        - val_df: Last val_days of data

    Raises:
        KeyError: If 'date' column is missing
        ValueError: If val_days exceeds available data range

    Example:
        >>> df['date'] = pd.to_datetime(df['date'])
        >>> train, val = temporal_train_val_split(df, val_days=90)
        >>> assert train['date'].max() < val['date'].min()  # No temporal overlap
        >>> print(f"Train: {len(train):,} rows, Val: {len(val):,} rows")
    """
    if "date" not in df.columns:
        raise KeyError("DataFrame must contain 'date' column")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("'date' column must be datetime type. Convert with pd.to_datetime() first.")

    date_range = (df["date"].max() - df["date"].min()).days
    if val_days > date_range:
        raise ValueError(
            f"val_days ({val_days}) exceeds available date range ({date_range} days)"
        )

    cutoff = df["date"].max() - pd.Timedelta(days=val_days)
    train = df[df["date"] <= cutoff].copy()
    val = df[df["date"] > cutoff].copy()

    return train, val
