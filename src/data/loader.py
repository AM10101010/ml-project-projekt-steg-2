"""Data loading utilities for sales forecasting"""

import pandas as pd
from pathlib import Path


def load_sales_data(
    filepath: str = "Dataset/train.csv",
    parse_dates: bool = True,
    date_format: str = "%Y-%m-%d"
) -> pd.DataFrame:
    """
    Load sales data from CSV file with proper type handling.

    Args:
        filepath: Path to CSV file (default: "Dataset/train.csv")
        parse_dates: Whether to parse date column to datetime (default: True)
        date_format: Date format string for parsing (default: "%Y-%m-%d")

    Returns:
        DataFrame with columns ['date', 'store', 'item', 'sales']

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing

    Example:
        >>> df = load_sales_data()
        >>> print(f"Loaded {len(df):,} records")
        >>> print(df.dtypes)
    """
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Validate required columns
    required_cols = ["date", "store", "item", "sales"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse dates if requested
    if parse_dates:
        df["date"] = pd.to_datetime(df["date"], format=date_format, errors="coerce")

    return df
