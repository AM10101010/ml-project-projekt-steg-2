"""Data quality validation functions for sales forecasting"""

import pandas as pd
from typing import Dict, Any


def run_min_checks(df: pd.DataFrame, date_format: str = "%Y-%m-%d") -> Dict[str, Any]:
    """
    Run comprehensive data quality checks on sales data.

    Validates column presence, data types, null values, duplicates,
    cardinality, completeness, and sales value sanity checks.

    Args:
        df: DataFrame with columns ['date', 'store', 'item', 'sales']
        date_format: Expected date format string (default: "%Y-%m-%d")

    Returns:
        Dictionary containing validation results with keys:
        - missing_columns: List of missing required columns
        - type_conversion_nulls: Failed type conversions per column
        - nulls_per_column: Null count per column
        - key_duplicates: Number of duplicate (date, store, item) keys
        - cardinality: Unique counts for stores, items, dates
        - bad_days_count: Days with incomplete records
        - bad_days_examples: Sample of incomplete dates
        - total_rows: Actual row count
        - total_expected: Expected row count (stores × items × dates)
        - total_match: Boolean indicating if counts match
        - sales_checks: Negative values and high outliers

    Example:
        >>> df = pd.read_csv('train.csv')
        >>> issues = run_min_checks(df)
        >>> if issues['key_duplicates'] > 0:
        ...     print("Warning: Duplicate records found!")
    """
    issues = {}

    # Column validation
    required = ["date", "store", "item", "sales"]
    missing_cols = [c for c in required if c not in df.columns]
    issues["missing_columns"] = missing_cols

    # Type conversion with error handling
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], format=date_format, errors="coerce")
    d["store"] = pd.to_numeric(d["store"], errors="coerce")
    d["item"] = pd.to_numeric(d["item"], errors="coerce")
    d["sales"] = pd.to_numeric(d["sales"], errors="coerce")

    type_fail = {
        "date": int(d["date"].isna().sum()),
        "store": int(d["store"].isna().sum()),
        "item": int(d["item"].isna().sum()),
        "sales": int(d["sales"].isna().sum()),
    }
    issues["type_conversion_nulls"] = type_fail

    # Null value detection
    nulls = d.isna().sum().to_dict()
    issues["nulls_per_column"] = {k: int(v) for k, v in nulls.items()}

    # Key uniqueness check
    dup_cnt = int(d.duplicated(["date", "store", "item"]).sum())
    issues["key_duplicates"] = dup_cnt

    # Cardinality validation
    n_stores = int(d["store"].nunique())
    n_items = int(d["item"].nunique())
    n_dates = int(d["date"].nunique())
    issues["cardinality"] = {"stores": n_stores, "items": n_items, "dates": n_dates}

    # Per-day completeness check
    expected_per_day = n_stores * n_items
    per_day = d.groupby("date").size()
    bad_days = per_day[per_day != expected_per_day]
    issues["bad_days_count"] = int(bad_days.size)
    issues["bad_days_examples"] = [str(idx.date()) for idx in bad_days.index[:5]]

    # Volume sanity check
    total_expected = expected_per_day * n_dates
    issues["total_rows"] = int(len(d))
    issues["total_expected"] = int(total_expected)
    issues["total_match"] = (len(d) == total_expected)

    # Sales value validation
    nonneg_fail = int((d["sales"] < 0).sum())
    upper = float(d["sales"].quantile(0.999) * 1.5)
    outliers_high = int((d["sales"] > upper).sum())
    issues["sales_checks"] = {
        "nonneg_fail": nonneg_fail,
        "upper_bound": upper,
        "outliers_high": outliers_high,
    }

    # Print summary
    print("=== Minimal Data Quality Checks ===")
    print("Missing columns:", issues["missing_columns"])
    print("Type conversion nulls:", issues["type_conversion_nulls"])
    print("Nulls per column:", issues["nulls_per_column"])
    print("Key duplicates:", issues["key_duplicates"])
    print("Cardinality:", issues["cardinality"])
    print(f"Per-day expected {expected_per_day}, bad days:", issues["bad_days_count"])
    print("Bad day examples:", issues["bad_days_examples"])
    print("Total rows:", issues["total_rows"], "| Total expected:", issues["total_expected"], "| Match:", issues["total_match"])
    print("Sales checks:", issues["sales_checks"])

    return issues
